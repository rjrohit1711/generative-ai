import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import os
import sys

# Hyperparameters
batch_size = 128
nz = 100                    # Size of latent vector (input to generator)
ngf = 64                    # Size of feature maps in the generator
ndf = 64                    # Size of feature maps in the discriminator
num_epochs = 1000
lr = 0.0002
beta1 = 0.4                 # Beta1 hyperparameter for Adam optimizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory for saving/loading models
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
netG_path = os.path.join(model_dir, 'netG.pth')
netD_path = os.path.join(model_dir, 'netD.pth')

# Generator Model – outputs 128x128 images
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: Z → (ngf*8 x 4 x 4)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4 → (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8 → (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16 → (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf) x 32 x 32 → (ngf//2) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            # (ngf//2) x 64 x 64 → (3) x 128 x 128
            nn.ConvTranspose2d(ngf // 2, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

# Discriminator Model – accepts 128x128 images
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (3) x 128 x 128 → output: 64x64 feature maps
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),  
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64 → 32x32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 → 16x16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 → 8x8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 → 4x4
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),  
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4 → 1x1, output raw logits (no Sigmoid as we use BCEWithLogitsLoss)
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
        )
    def forward(self, input):
        return self.main(input).view(-1)

# Weight initialization per DCGAN paper guidelines
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':
    # Ensure project root is in sys.path for module imports if needed
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Initialize models
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Load saved models if available
    if os.path.exists(netG_path) and os.path.exists(netD_path):
        netG.load_state_dict(torch.load(netG_path))
        netD.load_state_dict(torch.load(netD_path))
        print("Loaded saved models.")

    print("Generator Architecture:\n", netG)
    print("\nDiscriminator Architecture:\n", netD)

    # Loss function using raw logits; BCEWithLogitsLoss applies Sigmoid internally
    criterion = nn.BCEWithLogitsLoss()

    # Fixed noise vector for visualizing generator progress
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Setup optimizers (further reduced lr for discriminator if needed)
    optimizerD = optim.Adam(netD.parameters(), lr=lr * 0.01, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Import your data loader (make sure its transforms resize to 128x128)
    from scripts.data_loader import get_dataloader
    dataloader = get_dataloader(batch_size=batch_size)

    # from scripts.custom_dataset import get_custom_dataloader
    # image_dir = r"C:\Users\rjroh\Desktop\Rohit M\joshi g-20241226T125703Z-011\joshi g\data"
    # dataloader = get_custom_dataloader(image_dir)

    # Create directory to save generated images
    os.makedirs("output_images", exist_ok=True)

    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            ############################
            # (1) Update Discriminator
            ############################
            netD.zero_grad()
            real_images = real_images.to(device)
            b_size = real_images.size(0)
            # Label smoothing: random values between 0.7 and 1.0 for real labels
            label = 0.6 + 0.4 * torch.rand(b_size, device=device)
            output = netD(real_images)
            errD_real = criterion(output, label)
            errD_real.backward()

            # Generate fake images and compute discriminator loss for fakes
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(0.1)  # Fake labels
            output = netD(fake_images.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()

            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update Generator
            ############################
            # Update generator three times
            for _ in range(3):
                netG.zero_grad()
                label.fill_(1.0)  # the generator wants discriminator to see fakes as real
                # Regenerate noise and fake images if needed (this creates a fresh graph each time)
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake_images = netG(noise)
                output = netD(fake_images)
                errG = criterion(output, label)
                errG.backward()
                optimizerG.step()

            print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")

        # Save visualization images after each epoch
        with torch.no_grad():
            fake_images = netG(fixed_noise).detach().cpu()
        grid = vutils.make_grid(fake_images, padding=2, normalize=True)
        vutils.save_image(grid, f"output_images/epoch_{epoch:03d}.png")
        
        # Save model checkpoints
        torch.save(netG.state_dict(), netG_path)
        torch.save(netD.state_dict(), netD_path)
        print(f"Saved models at epoch {epoch}.")

    print("Training complete!")
