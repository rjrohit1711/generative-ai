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

if __name__ == '__main__':
    # Ensure project root is in sys.path for module imports if needed
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   
    netG = Generator().to(device) 
    if os.path.exists(netG_path):
        netG.load_state_dict(torch.load(netG_path))
        print("Loaded saved models.")
    netG.eval()  # Set generator to evaluation mode
    num_images = 64
    save_path = "generated_images"
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for i in range(num_images // 64):  # Assuming batch size 64
            noise = torch.randn(64, nz, 1, 1, device=device)
            fake_images = netG(noise).detach().cpu()
            for j in range(fake_images.size(0)):
                img = fake_images[j]
                vutils.save_image(img, os.path.join(save_path, f"image_{i*64+j:05d}.png"), normalize=True)
