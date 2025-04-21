import os, sys, argparse
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

# allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.generator import Generator
from scripts.discriminator import Discriminator
from scripts.data_loader import get_dataloader
from visualize import show_image_batch

def main():
    # ----------------------------
    # ▶︎ Parse command‑line args
    # ----------------------------
    parser = argparse.ArgumentParser(
        description="Train DCGAN: python scripts/train.py --data_dir PATH"
    )
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Path to folder of training images (no subfolders required)'
    )
    args = parser.parse_args()
    image_dir = args.data_dir
    if not os.path.isdir(image_dir):
        print(f"Error: --data_dir '{image_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(3)

    # Hyperparameters  # adjust as needed
    batch_size  = 128
    image_size  = 128
    nz          = 100
    ngf, ndf    = 64, 64
    num_epochs  = 100
    lr          = 0.0002
    beta1       = 0.5
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create models
    netG = Generator(nz=nz, ngf=ngf).to(device)
    netD = Discriminator(ndf=ndf).to(device)

    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if "Conv" in classname:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if "BatchNorm" in classname:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss and optimizers
    criterion = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr*0.02, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr,     betas=(beta1, 0.999))

    # Data loader
    dataloader = get_dataloader(image_dir, batch_size)

    # Checkpoint dir
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    pathG = os.path.join(model_dir, "netG.pth")
    pathD = os.path.join(model_dir, "netD.pth")

    # Training loop
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    for epoch in range(num_epochs):
        for i, (real, _) in enumerate(dataloader):
            real = real.to(device)
            bsz  = real.size(0)

            # Train Discriminator
            netD.zero_grad()
            # real
            label = 0.9 + 0.1 * torch.rand(bsz, device=device)
            outD_real = netD(real)
            errD_real = criterion(outD_real, label)
            errD_real.backward()
            # fake
            noise = torch.randn(bsz, nz, 1, 1, device=device)
            fake  = netG(noise)
            label.fill_(0.1)
            outD_fake = netD(fake.detach())
            errD_fake = criterion(outD_fake, label)
            errD_fake.backward()
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            label.fill_(0.9)
            outD = netD(fake)
            errG = criterion(outD, label)
            errG.backward()
            optimizerG.step()

            if i % 2 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] "
                    f"Loss_D: {(errD_real+errD_fake).item():.4f} "
                    f"Loss_G: {errG.item():.4f}")

        # save checkpoint
        torch.save(netG.state_dict(), pathG)
        torch.save(netD.state_dict(), pathD)

        # optional: visualize progress
        with torch.no_grad():
            samples = netG(fixed_noise).cpu()
        show_image_batch(samples, title=f"Epoch {epoch}")

    print("Training complete!")

if __name__ == '__main__':
    freeze_support()   # no-op on Unix, required on Windows spawn
    main()