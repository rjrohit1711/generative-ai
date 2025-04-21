import os, sys, torch
# allow imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.generator import Generator
from visualize import show_image_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(model_path="models/netG.pth", nz=100, num=64):
    netG = Generator(nz=nz).to(device)
    state = torch.load(model_path, map_location=device)
    netG.load_state_dict(state)
    netG.eval()

    noise = torch.randn(num, nz, 1, 1, device=device)
    with torch.no_grad():
        imgs = netG(noise).cpu()
    show_image_batch(imgs, title="Generated Samples")

if __name__ == "__main__":
    generate()
