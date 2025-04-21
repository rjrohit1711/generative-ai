import torch.nn as nn

class Discriminator(nn.Module):
    """
    DCGAN-style discriminator for 128×128 input.
    """
    def __init__(self, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            # Input 3×128×128 → 64×64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # → 32×32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            # → 16×16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            # → 8×8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, inplace=True),
            # → 4×4
            nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*16), nn.LeakyReLU(0.2, inplace=True),
            # → 1×1 (logit output)
            nn.Conv2d(ndf*16, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.main(x).view(-1)
