import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from sketchy_dataset import SketchyDataset

# 🛠️ Define dataset path
sketch_dir = r"C:\Users\rjroh\Desktop\Projects\Dataset\SketchyDataset\sketch\tx_000000000000"
photo_dir = r"C:\Users\rjroh\Desktop\Projects\Dataset\SketchyDataset\photo\tx_000000000000"

# 🧾 Create dataset & loader
dataset = SketchyDataset(sketch_root=sketch_dir, photo_root=photo_dir)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 🖼️ Fetch a batch
batch = next(iter(loader))
sketches = batch['sketch']
photos = batch['photo']

# 📊 Plot
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(6, 12))
for i in range(4):
    # Convert back from [-1, 1] to [0, 1]
    sketch_img = sketches[i].squeeze().detach().cpu().numpy() * 0.5 + 0.5
    photo_img = photos[i].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5

    axes[i][0].imshow(sketch_img, cmap='gray')
    axes[i][0].set_title("Sketch")
    axes[i][0].axis('off')

    axes[i][1].imshow(photo_img)
    axes[i][1].set_title("Photo")
    axes[i][1].axis('off')

plt.tight_layout()
plt.show()
