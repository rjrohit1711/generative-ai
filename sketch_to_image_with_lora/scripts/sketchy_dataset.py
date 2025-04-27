# image_generation_ai/datasets/sketchy_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SketchyDataset(Dataset):
    def __init__(self, sketch_root, photo_root, transform=None, image_size=256):
        """
        Args:
            sketch_root (str):
                e.g. ".../SketchyDataset/sketch/tx_000000000000"
            photo_root (str):
                e.g. ".../SketchyDataset/photo/tx_000000000000"
            transform (callable): applied to both sketch & photo
            image_size (int): final H/W
        """
        self.sketch_root = sketch_root
        self.photo_root = photo_root
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.data_pairs = self._make_pairs()
        if not self.data_pairs:
            raise RuntimeError(
                f"No sketch–photo pairs found under:\n"
                f"  sketches: {sketch_root}\n"
                f"  photos:   {photo_root}"
            )

    def _make_pairs(self):
        pairs = []
        # iterate over each class folder in sketch_root
        for class_name in os.listdir(self.sketch_root):
            sketch_class_dir = os.path.join(self.sketch_root, class_name)
            photo_class_dir  = os.path.join(self.photo_root,  class_name)
            if not (os.path.isdir(sketch_class_dir) and os.path.isdir(photo_class_dir)):
                continue

            # for each sketch file in that class
            for sketch_fname in os.listdir(sketch_class_dir):
                if not sketch_fname.lower().endswith(".png"):
                    continue

                # derive base name before the final dash
                base = sketch_fname.rsplit("-", 1)[0]  
                sketch_path = os.path.join(sketch_class_dir, sketch_fname)

                # look for photo match in the same class folder
                for ext in (".jpg", ".jpeg"):
                    photo_fname = base + ext
                    photo_path = os.path.join(photo_class_dir, photo_fname)
                    if os.path.isfile(photo_path):
                        pairs.append((sketch_path, photo_path))
                        break

        return pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        sketch_path, photo_path = self.data_pairs[idx]
        # load images
        sketch = Image.open(sketch_path).convert("RGB")
        photo  = Image.open(photo_path).convert("RGB")
        # apply transforms
        sketch = self.transform(sketch)
        photo  = self.transform(photo)
        return {"sketch": sketch, "photo": photo}
