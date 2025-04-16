import os
from PIL import Image
from torch.utils.data import Dataset

class celeba_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (e.g., 'data/celeba/img_align_celeba').
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.image_files = sorted([
            file for file in os.listdir(root_dir)
            if file.lower().endswith(('.jpg', '.png'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Construct the full path to the image file
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        # Open the image file and convert to RGB
        image = Image.open(img_path).convert("RGB")
        # Apply any defined transformations
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return a dummy label (0)