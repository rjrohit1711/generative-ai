import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, image_size=128):
        self.image_paths = [
            os.path.join(image_dir, fn)
            for fn in os.listdir(image_dir)
            if fn.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img), 0  # dummy label

def get_dataloader(image_dir, batch_size=128):
    dataset = CustomImageDataset(image_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
