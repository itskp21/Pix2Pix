import os
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        # Assuming the image is split into gray and color half
        width, height = image.size
        gray_image = image.crop((0, 0, width // 2, height))
        color_image = image.crop((width // 2, 0, width, height))

        # Convert to numpy arrays
        gray_image_np = np.array(gray_image)
        color_image_np = np.array(color_image)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=gray_image_np, image0=color_image_np)
            gray_image = transformed['image']
            color_image = transformed['image0']
        else:
            gray_image = ToTensorV2()(image=gray_image_np)['image']
            color_image = ToTensorV2()(image=color_image_np)['image']

        return gray_image, color_image
    
both_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    A.Resize(256, 256),  # Resize to match the input size of the model
    ToTensorV2()
])
