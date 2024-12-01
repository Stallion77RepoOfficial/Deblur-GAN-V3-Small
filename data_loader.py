import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class GoProDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.blur_images = []
        self.sharp_images = []
        self.transform = transform

        scenes = os.listdir(root_dir)
        for scene in scenes:
            scene_path = os.path.join(root_dir, scene)
            if not os.path.isdir(scene_path):
                continue  # Eğer scene bir dizin değilse, atla

            blur_dir = os.path.join(scene_path, 'blur')
            sharp_dir = os.path.join(scene_path, 'sharp')

            if not os.path.isdir(blur_dir) or not os.path.isdir(sharp_dir):
                continue  # Eğer blur veya sharp dizinleri yoksa, atla

            blur_files = sorted(os.listdir(blur_dir))
            sharp_files = sorted(os.listdir(sharp_dir))

            for b, s in zip(blur_files, sharp_files):
                self.blur_images.append(os.path.join(blur_dir, b))
                self.sharp_images.append(os.path.join(sharp_dir, s))

    def __len__(self):
        return len(self.blur_images)

    def __getitem__(self, idx):
        blur_image = Image.open(self.blur_images[idx]).convert('RGB')
        sharp_image = Image.open(self.sharp_images[idx]).convert('RGB')

        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)

        return blur_image, sharp_image
