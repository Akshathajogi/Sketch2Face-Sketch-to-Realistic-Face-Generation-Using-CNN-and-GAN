import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# ----------------------------------
# Image Transformations
# ----------------------------------
def get_sketch_transforms():
    """
    Transformations for sketch images (grayscale)
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_photo_transforms():
    """
    Transformations for RGB photos
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])


# ----------------------------------
# Custom Dataset Class
# ----------------------------------
class SketchToImageDataset(Dataset):
    """
    Strict paired dataset for Sketch-to-Image conversion
    Expected structure:
    data/train/sketches
    data/train/photos
    """

    def __init__(self, root_dir, mode="train"):
        self.sketch_dir = os.path.join(root_dir, mode, "sketches")
        self.photo_dir = os.path.join(root_dir, mode, "photos")

        self.sketch_files = sorted(os.listdir(self.sketch_dir))

        # Ensure exact filename matching
        self.photo_files = self.sketch_files

        self.sketch_transform = get_sketch_transforms()
        self.photo_transform = get_photo_transforms()

    def __len__(self):
        return len(self.sketch_files)

    def __getitem__(self, idx):
        sketch_name = self.sketch_files[idx]

        sketch_path = os.path.join(self.sketch_dir, sketch_name)
        photo_path = os.path.join(self.photo_dir, sketch_name)

        sketch = Image.open(sketch_path).convert("L")
        photo = Image.open(photo_path).convert("RGB")

        sketch = self.sketch_transform(sketch)
        photo = self.photo_transform(photo)

        return sketch, photo


# ----------------------------------
# DataLoader Helper
# ----------------------------------
def get_dataloader(data_dir, mode="train", batch_size=8, shuffle=True):
    """
    Returns DataLoader for given mode
    """
    dataset = SketchToImageDataset(data_dir, mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if mode == "train" else False,
        num_workers=2,
        pin_memory=True
    )


# ----------------------------------
# Save Generated Images
# ----------------------------------
def save_image(tensor, path):
    """
    Save a generated image tensor to disk
    """
    tensor = tensor.detach().cpu()
    tensor = (tensor + 1) / 2  # [-1,1] → [0,1]
    tensor = torch.clamp(tensor, 0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(path)
