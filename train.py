import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from models.generator import Generator
from models.discriminator import Discriminator

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data"
EPOCHS = 300
BATCH_SIZE = 8
LR = 0.0001
LAMBDA_L1 = 50          # 🔥 Reduced for sharper output
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "checkpoints"
SAMPLE_DIR = "outputs/comparison"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# -----------------------------
# Transforms
# -----------------------------
sketch_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])   # Grayscale: [-1,1]
])

photo_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB: [-1,1]
])

# -----------------------------
# Dataset
# -----------------------------
class SketchPhotoDataset(Dataset):
    def __init__(self, photo_dir, sketch_dir, sketch_transform=None, photo_transform=None):
        self.photo_dir = photo_dir
        self.sketch_dir = sketch_dir
        self.sketch_transform = sketch_transform
        self.photo_transform = photo_transform

        # Map IDs to files
        self.photos_dict = {f.split('_')[0]: f for f in os.listdir(photo_dir) if f.endswith(".jpg")}
        self.sketches_dict = {f.split('_')[0]: f for f in os.listdir(sketch_dir) if f.endswith(".jpg")}

        # Keep only IDs present in both
        self.ids = sorted(list(set(self.photos_dict.keys()) & set(self.sketches_dict.keys())))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        photo_path = os.path.join(self.photo_dir, self.photos_dict[id_])
        sketch_path = os.path.join(self.sketch_dir, self.sketches_dict[id_])

        photo = Image.open(photo_path).convert("RGB")
        sketch = Image.open(sketch_path).convert("L")  # Grayscale

        if self.photo_transform:
            photo = self.photo_transform(photo)
        if self.sketch_transform:
            sketch = self.sketch_transform(sketch)

        return sketch, photo  # Tensors

# -----------------------------
# DataLoader
# -----------------------------
def get_dataloader(data_dir, mode="train", batch_size=8, shuffle=True):
    photo_dir = os.path.join(data_dir, mode, "photos")
    sketch_dir = os.path.join(data_dir, mode, "sketches")

    dataset = SketchPhotoDataset(photo_dir, sketch_dir, sketch_transform=sketch_transform, photo_transform=photo_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return loader

train_loader = get_dataloader(DATA_DIR, mode="train", batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# Save image helper
# -----------------------------
def save_image(tensor, path):
    tensor = tensor.cpu().detach()
    tensor = tensor * 0.5 + 0.5  # [-1,1] -> [0,1]

    if tensor.shape[0] == 1:
        # Grayscale
        image = tensor.squeeze(0).numpy()  # H,W
        image = (image * 255).astype('uint8')
        Image.fromarray(image, mode='L').save(path)
    else:
        # RGB
        image = tensor.permute(1, 2, 0).numpy()  # C,H,W -> H,W,C
        image = (image * 255).astype('uint8')
        Image.fromarray(image).save(path)

# -----------------------------
# Initialize Models
# -----------------------------
generator = Generator(in_channels=1, out_channels=3).to(DEVICE)
discriminator = Discriminator(sketch_channels=1, image_channels=3).to(DEVICE)

# -----------------------------
# Loss Functions
# -----------------------------
adversarial_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

# -----------------------------
# Optimizers
# -----------------------------
optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):
    generator.train()
    discriminator.train()

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for sketches, real_images in loop:
        sketches = sketches.to(DEVICE)
        real_images = real_images.to(DEVICE)

        # -----------------------------
        # Train Discriminator
        # -----------------------------
        with torch.no_grad():
            fake_images = generator(sketches)

        optimizer_D.zero_grad()
        pred_real = discriminator(sketches, real_images)
        pred_fake = discriminator(sketches, fake_images)

        real_labels = torch.ones_like(pred_real)
        fake_labels = torch.zeros_like(pred_fake)

        d_loss_real = adversarial_loss(pred_real, real_labels)
        d_loss_fake = adversarial_loss(pred_fake, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        optimizer_D.step()

        # -----------------------------
        # Train Generator
        # -----------------------------
        optimizer_G.zero_grad()
        fake_images = generator(sketches)
        pred_fake = discriminator(sketches, fake_images)

        adv_loss = adversarial_loss(pred_fake, real_labels)
        pixel_loss = l1_loss(fake_images, real_images)
        g_loss = adv_loss + LAMBDA_L1 * pixel_loss
        g_loss.backward()
        optimizer_G.step()

        loop.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

    # -----------------------------
    # Save Checkpoints + Samples
    # -----------------------------
    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(),
                   os.path.join(CHECKPOINT_DIR, f"generator_epoch{epoch+1}.pth"))
        torch.save(discriminator.state_dict(),
                   os.path.join(CHECKPOINT_DIR, f"discriminator_epoch{epoch+1}.pth"))

        generator.eval()
        with torch.no_grad():
            sketches_sample, real_images_sample = next(iter(train_loader))
            sketches_sample = sketches_sample.to(DEVICE)
            real_images_sample = real_images_sample.to(DEVICE)

            fake_sample = generator(sketches_sample)

            for i in range(min(4, sketches_sample.size(0))):
                save_image(sketches_sample[i],
                           os.path.join(SAMPLE_DIR, f"epoch{epoch+1}_sketch{i+1}.png"))
                save_image(fake_sample[i],
                           os.path.join(SAMPLE_DIR, f"epoch{epoch+1}_generated{i+1}.png"))
                save_image(real_images_sample[i],
                           os.path.join(SAMPLE_DIR, f"epoch{epoch+1}_real{i+1}.png"))

print("Training completed successfully.")
