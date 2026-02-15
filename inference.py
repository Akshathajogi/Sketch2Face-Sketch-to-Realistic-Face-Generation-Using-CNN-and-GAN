import os
import torch
from PIL import Image
import torchvision.transforms as transforms

from models.generator import Generator
from utils import save_image


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "checkpoints"
INPUT_DIR = "data/test/sketches"
OUTPUT_DIR = "outputs/generated"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Load Generator
# -----------------------------
generator = Generator(in_channels=1, out_channels=3).to(DEVICE)

generator.load_state_dict(
    torch.load(os.path.join(CHECKPOINT_DIR, "generator.pth"),
               map_location=DEVICE)
)

generator.eval()


# -----------------------------
# Image Transform
# -----------------------------
sketch_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# -----------------------------
# Inference
# -----------------------------
with torch.no_grad():
    for file_name in os.listdir(INPUT_DIR):
        if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        sketch_path = os.path.join(INPUT_DIR, file_name)
        sketch = Image.open(sketch_path).convert("L")
        sketch = sketch_transform(sketch).unsqueeze(0).to(DEVICE)

        generated_image = generator(sketch)

        output_path = os.path.join(OUTPUT_DIR, file_name)
        save_image(generated_image[0], output_path)

        print(f"Generated image saved at: {output_path}")

print("Inference completed successfully.")
