import os
import sqlite3
from datetime import datetime

from flask import (
    Flask, render_template, request,
    redirect, url_for, session, send_from_directory
)

from PIL import Image, ImageEnhance
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from models.generator import Generator

# -------------------------------
# Flask Config
# -------------------------------
app = Flask(__name__)
app.secret_key = "sketch2face_secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
DB_PATH = os.path.join(BASE_DIR, "users.db")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Database Setup
# -------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    """)
    conn.close()

init_db()

# -------------------------------
# Load Generator ONLY
# -------------------------------
GENERATOR_PATH = os.path.join(
    CHECKPOINT_DIR,
    "generator_epoch_500.pth"
)

generator = Generator().to(DEVICE)
generator.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
generator.eval()

# -------------------------------
# Transforms
# -------------------------------
sketch_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def tensor_to_pil(tensor):
    tensor = (tensor.detach().cpu() + 1) / 2
    return transforms.ToPILImage()(tensor)

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("INSERT INTO users VALUES (?,?)", (u, p))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except:
            return render_template("signup.html", error="User already exists")

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form["username"]
        p = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (u, p)
        ).fetchone()
        conn.close()

        if user:
            session["user"] = u
            return redirect(url_for("generate"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route("/generate", methods=["GET", "POST"])
def generate():
    if "user" not in session:
        return redirect(url_for("login"))

    input_img = None
    output_img = None

    if request.method == "POST":
        if "sketch" not in request.files:
            return "Sketch not uploaded", 400

        file = request.files["sketch"]
        skin_factor = float(request.form.get("skin", 1.0))

        sketch = Image.open(file).convert("L")
        sketch_tensor = sketch_transform(sketch).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            fake = generator(sketch_tensor)
            fake = F.interpolate(fake, (256, 256), mode="bilinear")
            fake_pil = tensor_to_pil(fake.squeeze(0))

        fake_pil = ImageEnhance.Brightness(fake_pil).enhance(skin_factor)

        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        input_img = f"input_{ts}.png"
        output_img = f"output_{ts}.png"

        sketch.save(os.path.join(OUTPUT_DIR, input_img))
        fake_pil.save(os.path.join(OUTPUT_DIR, output_img))

    return render_template(
        "generate.html",
        input_img=input_img,
        output_img=output_img
    )

@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
