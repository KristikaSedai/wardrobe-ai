import kagglehub
import os
import csv
from PIL import Image
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")

# ── 1. Check contents of myntradataset folder ─────────────────────────────────
print("=== MYNTRADATASET FOLDER ===")
myntra_path = os.path.join(path, "myntradataset")
for item in os.listdir(myntra_path):
    print(item)

# ── 2. Read styles.csv ────────────────────────────────────────────────────────
print("\n=== STYLES.CSV ===")
styles_path = os.path.join(path, "styles.csv")
with open(styles_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total items: {len(rows)}")
print(f"Columns: {list(rows[0].keys())}")
print("\nFirst 3 rows:")
for row in rows[:3]:
    print(row)

# ── 3. Display sample images ──────────────────────────────────────────────────
print("\n=== SAMPLE IMAGES ===")
images_path = os.path.join(path, "images")
sample_images = os.listdir(images_path)[:6]
print(f"Total images: {len(os.listdir(images_path))}")

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
axes = axes.flatten()

for i, img_file in enumerate(sample_images):
    img = Image.open(os.path.join(images_path, img_file))
    axes[i].imshow(img)
    axes[i].set_title(img_file)
    axes[i].axis("off")

plt.tight_layout()
plt.show()