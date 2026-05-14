import kagglehub
import os
import torch
import open_clip
from PIL import Image
import matplotlib.pyplot as plt
import csv

# ── 1. Load CLIP model ────────────────────────────────────────────────────────
print("Loading CLIP model...")
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()
print("Model loaded!")

# ── 2. Load dataset ───────────────────────────────────────────────────────────
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
images_path = os.path.join(path, "images")

# Load styles.csv so we know what each image actually is
styles = {}
with open(os.path.join(path, "styles.csv"), newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        styles[row['id']] = row

# ── 3. Text descriptions from actual dataset categories ───────────────────────
text_descriptions = [
    "a shirt", "a t-shirt", "jeans", "trousers", "a skirt",
    "a dress", "a jacket", "a sweater", "sports shoes", "casual shoes",
    "a watch", "a handbag", "shorts", "a saree", "a kurta"
]
text = tokenizer(text_descriptions)

# ── 4. Test on 6 images ───────────────────────────────────────────────────────
sample_images = os.listdir(images_path)[:6]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for i, img_file in enumerate(sample_images):
    img_id = img_file.replace(".jpg", "")
    img_path = os.path.join(images_path, img_file)

    # What styles.csv says this item actually is
    actual = styles.get(img_id, {})
    actual_type = actual.get('articleType', 'Unknown')
    actual_colour = actual.get('baseColour', '')

    # Run CLIP
    image = preprocess_val(Image.open(img_path)).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Top prediction
    top_value, top_index = similarity[0].topk(1)
    predicted = text_descriptions[top_index[0]]
    confidence = top_value[0].item()

    print(f"\nImage: {img_file}")
    print(f"  Actual:    {actual_colour} {actual_type}")
    print(f"  Predicted: {predicted} ({confidence:.1f}%)")

    # Show image
    img = Image.open(img_path)
    axes[i].imshow(img)
    axes[i].set_title(f"Actual: {actual_type}\nCLIP: {predicted} ({confidence:.1f}%)", fontsize=8)
    axes[i].axis("off")

plt.tight_layout()
plt.show()