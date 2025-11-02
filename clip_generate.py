import torch
import clip
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Paths
dataset_dir = "dataset"
output_dir = "embeddings"
os.makedirs(output_dir, exist_ok=True)

embeddings = []

# Iterate over brand folders
for brand in os.listdir(dataset_dir):
    brand_path = os.path.join(dataset_dir, brand)
    if not os.path.isdir(brand_path):
        continue
    
    print(f"Processing brand: {brand}")
    for img_name in tqdm(os.listdir(brand_path)):
        img_path = os.path.join(brand_path, img_name)
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embeddings.append({
                "brand": brand,
                "image": img_name,
                "embedding": image_features.cpu().numpy().flatten().tolist()
            })
        except Exception as e:
            print(f"Error with {img_path}: {e}")

# Save embeddings
df = pd.DataFrame(embeddings)
df.to_pickle(os.path.join(output_dir, "clip_embeddings.pkl"))

print("✅ Embeddings generated and saved!")
