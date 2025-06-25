import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from siamese_model import SiameseNetwork

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def get_embedding(model, device, image_path, transform):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.forward_once(img_tensor)
    return embedding.cpu().numpy().flatten()

def gather_identity_images(identity_dir):
    image_paths = []
    for root, _, files in os.walk(identity_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, f))
    return image_paths

def main(test_image_path, identity_dir, model_path, threshold=0.9):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    model, device = load_model(model_path)

    # Get embedding for test image
    test_emb = get_embedding(model, device, test_image_path, transform)
    test_emb = normalize(test_emb.reshape(1, -1))[0]

    # Get embeddings for identity images
    identity_image_paths = gather_identity_images(identity_dir)
    if len(identity_image_paths) == 0:
        print("No identity images found in the specified directory.")
        return

    identity_embs = []
    for img_path in identity_image_paths:
        emb = get_embedding(model, device, img_path, transform)
        identity_embs.append(emb)
    identity_embs = normalize(np.array(identity_embs))

    # Compute cosine similarities
    sims = cosine_similarity(test_emb.reshape(1, -1), identity_embs)[0]

    # Determine if any similarity exceeds threshold
    max_sim_idx = np.argmax(sims)
    max_sim = sims[max_sim_idx]
    if max_sim >= threshold:
        label = 1
        matched_identity = os.path.basename(os.path.dirname(identity_image_paths[max_sim_idx]))
        print(f"Positive match (label=1). Best matching identity folder: {matched_identity} with similarity {max_sim:.4f}")
    else:
        label = 0
        print(f"Negative match (label=0). No identity matched above threshold {threshold}.")

    return label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face verification test script")
    parser.add_argument("test_image", help="Path to the test image")
    parser.add_argument("identity_dir", help="Path to the directory containing identity folders")
    parser.add_argument("model_path", help="Path to the trained Siamese model file")
    parser.add_argument("--threshold", type=float, default=0.9, help="Similarity threshold for positive match")
    args = parser.parse_args()

    main(args.test_image, args.identity_dir, args.model_path, args.threshold)
