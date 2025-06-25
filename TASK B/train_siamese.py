import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import random
from siamese_model import SiameseNetwork
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Config
train_dir = "C:/workspace-python/hackathon-jadavpur/siamese_face/train"
val_dir = "C:/workspace-python/hackathon-jadavpur/siamese_face/val"
batch_size = 16
num_epochs = 10  # Increased epochs for longer training
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset class
class TripletFaceDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.triplets = []
        self.label_map = {}

        # Gather all identities
        self.identities = [d for d in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, d))]

        for idx, identity in enumerate(self.identities):
            self.label_map[identity] = idx

        # Build triplets (anchor, positive, negative)
        for identity in self.identities:
            id_path = os.path.join(root_dir, identity)
            clean_img = None
            for f in os.listdir(id_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    clean_img = os.path.join(id_path, f)
                    break
            if not clean_img:
                continue

            distortion_path = os.path.join(id_path, "distortion")
            if not os.path.isdir(distortion_path):
                continue

            distorted_imgs = [
                os.path.join(distortion_path, f)
                for f in os.listdir(distortion_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            # Create triplets
            for pos_img in distorted_imgs:
                neg_identity = random.choice(
                    [i for i in self.identities if i != identity])
                neg_path = os.path.join(root_dir, neg_identity)
                neg_clean = None
                for f in os.listdir(neg_path):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        neg_clean = os.path.join(neg_path, f)
                        break
                if neg_clean:
                    self.triplets.append((clean_img, pos_img, neg_clean))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, pos_path, neg_path = self.triplets[idx]
        anchor_img = Image.open(anchor_path).convert("RGB")
        pos_img = Image.open(pos_path).convert("RGB")
        neg_img = Image.open(neg_path).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img

def evaluate(model, data_loader, criterion, device, threshold=1.0):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for anchor, positive, negative in data_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_emb = model.forward_once(anchor)
            positive_emb = model.forward_once(positive)
            negative_emb = model.forward_once(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()

            # Calculate distances
            pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb)
            neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb)

            # Labels: 1 for positive pair, 0 for negative pair
            labels = torch.cat([torch.ones(pos_dist.size(0)), torch.zeros(neg_dist.size(0))]).cpu().numpy()
            # Predictions based on threshold
            preds = torch.cat([(pos_dist < threshold).float(), (neg_dist > threshold).float()]).cpu().numpy()

            all_labels.extend(labels)
            all_preds.extend(preds)

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return avg_loss, accuracy, precision, recall, f1

# Load Datasets
train_dataset = TripletFaceDataset(train_dir, transform=transform)
val_dataset = TripletFaceDataset(val_dir, transform=transform)

if len(train_dataset) == 0:
    raise ValueError("[❌] No training data found. Check your folder structure.")
if len(val_dataset) == 0:
    raise ValueError("[❌] No validation data found. Check your folder structure.")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = SiameseNetwork().to(device)
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
print(f"[INFO] Training on {len(train_dataset)} triplets...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        anchor_emb = model.forward_once(anchor)
        positive_emb = model.forward_once(positive)
        negative_emb = model.forward_once(negative)

        loss = criterion(anchor_emb, positive_emb, negative_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print(f"[DEBUG] Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    train_loss, train_acc, train_prec, train_rec, train_f1 = evaluate(model, train_loader, criterion, device)
    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1 Score: {train_f1:.4f}")
    print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1 Score: {val_f1:.4f}")

# Save Model
save_dir = "C:/workspace-python/hackathon-jadavpur/try/saved_models"
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, "siamese_face_matcher.pth"))
print("[✅] Model saved to 'siamese_face/saved_models/siamese_face_matcher.pth'")
