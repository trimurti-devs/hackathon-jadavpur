import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from model_gender import GenderClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Config
train_dir = "hackathon-jadavpur/gender_classification/train"
val_dir = "hackathon-jadavpur/gender_classification/val"
batch_size = 16
num_epochs = 10
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Custom Dataset
class GenderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for gender_label, gender_name in enumerate(['female', 'male']):  # 0: female, 1: male
            gender_folder = os.path.join(root_dir, gender_name)
            if not os.path.isdir(gender_folder):
                continue
            # Check if images are directly inside gender_folder
            direct_images = [f for f in os.listdir(gender_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if direct_images:
                for img_file in direct_images:
                    img_path = os.path.join(gender_folder, img_file)
                    self.samples.append((img_path, gender_label))
            else:
                for person in os.listdir(gender_folder):
                    person_folder = os.path.join(gender_folder, person)
                    if not os.path.isdir(person_folder):
                        continue
                    for img_file in os.listdir(person_folder):
                        if img_file.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(person_folder, img_file)
                            self.samples.append((img_path, gender_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Load Datasets
train_dataset = GenderDataset(train_dir, transform=transform)
val_dataset = GenderDataset(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load Model
model = GenderClassifier().to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(all_labels, all_preds)
    train_prec = precision_score(all_labels, all_preds, zero_division=0)
    train_rec = recall_score(all_labels, all_preds, zero_division=0)
    train_f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, "
          f"Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1-score: {train_f1:.4f}")

    # Validation Loop
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_prec = precision_score(val_labels, val_preds, zero_division=0)
    val_rec = recall_score(val_labels, val_preds, zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, zero_division=0)

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, "
          f"Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1-score: {val_f1:.4f}")

# Save Model
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "hackathon-jadavpur/gender_classification/saved_models/gender_classifier.pth")
print("Model saved successfully to saved_models/gender_classifier.pth")
