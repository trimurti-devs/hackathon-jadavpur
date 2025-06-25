import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import classification_report
from PIL import Image
from model_gender import GenderClassifier

# Config
val_dir = "C:/workspace-python/hackathon-jadavpur/gender_classification/val"
model_path = "C:/workspace-python/hackathon-jadavpur/gender_classification/saved_models/gender_classifier.pth"
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Custom Dataset Class for flat folder structure
class GenderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for gender_name in os.listdir(root_dir):
            gender_folder = os.path.join(root_dir, gender_name)
            if not os.path.isdir(gender_folder):
                continue

            label = 0 if gender_name.lower() == 'female' else 1

            for img_file in os.listdir(gender_folder):
                img_path = os.path.join(gender_folder, img_file)
                if os.path.isfile(img_path) and img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Load Validation Data
val_dataset = GenderDataset(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"[INFO] Validation samples loaded: {len(val_dataset)}")

# Exit early if dataset is empty
if len(val_dataset) == 0:
    print("[ERROR] No validation samples found. Check folder paths and structure.")
    exit()

# Load Model
model = GenderClassifier().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Evaluate
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Report
if len(all_labels) == 0:
    print("[ERROR] No predictions were made.")
else:
    print("📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['female', 'male']))
