import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from train_siamese import TripletFaceDataset, transform
from siamese_model import SiameseNetwork

def evaluate(model, data_loader, criterion, device, threshold=1.0):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for anchor, positive, negative in data_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Get embeddings
            anchor_emb = model.forward_once(anchor)
            positive_emb = model.forward_once(positive)
            negative_emb = model.forward_once(negative)

            # Triplet loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()

            # Distance between anchor-positive and anchor-negative
            pos_dist = F.pairwise_distance(anchor_emb, positive_emb)
            neg_dist = F.pairwise_distance(anchor_emb, negative_emb)

            # Predict: correct if pos_dist < neg_dist
            preds = (pos_dist < neg_dist).int()
            labels = torch.ones_like(preds)  # expected outcome

            correct += (preds == labels).sum().item()
            total += preds.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total if total > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return avg_loss, accuracy, precision, recall, f1

def main():
    # Parameters
    val_dir = "C:/workspace-python/hackathon-jadavpur/siamese_face/val"
    model_path = "C:/workspace-python/hackathon-jadavpur/try/siamese_face_matcher.pth"
    batch_size = 16
    threshold = 1.0
    margin = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation dataset
    val_dataset = TripletFaceDataset(val_dir, transform=transform)
    if len(val_dataset) == 0:
        raise ValueError("[❌] No validation data found. Check your folder structure.")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Loss function
    criterion = nn.TripletMarginLoss(margin=margin, p=2)

    # Evaluate
    avg_loss, accuracy, precision, recall, f1 = evaluate(model, val_loader, criterion, device, threshold)

    # Print results
    print("Evaluation Results:")
    print(f"{'Metric':<10} | {'Value':<10}")
    print("-" * 25)
    print(f"{'Loss':<10} | {avg_loss:<10.4f}")
    print(f"{'Accuracy':<10} | {accuracy:<10.4f}")
    print(f"{'Precision':<10} | {precision:<10.4f}")
    print(f"{'Recall':<10} | {recall:<10.4f}")
    print(f"{'F1 Score':<10} | {f1:<10.4f}")
    print("\nEvaluation Parameters:")
    print(f"Validation Directory: {val_dir}")
    print(f"Model Path: {model_path}")
    print(f"Batch Size: {batch_size}")
    print(f"Threshold: {threshold}")
    print(f"Margin: {margin}")
    print(f"Device: {device}")

if __name__ == "__main__":
    main()
