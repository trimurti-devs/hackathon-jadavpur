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
    train_dir = "C:/workspace-python/hackathon-jadavpur/siamese_face/train"
    val_dir = "C:/workspace-python/hackathon-jadavpur/siamese_face/val"
    model_path = "C:/workspace-python/hackathon-jadavpur/try/siamese_face_matcher.pth"
    batch_size = 16
    threshold = 1.0
    margin = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = TripletFaceDataset(train_dir, transform=transform)
    val_dataset = TripletFaceDataset(val_dir, transform=transform)

    if len(train_dataset) == 0:
        raise ValueError("[❌] No training data found. Check your folder structure.")
    if len(val_dataset) == 0:
        raise ValueError("[❌] No validation data found. Check your folder structure.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Loss function
    criterion = nn.TripletMarginLoss(margin=margin, p=2)

    # Evaluate on train dataset
    train_loss, train_acc, train_prec, train_rec, train_f1 = evaluate(model, train_loader, criterion, device, threshold)

    # Evaluate on val dataset
    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device, threshold)

    # Print results for train dataset
    print("Train Dataset Evaluation Results:")
    print(f"{'Metric':<10} | {'Value':<10}")
    print("-" * 25)
    print(f"{'Loss':<10} | {train_loss:<10.4f}")
    print(f"{'Accuracy':<10} | {train_acc:<10.4f}")
    print(f"{'Precision':<10} | {train_prec:<10.4f}")
    print(f"{'Recall':<10} | {train_rec:<10.4f}")
    print(f"{'F1 Score':<10} | {train_f1:<10.4f}")
    print("\nTrain Dataset Parameters:")
    print(f"Training Directory: {train_dir}")
    print(f"Batch Size: {batch_size}")
    print(f"Threshold: {threshold}")
    print(f"Margin: {margin}")
    print(f"Device: {device}")
    print("\n" + "="*40 + "\n")

    # Print results for val dataset
    print("Validation Dataset Evaluation Results:")
    print(f"{'Metric':<10} | {'Value':<10}")
    print("-" * 25)
    print(f"{'Loss':<10} | {val_loss:<10.4f}")
    print(f"{'Accuracy':<10} | {val_acc:<10.4f}")
    print(f"{'Precision':<10} | {val_prec:<10.4f}")
    print(f"{'Recall':<10} | {val_rec:<10.4f}")
    print(f"{'F1 Score':<10} | {val_f1:<10.4f}")
    print("\nValidation Dataset Parameters:")
    print(f"Validation Directory: {val_dir}")
    print(f"Batch Size: {batch_size}")
    print(f"Threshold: {threshold}")
    print(f"Margin: {margin}")
    print(f"Device: {device}")

if _name_ == "_main_":
    main()
