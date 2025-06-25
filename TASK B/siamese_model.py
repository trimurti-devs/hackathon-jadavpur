# siamese_model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=256):
        super(SiameseNetwork, self).__init__()

        # Pretrained ResNet18 as feature extractor
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base_model.fc = nn.Linear(base_model.fc.in_features, embedding_dim)
        self.embedding_net = base_model

        # Final classifier for similarity
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        return self.embedding_net(x)

    def forward(self, x1, x2):
        # Extract embeddings
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)

        # Compute absolute difference
        diff = torch.abs(out1 - out2)

        # Predict similarity
        similarity = self.classifier(diff)
        return similarity
