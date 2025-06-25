import torch.nn as nn
import torchvision.models as models

class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, 2)  # 2 classes: male/female

    def forward(self, x):
        return self.base_model(x)
