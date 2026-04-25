import torch
from smoke_detection.models.classifier_resnet import build_classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_classifier(in_channels=4, pretrained=True).to(device)
