import torch
from smoke_detection.models.segmenter_unet import build_segmenter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_segmenter(in_channels=4, n_classes=1).to(device)
