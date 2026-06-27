from transformers import VideoMAEModel
import torch
import torch.nn as nn

class VideoMAEFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        features = outputs.last_hidden_state
        features = features.mean(dim=1)  # Media lungo la dimensione temporale
        return features