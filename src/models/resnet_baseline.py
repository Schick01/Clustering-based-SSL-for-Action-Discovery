import torch
import torch.nn as nn
import torchvision

class ResNetFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights = 'DEFAULT')
        self.backbone.fc = nn.Identity()    

    def forward(self, x):
        # x arriva con forma: (Batch, Canali, Tempo, Altezza, Larghezza)
        B, C, T, H, W = x.shape

        # Da (B, C, T, H, W) diventa (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B*T, C, H, W)

        features = self.backbone(x)
        features = features.reshape(B, T, -1)

        final_video_features = features.mean(dim=1)

        return final_video_features


        
