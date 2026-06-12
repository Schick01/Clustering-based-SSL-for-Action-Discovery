import torch
from torch.utils.data import DataLoader

from datasets.video_dataset import VideoKineticsDataset
from models.resnet_baseline import ResNetFeatureExtractor

if __name__ == '__main__':

    dataset = VideoKineticsDataset()
    kinetics_loader = DataLoader(dataset, batch_size = 2)

    model = ResNetFeatureExtractor()

    batch_video, batch_labels = next(iter(kinetics_loader))

    features = model(batch_video)
    print(features.shape)