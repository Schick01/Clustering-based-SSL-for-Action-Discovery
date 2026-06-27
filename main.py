from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.datasets.video_dataset import VideoKineticsDataset
from src.models.resnet_baseline import ResNetFeatureExtractor


if __name__ == '__main__':
    print("Inizializzazione del dataset e del modello...")
    dataset = VideoKineticsDataset()

    kinetics_loader = DataLoader(dataset, batch_size = 4, num_workers = 4)

    model = ResNetFeatureExtractor()

    features_list = []
    labels_list = []
    
    model.eval()

    print("Estrazione feature in corso...")

    with torch.no_grad():
        for video, label in tqdm(kinetics_loader):
            feature = model(video)
            features_list.append(feature)
            labels_list.extend(label)
    
    print("Salvataggio delle feature e delle etichette...")
    
    features_list = torch.cat(features_list, dim=0)

    torch.save(features_list, 'features.pt')
    torch.save(labels_list, 'labels.pt')

    print(f"Finito! Feature salvate con forma: {features_list.shape}")
    print(f"Etichette salvate: {len(labels_list)}")
