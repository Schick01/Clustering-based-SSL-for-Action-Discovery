import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.datasets.video_dataset import VideoKineticsDataset
from src.models.resnet_baseline import ResNetFeatureExtractor
from src.models.videomae_extractor import VideoMAEFeatureExtractor
from transformers import VideoMAEImageProcessor, VideoMAEModel

def main():
    print("Inizio estrazione feature dai video tramite il modello ResNet18 pre-addestrato su ImageNet...")
    dataset_resnet = VideoKineticsDataset()

    loader_resnet = DataLoader(dataset_resnet, batch_size = 4, shuffle = False)

    print("Controllo esistenza feature e etichette salvate...")
    path_features_resnet = 'data/features_resnet18.pt'
    path_labels_resnet = 'data/labels_resnet18.pt'

    if os.path.exists(path_features_resnet) and os.path.exists(path_labels_resnet):
        print("Feature e etichette già salvate. Caricamento...")
        features_resnet_list = torch.load(path_features_resnet)
        labels_resnet_list = torch.load(path_labels_resnet)
        print(f"Feature caricate con forma: {features_resnet_list.shape}")
        print(f"Etichette caricate: {len(labels_resnet_list)}")
    else:
        print("Feature e etichette non trovate. Estrazione in corso...")
        model = ResNetFeatureExtractor()

        features_resnet_list = []
        labels_resnet_list = []

        model.eval()

        print("Estrazione feature in corso...")

        with torch.no_grad():
            for video, label in tqdm(loader_resnet):
                feature = model(video)
                features_resnet_list.append(feature)
                labels_resnet_list.extend(label)

        print("Salvataggio delle feature e delle etichette...")

        features_resnet_list = torch.cat(features_resnet_list, dim=0)

        torch.save(features_resnet_list, 'data/features_resnet18.pt')
        torch.save(labels_resnet_list, 'data/labels_resnet18.pt')

        print(f"Finito! Feature salvate con forma: {features_resnet_list.shape}")
        print(f"Etichette salvate: {len(labels_resnet_list)}")

    print("Inizio estrazione feature dai video tramite il modello VideoMAE...")

    videomae_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    dataset_videomae = VideoKineticsDataset(processor=videomae_processor)
    loader_videomae = DataLoader(dataset_videomae, batch_size=4, shuffle=False)

    path_features_videomae = 'data/features_videomae.pt'
    path_labels_videomae = 'data/labels_videomae.pt'

    if os.path.exists(path_features_videomae) and os.path.exists(path_labels_videomae):
        print("Feature e etichette già salvate. Caricamento...")
        features_videomae_list = torch.load(path_features_videomae)
        labels_videomae_list = torch.load(path_labels_videomae)
        
        print(f"Feature caricate con forma: {features_videomae_list.shape}")
        print(f"Etichette caricate: {len(labels_videomae_list)}")
    else:
        print("Feature e etichette non trovate. Estrazione in corso...")
        model_videomae = VideoMAEFeatureExtractor()

        features_videomae_list = []
        labels_videomae_list = []

        model_videomae.eval()

        print("Estrazione feature in corso...")
        with torch.no_grad():
            for video, label in tqdm(loader_videomae):
                feature = model_videomae(video)
                features_videomae_list.append(feature)
                labels_videomae_list.extend(label)

        print("Salvataggio delle feature e delle etichette...")
        features_videomae_list = torch.cat(features_videomae_list)
        torch.save(features_videomae_list, 'data/features_videomae.pt')
        torch.save(labels_videomae_list, 'data/labels_videomae.pt')
        
        print(f"Finito! Feature salvate con forma: {features_videomae_list.shape}")
        print(f"Etichette salvate: {len(labels_videomae_list)}")

if __name__ == "__main__":
    main()