import os
from torch.utils.data import Dataset
import torch
import cv2
import numpy as np

def get_dataset(path = "data"):
    """
    Prende la cartella "data" e torna un dizionario i cui elementi sono delle coppie percorso:classe
    """
    data_path = path
    
    dict = {}
    
    for dir in os.listdir(data_path):
        action = dir
        action_dir = os.path.join(data_path,dir)
        if os.path.isdir(action_dir):
            for video in os.listdir(action_dir):
                if video.endswith("mp4"):
                    dict[os.path.join(action_dir,video)] = action
            
    return dict

class VideoKineticsDataset(Dataset):

    def __init__(self, path="data", num_frames=16):
        self.video_dict = list(get_dataset(path).items())
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_dict)


    def __getitem__(self, idx):
        '''
        Prende un indice e torna i frames del video corrispondente a quell'indice e la classe del video
        '''
        path,action = self.video_dict[idx]
        
        capture = cv2.VideoCapture(path)
        frames = []

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = [int(i * total_frames / self.num_frames) for i in range(self.num_frames)]

        for frame_idx in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = capture.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
                frames.append(frame)  
            else:
                break

        capture.release()
        # Passiamo da (Tempo, Altezza, Larghezza, Canali) a (Canali, Tempo, Altezza, Larghezza)
        video_array = np.array(frames) # Questo passaggio si fa perche perche torch impiega meno tempo a convertire un array numpy in un tensore
        video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2).float() / 255.0
        return video_tensor, action
        