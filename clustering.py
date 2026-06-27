import torch
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter

if __name__ == '__main__':

    print("1. Caricamento dati")
    features_tensor = torch.load("data/features.pt")
    labels = torch.load("data/labels.pt")

    print("2. Conversione del tensore con numpy")
    X = features_tensor.numpy()

    print("3. Conteggio classi")
    unique_classes = set(labels)
    k_cluster = len(unique_classes)

    print("4. Applichiamo il K-means")
    kmeans = KMeans(n_clusters=k_cluster, random_state=42).fit(X)
    kmeans_labels = kmeans.labels_

    labels_np = np.array(labels)
    video_corretti = 0
    video_totali = len(labels_np)

    for i in range(k_cluster):
        #Prende gli elementi (indici) che appartengono allo stesso cluster
        elementi_che_appartengono_allo_stesso_cluster = np.where(kmeans_labels==i)[0]
        
        #Prende le etichette reali, ovvero la cartella a cui appartengono
        etichette_reali_degli_elementi_nel_cluster = labels_np[elementi_che_appartengono_allo_stesso_cluster]
        
        #Conta quante volte appare ogni etichetta reale all'interno del cluster
        conteggio_etichette_reali = Counter(etichette_reali_degli_elementi_nel_cluster)

        #Prende l'etichetta reale più frequente all'interno del cluster
        etichetta_reale_piu_frequente = conteggio_etichette_reali.most_common(1)[0][0]

        #Contiamo quante volte appare l'etichetta reale più frequente all'interno del cluster
        video_corretti += conteggio_etichette_reali[etichetta_reale_piu_frequente]

        print(f"Cluster {i}: Etichetta reale più frequente: {etichetta_reale_piu_frequente}, Conteggio: {conteggio_etichette_reali[etichetta_reale_piu_frequente]}")


accuracy = video_corretti / video_totali * 100
print(f"Accuratezza del clustering: {accuracy:.2f}%")
