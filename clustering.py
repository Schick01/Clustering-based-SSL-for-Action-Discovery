import torch
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def clustering(path_to_features, path_to_labels):

    print("1. Loading data...")
    features_tensor = torch.load(path_to_features)
    labels = torch.load(path_to_labels)

    print("2. Converting tensor to NumPy...")
    X = features_tensor.numpy()

    print("3. Analyzing classes...")
    unique_classes = set(labels)
    num_clusters = len(unique_classes)

    print("4. Applying K-Means clustering...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(X)
    cluster_labels = kmeans.labels_

    # Valutazione
    true_labels = np.array(labels)
    correct_samples = 0
    total_samples = len(true_labels)

    for i in range(num_clusters):
        # 1. Trova gli indici dei campioni assegnati a questo cluster
        cluster_indices = np.where(cluster_labels == i)[0]
        
        # 2. Estrai le etichette reali per quegli indici
        true_labels_in_cluster = true_labels[cluster_indices]
        
        # 3. Conta le frequenze delle etichette
        label_counts = Counter(true_labels_in_cluster)

        # 4. Identifica la classe maggioritaria
        most_common_label = label_counts.most_common(1)[0][0]

        # 5. Aggiorna il contatore delle previsioni corrette
        correct_samples += label_counts[most_common_label]

        print(f"Cluster {i} | Assigned to: '{most_common_label}' | Correct matches: {label_counts[most_common_label]}")

    accuracy = (correct_samples / total_samples) * 100
    print(f"Clustering Accuracy: {accuracy:.2f}%")

def iterative_clustering(path_to_features, path_to_labels, num_iterations=5):