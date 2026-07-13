"""
Testa di proiezione MLP per il metodo a backbone congelato.

Nel metodo alternativo il backbone non viene mai aggiornato: le sue
feature sono fisse e il progresso tra le iterazioni si accumula in
questo MLP, che ri-proietta le feature in uno spazio che il loop puo'
riorganizzare. Il suo ruolo e' quindi speculare a quello del backbone
nel metodo completo: persiste tra le iterazioni, mentre il
classificatore lineare finale resta usa-e-getta (gli ID dei cluster
permutano a ogni giro).

Il clustering viene eseguito sull'uscita della proiezione, non sui
logit del classificatore.
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    MLP di proiezione: Linear(D, D) + ReLU + Linear(D, proj_dim).

    Capacita' volutamente minima (un solo strato nascosto): con feature
    fisse e pseudo-label rumorose, un MLP piu' grande memorizzerebbe le
    pseudo-label invece di riorganizzare lo spazio.

    Args:
        feature_dim: dimensione delle feature del backbone congelato
            (512 per ResNet18, 768 per VideoMAE-base).
        projection_dim: dimensione dello spazio di proiezione su cui
            viene eseguito il clustering.
    """

    def __init__(self, feature_dim: int, projection_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Proietta le feature nello spazio di clustering.

        Args:
            features: tensore (N, feature_dim).

        Returns:
            Tensore (N, projection_dim).
        """
        return self.net(features)
