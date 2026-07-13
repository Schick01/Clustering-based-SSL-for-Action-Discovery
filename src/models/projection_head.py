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

    Variante residua (diagnostica): con ``residual=True`` la proiezione
    diventa ``x + MLP(x)`` a dimensione invariata, con l'ultimo strato
    inizializzato a zero cosi' da partire ESATTAMENTE dall'identita':
    all'iterazione 1 lo spazio proiettato coincide con le feature
    originali (geometria della baseline preservata) e il training puo'
    solo deformarlo gradualmente. E' l'analogo fedele del metodo
    completo, che parte dal backbone pre-addestrato e lo ritocca.

    Args:
        feature_dim: dimensione delle feature del backbone congelato
            (512 per ResNet18, 768 per VideoMAE-base).
        projection_dim: dimensione dello spazio di proiezione su cui
            viene eseguito il clustering (ignorata con ``residual=True``,
            dove l'uscita resta a ``feature_dim``).
        residual: attiva la variante residua a identita' iniziale.
    """

    def __init__(
        self,
        feature_dim: int,
        projection_dim: int = 256,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual
        output_dim = feature_dim if residual else projection_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, output_dim),
        )
        if residual:
            # Ultimo strato a zero: la correzione parte nulla e la
            # proiezione iniziale e' l'identita' esatta
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    @property
    def output_dim(self) -> int:
        """Dimensione dello spazio di uscita della proiezione."""
        return self.net[-1].out_features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Proietta le feature nello spazio di clustering.

        Args:
            features: tensore (N, feature_dim).

        Returns:
            Tensore (N, output_dim); con ``residual=True`` l'uscita e'
            ``features + correzione``.
        """
        if self.residual:
            return features + self.net(features)
        return self.net(features)
