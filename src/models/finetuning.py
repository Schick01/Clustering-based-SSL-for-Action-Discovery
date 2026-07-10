"""
Preparazione dei backbone al fine-tuning sulle pseudo-label.

Con un dataset piccolo e pseudo-label rumorose, il fine-tuning completo
rischia di distruggere le rappresentazioni del pre-training: qui si
sbloccano solo i layer semantici di alto livello, lasciando congelato
il resto del backbone.

- ResNet18: solo ``layer4``; le statistiche delle BatchNorm restano
  congelate (vedi ``freeze_batchnorm_stats``) perché con batch piccoli
  di frame fortemente correlati verrebbero stimate male;
- VideoMAE: ultimi ``num_blocks`` blocchi transformer dell'encoder,
  più la LayerNorm finale se presente (nella configurazione standard,
  che usa il mean pooling, non esiste).

La testa di classificazione è un singolo layer lineare, da
reinizializzare a ogni iterazione del clustering: gli ID dei cluster
cambiano arbitrariamente tra un'iterazione e l'altra, quindi riusare la
testa precedente non avrebbe senso.
"""

import torch.nn as nn

from src.models.resnet_baseline import ResNetFeatureExtractor
from src.models.videomae_extractor import VideoMAEFeatureExtractor


def freeze_all_parameters(model: nn.Module) -> None:
    """
    Disabilita il gradiente su tutti i parametri del modello.

    Args:
        model: modello da congelare.
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_resnet_top(model: ResNetFeatureExtractor) -> list[nn.Parameter]:
    """
    Congela la ResNet e sblocca solo ``layer4``.

    Args:
        model: estrattore ResNet da preparare al fine-tuning.

    Returns:
        Lista dei parametri allenabili del backbone.
    """
    freeze_all_parameters(model)

    trainable_params = []
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    return trainable_params


def unfreeze_videomae_top(
    model: VideoMAEFeatureExtractor,
    num_blocks: int = 2,
) -> list[nn.Parameter]:
    """
    Congela VideoMAE e sblocca gli ultimi ``num_blocks`` blocchi encoder.

    Args:
        model: estrattore VideoMAE da preparare al fine-tuning.
        num_blocks: numero di blocchi transformer finali da sbloccare.

    Returns:
        Lista dei parametri allenabili del backbone.
    """
    freeze_all_parameters(model)

    trainable_params = []
    for block in model.model.encoder.layer[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
            trainable_params.append(param)

    # La LayerNorm finale esiste solo se il modello non usa il mean
    # pooling: con la configurazione standard è None e si salta
    if model.model.layernorm is not None:
        for param in model.model.layernorm.parameters():
            param.requires_grad = True
            trainable_params.append(param)

    return trainable_params


def freeze_batchnorm_stats(model: nn.Module) -> None:
    """
    Porta tutti i moduli BatchNorm del modello in modalità eval.

    Da chiamare dopo ``model.train()``: i pesi affini delle BatchNorm
    restano allenabili (se sbloccati), ma le statistiche running non
    vengono più aggiornate dai batch di training.

    Args:
        model: modello i cui moduli BatchNorm vanno congelati.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def build_linear_head(feature_dim: int, num_clusters: int) -> nn.Linear:
    """
    Crea la testa lineare di classificazione feature -> cluster.

    Args:
        feature_dim: dimensione delle feature del backbone
            (512 per ResNet18, 768 per VideoMAE-base).
        num_clusters: numero di cluster K (classi della testa).

    Returns:
        Layer lineare con inizializzazione di default.
    """
    return nn.Linear(feature_dim, num_clusters)
