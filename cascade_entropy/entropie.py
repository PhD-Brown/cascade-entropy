"""Mesures informationnelles.

Module du fil B. Il prend en entrée une série de nombres et n'a aucune
connaissance du modèle électrique : il ne doit importer aucun module du fil A.

Critères de validation : entropie de permutation basse sur un signal périodique et
proche du maximum sur un bruit non corrélé ; en multiéchelle, croisement entre le
bruit non corrélé et le bruit corrélé à longue portée.
"""

from __future__ import annotations

import numpy as np


def permutation(serie: np.ndarray, dimension: int = 3, delai: int = 1) -> float:
    """Entropie de permutation, normalisée entre 0 et 1."""
    raise NotImplementedError


def echantillon(serie: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Entropie d'échantillon (SampEn), r exprimé en écarts-types."""
    raise NotImplementedError


def multiechelle(serie: np.ndarray, echelles: int = 20, m: int = 2, r: float = 0.2):
    """Entropie multiéchelle : SampEn après granularisation à chaque échelle."""
    raise NotImplementedError
