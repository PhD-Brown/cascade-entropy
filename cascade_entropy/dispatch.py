"""Dispatch de la production sous contraintes.

Deuxième module du fil A. Le problème est linéaire : on minimise la production
totale en pénalisant fortement le délestage, sous quatre contraintes (bornes de
production, charges négatives, limites de flux, équilibre global).

Critère de validation : sur un réseau réduit résolu à la main, les flux doivent
coïncider ; à faible charge, la solution ne doit comporter ni délestage ni ligne
saturée.
"""

from __future__ import annotations

import numpy as np


def resoudre(reseau, demande: np.ndarray, poids_delestage: float = 100.0):
    """Résout le dispatch et retourne les flux et la production par nœud."""
    raise NotImplementedError


def taux_de_charge(flux: np.ndarray, limites: np.ndarray) -> np.ndarray:
    """Taux de charge M_ij = F_ij / F_ij_max, ligne par ligne."""
    raise NotImplementedError
