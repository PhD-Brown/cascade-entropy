"""Construction du réseau : topologie, matrice de flux, limites de lignes.

Premier module du fil A. Ne dépend d'aucun autre module du projet.

Critère de validation : la somme des injections de puissance doit être nulle, le
réseau doit être connexe, et la matrice reliant les injections aux flux doit avoir
la dimension attendue.
"""

from __future__ import annotations

import numpy as np


def arbre(n_generations: int, generation_des_sources: int = 3):
    """Réseau en arbre à trois branches par nœud, générateurs à un niveau donné.

    Retourne la topologie, la liste des nœuds générateurs et la liste des charges.
    """
    raise NotImplementedError


def matrice_de_flux(topologie, reactances: np.ndarray) -> np.ndarray:
    """Matrice A telle que F = A P, où P exclut le générateur de référence.

    L'exclusion du nœud de référence évite la singularité qui découlerait de
    l'équilibre global des puissances.
    """
    raise NotImplementedError
