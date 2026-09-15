"""Contrôles communs aux mesures : aucune donnée manquante ignorée."""

from numbers import Integral

import numpy as np


def serie_finie(serie):
    """Exige une série réelle, non vide, unidimensionnelle et finie."""
    if np.iscomplexobj(serie):
        raise ValueError("La série doit être réelle.")
    valeurs = np.asarray(serie, dtype=float)
    if valeurs.ndim != 1 or valeurs.size == 0:
        raise ValueError("La série doit être un tableau 1D non vide.")
    if not np.isfinite(valeurs).all():
        raise ValueError("La série contient des NaN ou des valeurs infinies.")
    return valeurs


def entier_positif(valeur, nom, minimum=1):
    if isinstance(valeur, (bool, np.bool_)) or not isinstance(valeur, Integral):
        raise ValueError(f"{nom} doit être un entier >= {minimum}.")
    if valeur < minimum:
        raise ValueError(f"{nom} doit être un entier >= {minimum}.")
    return int(valeur)
