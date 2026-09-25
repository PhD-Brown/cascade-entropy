"""
Contrôles communs aux mesures : aucune donnée manquante ignorée.
Sert surtout à valider les entrées des fonctions d'entropie.
"""

from numbers import Integral

import numpy as np


def serie_finie(serie):
    """
    Exige une série réelle, non vide, unidimensionnelle et finie.

    La fonction retourne un tableau NumPy de type flottant afin que les fonctions
    appelantes travaillent toutes avec une représentation homogène. Elle refuse
    explicitement les données complexes, vides, multidimensionnelles ou non
    finies, plutôt que de les corriger silencieusement.
    """
    # Les nombres complexes ne sont pas acceptés implicitement : les convertir en
    # flottants pourrait supprimer leur partie imaginaire sans avertissement.
    if np.iscomplexobj(serie):
        raise ValueError("La série doit être réelle.")

    # La conversion uniformise les listes Python, tuples et tableaux NumPy en un
    # tableau numérique de flottants.
    valeurs = np.asarray(serie, dtype=float)
    # Les mesures analysent une seule série. On refuse donc les matrices et les
    # tableaux vides plutôt que de choisir arbitrairement un axe.
    if valeurs.ndim != 1 or valeurs.size == 0:
        raise ValueError("La série doit être un tableau 1D non vide.")
    # Les NaN et les infinis contamineraient ensuite les sommes, logarithmes ou
    # régressions utilisés par les mesures statistiques.
    if not np.isfinite(valeurs).all():
        raise ValueError("La série contient des NaN ou des valeurs infinies.")
    return valeurs


def entier_positif(valeur, nom, minimum=1):
    """
    Vérifie qu'une valeur est un entier supérieur ou égal à un minimum.

    Les booléens sont explicitement exclus, même si Python les considère comme
    une sous-classe des entiers (`True` vaut 1 et `False` vaut 0). Cette
    distinction évite qu'une dimension ou une échelle accepte une valeur logique.
    """
    # Le nom du paramètre permet de produire un message d'erreur précis pour
    # l'appelant.
    if isinstance(valeur, (bool, np.bool_)) or not isinstance(valeur, Integral):
        raise ValueError(f"{nom} doit être un entier >= {minimum}.")
    # La borne minimale protège les tailles de fenêtre, dimensions et répétitions.
    if valeur < minimum:
        raise ValueError(f"{nom} doit être un entier >= {minimum}.")
    # On retourne un int Python standard, même si l'entrée était un entier NumPy.
    return int(valeur)
