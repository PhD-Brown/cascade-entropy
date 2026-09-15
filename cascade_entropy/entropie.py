"""Mesures informationnelles.

Module du fil B. Chaque fonction prend une série de nombres et retourne une
mesure. Aucun module du fil A n'est importé ici : ces outils ne savent pas d'où
viennent les séries qu'ils analysent.

Trois familles, qui ne mesurent pas la même chose.

L'entropie de permutation caractérise la complexité à partir de l'ordre relatif
des valeurs successives, en ignorant complètement leurs amplitudes.

L'entropie d'échantillon mesure la régularité : la probabilité que deux segments
qui se ressemblaient sur m points se ressemblent encore au point suivant.

L'entropie multiéchelle applique la précédente à des versions granularisées de la
série, ce qui distingue l'irrégularité de la complexité structurelle.
"""

from __future__ import annotations

from math import factorial, log

import numpy as np

TAILLE_BLOC = 512


def permutation(
    serie: np.ndarray,
    dimension: int = 3,
    delai: int = 1,
    normaliser: bool = True,
) -> float:
    """Entropie de permutation d'une série.

    Chaque fenêtre de `dimension` valeurs espacées de `delai` est remplacée par
    l'ordre de ses éléments. On applique ensuite l'entropie de Shannon à la
    distribution des ordres observés.

    Normalisée, la mesure vaut 0 pour une série parfaitement monotone et tend vers
    1 pour une série non corrélée.
    """
    serie = np.asarray(serie, dtype=float)
    n_fenetres = serie.size - (dimension - 1) * delai
    if n_fenetres < 2:
        raise ValueError("Série trop courte pour cette dimension et ce délai.")

    indices = np.arange(dimension) * delai
    fenetres = serie[np.arange(n_fenetres)[:, None] + indices]
    ordres = np.argsort(fenetres, axis=1, kind="stable")

    # Chaque motif ordinal devient un entier unique, ce qui permet de compter les
    # occurrences sans construire de dictionnaire.
    poids = dimension ** np.arange(dimension)
    codes = ordres @ poids
    _, effectifs = np.unique(codes, return_counts=True)

    p = effectifs / effectifs.sum()
    h = -np.sum(p * np.log(p))
    return h / log(factorial(dimension)) if normaliser else h


def _compter_paires(modeles: np.ndarray, r: float) -> tuple[int, int]:
    """Compte les paires de modèles proches en distance de Tchebychev.

    Retourne le compte pour les modèles de longueur m et pour ceux de longueur
    m+1, calculés ensemble puisque la distance en m+1 est le maximum de la
    distance en m et de l'écart sur le point supplémentaire.

    Les modèles passés incluent ce point supplémentaire en dernière colonne. Le
    calcul se fait par blocs de lignes pour éviter d'allouer une matrice de
    distances complète.
    """
    n, largeur = modeles.shape
    m = largeur - 1
    court, supplement = modeles[:, :m], modeles[:, m]

    total_m = 0
    total_m1 = 0
    for debut in range(0, n, TAILLE_BLOC):
        bloc = court[debut : debut + TAILLE_BLOC]
        distances = np.abs(bloc[:, None, :] - court[None, :, :]).max(axis=2)
        proches_m = distances <= r

        supp_bloc = supplement[debut : debut + TAILLE_BLOC]
        ecart_supp = np.abs(supp_bloc[:, None] - supplement[None, :])
        proches_m1 = proches_m & (ecart_supp <= r)

        total_m += int(proches_m.sum())
        total_m1 += int(proches_m1.sum())

    # On retire les appariements d'un modèle avec lui-même.
    return total_m - n, total_m1 - n


def echantillon(
    serie: np.ndarray,
    m: int = 2,
    r: float = 0.15,
    ecart_reference: float | None = None,
) -> float:
    """Entropie d'échantillon (SampEn).

    `r` est exprimé en fraction d'un écart-type : celui de la série analysée par
    défaut, ou celui passé par `ecart_reference`. Cette seconde possibilité est
    nécessaire en multiéchelle, où la tolérance doit rester fixée sur la série
    d'origine.

    Retourne NaN lorsqu'aucun appariement n'est trouvé, cas où la mesure est
    indéfinie plutôt que nulle.
    """
    serie = np.asarray(serie, dtype=float)
    n = serie.size
    if n < m + 2:
        raise ValueError("Série trop courte pour cette dimension d'immersion.")

    ecart = serie.std(ddof=0) if ecart_reference is None else ecart_reference
    if ecart == 0:
        return np.nan
    tolerance = r * ecart

    n_modeles = n - m
    indices = np.arange(m + 1)
    modeles = serie[np.arange(n_modeles)[:, None] + indices]

    compte_m, compte_m1 = _compter_paires(modeles, tolerance)
    if compte_m <= 0 or compte_m1 <= 0:
        return np.nan
    return -log(compte_m1 / compte_m)


def granulariser(serie: np.ndarray, echelle: int) -> np.ndarray:
    """Moyennes sur des fenêtres disjointes de longueur `echelle`.

    C'est l'opération de granularisation de l'entropie multiéchelle : à l'échelle
    1 la série est inchangée, à l'échelle 2 chaque paire de points est remplacée
    par sa moyenne, et ainsi de suite.
    """
    serie = np.asarray(serie, dtype=float)
    n = serie.size // echelle
    if n == 0:
        raise ValueError("Échelle plus grande que la série.")
    return serie[: n * echelle].reshape(n, echelle).mean(axis=1)


def multiechelle(
    serie: np.ndarray,
    echelles: int = 20,
    m: int = 2,
    r: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Entropie multiéchelle : SampEn de la série granularisée à chaque échelle.

    La tolérance est calculée une seule fois, sur l'écart-type de la série
    d'origine, et reste fixée pour toutes les échelles. C'est ce qui rend les
    valeurs comparables entre elles : la granularisation réduit la variance, et
    recalculer la tolérance à chaque échelle masquerait précisément l'effet que la
    mesure cherche à révéler.

    Retourne les échelles retenues et les entropies correspondantes.
    """
    serie = np.asarray(serie, dtype=float)
    ecart = serie.std(ddof=0)

    liste_echelles, valeurs = [], []
    for echelle in range(1, echelles + 1):
        granulee = granulariser(serie, echelle)
        if granulee.size < 10 * (m + 1):
            break
        liste_echelles.append(echelle)
        valeurs.append(echantillon(granulee, m=m, r=r, ecart_reference=ecart))

    return np.array(liste_echelles, dtype=float), np.array(valeurs, dtype=float)