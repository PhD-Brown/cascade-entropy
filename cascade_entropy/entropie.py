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

from ._validation import entier_positif, serie_finie

TAILLE_BLOC = 512


def shannon(probabilites: np.ndarray) -> float:
    """Entropie de Shannon d'une distribution de probabilité, en nats.

    Les probabilités doivent être positives et sommer à 1. Les termes nuls sont
    ignorés, conformément à la convention 0 log 0 = 0.
    """
    p = serie_finie(probabilites)
    if np.any(p < 0):
        raise ValueError("Une probabilité ne peut pas être négative.")
    if not np.isclose(p.sum(), 1.0, atol=1e-9):
        raise ValueError(f"Les probabilités ne somment pas à 1 (somme = {p.sum():.3e}).")
    non_nuls = p[p > 0]
    return float(-np.sum(non_nuls * np.log(non_nuls)))


def entropie_repartition(valeurs: np.ndarray, normaliser: bool = False) -> float:
    """Entropie de la répartition d'une quantité positive entre plusieurs éléments.

    Appliquée aux flux de puissance, elle mesure le degré de concentration de la
    puissance dans le réseau : maximale lorsque toutes les lignes portent la même
    charge, minimale lorsqu'une seule les porte toutes.

    Contrairement aux mesures temporelles, cette entropie porte sur un seul
    instant : elle décrit une répartition dans l'espace, pas une dynamique. C'est
    elle qui produit, jour après jour, la série temporelle à analyser ensuite.

    Avec `normaliser`, le résultat est divisé par log(N) et se lit entre 0 et 1.
    """
    valeurs = np.abs(serie_finie(valeurs))
    total = valeurs.sum()
    if total == 0:
        return np.nan
    h = shannon(valeurs / total)
    return h / np.log(valeurs.size) if normaliser else h


def nombre_effectif(valeurs: np.ndarray) -> float:
    """Nombre effectif d'éléments portant une répartition, exp(H).

    Interprétation directe : si dix lignes se partagent la puissance également,
    la valeur est 10 ; si une seule ligne porte tout, elle est 1. Deux réseaux
    transportant la même puissance totale mais avec des nombres effectifs de 5 et
    de 2,2 n'ont pas la même fragilité.
    """
    h = entropie_repartition(valeurs)
    return float(np.exp(h)) if np.isfinite(h) else np.nan


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
    1 pour une série non corrélée à valeurs continues.

    Les égalités exactes sont départagées par ordre temporel (tri stable) :
    le point le plus ancien passe en premier. Aucun bruit artificiel n’est ajouté.
    Une série constante a ainsi une PE nulle par convention. Pour des données
    comportant beaucoup de zéros, cette convention doit accompagner les résultats.
    """
    serie = serie_finie(serie)
    dimension = entier_positif(dimension, "dimension", minimum=2)
    delai = entier_positif(delai, "delai")
    n_fenetres = serie.size - (dimension - 1) * delai
    if n_fenetres < 2:
        raise ValueError("Série trop courte pour cette dimension et ce délai.")

    indices = np.arange(dimension) * delai
    fenetres = serie[np.arange(n_fenetres)[:, None] + indices]
    ordres = np.argsort(fenetres, axis=1, kind="stable")

    # Comptage direct des motifs, sans risque de dépassement d’un code entier.
    _, effectifs = np.unique(ordres, axis=0, return_counts=True)

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
    serie = serie_finie(serie)
    m = entier_positif(m, "m")
    if not np.isfinite(r) or r <= 0:
        raise ValueError("r doit être fini et strictement positif.")
    if ecart_reference is not None and (
        not np.isfinite(ecart_reference) or ecart_reference < 0
    ):
        raise ValueError("ecart_reference doit être fini et positif ou nul.")
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
    serie = serie_finie(serie)
    echelle = entier_positif(echelle, "echelle")
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
    serie = serie_finie(serie)
    echelles = entier_positif(echelles, "echelles")
    m = entier_positif(m, "m")
    if not np.isfinite(r) or r <= 0:
        raise ValueError("r doit être fini et strictement positif.")
    ecart = serie.std(ddof=0)

    liste_echelles, valeurs = [], []
    for echelle in range(1, echelles + 1):
        granulee = granulariser(serie, echelle)
        if granulee.size < 10 * (m + 1):
            break
        liste_echelles.append(echelle)
        valeurs.append(echantillon(granulee, m=m, r=r, ecart_reference=ecart))

    return np.array(liste_echelles, dtype=float), np.array(valeurs, dtype=float)
