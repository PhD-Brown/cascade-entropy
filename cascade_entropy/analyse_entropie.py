"""
Batterie de mesures d'entropie avec références de substitution (fil B, SO3).

SO3 pose deux questions sur les séries de la dynamique lente :

1. Une entropie distingue-t-elle les régimes (valeurs de G) ? On la compare à
   la même mesure sur des MÉLANGES de la série (distribution conservée, ordre
   détruit) : un écart signale une structure temporelle.
2. Apporte-t-elle une information complémentaire à Hurst et aux corrélations
   linéaires ? On la compare à des substituts IAAFT (distribution ET spectre
   conservés, phases aléatoires, `substituts.substitut_iaaft`) : un écart
   signale une structure que le spectre n'explique pas.

Mesures calculées pour chaque série et chaque échelle s (moyennes sur des blocs
disjoints de s jours, `entropie.granulariser`) :

- PE de dimension d (Bandt et Pompe 2002) sur la série granularisée
  (entropie de permutation multiéchelle). Une échelle est rendue NaN quand la
  série granularisée compte moins de 10·d! points : il faut beaucoup plus de
  fenêtres que de motifs possibles pour estimer leur distribution.
- SampEn (Richman et Moorman 2000) sur la série granularisée, tolérance FIXE
  r·σ de la série d'origine (Costa et al. 2002) : c'est la MSE. La tolérance
  ne suit pas la variance de la série granularisée ; une MSE plus haute à
  grande échelle peut donc venir simplement d'une variance qui décroît moins
  vite que 1/s, c'est-à-dire du spectre. C'est précisément ce que la
  référence IAAFT permet de départager.

Avant toute interprétation, `diagnostic_egalites` mesure la part de zéros et
d'égalités : sur une série riche en zéros exacts, PE et SampEn mesurent la
fréquence des zéros, pas la dynamique (docs/journal.md, « Point de
vigilance »).

Ce module ne connaît rien du réseau : il reçoit des tableaux NumPy.
Validation : tests/test_analyse_entropie.py.
"""

from __future__ import annotations

import signal
import time
from collections.abc import Callable, Sequence
from math import factorial

import numpy as np

from ._validation import entier_positif, serie_finie
from .entropie import METHODES_SAMPEN, granulariser, multiechelle, permutation
from .substituts import ITERATIONS_IAAFT, ecart_spectral, substitut_iaaft

ECHELLES_DEFAUT = (1, 2, 5, 10, 20, 50, 100, 200)
DIMENSIONS_DEFAUT = (3, 4, 5, 6)
# Nombre minimal de points de la série granularisée, en multiples de d!.
FACTEUR_MIN_PE = 10


def _echelles_valides(echelles) -> list[int]:
    if isinstance(echelles, (str, bytes)) or not isinstance(echelles, Sequence):
        raise ValueError("echelles doit être une liste d'entiers strictement croissante.")
    liste = [entier_positif(e, "echelle") for e in echelles]
    if not liste or any(b <= a for a, b in zip(liste, liste[1:])):
        raise ValueError("echelles doit être une liste non vide strictement croissante.")
    return liste


def _dimensions_valides(dimensions) -> list[int]:
    if isinstance(dimensions, (str, bytes)) or not isinstance(dimensions, Sequence):
        raise ValueError("dimensions doit être une liste d'entiers >= 2.")
    liste = [entier_positif(d, "dimension", minimum=2) for d in dimensions]
    if not liste or len(set(liste)) != len(liste):
        raise ValueError("dimensions doit être une liste non vide sans doublon.")
    return liste


def diagnostic_egalites(serie) -> dict[str, float]:
    """
    Part de zéros exacts, d'égalités consécutives et de valeurs répétées.

    - `fraction_zeros` : part des points égaux à 0 ;
    - `fraction_egalites_consecutives` : part des pas t → t+1 sans changement ;
    - `fraction_valeurs_repetees` : 1 − (nombre de valeurs distinctes) / n.

    Les trois sont des fractions dans [0, 1]. Une valeur élevée signale que PE
    et SampEn risquent de mesurer les égalités plutôt que la dynamique.
    """
    x = serie_finie(serie)
    egal = float(np.mean(np.diff(x) == 0)) if x.size > 1 else 0.0
    return {
        "fraction_zeros": float(np.mean(x == 0)),
        "fraction_egalites_consecutives": egal,
        "fraction_valeurs_repetees": float(1.0 - np.unique(x).size / x.size),
    }


def permutation_multiechelle(serie, echelles=ECHELLES_DEFAUT, dimension: int = 4,
                             delai: int = 1) -> np.ndarray:
    """
    PE normalisée de la série granularisée, une valeur par échelle.

    Une échelle dont la série granularisée compte moins de 10·d! points vaut
    NaN (estimation de la distribution des d! motifs non fiable).
    """
    x = serie_finie(serie)
    echelles = _echelles_valides(echelles)
    dimension = entier_positif(dimension, "dimension", minimum=2)
    delai = entier_positif(delai, "delai")
    minimum = FACTEUR_MIN_PE * factorial(dimension)
    sortie = np.full(len(echelles), np.nan)
    for j, s in enumerate(echelles):
        if x.size // s < minimum:
            continue
        sortie[j] = permutation(granulariser(x, s), dimension=dimension, delai=delai)
    return sortie


def mesures_entropie(serie, echelles=ECHELLES_DEFAUT, dimensions=DIMENSIONS_DEFAUT,
                     m: int = 2, r: float = 0.15, methode: str = "arbre") -> dict:
    """
    PE multiéchelle (une ligne par dimension) et MSE d'une série.

    Retourne {"pe": tableau (len(dimensions), len(echelles)),
              "mse": tableau (len(echelles),)}.
    La MSE utilise la tolérance fixe r·σ de `serie` (`entropie.multiechelle`
    avec une liste d'échelles) ; `methode` choisit le comptage des paires
    (« arbre » par défaut ici, indispensable sur 10^5 points).
    """
    x = serie_finie(serie)
    echelles = _echelles_valides(echelles)
    dimensions = _dimensions_valides(dimensions)
    if methode not in METHODES_SAMPEN:
        raise ValueError(f"methode doit valoir {' ou '.join(METHODES_SAMPEN)}.")
    pe = np.array([permutation_multiechelle(x, echelles, d) for d in dimensions])
    _, mse = multiechelle(x, echelles=echelles, m=m, r=r, methode=methode)
    return {"pe": pe, "mse": mse}


def mesures_substituts(serie, generer: Callable[[np.ndarray, np.random.Generator], np.ndarray],
                       n: int, rng: np.random.Generator, **parametres) -> dict:
    """
    `mesures_entropie` sur `n` substituts produits par `generer(serie, rng)`.

    Retourne {"pe": (n, len(dimensions), len(echelles)), "mse": (n, len(echelles))}.
    `generer` est par exemple `lambda x, g: g.permutation(x)` (mélange) ou
    `substituts.substitut_iaaft`. Le même `rng` sert aux n tirages, dans
    l'ordre : le résultat est reproductible pour une graine donnée.
    """
    x = serie_finie(serie)
    n = entier_positif(n, "n")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng doit être un numpy.random.Generator.")
    if not callable(generer):
        raise TypeError("generer doit être appelable.")
    pe, mse = [], []
    for _ in range(n):
        y = np.asarray(generer(x, rng), dtype=float)
        if y.shape != x.shape:
            raise ValueError("Un substitut doit avoir la forme de la série.")
        res = mesures_entropie(y, **parametres)
        pe.append(res["pe"])
        mse.append(res["mse"])
    return {"pe": np.array(pe), "mse": np.array(mse)}


def resume_substituts(valeur, references, niveau: float = 0.95) -> dict[str, np.ndarray]:
    """
    Position d'une mesure par rapport à la distribution de ses substituts.

    `valeur` : mesure de l'original (scalaire ou tableau). `references` : même
    mesure sur les substituts, empilés selon l'axe 0. Retourne, élément par
    élément : moyenne, écart-type (ddof = 1), bornes de l'intervalle central de
    niveau `niveau`, et l'écart réduit z = (valeur − moyenne) / écart-type.
    z vaut NaN si moins de deux substituts sont finis ou si leur écart-type est
    nul. Les NaN des substituts (échelles non estimables) sont ignorés.
    """
    valeur = np.asarray(valeur, dtype=float)
    references = np.asarray(references, dtype=float)
    if references.ndim < 1 or references.shape[1:] != valeur.shape:
        raise ValueError("references doit empiler, selon l'axe 0, des tableaux de la forme de valeur.")
    if not 0 < float(niveau) < 1:
        raise ValueError("niveau doit être dans ]0, 1[.")
    alpha = (1.0 - float(niveau)) / 2.0
    plat = references.reshape(references.shape[0], -1)
    moy, ect, bas, haut = (np.full(plat.shape[1], np.nan) for _ in range(4))
    for j in range(plat.shape[1]):
        col = plat[:, j][np.isfinite(plat[:, j])]
        if col.size == 0:
            continue
        moy[j] = col.mean()
        bas[j], haut[j] = np.quantile(col, [alpha, 1.0 - alpha])
        if col.size >= 2:
            ect[j] = col.std(ddof=1)
    forme = valeur.shape
    moy, ect, bas, haut = (a.reshape(forme) for a in (moy, ect, bas, haut))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(np.isfinite(ect) & (ect > 0), (valeur - moy) / ect, np.nan)
    return {"moyenne": moy, "ecart_type": ect, "bas": bas, "haut": haut, "z": z}


def rapport_variance_blocs(serie, taille: int = 1000) -> float:
    """
    Variance des moyennes de blocs de `taille` points, rapportée à celle d'un bruit blanc.

    Pour une série sans mémoire, la variance des moyennes de blocs vaut σ²/taille :
    le rapport vaut 1 (à l'erreur statistique près). Un rapport > 1 signale une
    composante lente qui ne se moyenne pas. Rendu comme rapport d'écarts-types
    (√ du rapport des variances), NaN s'il y a moins de deux blocs ou si σ = 0.
    """
    x = serie_finie(serie)
    taille = entier_positif(taille, "taille")
    n_blocs = x.size // taille
    sigma = x.std(ddof=0)
    if n_blocs < 2 or sigma == 0:
        return float("nan")
    moyennes = x[: n_blocs * taille].reshape(n_blocs, taille).mean(axis=1)
    return float(moyennes.std(ddof=1) / (sigma / np.sqrt(taille)))


def autocorrelation(serie, decalage: int = 1) -> float:
    """Autocorrélation empirique au décalage donné (NaN pour une série constante)."""
    x = serie_finie(serie)
    decalage = entier_positif(decalage, "decalage")
    if decalage >= x.size:
        raise ValueError("decalage doit être plus petit que la longueur de la série.")
    y = x - x.mean()
    denom = float(np.dot(y, y))
    if denom == 0:
        return float("nan")
    return float(np.dot(y[:-decalage], y[decalage:]) / denom)


def analyser_avec_references(serie, echelles=ECHELLES_DEFAUT, dimensions=DIMENSIONS_DEFAUT,
                             m: int = 2, r: float = 0.15, n_melanges: int = 20,
                             n_iaaft: int = 20, iterations_iaaft: int = ITERATIONS_IAAFT,
                             graine_melange: int = 0, graine_iaaft: int = 1,
                             taille_blocs: int = 1000) -> dict:
    """
    Analyse complète d'une série : mesures, deux références, statistiques simples.

    C'est l'unité de travail de `scripts/09_analyse_so3.py` (une par cas et
    observable). Elle vit dans le paquet pour pouvoir être exécutée dans un
    processus séparé, y compris sous Windows où chaque processus réimporte les
    fonctions par leur module.

    Retourne les tableaux "pe", "mse" de l'original, "pe_melange", "mse_melange",
    "pe_iaaft", "mse_iaaft" des substituts (empilés selon l'axe 0),
    "ecart_spectral_iaaft" (un par substitut IAAFT), puis "moyenne",
    "ecart_type", "autocorr_1", "autocorr_10", "rapport_variance_blocs",
    "n_points" et "secondes". Les graines fixent entièrement le résultat.
    """
    debut = time.perf_counter()
    x = serie_finie(serie)
    for nom, graine in (("graine_melange", graine_melange), ("graine_iaaft", graine_iaaft)):
        entier_positif(graine, nom, minimum=0)
    reglages = {"echelles": echelles, "dimensions": dimensions, "m": m, "r": r,
                "methode": "arbre"}
    original = mesures_entropie(x, **reglages)
    melange = mesures_substituts(x, lambda s, g: g.permutation(s), n_melanges,
                                 np.random.default_rng(graine_melange), **reglages)
    ecarts: list[float] = []

    def generer_iaaft(s: np.ndarray, g: np.random.Generator) -> np.ndarray:
        y = substitut_iaaft(s, g, iterations=iterations_iaaft)
        ecarts.append(ecart_spectral(s, y))
        return y

    iaaft = mesures_substituts(x, generer_iaaft, n_iaaft,
                               np.random.default_rng(graine_iaaft), **reglages)
    return {
        "pe": original["pe"], "mse": original["mse"],
        "pe_melange": melange["pe"], "mse_melange": melange["mse"],
        "pe_iaaft": iaaft["pe"], "mse_iaaft": iaaft["mse"],
        "ecart_spectral_iaaft": np.array(ecarts),
        "moyenne": float(x.mean()), "ecart_type": float(x.std(ddof=0)),
        "autocorr_1": autocorrelation(x, 1), "autocorr_10": autocorrelation(x, 10),
        "rapport_variance_blocs": rapport_variance_blocs(x, taille_blocs),
        "n_points": int(x.size), "secondes": time.perf_counter() - debut,
    }


def ignorer_interruption() -> None:
    """
    Initialisation d'un processus de calcul : il ignore Ctrl+C (SIGINT).

    Seul le processus principal doit intercepter Ctrl+C, puis arrêter
    proprement les autres. Cette fonction vit dans le paquet (et non dans le
    script) pour qu'un processus lancé par la méthode « spawn » de Windows
    puisse la réimporter par son module.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass
