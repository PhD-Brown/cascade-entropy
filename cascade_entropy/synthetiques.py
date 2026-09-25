"""
Séries synthétiques de comportement connu.

Ces séries servent à valider les modules d'analyse (`entropie`, `indicateurs`)
indépendamment de tout modèle de réseau électrique. Chaque générateur produit une
série dont le comportement attendu est documenté, ce qui permet de vérifier que
les mesures répondent comme la littérature le prévoit.
"""

from __future__ import annotations

import numpy as np


def periodique(n: int, periode: float = 50.0, amplitude: float = 1.0) -> np.ndarray:
    """
    Génère un signal sinusoïdal pur.

    Ce type de signal est très régulier : il répète exactement le même motif à
    intervalle constant. Du point de vue des mesures d'entropie, cela signifie
    qu'il n'y a que quelques motifs ordinaux possibles, donc une entropie de
    permutation très faible.
    """
    t = np.arange(n)
    # On crée une sinusoïde standard avec période `periode` et amplitude donnée.
    return amplitude * np.sin(2.0 * np.pi * t / periode)


def bruit_puissance(
    n: int,
    beta: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Génère un bruit gaussien dont la densité spectrale suit une loi de puissance.

    La stratégie est la suivante :
    - on part d'un bruit blanc,
    - on modifie son spectre selon la loi f^(-beta),
    - on revient dans le domaine temporel,
    - puis on centre et normalise la série.

    Cela permet de contrôler le degré de corrélation :
    - beta = 0 : bruit blanc, pas de mémoire,
    - beta = 1 : bruit rose, corrélation longue portée,
    - beta = 2 : bruit brun, encore plus organisé à long terme.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Bruit blanc gaussien de base, centré et de variance ~ 1.
    blanc = rng.standard_normal(n)
    if beta == 0.0:
        serie = blanc
    else:
        # On travaille en fréquence pour imposer la pente spectrale voulue.
        spectre = np.fft.rfft(blanc)
        freq = np.fft.rfftfreq(n, d=1.0)
        facteur = np.ones_like(freq)
        # La composante continue est gardée telle quelle pour éviter division par zéro.
        facteur[1:] = freq[1:] ** (-beta / 2.0)
        serie = np.fft.irfft(spectre * facteur, n=n)

    # On centre la série pour enlever la moyenne et la normalise pour fixer la variance.
    serie = serie - serie.mean()
    ecart = serie.std()
    return serie / ecart if ecart > 0 else serie


def bruit_blanc(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Bruit non corrélé, correspondant à beta = 0.

    C'est le signal de référence du bruit totalement aléatoire : il sert de point
    de comparaison pour les séries corrélées et pour le cas périodique.
    """
    return bruit_puissance(n, beta=0.0, rng=rng)


def bruit_rose(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Bruit rose, correspondant à beta = 1.

    Il est corrélé à longue portée, ce qui le rend plus structuré que le bruit
    blanc. Ce type de signal est souvent utilisé comme point intermédiaire entre
    un processus complètement aléatoire et un comportement très régulier.
    """
    return bruit_puissance(n, beta=1.0, rng=rng)
