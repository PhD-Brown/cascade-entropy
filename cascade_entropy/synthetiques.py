"""Séries synthétiques de comportement connu.

Ces séries servent à valider les modules d'analyse (`entropie`, `indicateurs`)
indépendamment de tout modèle de réseau électrique. Chaque générateur produit une
série dont le comportement attendu est documenté, ce qui permet de vérifier que
les mesures répondent comme la littérature le prévoit.
"""

from __future__ import annotations

import numpy as np


def periodique(n: int, periode: float = 50.0, amplitude: float = 1.0) -> np.ndarray:
    """Signal sinusoïdal pur.

    Comportement attendu : entropie de permutation très basse (peu de motifs
    ordinaux distincts), exposant de Hurst non interprétable car la série n'est
    pas stochastique.
    """
    t = np.arange(n)
    return amplitude * np.sin(2.0 * np.pi * t / periode)


def bruit_puissance(
    n: int,
    beta: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Bruit gaussien de densité spectrale S(f) proportionnelle à f**(-beta).

    Synthèse spectrale : on tire un bruit blanc, on met sa transformée de Fourier
    à l'échelle par f**(-beta/2), puis on revient dans le domaine temporel.

    beta = 0 donne un bruit blanc (non corrélé).
    beta = 1 donne un bruit en 1/f (corrélé à longue portée).
    beta = 2 donne une marche aléatoire.

    La série est centrée et normalisée à variance unitaire.
    """
    if rng is None:
        rng = np.random.default_rng()

    blanc = rng.standard_normal(n)
    if beta == 0.0:
        serie = blanc
    else:
        spectre = np.fft.rfft(blanc)
        freq = np.fft.rfftfreq(n, d=1.0)
        facteur = np.ones_like(freq)
        # La composante continue est laissée intacte pour éviter une division par zéro.
        facteur[1:] = freq[1:] ** (-beta / 2.0)
        serie = np.fft.irfft(spectre * facteur, n=n)

    serie = serie - serie.mean()
    ecart = serie.std()
    return serie / ecart if ecart > 0 else serie


def bruit_blanc(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Bruit non corrélé. Exposant de Hurst attendu voisin de 0,5."""
    return bruit_puissance(n, beta=0.0, rng=rng)


def bruit_rose(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Bruit en 1/f, corrélé à longue portée.

    Comportement attendu : exposant de Hurst nettement supérieur à 0,5, et
    entropie multiéchelle qui se maintient là où celle du bruit blanc s'effondre.
    """
    return bruit_puissance(n, beta=1.0, rng=rng)
