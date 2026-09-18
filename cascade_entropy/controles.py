"""Contrôles méthodologiques sur les mesures d'analyse.

Ces procédures ne mesurent pas le système étudié : elles vérifient ce à quoi une
mesure répond réellement. Le contrôle par permutation compare une série à des
réarrangements de ses propres valeurs, ce qui conserve exactement l'histogramme,
les extrema et tous les moments, et ne modifie que l'ordre temporel. Si la mesure
bouge, c'est bien à l'ordre qu'elle répond.

Le code vit ici plutôt que dans un script afin d'être testable et réutilisable.
Les scripts de `scripts/` ne font qu'orchestrer et tracer.
"""

from __future__ import annotations

import numpy as np

from ._validation import entier_positif
from .entropie import multiechelle
from .synthetiques import bruit_rose


def analyser(n=8192, n_melanges=20, echelles=10, m=2, r=0.15,
             graine=20260915, progression=True):
    """Retourne la série originale et toutes les courbes, sans écrire de fichier."""
    n = entier_positif(n, "n")
    n_melanges = entier_positif(n_melanges, "n_melanges", minimum=2)
    echelles = entier_positif(echelles, "echelles")
    m = entier_positif(m, "m")
    if not np.isfinite(r) or r <= 0:
        raise ValueError("r doit être fini et strictement positif.")
    if n // echelles < 10 * (m + 1):
        raise ValueError("Trop peu de points à la dernière échelle : augmenter n "
                         "ou réduire echelles.")
    # Deux flux indépendants : changer le nombre de mélanges ne change pas x.
    semences = np.random.SeedSequence(graine).spawn(2)
    rng_signal = np.random.default_rng(semences[0])
    rng_melanges = np.random.default_rng(semences[1])
    originale = bruit_rose(n, rng_signal)
    triee = np.sort(originale)
    if progression:
        print(f"Originale : N={n}, {echelles} échelles", flush=True)
    tau, mse_originale = multiechelle(originale, echelles=echelles, m=m, r=r)
    courbes = []
    for i in range(n_melanges):
        melangee = rng_melanges.permutation(originale)
        # Vérification exacte du multiensemble, plus forte qu'un histogramme.
        if not np.array_equal(np.sort(melangee), triee):
            raise RuntimeError("Un mélange a modifié les valeurs de la série.")
        tau_m, mse = multiechelle(melangee, echelles=echelles, m=m, r=r)
        if not np.array_equal(tau_m, tau):
            raise RuntimeError("Les échelles diffèrent entre les courbes.")
        courbes.append(mse)
        if progression:
            print(f"Mélange {i + 1}/{n_melanges} terminé", flush=True)
    courbes = np.asarray(courbes)
    if not np.isfinite(mse_originale).all() or not np.isfinite(courbes).all():
        raise ValueError("SampEn indéfinie à au moins une échelle : augmenter N "
                         "ou revoir r. Aucune courbe n'est omise de la moyenne.")
    return dict(serie=originale, echelles=tau, originale=mse_originale,
                melanges=courbes, moyenne=courbes.mean(axis=0),
                ecart_type=courbes.std(axis=0, ddof=1))
