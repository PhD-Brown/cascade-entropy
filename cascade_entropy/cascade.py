"""Boucle de cascade sur une journée.

Troisième module du fil A. Une ligne est dite surchargée lorsqu'elle est à moins
de 1 % de sa limite : le dispatch n'autorise jamais le dépassement, c'est la
saturation qui déclenche l'avarie. Chaque ligne surchargée tombe avec une
probabilité p1, le dispatch est relancé, et le processus se répète jusqu'à
stabilisation.

Critère de validation : avec p1 nul, aucune panne ne doit survenir ; avec une
graine aléatoire fixée, une même cascade doit se rejouer à l'identique.
"""

from __future__ import annotations

import numpy as np

SEUIL_SURCHARGE = 0.99


def journee(reseau, demande: np.ndarray, p1: float, rng: np.random.Generator):
    """Simule une journée et retourne le délestage et les lignes tombées."""
    raise NotImplementedError
