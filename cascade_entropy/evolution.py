"""Dynamique lente sur N journées.

Quatrième module du fil A, et seul point de contact avec le fil B : il écrit les
séries temporelles sur disque. Trois règles s'appliquent entre deux journées, la
croissance de la demande, le renforcement des lignes ayant été surchargées lors
d'un blackout, et l'augmentation de la capacité de production lorsque la marge
descend sous un seuil.

Critère de validation : le taux de charge moyen doit converger vers un plateau, et
la distribution des tailles de blackout doit présenter une queue en loi de
puissance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def simuler(reseau, n_jours: int, parametres, rng: np.random.Generator):
    """Fait évoluer le réseau et retourne les séries temporelles produites."""
    raise NotImplementedError


def ecrire(series: dict[str, np.ndarray], chemin: Path) -> None:
    """Écrit les séries dans un fichier .npz, seul lien avec le fil B."""
    raise NotImplementedError
