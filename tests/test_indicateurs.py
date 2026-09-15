"""Validation du module `indicateurs`.

Critère du plan de travail : l'exposant de Hurst d'un bruit non corrélé doit être
voisin de 0,5. On vérifie aussi que des séries corrélées à longue portée donnent
un exposant nettement supérieur, faute de quoi la mesure ne discriminerait rien.

Lancer depuis la racine du dépôt : pytest
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.indicateurs import hurst
from cascade_entropy.synthetiques import bruit_puissance

N = 8192
N_REALISATIONS = 30
GRAINE = 20260915

# Tolérance sur le bruit non corrélé. Volontairement large : la méthode R/S
# surestime H sur des séries finies, et ce biais est lui-même un objet d'étude du
# projet avant toute interprétation de persistance.
TOLERANCE_BRUIT_BLANC = 0.06


def exposants(beta: float) -> np.ndarray:
    """Exposants obtenus sur plusieurs réalisations d'un même type de bruit."""
    rng = np.random.default_rng(GRAINE)
    return np.array(
        [hurst(bruit_puissance(N, beta=beta, rng=rng)) for _ in range(N_REALISATIONS)]
    )


def test_bruit_non_correle_proche_de_un_demi():
    valeurs = exposants(beta=0.0)
    assert abs(valeurs.mean() - 0.5) < TOLERANCE_BRUIT_BLANC


def test_bruit_en_un_sur_f_persistant():
    assert exposants(beta=1.0).mean() > 0.7


def test_bruit_brun_pente_rs_elevee():
    assert exposants(beta=2.0).mean() > 0.85


def test_ordre_des_regimes():
    """L'exposant doit croître avec le degré de corrélation de la série."""
    assert exposants(0.0).mean() < exposants(1.0).mean() < exposants(2.0).mean()


def test_reproductibilite_avec_graine_fixee():
    a = hurst(bruit_puissance(N, 0.0, np.random.default_rng(1)))
    b = hurst(bruit_puissance(N, 0.0, np.random.default_rng(1)))
    assert a == b


def test_serie_trop_courte_rejetee():
    with pytest.raises(ValueError):
        hurst(np.arange(10, dtype=float))
