"""
Validation du module `indicateurs`.

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
    """
    Calcule l'exposant de Hurst sur plusieurs réalisations d'un même bruit.
    Cela permet de lisser la variabilité d'échantillonnage et d'étudier la tendance
    moyenne du comportement du signal.
    """
    rng = np.random.default_rng(GRAINE)
    return np.array(
        [hurst(bruit_puissance(N, beta=beta, rng=rng)) for _ in range(N_REALISATIONS)]
    )


def test_bruit_non_correle_proche_de_un_demi():
    """
    Vérifie que le bruit non corrélé a un exposant de Hurst proche de 0,5, ce qui
    correspond à un comportement aléatoire sans mémoire persistante ni antipersistante.
    """
    valeurs = exposants(beta=0.0)
    assert abs(valeurs.mean() - 0.5) < TOLERANCE_BRUIT_BLANC


def test_bruit_en_un_sur_f_persistant():
    """
    Vérifie que le bruit 1/f, plus corrélé à longue portée, donne un exposant de Hurst
    nettement supérieur à 0,5, signe d'une persistance statistique.
    """
    assert exposants(beta=1.0).mean() > 0.7


def test_bruit_brun_pente_rs_elevee():
    """
    Vérifie que le bruit brun, encore plus corrélé à longue portée, donne un exposant
    de Hurst encore plus élevé, confirmant la montée de la persistance avec la pente.
    """
    assert exposants(beta=2.0).mean() > 0.85


def test_ordre_des_regimes():
    """
    Vérifie que l'exposant de Hurst croît avec le degré de corrélation du bruit.
    Le bruit blanc doit être inférieur au 1/f, lui-même inférieur au bruit brun.
    """
    assert exposants(0.0).mean() < exposants(1.0).mean() < exposants(2.0).mean()


def test_reproductibilite_avec_graine_fixee():
    """
    Vérifie la reproductibilité du calcul Hurst avec une graine fixe. Deux séries
    identiques générées avec la même graine doivent donner exactement le même résultat.
    """
    a = hurst(bruit_puissance(N, 0.0, np.random.default_rng(1)))
    b = hurst(bruit_puissance(N, 0.0, np.random.default_rng(1)))
    assert a == b


def test_serie_trop_courte_rejetee():
    """
    Vérifie que la fonction rejette une série trop courte pour estimer un exposant de Hurst,
    en levant une ValueError pour éviter un calcul non fiable.
    """
    with pytest.raises(ValueError):
        hurst(np.arange(10, dtype=float))
