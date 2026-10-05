"""Fenêtrage des événements et détection du régime stationnaire (SO2)."""

from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.indicateurs import comptes_par_fenetre, troncature_mser

GRAINE = 20261005


def test_comptes_par_fenetre_exact():
    ev = np.array([1, 0, 1, 1, 0, 0, 1])
    np.testing.assert_array_equal(comptes_par_fenetre(ev, 3), [2, 1])
    np.testing.assert_array_equal(comptes_par_fenetre(ev.astype(bool), 1), ev[:7])


@pytest.mark.parametrize("ev,largeur", [
    (np.array([]), 3), (np.array([0, 2, 1]), 1), (np.ones((2, 2)), 1),
    (np.ones(5), 0), (np.ones(2), 3),
])
def test_comptes_par_fenetre_entrees_invalides(ev, largeur):
    with pytest.raises(ValueError):
        comptes_par_fenetre(ev, largeur)


def _mser_naif(ev, fenetre=300):
    y = comptes_par_fenetre(ev, fenetre).astype(float)
    n = y.size
    valeurs = [((y[d:] - y[d:].mean()) ** 2).sum() / (n - d) ** 2 for d in range(n // 2)]
    return int(np.argmin(valeurs)) * fenetre


def test_mser_egal_a_la_definition():
    rng = np.random.default_rng(GRAINE)
    t = np.arange(60_000)
    ev = rng.random(t.size) < np.where(t < 15_000, 0.01, 0.05)
    assert troncature_mser(ev) == _mser_naif(ev)


def test_serie_stationnaire_tronque_tres_peu():
    rng = np.random.default_rng(GRAINE)
    debuts = [troncature_mser(rng.random(120_000) < 0.04) for _ in range(20)]
    assert np.median(debuts) <= 1_500
    assert max(debuts) <= 10_000


def test_transitoire_detecte():
    """Fréquence 0.005 pendant 20 000 jours, puis 0.04 : saut net à 20 000."""
    rng = np.random.default_rng(GRAINE)
    t = np.arange(120_000)
    ev = rng.random(t.size) < np.where(t < 20_000, 0.005, 0.04)
    assert 19_000 <= troncature_mser(ev) <= 21_000


def test_parametres_invalides():
    with pytest.raises(ValueError):
        troncature_mser(np.zeros(900), fenetre=300)
    with pytest.raises(ValueError):
        troncature_mser(np.zeros(10_000), fraction_max=0.0)
