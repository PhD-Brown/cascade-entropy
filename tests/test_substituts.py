"""
Substituts IAAFT (`cascade_entropy.substituts`).

Critères : distribution conservée exactement ; spectre conservé à mieux que
2 % (un mélange le détruit) ; reproductible pour une graine ; un processus
linéaire gaussien AR(1) est indiscernable de ses substituts par SampEn ; une
dynamique déterministe non linéaire (application logistique) en est
nettement distinguée.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.entropie import echantillon
from cascade_entropy.substituts import ecart_spectral, substitut_iaaft

GRAINE = 20261005
N = 3000


def _ar1(n=N, phi=0.8, graine=GRAINE):
    rng = np.random.default_rng(graine)
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _logistique(n=N):
    x = np.empty(n)
    x[0] = 0.3
    for i in range(1, n):
        x[i] = 4.0 * x[i - 1] * (1.0 - x[i - 1])
    return x


def test_distribution_conservee_exactement():
    x = _ar1()
    y = substitut_iaaft(x, np.random.default_rng(1))
    np.testing.assert_array_equal(np.sort(y), np.sort(x))
    assert not np.array_equal(y, x)


def test_spectre_conserve_et_melange_le_detruit():
    x = _ar1()
    rng = np.random.default_rng(2)
    assert ecart_spectral(x, substitut_iaaft(x, rng)) < 0.02
    assert ecart_spectral(x, rng.permutation(x)) > 0.3


def test_reproductible_pour_une_graine():
    x = _ar1()
    a = substitut_iaaft(x, np.random.default_rng(3))
    b = substitut_iaaft(x, np.random.default_rng(3))
    c = substitut_iaaft(x, np.random.default_rng(4))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def _z_sampen(x, n_substituts=12):
    rng = np.random.default_rng(5)
    ref = np.array([echantillon(substitut_iaaft(x, rng), methode="arbre")
                    for _ in range(n_substituts)])
    return (echantillon(x, methode="arbre") - ref.mean()) / ref.std(ddof=1)


def test_processus_lineaire_indiscernable():
    """AR(1) gaussien : tout est dans le spectre, SampEn reste dans la référence."""
    assert abs(_z_sampen(_ar1())) < 4


def test_dynamique_non_lineaire_distinguee():
    """Application logistique : déterministe, bien plus régulière que ses substituts."""
    assert _z_sampen(_logistique()) < -10


def test_serie_constante_rendue_telle_quelle():
    x = np.full(100, 2.5)
    y = substitut_iaaft(x, np.random.default_rng(0))
    np.testing.assert_array_equal(x, y)
    assert y is not x


def test_ecart_spectral_nul_pour_series_identiques():
    x = _ar1(500)
    assert ecart_spectral(x, x) == 0.0


@pytest.mark.parametrize("mauvais", [
    dict(serie=[1.0, 2.0, 3.0]),                       # trop courte
    dict(serie=[1.0, np.nan, 2.0, 3.0, 4.0]),          # NaN
    dict(serie=np.ones((4, 4))),                       # 2D
    dict(iterations=0),
    dict(iterations=True),
    dict(iterations=2.5),
])
def test_entrees_invalides(mauvais):
    arguments = {"serie": _ar1(64), "rng": np.random.default_rng(0), "iterations": 10}
    arguments.update(mauvais)
    with pytest.raises(ValueError):
        substitut_iaaft(**arguments)


def test_rng_obligatoire():
    with pytest.raises(TypeError):
        substitut_iaaft(_ar1(64), rng=3)


def test_ecart_spectral_longueurs_differentes():
    with pytest.raises(ValueError):
        ecart_spectral(np.arange(10.0), np.arange(12.0))
