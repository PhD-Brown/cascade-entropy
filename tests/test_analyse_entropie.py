"""
Batterie de mesures de SO3 (`cascade_entropy.analyse_entropie`).

Critères : diagnostics d'égalités exacts sur un exemple construit ; PE
multiéchelle égale à `permutation` sur la série granularisée, NaN sous 10·d!
points ; MSE égale à `multiechelle` ; substituts reproductibles ; écart réduit
conforme à sa formule ; rapport de variance des blocs ≈ 1 sur un bruit blanc
et > 1 avec une composante lente ; autocorrélation d'un AR(1).
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.analyse_entropie import (
    autocorrelation,
    diagnostic_egalites,
    mesures_entropie,
    mesures_substituts,
    permutation_multiechelle,
    rapport_variance_blocs,
    resume_substituts,
)
from cascade_entropy.entropie import granulariser, multiechelle, permutation

GRAINE = 20261005


def _bruit(n=4000, graine=GRAINE):
    return np.random.default_rng(graine).standard_normal(n)


def test_diagnostic_egalites_sur_exemple_construit():
    d = diagnostic_egalites([0.0, 0.0, 1.0, 1.0, 2.0, 0.0])
    assert d["fraction_zeros"] == pytest.approx(3 / 6)
    assert d["fraction_egalites_consecutives"] == pytest.approx(2 / 5)
    assert d["fraction_valeurs_repetees"] == pytest.approx(1 - 3 / 6)


def test_diagnostic_egalites_bruit_continu():
    d = diagnostic_egalites(_bruit())
    assert d == {"fraction_zeros": 0.0, "fraction_egalites_consecutives": 0.0,
                 "fraction_valeurs_repetees": 0.0}


def test_permutation_multiechelle_egale_permutation_granularisee():
    x = _bruit()
    pe = permutation_multiechelle(x, echelles=[1, 4], dimension=3)
    assert pe[0] == permutation(x, dimension=3)
    assert pe[1] == permutation(granulariser(x, 4), dimension=3)


def test_permutation_multiechelle_nan_sous_dix_fois_d_factorielle():
    x = _bruit(2000)
    pe = permutation_multiechelle(x, echelles=[1, 10], dimension=4)  # 10 · 4! = 240
    assert np.isfinite(pe[0])
    assert np.isnan(pe[1])  # 2000 // 10 = 200 < 240


def test_mesures_entropie_formes_et_mse():
    x = _bruit(3000)
    res = mesures_entropie(x, echelles=[1, 2, 5], dimensions=[3, 4], methode="arbre")
    assert res["pe"].shape == (2, 3)
    assert res["mse"].shape == (3,)
    _, attendu = multiechelle(x, echelles=[1, 2, 5], methode="blocs")
    np.testing.assert_array_equal(res["mse"], attendu)


def test_mesures_substituts_reproductibles_et_formes():
    x = _bruit(1500)
    reglages = dict(echelles=[1, 2], dimensions=[3], methode="arbre")
    a = mesures_substituts(x, lambda s, g: g.permutation(s), 3, np.random.default_rng(1), **reglages)
    b = mesures_substituts(x, lambda s, g: g.permutation(s), 3, np.random.default_rng(1), **reglages)
    assert a["pe"].shape == (3, 1, 2) and a["mse"].shape == (3, 2)
    np.testing.assert_array_equal(a["mse"], b["mse"])
    np.testing.assert_array_equal(a["pe"], b["pe"])


def test_mesures_substituts_refuse_un_substitut_de_mauvaise_forme():
    with pytest.raises(ValueError):
        mesures_substituts(_bruit(500), lambda s, g: s[:-1], 2, np.random.default_rng(0),
                           echelles=[1], dimensions=[3])


def test_mesures_substituts_validations():
    with pytest.raises(TypeError):
        mesures_substituts(_bruit(500), lambda s, g: s, 2, rng=1, echelles=[1], dimensions=[3])
    with pytest.raises(TypeError):
        mesures_substituts(_bruit(500), "pas une fonction", 2, np.random.default_rng(0))
    with pytest.raises(ValueError):
        mesures_substituts(_bruit(500), lambda s, g: s, 0, np.random.default_rng(0))


def test_resume_substituts_formule():
    refs = np.array([[1.0, 10.0], [2.0, 10.0], [3.0, 10.0]])
    r = resume_substituts(np.array([4.0, 10.0]), refs)
    assert r["moyenne"][0] == pytest.approx(2.0)
    assert r["ecart_type"][0] == pytest.approx(1.0)
    assert r["z"][0] == pytest.approx(2.0)
    assert np.isnan(r["z"][1])  # écart-type nul


def test_resume_substituts_ignore_les_nan():
    refs = np.array([[1.0, np.nan], [3.0, np.nan], [np.nan, 5.0]])
    r = resume_substituts(np.array([2.0, 5.0]), refs)
    assert r["moyenne"][0] == pytest.approx(2.0)
    assert r["z"][0] == pytest.approx(0.0)
    assert np.isnan(r["z"][1])  # un seul substitut fini


def test_resume_substituts_forme_incompatible():
    with pytest.raises(ValueError):
        resume_substituts(np.zeros(3), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        resume_substituts(np.zeros(2), np.zeros((4, 2)), niveau=1.5)


def test_rapport_variance_blocs():
    blanc = _bruit(100_000)
    assert 0.75 < rapport_variance_blocs(blanc, 1000) < 1.25
    t = np.arange(100_000)
    lent = blanc + 0.3 * np.sin(2 * np.pi * t / 20_000)
    assert rapport_variance_blocs(lent, 1000) > 3
    assert np.isnan(rapport_variance_blocs(blanc[:1500], 1000))
    assert np.isnan(rapport_variance_blocs(np.ones(5000), 1000))


def test_autocorrelation_ar1():
    rng = np.random.default_rng(GRAINE)
    e = rng.standard_normal(50_000)
    x = np.zeros_like(e)
    for i in range(1, e.size):
        x[i] = 0.8 * x[i - 1] + e[i]
    assert autocorrelation(x, 1) == pytest.approx(0.8, abs=0.02)
    assert abs(autocorrelation(e, 1)) < 0.02
    assert np.isnan(autocorrelation(np.ones(10), 1))
    with pytest.raises(ValueError):
        autocorrelation(x[:5], 5)


@pytest.mark.parametrize("echelles", [[], [3, 1], [0], "12", [1, 1]])
def test_echelles_invalides(echelles):
    with pytest.raises(ValueError):
        permutation_multiechelle(_bruit(500), echelles=echelles)


@pytest.mark.parametrize("dimensions", [[], [1], [3, 3], "3"])
def test_dimensions_invalides(dimensions):
    with pytest.raises(ValueError):
        mesures_entropie(_bruit(500), echelles=[1], dimensions=dimensions)


def test_methode_invalide():
    with pytest.raises(ValueError):
        mesures_entropie(_bruit(500), echelles=[1], dimensions=[3], methode="kd")


def test_analyser_avec_references_reproductible_et_complet():
    from cascade_entropy.analyse_entropie import analyser_avec_references

    x = _bruit(2500)
    reglages = dict(echelles=[1, 5], dimensions=[3], n_melanges=3, n_iaaft=3,
                    iterations_iaaft=20, graine_melange=7, graine_iaaft=8, taille_blocs=250)
    a = analyser_avec_references(x, **reglages)
    b = analyser_avec_references(x, **reglages)
    for cle in ("pe", "mse", "pe_melange", "mse_melange", "pe_iaaft", "mse_iaaft",
                "ecart_spectral_iaaft"):
        np.testing.assert_array_equal(a[cle], b[cle])
    assert a["mse_iaaft"].shape == (3, 2) and a["pe_melange"].shape == (3, 1, 2)
    assert a["n_points"] == 2500 and np.all(a["ecart_spectral_iaaft"] < 0.05)
    with pytest.raises(ValueError):
        analyser_avec_references(x, **{**reglages, "graine_iaaft": -1})


def test_analyser_avec_references_executable_dans_un_processus_spawn():
    """La fonction doit être importable par un processus neuf (méthode Windows)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from cascade_entropy.analyse_entropie import analyser_avec_references

    x = _bruit(800)
    reglages = dict(echelles=[1], dimensions=[3], n_melanges=2, n_iaaft=2,
                    iterations_iaaft=5, graine_melange=1, graine_iaaft=2, taille_blocs=100)
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        distant = ex.submit(analyser_avec_references, x, **reglages).result()
    local = analyser_avec_references(x, **reglages)
    np.testing.assert_array_equal(distant["mse_iaaft"], local["mse_iaaft"])


def test_ignorer_interruption_dans_un_processus_spawn():
    """L'initialiseur des processus doit être importable par un processus neuf (Windows)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from cascade_entropy.analyse_entropie import autocorrelation, ignorer_interruption

    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
                             initializer=ignorer_interruption) as ex:
        assert ex.submit(autocorrelation, np.arange(10.0), 1).result() > 0
