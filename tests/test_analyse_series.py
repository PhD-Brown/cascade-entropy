"""Outils d'analyse de la mission 3 de SO2 (`cascade_entropy.analyse_series`).

Séries synthétiques dont la réponse est connue :
- `rs_vectorise` doit égaler `indicateurs.rs_par_echelle` (même définition) ;
- bruit blanc : la pente R/S courte dépasse 0.5 (biais connu) mais tombe dans
  l'intervalle de référence par mélange ;
- bruit corrélé (spectre 1/f) : la pente sort de l'intervalle ;
- sinus de période 1000 : période retrouvée, cassure de R/S autour de 1000 ;
- échantillon de Pareto : indice de la densité retrouvé.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.analyse_series import (
    ccdf_logarithmique,
    echelles_log,
    pdf_logarithmique,
    pentes_depuis_courbe,
    pentes_rs,
    periode_dominante,
    plage_loi_puissance,
    references_melange,
    resume_reference,
    rs_vectorise,
)
from cascade_entropy.indicateurs import rs_par_echelle
from cascade_entropy.synthetiques import bruit_puissance

GRAINE = 20261005
PLAGES = [(10, 365), (600, 5000)]


@pytest.fixture(scope="module")
def blanc():
    return np.random.default_rng(GRAINE).standard_normal(20_000)


# --------------------------------------------------------------------------
# R/S
# --------------------------------------------------------------------------


def test_rs_vectorise_egal_a_la_reference(blanc):
    # Série avec des blocs constants (zéros), comme une série de délestage.
    x = blanc.copy()
    x[x < 0.5] = 0.0
    e = np.unique(np.logspace(np.log10(4), np.log10(15_000), 25).astype(int))
    t1, v1 = rs_par_echelle(x, e)
    t2, v2 = rs_vectorise(x, e)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_allclose(v1, v2, rtol=1e-12)


def test_echelles_log_bornes():
    e = echelles_log(100_000)
    assert e[0] == 8 and e[-1] == 25_000 and np.all(np.diff(e) > 0)
    with pytest.raises(ValueError):
        echelles_log(20)


def test_bruit_blanc_biais_court_dans_la_reference(blanc):
    h = pentes_rs(blanc, PLAGES)
    assert 0.5 < h[0] < 0.62  # biais d'Anis-Lloyd aux petites échelles
    ref = references_melange(blanc, PLAGES, 40, np.random.default_rng(1))
    med, _, _ = resume_reference(ref)
    # Le bruit blanc est statistiquement identique à ses mélanges : sa pente
    # est un tirage de la même loi (écart à la médiane < 4 écarts-types).
    ecart_type = np.nanstd(ref, axis=0)
    assert np.all(np.abs(h - med) < 4 * ecart_type)
    assert 0.5 < med[0] < 0.62  # le biais est aussi présent dans la référence


def test_bruit_correle_sort_de_la_reference():
    x = bruit_puissance(20_000, beta=0.8, rng=np.random.default_rng(GRAINE))
    h = pentes_rs(x, PLAGES)
    _, _, haut = resume_reference(references_melange(x, PLAGES, 20,
                                                     np.random.default_rng(2)))
    assert h[0] > haut[0] + 0.05 and h[1] > haut[1] + 0.05


def test_sinus_casse_la_courbe_rs():
    t = np.arange(40_000)
    x = np.sin(2 * np.pi * t / 1000) + 0.3 * np.random.default_rng(3).standard_normal(t.size)
    avant, apres = pentes_rs(x, [(200, 900), (2000, 10_000)])
    assert avant > 0.8 and apres < 0.3


def test_pentes_depuis_courbe_egale_pentes_rs(blanc):
    t, v = rs_vectorise(blanc, echelles_log(blanc.size))
    np.testing.assert_array_equal(pentes_depuis_courbe(t, v, PLAGES), pentes_rs(blanc, PLAGES))
    with pytest.raises(ValueError):
        pentes_depuis_courbe(t, v[:-1], PLAGES)


def test_plage_trop_longue_donne_nan(blanc):
    h = pentes_rs(blanc, [(10, 365), (6000, 100_000)])
    assert np.isfinite(h[0]) and np.isnan(h[1])


def test_reference_reproductible(blanc):
    a = references_melange(blanc, PLAGES, 3, np.random.default_rng(7))
    b = references_melange(blanc, PLAGES, 3, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (3, 2)


@pytest.mark.parametrize("plages", [[], [(10,)], [(365, 10)], [(0, 10)], [(10, np.nan)]])
def test_plages_invalides(blanc, plages):
    with pytest.raises(ValueError):
        pentes_rs(blanc, plages)


def test_arguments_invalides(blanc):
    with pytest.raises(ValueError):
        references_melange(blanc, PLAGES, 0, np.random.default_rng(0))
    with pytest.raises(TypeError):
        references_melange(blanc, PLAGES, 2, 42)
    with pytest.raises(ValueError):
        resume_reference(np.ones((3, 2)), niveau=1.0)
    with pytest.raises(ValueError):
        resume_reference(np.ones(3))


def test_resume_reference_ignore_les_nan():
    ref = np.array([[0.5, np.nan], [0.6, np.nan], [0.7, np.nan]])
    med, bas, haut = resume_reference(ref, niveau=0.5)
    assert med[0] == pytest.approx(0.6) and bas[0] < 0.6 < haut[0]
    assert np.isnan(med[1]) and np.isnan(bas[1]) and np.isnan(haut[1])


# --------------------------------------------------------------------------
# Période dominante
# --------------------------------------------------------------------------


def test_periode_dominante_sinus():
    t = np.arange(100_000)
    rng = np.random.default_rng(4)
    x = np.sin(2 * np.pi * t / 1000) + rng.standard_normal(t.size)
    periode, fraction = periode_dominante(x, 50, 20_000)
    assert periode == pytest.approx(1000, rel=0.02)
    assert fraction > 0.5


def test_periode_dominante_bruit_sans_pic_marque(blanc):
    _, fraction = periode_dominante(blanc, 50, 5000)
    assert fraction < 0.25


def test_periode_dominante_serie_constante():
    periode, fraction = periode_dominante(np.ones(5000), 10, 1000)
    assert np.isnan(periode) and fraction == 0.0


@pytest.mark.parametrize("bornes", [(1, 100), (100, 50), (10, 10 ** 7), (-5, 100)])
def test_periode_dominante_bornes_invalides(blanc, bornes):
    with pytest.raises(ValueError):
        periode_dominante(blanc, *bornes)


# --------------------------------------------------------------------------
# Queue de distribution
# --------------------------------------------------------------------------


def test_pdf_logarithmique_compte_toutes_les_valeurs():
    """Régression : 10**log10(max) < max excluait la plus grande valeur (Windows)."""
    # Pour ce maximum, la dernière borne calculée par logspace est inférieure
    # d'un ulp à la valeur elle-même, sur toute plateforme.
    valeurs = np.array([0.0, 1e-3, 0.5, 12.0, 6369.617873214544])
    _, _, effectifs = pdf_logarithmique(valeurs, 25)
    assert effectifs.sum() == 4
    rng = np.random.default_rng(11)
    for _ in range(50):
        x = rng.random(rng.integers(2, 500)) * 10.0 ** rng.integers(-4, 5)
        _, _, effectifs = pdf_logarithmique(x, int(rng.integers(3, 40)))
        assert effectifs.sum() == np.count_nonzero(x > 0)


def test_pareto_indice_retrouve():
    rng = np.random.default_rng(5)
    alpha = 0.6  # densité ∝ x^-(1 + alpha) sur [1, ∞[
    x = rng.pareto(alpha, 200_000) + 1.0
    x = x[x < 1e4]
    centres, densite, effectifs = pdf_logarithmique(np.concatenate([np.zeros(500), x]), 30)
    assert effectifs.sum() == x.size  # les zéros sont exclus
    res = plage_loi_puissance(centres, densite, effectifs, r2_min=0.99)
    assert res["pente"] == pytest.approx(-(1 + alpha), abs=0.1)
    assert res["etendue"] > 100 and res["r2"] >= 0.99


def test_pente_max_exclut_les_plages_croissantes():
    centres = np.logspace(0, 3, 10)
    densite = np.concatenate([centres[:6] ** 1.0, centres[5] * (centres[6:] / centres[5]) ** -1.0])
    effectifs = np.full(10, 100.0)
    libre = plage_loi_puissance(centres, densite, effectifs, points_min=4)
    queue = plage_loi_puissance(centres, densite, effectifs, points_min=4, pente_max=0.0)
    assert libre["pente"] == pytest.approx(1.0)
    assert queue["pente"] == pytest.approx(-1.0) and queue["n_classes"] == 5
    with pytest.raises(ValueError):
        plage_loi_puissance(centres, densite, effectifs, pente_max=np.inf)


def test_aucune_plage_lineaire():
    centres = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    densite = np.array([1.0, 10.0, 1.0, 10.0, 1.0])
    res = plage_loi_puissance(centres, densite, np.full(5, 100.0))
    assert res["n_classes"] == 0 and np.isnan(res["pente"])


def test_queue_arguments_invalides():
    with pytest.raises(ValueError):
        pdf_logarithmique(np.zeros(10))
    with pytest.raises(ValueError):
        plage_loi_puissance(np.ones(3), np.ones(4), np.ones(3))
    with pytest.raises(ValueError):
        plage_loi_puissance(np.ones(5), np.ones(5), np.ones(5), r2_min=0.0)


# --------------------------------------------------------------------------
# Fréquence cumulée (Table I de Carreras 2004)
# --------------------------------------------------------------------------


def test_ccdf_logarithmique_definition():
    x, f, n = ccdf_logarithmique(np.array([0.0, 1.0, 2.0, 2.0, 4.0]), 3)
    np.testing.assert_allclose(x, [1.0, 2.0, 4.0])
    np.testing.assert_allclose(f, [1.0, 0.75, 0.25])
    np.testing.assert_array_equal(n, [4, 3, 1])


def test_ccdf_pareto_pente_cumulee():
    """Densité x^-(1+a) : pente cumulée −a, soit la pente de densité + 1."""
    alpha = 0.55
    # Pas de troncature : elle courberait la fréquence cumulée vers le bas.
    x = np.random.default_rng(8).pareto(alpha, 300_000) + 1.0
    grille, f, n = ccdf_logarithmique(x, 40)
    res = plage_loi_puissance(grille, f, n, r2_min=0.99, effectif_min=50, pente_max=0.0)
    assert res["pente"] == pytest.approx(-alpha, abs=0.07)
    assert res["etendue"] > 100


def test_ccdf_arguments_invalides():
    with pytest.raises(ValueError):
        ccdf_logarithmique(np.zeros(5))
    with pytest.raises(ValueError):
        ccdf_logarithmique(np.array([1.0, 2.0]), 2)
