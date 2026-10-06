"""
Option `methode="arbre"` de `entropie.echantillon` et liste d'échelles de `multiechelle`.

Critère : l'arbre k-d compte EXACTEMENT les mêmes paires que le calcul par
blocs historique, y compris sur des séries à égalités (entiers, zéros) et
quand la tolérance tombe pile sur un écart existant. Les défauts historiques
(`methode="blocs"`, `echelles` entier) ne changent pas.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.entropie import (
    METHODES_SAMPEN,
    _compter_paires,
    _compter_paires_arbre,
    echantillon,
    multiechelle,
)

GRAINE = 20261005


def _series():
    rng = np.random.default_rng(GRAINE)
    n = 1500
    return {
        "gauss": rng.standard_normal(n),
        "entiers": rng.integers(0, 6, n).astype(float),
        "zeros": np.where(rng.random(n) < 0.9, 0.0, rng.lognormal(size=n)),
        "marche": np.cumsum(rng.standard_normal(n)),
    }


@pytest.mark.parametrize("nom", ["gauss", "entiers", "zeros", "marche"])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_comptes_identiques_blocs_et_arbre(nom, m):
    """Mêmes comptes de paires, pour des tolérances relatives et entières exactes."""
    x = _series()[nom]
    modeles = np.lib.stride_tricks.sliding_window_view(x, m + 1)[: x.size - m]
    for tolerance in (0.15 * x.std(), 1.0, 2.0):
        assert _compter_paires(modeles, tolerance) == _compter_paires_arbre(modeles, tolerance)


@pytest.mark.parametrize("nom", ["gauss", "entiers", "zeros", "marche"])
def test_sampen_identique(nom):
    x = _series()[nom]
    a = echantillon(x, methode="blocs")
    b = echantillon(x, methode="arbre")
    assert (np.isnan(a) and np.isnan(b)) or a == b


def test_defaut_historique_inchange():
    x = _series()["gauss"]
    assert echantillon(x) == echantillon(x, methode="blocs")
    e1, v1 = multiechelle(x, echelles=5)
    e2, v2 = multiechelle(x, echelles=5, methode="blocs")
    np.testing.assert_array_equal(e1, e2)
    np.testing.assert_array_equal(v1, v2)


def test_cas_sans_paire_et_constant_donnent_nan_dans_les_deux_methodes():
    constante = np.ones(200)
    assert np.isnan(echantillon(constante, methode="arbre"))
    assert np.isnan(echantillon(constante, methode="blocs"))
    rampe = np.arange(50, dtype=float)  # aucune paire à moins de r·σ pour m + 1 points
    assert np.isnan(echantillon(rampe, r=0.01, methode="arbre"))
    assert np.isnan(echantillon(rampe, r=0.01, methode="blocs"))


@pytest.mark.parametrize("mauvaise", ["kd", "", None, 1, "ARBRE"])
def test_methode_invalide_refusee(mauvaise):
    x = _series()["gauss"]
    with pytest.raises(ValueError):
        echantillon(x, methode=mauvaise)
    with pytest.raises(ValueError):
        multiechelle(x, echelles=3, methode=mauvaise)


def test_methodes_declarees():
    assert METHODES_SAMPEN == ("blocs", "arbre")


def test_multiechelle_liste_egale_au_chemin_entier():
    """Une liste d'échelles redonne exactement les valeurs du chemin historique."""
    x = _series()["gauss"]
    e_int, v_int = multiechelle(x, echelles=6)
    e_lst, v_lst = multiechelle(x, echelles=[1, 3, 6], methode="arbre")
    np.testing.assert_array_equal(e_lst, [1, 3, 6])
    for s, v in zip(e_lst, v_lst):
        assert v == v_int[list(e_int).index(s)]


def test_multiechelle_liste_rend_nan_pour_les_echelles_trop_courtes():
    x = _series()["gauss"][:300]
    echelles, valeurs = multiechelle(x, echelles=[1, 5, 20], methode="arbre")
    np.testing.assert_array_equal(echelles, [1, 5, 20])
    assert np.isfinite(valeurs[0]) and np.isfinite(valeurs[1])
    assert np.isnan(valeurs[2])  # 300 // 20 = 15 < 10 (m + 1) = 30


@pytest.mark.parametrize("liste", [[], [2, 1], [1, 1], [0, 2], [1, 2.5], [True, 2]])
def test_liste_d_echelles_invalide(liste):
    with pytest.raises(ValueError):
        multiechelle(_series()["gauss"], echelles=liste)
