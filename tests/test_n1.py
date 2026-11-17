"""Tests N−1 avec uniquement des graphes synthétiques (aucune donnée restreinte)."""
from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.n1 import (
    composantes_actives,
    evaluer_retrait_n1,
    redispatch_exact,
)
from cascade_entropy.reseau import Reseau, matrice_incidence


def triangle_feuille() -> Reseau:
    # e0=(0,1), e1=(1,2), e2=(2,0) : cycle ; e3=(2,3) : pont.
    return Reseau(
        n_noeuds=4,
        lignes=np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=int),
        reactances=np.array([1.0, 1.4, 0.8, 1.0]),
        generateurs=np.array([0], dtype=int),
        charges=np.array([1, 2, 3], dtype=int),
        limites=np.full(4, 6.0),
        reference=0,
    )


@pytest.mark.parametrize("ligne,delestage,nb_ilots", [
    (None, 0.0, 1),
    (0, 0.0, 1),
    (1, 0.0, 1),
    (2, 0.0, 1),
    (3, 1.0, 2),
])
def test_banc_triangle_feuille(ligne, delestage, nb_ilots):
    r = triangle_feuille()
    sol = evaluer_retrait_n1(r, np.ones(3), np.array([4.0]), ligne)
    assert len(sol.ilots) == nb_ilots
    assert sol.delestage_total == pytest.approx(delestage, abs=1e-8)
    assert sol.fraction_delestee == pytest.approx(delestage / 3.0, abs=1e-8)
    assert np.all(sol.taux_de_charge[~sol.hors_service] <= 1.0 + 1e-8)
    if ligne is not None:
        assert sol.flux[ligne] == 0.0
        assert np.isnan(sol.taux_de_charge[ligne])


def test_e2_au_lieu_du_delestage_artificiel():
    r = triangle_feuille()
    sol = evaluer_retrait_n1(r, np.ones(3), np.array([4.0]), 2)
    assert sol.delestage_total == pytest.approx(0.0, abs=1e-8)
    assert sol.charge_servie.sum() == pytest.approx(3.0, abs=1e-8)


def test_pont_ilot_sans_generateur():
    sol = evaluer_retrait_n1(triangle_feuille(), np.ones(3), np.array([4.0]), 3)
    assert [x.tolist() for x in sol.ilots] == [[0, 1, 2], [3]]
    assert sol.delestage[-1] == pytest.approx(1.0)
    assert "sans_generateur" in sol.statut_ilots


def test_conservation_nodale_apres_retrait():
    r = triangle_feuille()
    M = matrice_incidence(r)
    for ligne in range(r.n_lignes):
        sol = evaluer_retrait_n1(r, np.ones(3), np.array([4.0]), ligne)
        injection = np.zeros(r.n_noeuds)
        injection[r.generateurs] = sol.production
        injection[r.charges] = -sol.charge_servie
        assert np.allclose(M.T @ sol.flux, injection, atol=1e-8)
        assert sol.production.sum() == pytest.approx(sol.charge_servie.sum(), abs=1e-8)


def test_pas_de_mutation_et_remise_a_zero():
    r = triangle_feuille()
    copie_lignes, copie_x, copie_cap = r.lignes.copy(), r.reactances.copy(), r.limites.copy()
    sol2 = evaluer_retrait_n1(r, np.ones(3), np.array([4.0]), 2)
    sol3 = evaluer_retrait_n1(r, np.ones(3), np.array([4.0]), 3)
    bis2 = evaluer_retrait_n1(r, np.ones(3), np.array([4.0]), 2)
    assert np.allclose(sol2.flux, bis2.flux)
    assert sol2.delestage_total == bis2.delestage_total
    assert sol3.delestage_total > sol2.delestage_total
    assert np.array_equal(r.lignes, copie_lignes)
    assert np.array_equal(r.reactances, copie_x)
    assert np.array_equal(r.limites, copie_cap)


def test_deux_ilots_avec_generateurs_autonomes():
    # Deux paires reliées par une ligne-pont ; après retrait, deux îlots
    # possèdent chacun un générateur et une charge et doivent servir leur demande.
    r = Reseau(
        n_noeuds=4,
        lignes=np.array([[0, 1], [1, 2], [2, 3]], dtype=int),
        reactances=np.ones(3),
        generateurs=np.array([0, 2]),
        charges=np.array([1, 3]),
        limites=np.full(3, 5.0),
        reference=0,
    )
    sol = evaluer_retrait_n1(r, np.array([1., 1.]), np.array([2., 2.]), 1)
    assert sol.delestage_total == pytest.approx(0.0, abs=1e-8)
    assert [x.tolist() for x in sol.ilots] == [[0, 1], [2, 3]]
    assert sol.production.sum() == pytest.approx(2.0, abs=1e-8)


def test_toutes_lignes_retirees():
    r = triangle_feuille()
    sol = redispatch_exact(r, np.ones(3), np.array([4.0]), np.ones(4, dtype=bool))
    assert len(sol.ilots) == 4
    assert sol.delestage_total == pytest.approx(3.0)
    assert np.array_equal(sol.flux, np.zeros(4))


def test_masque_invalide_refuse():
    with pytest.raises(ValueError):
        composantes_actives(triangle_feuille(), np.array([0, 0, 1, 0]))
    with pytest.raises(ValueError):
        evaluer_retrait_n1(triangle_feuille(), np.ones(3), np.array([4.0]), 4)


def test_fraction_indefinie_si_demande_nulle():
    sol = evaluer_retrait_n1(triangle_feuille(), np.zeros(3), np.array([4.0]), 2)
    with pytest.raises(ValueError):
        _ = sol.fraction_delestee
