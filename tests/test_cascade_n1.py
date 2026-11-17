"""Validation A6 : cascades N−1 exactes sur réseaux 100 % synthétiques."""
from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.cascade_n1 import (
    simuler_cascade_n1,
    simuler_tous_retraits_n1,
)
from cascade_entropy.reseau import Reseau, matrice_incidence


def triangle_simple() -> Reseau:
    # e0=(0,1), e1=(1,2), e2=(0,2), toutes les réactances sont 1 pu.
    # La génération au nœud 0 sert 1 pu aux nœuds 1 et 2.
    return Reseau(
        n_noeuds=3,
        lignes=np.array([[0, 1], [1, 2], [0, 2]], dtype=int),
        reactances=np.ones(3),
        generateurs=np.array([0]),
        charges=np.array([1, 2]),
        limites=np.array([1.2, 2.0, 1.2]),
        reference=0,
    )


def triangle_limites_hautes() -> Reseau:
    r = triangle_simple()
    r.limites = np.array([10., 10., 10.])
    return r


def deux_vagues() -> Reseau:
    # Exemple synthétique obtenu par recherche de cas test, fixé ici.
    # Permet de vérifier le recalcul de redispatch *après chaque vague*.
    return Reseau(
        n_noeuds=5,
        lignes=np.array([[0,1],[1,2],[2,3],[3,4],[4,0],[0,2],[1,3]], dtype=int),
        reactances=np.array([0.37, 2.10, 0.60, 1.35, 0.97, 1.78, 2.14]),
        generateurs=np.array([0]),
        charges=np.array([1,2,3,4]),
        limites=np.array([2.93, 0.53, 0.40, 0.48, 1.88, 1.27, 0.51]),
        reference=0,
    )


def test_sans_saturation_pas_de_cascade_secondaire():
    c = simuler_cascade_n1(triangle_limites_hautes(), np.ones(2), np.array([3.]), 0, p1=1)
    assert c.delestage_total == pytest.approx(0, abs=1e-8)
    assert c.n_vagues_secondaires == 0
    assert c.n_redispatchs == 1
    assert c.lignes_tombees.tolist() == [0]
    assert c.hors_service.tolist() == [True, False, False]


def test_p1_zero_ne_propage_pas_meme_si_ligne_saturee():
    c = simuler_cascade_n1(triangle_simple(), np.ones(2), np.array([3.]), 0, p1=0)
    assert c.n_vagues_secondaires == 0
    assert c.delestage_total == pytest.approx(0.8, abs=1e-8)
    assert c.etat_final.taux_de_charge[2] >= 0.99
    assert c.delestage_initial == pytest.approx(0)


def test_p1_un_fait_tomber_une_ligne_secondaire():
    c = simuler_cascade_n1(triangle_simple(), np.ones(2), np.array([3.]), 0, p1=1)
    assert c.n_vagues_secondaires == 1
    assert [v.tolist() for v in c.vagues_secondaires] == [[2]]
    assert c.lignes_tombees.tolist() == [0, 2]
    assert c.hors_service.tolist() == [True, False, True]
    assert c.n_redispatchs == 2
    assert c.delestage_total == pytest.approx(2.0, abs=1e-8)
    assert c.fraction_delestee == pytest.approx(1.0)
    assert c.delestage_incremental == pytest.approx(2.0)
    assert c.etat_final.flux[0] == 0.0
    assert c.etat_final.flux[2] == 0.0


def test_recalcule_apres_deux_vagues_sans_retombee():
    c = simuler_cascade_n1(deux_vagues(), np.ones(4), np.array([5.]), 0, p1=1)
    assert [v.tolist() for v in c.vagues_secondaires] == [[3], [5]]
    assert len(c.etats) == 3
    assert c.lignes_tombees.tolist() == [0, 3, 5]
    assert c.delestage_total == pytest.approx(3.0, abs=1e-8)
    assert c.delestage_initial == pytest.approx(0.0, abs=1e-8)
    assert c.etats[0].delestage_total < c.etats[1].delestage_total < c.etats[2].delestage_total
    assert not np.any(c.etat_final.taux_de_charge[~c.hors_service] >= 0.99)


def test_conservation_de_la_puissance_pour_chaque_etat():
    r = deux_vagues()
    c = simuler_cascade_n1(r, np.ones(4), np.array([5.]), 0, p1=1)
    M = matrice_incidence(r)
    for etat in [c.etat_reference, *c.etats]:
        injections = np.zeros(r.n_noeuds)
        injections[r.generateurs] = etat.production
        injections[r.charges] = -etat.charge_servie
        assert np.allclose(M.T @ etat.flux, injections, atol=1e-7)
        assert np.all(etat.flux[etat.hors_service] == 0.0)
        assert np.all(np.isnan(etat.taux_de_charge[etat.hors_service]))
        assert np.all(etat.taux_de_charge[~etat.hors_service] <= 1.0 + 1e-8)


def test_n1_independant_et_aucune_mutation_des_entrees():
    r = triangle_simple()
    lignes = r.lignes.copy(); x = r.reactances.copy(); limites = r.limites.copy()
    demande = np.ones(2); pmax = np.array([3.]); demande_c = demande.copy(); pmax_c = pmax.copy()
    a = simuler_cascade_n1(r, demande, pmax, 0, p1=1)
    b = simuler_cascade_n1(r, demande, pmax, 1, p1=1)
    c = simuler_cascade_n1(r, demande, pmax, 0, p1=1)
    assert a.lignes_tombees.tolist() == c.lignes_tombees.tolist()
    assert a.delestage_total == c.delestage_total
    assert b.delestage_total != a.delestage_total
    assert np.array_equal(r.lignes, lignes)
    assert np.array_equal(r.reactances, x)
    assert np.array_equal(r.limites, limites)
    assert np.array_equal(demande, demande_c)
    assert np.array_equal(pmax, pmax_c)


def test_campagne_n1_chaque_ligne_une_fois():
    r = triangle_simple()
    resultats = simuler_tous_retraits_n1(r, np.ones(2), np.array([3.]), p1=1)
    assert [c.ligne_initiale for c in resultats] == [0, 1, 2]
    assert len(resultats) == r.n_lignes
    for c in resultats:
        direct = simuler_cascade_n1(r, np.ones(2), np.array([3.]), c.ligne_initiale, p1=1)
        assert c.lignes_tombees.tolist() == direct.lignes_tombees.tolist()
        assert c.delestage_total == pytest.approx(direct.delestage_total, abs=1e-8)


def test_avec_p1_stochastique_graine_reproductible():
    a = simuler_cascade_n1(triangle_simple(), np.ones(2), np.array([3.]), 0,
                            p1=0.5, rng=np.random.default_rng(20261113))
    b = simuler_cascade_n1(triangle_simple(), np.ones(2), np.array([3.]), 0,
                            p1=0.5, rng=np.random.default_rng(20261113))
    assert [v.tolist() for v in a.vagues_secondaires] == [v.tolist() for v in b.vagues_secondaires]
    assert a.delestage_total == pytest.approx(b.delestage_total)


def test_campagne_stochastique_reproductible():
    r = triangle_simple()
    a = simuler_tous_retraits_n1(r, np.ones(2), np.array([3.]), p1=0.5, graine=41)
    b = simuler_tous_retraits_n1(r, np.ones(2), np.array([3.]), p1=0.5, graine=41)
    assert [[v.tolist() for v in c.vagues_secondaires] for c in a] == [
        [v.tolist() for v in c.vagues_secondaires] for c in b
    ]


@pytest.mark.parametrize("ligne", [-1, 3, 1.2, True])
def test_declencheur_invalide_refuse(ligne):
    with pytest.raises(ValueError):
        simuler_cascade_n1(triangle_simple(), np.ones(2), np.array([3.]), ligne, p1=1)


@pytest.mark.parametrize("p1", [-0.1, 1.1, np.nan, np.inf, True, "1"])
def test_p1_invalide_refuse(p1):
    with pytest.raises(ValueError):
        simuler_cascade_n1(triangle_simple(), np.ones(2), np.array([3.]), 0, p1=p1)


def test_p1_fractionnaire_exige_generateur_aleatoire():
    with pytest.raises(ValueError, match="rng explicite"):
        simuler_cascade_n1(triangle_simple(), np.ones(2), np.array([3.]), 0, p1=0.5)
    with pytest.raises(TypeError):
        simuler_cascade_n1(triangle_simple(), np.ones(2), np.array([3.]), 0, p1=1, rng=4)


def test_campagne_aleatoire_exige_graine_et_graine_valide():
    with pytest.raises(ValueError, match="graine obligatoire"):
        simuler_tous_retraits_n1(triangle_simple(), np.ones(2), np.array([3.]), p1=0.4)
    with pytest.raises(ValueError, match="graine doit"):
        simuler_tous_retraits_n1(triangle_simple(), np.ones(2), np.array([3.]), p1=1, graine=-10)


def test_fraction_nulle_non_definie_si_demande_nulle():
    r = triangle_simple()
    c = simuler_cascade_n1(r, np.zeros(2), np.array([3.]), 0, p1=1)
    assert c.delestage_total == 0
    with pytest.raises(ValueError):
        _ = c.fraction_delestee


def test_echec_du_redispatch_non_masque(monkeypatch):
    import cascade_entropy.cascade_n1 as mod
    original = mod.redispatch_exact
    calls = []

    def echouer_apres_reference(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("échec numérique simulé")
        return original(*args, **kwargs)

    monkeypatch.setattr(mod, "redispatch_exact", echouer_apres_reference)
    with pytest.raises(RuntimeError, match="échec numérique simulé"):
        mod.simuler_cascade_n1(triangle_simple(), np.ones(2), np.array([3.]), 0, p1=1)


def test_ligne_pont_cree_ilot_sans_generateur_sans_cascade_secondaire():
    r = Reseau(
        n_noeuds=4,
        lignes=np.array([[0, 1], [1, 2], [2, 0], [2, 3]]),
        reactances=np.ones(4),
        generateurs=np.array([0]),
        charges=np.array([1, 2, 3]),
        limites=np.full(4, 6.),
        reference=0,
    )
    c = simuler_cascade_n1(r, np.ones(3), np.array([4.]), 3, p1=1)
    assert c.n_vagues_secondaires == 0
    assert c.delestage_total == pytest.approx(1.0, abs=1e-8)
    assert c.etat_final.statut_ilots == ["resolu", "sans_generateur"]
    assert c.lignes_tombees.tolist() == [3]


@pytest.mark.parametrize("p1", [np.nan, np.inf, -1, 2, True, "abc"])
def test_campagne_p1_invalide_refuse(p1):
    with pytest.raises(ValueError):
        simuler_tous_retraits_n1(triangle_simple(), np.ones(2), np.array([3.]), p1=p1)
