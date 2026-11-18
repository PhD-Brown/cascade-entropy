"""Oracles indépendants des options N−1, sur graphes synthétiques seulement."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from cascade_entropy import attribution_n1 as B, topologies_n1 as T
from cascade_entropy import diagnostics_dispatch_n1 as D
from cascade_entropy import _optimisation_dispatch_n1 as O
from cascade_entropy.cascade_n1 import simuler_cascade_n1, simuler_tous_retraits_n1
from cascade_entropy.dispatch import TOLERANCE_SERVIE
from cascade_entropy.n1 import redispatch_exact
from cascade_entropy.reseau import Reseau, matrice_incidence


def radial(capacite=2.):
    return Reseau(3, np.array([[0, 2], [1, 2]]), np.array([.4, .7]),
                  np.array([0, 1]), np.array([2]), limites=np.full(2, capacite))


def attribuer_fixe(**options):
    t = T.anneau_cordes(8, 4, graine=7)
    return B.attribuer(t, placement=B.placement_aleatoire(t, n_generateurs=2, n_charges=4, graine=11),
        demande=B.Valeurs("uniforme", [.5, 1.5], graine=12),
        puissance_max=B.Valeurs("constante", [5.]), reserves_minimales=B.Valeurs("constante", [0.]),
        reactances=B.Valeurs("uniforme", [.2, .8], graine=13),
        capacites=B.Capacites("flux_reference", B.Valeurs("constante", [100.]), marge=1.5, plancher=.01),
        base_mva=100., unite_puissance="pu", unite_reactance="pu", **options)


def _egal_octets(valeur, temoin):
    """Compare aussi les bits des flottants et NaN, dans l'environnement courant."""
    if isinstance(temoin, np.ndarray):
        assert valeur.dtype == temoin.dtype
        assert valeur.shape == temoin.shape
        assert valeur.tobytes() == temoin.tobytes()
    elif isinstance(temoin, dict):
        assert valeur.keys() == temoin.keys()
        for k in temoin:
            _egal_octets(valeur[k], temoin[k])
    elif isinstance(temoin, (list, tuple)):
        assert len(valeur) == len(temoin)
        for a, b in zip(valeur, temoin):
            _egal_octets(a, b)
    else:
        assert valeur == temoin


def test_defauts_identiques_octets_au_commit_initial(monkeypatch):
    temoin = json.loads((Path(__file__).parent / "fixtures/departage_bebe96d.json").read_text(encoding="utf-8"))
    # Rejouer les sources figées sous le solveur courant, plutôt que figer un
    # sommet HiGHS d'une autre plateforme. Aucun accès à Git ni au réseau requis.
    anciens = {}
    for nom, source in temoin["sources"].items():
        module = types.ModuleType(f"cascade_entropy._temoin_{nom}")
        module.__package__ = "cascade_entropy"
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(compile(source, f"bebe96d/{nom}.py", "exec"), module.__dict__)
        if nom != "n1":
            module.redispatch_exact = anciens["n1"].redispatch_exact
        anciens[nom] = module
    e = attribuer_fixe()
    cfg = e.configuration
    ab = anciens["attribution_n1"]
    attendu = ab.attribuer(T.Topologie(**cfg["topologie"]), placement=ab.Placement(**cfg["placement"]),
        demande=ab.Valeurs(**cfg["demande"]), puissance_max=ab.Valeurs(**cfg["puissance_max"]),
        reserves_minimales=ab.Valeurs(**cfg["reserves_minimales"]), reactances=ab.Valeurs(**cfg["reactances"]),
        capacites=ab.Capacites("flux_reference", ab.Valeurs("constante", [100.]), marge=1.5, plancher=.01),
        base_mva=100., unite_puissance="pu", unite_reactance="pu")
    _egal_octets(asdict(e), asdict(attendu))
    for l in range(e.reseau.n_lignes):
        masque = np.zeros(e.reseau.n_lignes, dtype=bool)
        masque[l] = True
        avant = anciens["n1"].redispatch_exact(e.reseau, e.demande, e.puissance_max, masque)
        _egal_octets(asdict(redispatch_exact(e.reseau, e.demande, e.puissance_max, masque)), asdict(avant))
        avant = anciens["cascade_n1"].simuler_cascade_n1(e.reseau, e.demande, e.puissance_max, l, p1=1.)
        _egal_octets(asdict(simuler_cascade_n1(e.reseau, e.demande, e.puissance_max, l, p1=1.)), asdict(avant))


def test_minimax_radial_analytique_sans_hypothese_sur_highs():
    r = radial()
    s = redispatch_exact(r, [2.], [2., 2.], np.zeros(2, bool), departage="min_charge_max")
    np.testing.assert_allclose(s.production, [1., 1.], rtol=0, atol=2e-9)
    assert max(s.taux_de_charge) == pytest.approx(.5, abs=1e-9)


def verifier_physique(r, d, p, s):
    """Bilan nodal, bilan d'îlot et angles indépendants de la matrice du PL."""
    assert np.all(s.production >= -1e-8)
    assert np.all(s.production <= p + 1e-8)
    assert np.all(s.charge_servie >= -1e-8)
    assert np.all(s.charge_servie <= d + 1e-8)
    assert np.all(s.flux[s.hors_service] == 0.)
    assert np.all(np.isnan(s.taux_de_charge[s.hors_service]))
    assert np.all(np.abs(s.flux) <= r.limites + 1e-8)
    np.testing.assert_allclose(s.delestage, d - s.charge_servie, atol=1e-8)
    inj = np.zeros(r.n_noeuds)
    inj[r.generateurs] = s.production
    inj[r.charges] = -s.charge_servie
    np.testing.assert_allclose(matrice_incidence(r).T @ s.flux, inj, atol=1e-8)
    for noeuds in s.ilots:
        assert abs(inj[noeuds].sum()) < 1e-8
        ids = np.flatnonzero(~s.hors_service & np.isin(r.lignes[:, 0], noeuds))
        if len(noeuds) == 1:
            continue
        M = matrice_incidence(r)[ids][:, noeuds]
        lap = M.T @ ((1 / r.reactances[ids])[:, None] * M)
        theta = np.r_[0., np.linalg.solve(lap[1:, 1:], inj[noeuds][1:])]
        np.testing.assert_allclose((M @ theta) / r.reactances[ids], s.flux[ids], atol=1e-8)


@pytest.mark.parametrize("regle", ["highs", "min_charge_max", "min_redispatch"])
@pytest.mark.parametrize("retrait", [None, 0, 1, 2])
def test_meme_service_physique_et_objectif(regle, retrait):
    e = attribuer_fixe()
    r, d, p = e.reseau, e.demande, e.puissance_max
    masque = np.zeros(r.n_lignes, bool)
    if retrait is not None:
        masque[retrait] = True
    base = redispatch_exact(r, d, p, masque)
    precedente = np.array([1.5, 2.5])
    options = {"production_precedente": precedente} if regle == "min_redispatch" else {}
    s = redispatch_exact(r, d, p, masque, departage=regle, **options)
    verifier_physique(r, d, p, s)
    assert abs(s.charge_servie.sum() - base.charge_servie.sum()) <= 1.1 * TOLERANCE_SERVIE * max(1, d.sum())
    if regle == "min_charge_max":
        v = D.variabilite_flux(r, d, p, masque)
        assert np.nanmax(s.taux_de_charge) <= np.nanmax(base.taux_de_charge) + 1e-8
        # HiGHS emploie une tolérance de faisabilité absolue de l'ordre de 1e-7.
        assert np.nanmax(s.taux_de_charge) == pytest.approx(max(v.charge_max_minimale), abs=1e-7)
    if regle == "min_redispatch":
        assert abs(s.production - precedente).sum() <= abs(base.production - precedente).sum() + 1e-8


def test_minredispatch_oracle_analytique():
    r = radial()
    precedente = np.array([.3, 1.7])
    s = redispatch_exact(r, [2.], [2., 2.], np.zeros(2, bool),
                        departage="min_redispatch", production_precedente=precedente)
    np.testing.assert_allclose(s.production, precedente, rtol=0, atol=3e-9)


def test_plages_connues_evitable_et_forcee():
    r = radial()
    v = D.variabilite_flux(r, [2.], [2., 2.], np.zeros(2, bool))
    np.testing.assert_allclose(v.flux_min, 0., atol=1e-8)
    np.testing.assert_allclose(v.flux_max, 2., atol=1e-8)
    assert v.n_flux_non_uniques == 2
    assert v.charge_max_minimale == pytest.approx([.5])
    # Un sommet construit, sans supposer lequel le solveur choisit.
    s = redispatch_exact(r, [2.], [2., 2.], np.zeros(2, bool))
    s.production[:] = [2., 0.]
    s.flux[:] = [2., 0.]
    s.taux_de_charge[:] = [1., 0.]
    assert D.classer_saturations(s, v) == ([0], [])
    # Une seule source capable de produire : le flux 0 est physiquement forcé.
    v = D.variabilite_flux(r, [2.], [2., 0.], np.zeros(2, bool))
    assert D.classer_saturations(s, v) == ([], [0])
    np.testing.assert_allclose(v.flux_min, [2., 0.], atol=1e-8)
    assert v.n_flux_non_uniques == 0


def test_intervalle_traverse_zero_et_limites_explicites():
    r = Reseau(3, np.array([[0, 1], [0, 2], [1, 2]]), np.ones(3),
               np.array([0, 1]), np.array([2]), limites=np.full(3, 10.))
    C = np.full(3, 2.)
    v = D.variabilite_flux(r, [2.], [2., 2.], np.zeros(3, bool), limites=C)
    assert v.flux_min[0] == pytest.approx(-2/3)
    assert v.flux_max[0] == pytest.approx(2/3)
    assert v.taux_min_possible[0] == 0.
    assert v.charge_max_minimale[0] == pytest.approx(.5)


def test_ilots_triviaux_nan_retrait_et_statuts():
    r = Reseau(6, np.array([[0, 1], [1, 2], [3, 4]]), np.ones(3),
               np.array([0, 3]), np.array([2]), limites=np.ones(3))
    masque = np.array([False, True, False])
    v = D.variabilite_flux(r, [2.], [2., 2.], masque)
    s = redispatch_exact(r, [2.], [2., 2.], masque)
    assert v.statut_ilots == s.statut_ilots
    assert np.isnan(v.flux_min[1]) and np.isnan(v.taux_min_possible[1])
    assert v.flux_min[[0, 2]].tolist() == [0., 0.]
    assert np.all(v.charge_max_minimale == 0)


@pytest.mark.parametrize("regle", ["highs", "min_charge_max", "min_redispatch"])
def test_determinisme_non_mutation(regle):
    r, d, p, masque = radial(), np.array([2.]), np.array([2., 2.]), np.zeros(2, bool)
    avant = [x.copy() for x in (r.lignes, r.reactances, r.limites, d, p, masque)]
    precedente = np.array([.4, 1.6])
    options = {"production_precedente": precedente} if regle == "min_redispatch" else {}
    a = redispatch_exact(r, d, p, masque, departage=regle, **options)
    b = redispatch_exact(r, d, p, masque, departage=regle, **options)
    for nom in ("production", "charge_servie", "flux", "taux_de_charge"):
        np.testing.assert_array_equal(getattr(a, nom), getattr(b, nom))
    va, vb = D.variabilite_flux(r, d, p, masque), D.variabilite_flux(r, d, p, masque)
    np.testing.assert_array_equal(va.flux_min, vb.flux_min)
    D.etat_intact_prorata(r, d, p)
    for x, y in zip((r.lignes, r.reactances, r.limites, d, p, masque), avant):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(precedente, [.4, 1.6])


@pytest.mark.parametrize("regle,precedente", [("inconnu", None), (None, None), ([], None),
    ("min_redispatch", None), ("min_redispatch", [1.]), ("min_redispatch", [np.nan, 1.]),
    ("min_redispatch", [-1., 1.]), ("highs", [1., 1.]), ("min_charge_max", [1., 1.])])
def test_options_invalides(regle, precedente):
    with pytest.raises(ValueError):
        redispatch_exact(radial(), [2.], [2., 2.], np.zeros(2, bool),
                         departage=regle, production_precedente=precedente)


@pytest.mark.parametrize("seuil", [0., -1., np.inf, np.nan, True, "0.99"])
def test_seuil_invalide(seuil):
    with pytest.raises(ValueError):
        D.lignes_saturees(redispatch_exact(radial(), [2.], [2., 2.], np.zeros(2, bool)), seuil)


def test_echec_secondaire_remonte_sans_repli(monkeypatch):
    def echouer(*args, **kwargs):
        raise RuntimeError("échec secondaire injecté")
    monkeypatch.setattr(O, "resoudre_pl", echouer)
    for regle in ("min_charge_max", "min_redispatch"):
        options = {"production_precedente": [1., 1.]} if regle == "min_redispatch" else {}
        with pytest.raises(RuntimeError, match="échec secondaire injecté"):
            redispatch_exact(radial(), [2.], [2., 2.], np.zeros(2, bool), departage=regle, **options)


def test_prorata_sans_solveur_et_infaisabilite_signalee(monkeypatch):
    def interdit(*args, **kwargs):
        raise AssertionError("Aucun PL permis pour le prorata")
    monkeypatch.setattr(D, "redispatch_exact", interdit)
    r = radial()
    a = D.etat_intact_prorata(r, [2.], [1., 3.])
    assert a.faisable
    np.testing.assert_allclose(a.resultat.production, [.5, 1.5])
    B._controler(r, np.array([2.]), np.array([1., 3.]), a.resultat)
    b = D.etat_intact_prorata(radial(.4), [2.], [2., 2.])
    assert not b.faisable and b.violations == ("limites_transport",)
    assert max(b.resultat.taux_de_charge) == pytest.approx(2.5)
    c = D.etat_intact_prorata(r, [2.], [.2, .2])
    assert not c.faisable and "ilot_0_production_insuffisante" in c.violations
    np.testing.assert_allclose(c.resultat.charge_servie, [2.])


def test_prorata_bilan_impossible():
    a = D.etat_intact_prorata(radial(), [2.], [0., 0.])
    assert not a.faisable and a.violations == ("ilot_0_bilan_impossible",)
    assert np.isnan(a.resultat.flux).all()


def test_suivi_production_precedente_et_transmission(monkeypatch):
    import cascade_entropy.cascade_n1 as C
    original = C.redispatch_exact
    appels = []
    def tracer(*args, **kwargs):
        s = original(*args, **kwargs)
        appels.append((kwargs, s))
        return s
    monkeypatch.setattr(C, "redispatch_exact", tracer)
    e = attribuer_fixe()
    r = C.simuler_cascade_n1(e.reseau, e.demande, e.puissance_max, 0, p1=1.,
                             departage="min_redispatch", departage_intact="min_charge_max")
    assert appels[0][0]["departage"] == "min_charge_max"
    assert len(appels) >= 2
    for (avant, sol_avant), (apres, _) in zip(appels, appels[1:]):
        np.testing.assert_array_equal(apres["production_precedente"], sol_avant.production)
    verifier_physique(e.reseau, e.demande, e.puissance_max, r.etat_final)
    tous = C.simuler_tous_retraits_n1(radial(), [2.], [2., 2.], p1=1.,
                                     departage="min_redispatch", departage_intact="min_charge_max")
    assert len(tous) == 2
    np.testing.assert_allclose(tous[0].etat_reference.production, [1., 1.], atol=2e-9)


def test_diagnostic_vagues_conditionnelles_et_pas_declencheur():
    # Après retrait d'une ligne du radial, l'autre est forcée à saturation.
    r = simuler_cascade_n1(radial(), [2.], [2., 2.], 0, p1=1.)
    a = D.diagnostiquer_cascade(r, radial(), [2., 2.])
    assert a.n_forcees == 1 and a.n_evitables == 0
    assert a.vagues[0].forcees == [1]
    assert a.part_premiere_vague_uniquement_evitable == 0.
    assert a.n_evitables + a.n_forcees == len(r.lignes_tombees) - 1
    b = D.diagnostiquer_cascade(simuler_cascade_n1(radial(), [2.], [2., 2.], 0, p1=0.), radial(), [2., 2.])
    assert b.part_premiere_vague_uniquement_evitable is None


@pytest.mark.parametrize("regle", ["min_charge_max", "min_redispatch"])
def test_attribution_memes_options_reference_finale_et_rejeu(tmp_path, regle):
    options = {"production_precedente": [1.5, 2.5]} if regle == "min_redispatch" else {}
    e = attribuer_fixe(departage=regle, **options)
    assert e.configuration["departage"] == regle
    assert e.resultat_reference is not None
    B._controler(e.reseau, e.demande, e.puissance_max, e.resultat)
    chemin = tmp_path / "etat.json"
    B.sauver_configuration(e, chemin)
    rejeu = B.charger_configuration(chemin)
    np.testing.assert_array_equal(rejeu.resultat.flux, e.resultat.flux)
    assert rejeu.avertissements == e.avertissements


@pytest.mark.parametrize("mauvais", ["min_redispatch", None, [], "autre"])
def test_departage_intact_invalide(mauvais):
    with pytest.raises(ValueError):
        simuler_cascade_n1(radial(), [2.], [2., 2.], 0, p1=1.,
                           departage="min_redispatch", departage_intact=mauvais)


def test_evitable_et_forcee_dans_un_meme_etat():
    r = Reseau(4, np.array([[0, 2], [1, 2], [2, 3]]), np.ones(3),
               np.array([0, 1]), np.array([2, 3]), limites=np.array([2., 2., 1.]))
    s = redispatch_exact(r, [1., 1.], [2., 2.], np.zeros(3, bool))
    s.production[:] = [2., 0.]
    s.flux[:] = [2., 0., 1.]
    s.taux_de_charge[:] = [1., 0., 1.]
    verifier_physique(r, np.ones(2), np.full(2, 2.), s)
    v = D.variabilite_flux(r, [1., 1.], [2., 2.], np.zeros(3, bool))
    assert D.classer_saturations(s, v) == ([0], [2])


def test_diagnostic_etat_precedent_et_marqueur_intact():
    from cascade_entropy.cascade_n1 import ResultatCascadeN1
    r = Reseau(4, np.array([[0, 2], [1, 2], [2, 3]]), np.ones(3),
               np.array([0, 1]), np.array([2]), limites=np.full(3, 2.))
    etats = [redispatch_exact(r, [2.], [2., 2.], np.array(m, bool))
             for m in ([False, False, False], [False, False, True],
                       [True, False, True], [True, True, True])]
    # Fixer un sommet admissible connu, indépendamment du choix du solveur.
    for s in etats[:2]:
        s.production[:] = [2., 0.]
        s.flux[:] = [2., 0., 0.]
        s.taux_de_charge[~s.hors_service] = np.array([1., 0., 0.])[~s.hors_service]
    cascade = ResultatCascadeN1(2, 1., np.array([2.]), etats[0], etats[1:],
                                [np.array([0]), np.array([1])])
    a = D.diagnostiquer_cascade(cascade, r, [2., 2.])
    assert (a.n_evitables, a.n_forcees) == (1, 1)
    assert a.vagues[0].evitables == [0]
    assert a.vagues[0].deja_saturees_intact == [0]
    assert a.vagues[1].forcees == [1]
    assert a.vagues[1].deja_saturees_intact == []
    assert a.part_premiere_vague_uniquement_evitable == 1.


def test_non_unicite_residuelle_face_minimax():
    # La branche vers 4 impose t=1 ; le partage entre 0 et 1 reste libre.
    r = Reseau(5, np.array([[0, 3], [1, 3], [2, 3], [3, 4]]), np.ones(4),
               np.array([0, 1, 2]), np.array([3, 4]), limites=np.array([2., 2., 2., 1.]))
    v = D.variabilite_flux(r, [1., 1.], [2., 2., 0.], np.zeros(4, bool), departage="min_charge_max")
    assert v.charge_max_minimale[0] == pytest.approx(1., abs=1e-7)
    assert v.n_flux_non_uniques == 2
    assert v.flux_max[0] - v.flux_min[0] == pytest.approx(2., abs=1e-6)
    # Le cas symétrique à deux branches, lui, devient unique à la tolérance du diagnostic.
    w = D.variabilite_flux(radial(), [2.], [2., 2.], np.zeros(2, bool), departage="min_charge_max")
    assert w.n_flux_non_uniques == 0


@pytest.mark.parametrize("fonction", [D.variabilite_flux, D.etat_intact_prorata])
@pytest.mark.parametrize("d,p,C", [([np.nan], [2., 2.], [2., 2.]),
    ([-1.], [2., 2.], [2., 2.]), ([2.], [2.], [2., 2.]),
    ([2.], [2., np.inf], [2., 2.]), ([2.], [2., 2.], [0., 2.]),
    ([2.], [2., 2.], [2.])])
def test_diagnostics_entrees_invalides(fonction, d, p, C):
    args = (np.zeros(2, bool),) if fonction is D.variabilite_flux else ()
    with pytest.raises(ValueError):
        fonction(radial(), d, p, *args, limites=C)


def test_classification_masque_incoherent_et_taux_invalides():
    r = radial()
    s = redispatch_exact(r, [2.], [2., 2.], np.zeros(2, bool))
    v = D.variabilite_flux(r, [2.], [2., 2.], np.ones(2, bool))
    with pytest.raises(ValueError, match="même état"):
        D.classer_saturations(s, v)
    s.taux_de_charge[0] = np.nan
    with pytest.raises(ValueError, match="finis"):
        D.lignes_saturees(s)


def test_cache_ilots_cle_physique_et_valeurs_independantes():
    D._plages_ilot.cache_clear()
    a = D.variabilite_flux(radial(), [2.], [2., 2.], np.zeros(2, bool))
    attendu = a.flux_max.copy()
    a.flux_max[:] = 123.
    b = D.variabilite_flux(radial(), [2.], [2., 2.], np.zeros(2, bool))
    np.testing.assert_array_equal(b.flux_max, attendu)
    assert D._plages_ilot.cache_info().hits == 1
    c = D.variabilite_flux(radial(), [2.], [2., 0.], np.zeros(2, bool))
    assert D._plages_ilot.cache_info().misses == 2
    assert c.n_flux_non_uniques == 0


def test_precision_secondaire_permet_chainer_production_anterieure():
    # Cas rencontré par le diagnostic : les tolérances historiques pouvaient
    # produire -2.3e-8 pu, ensuite refusé comme production précédente.
    t = T.anneau_cordes()
    e = B.attribuer(t, placement=B.placement_aleatoire(t, n_generateurs=8, n_charges=20, graine=11),
        demande=B.Valeurs("uniforme", [.5, 1.5], graine=12),
        puissance_max=B.Valeurs("constante", [5.]), reserves_minimales=B.Valeurs("constante", [0.]),
        reactances=B.Valeurs("uniforme", [.2, .8], graine=13),
        capacites=B.Capacites("flux_reference", B.Valeurs("constante", [100.]), marge=1.5, plancher=.01),
        base_mva=100., unite_puissance="pu", unite_reactance="pu")
    r = simuler_cascade_n1(e.reseau, e.demande, e.puissance_max, 0, p1=1.,
                           departage="min_redispatch", departage_intact="min_charge_max")
    for s in [r.etat_reference, *r.etats]:
        assert np.all(s.production >= 0.) and np.all(s.production <= e.puissance_max)
        verifier_physique(e.reseau, e.demande, e.puissance_max, s)


def test_secours_secondaire_garde_objectif_et_est_visible(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(O, "linprog", lambda *a, **k: SimpleNamespace(success=False))
    s = redispatch_exact(radial(), [2.], [2., 2.], np.zeros(2, bool), departage="min_charge_max")
    np.testing.assert_allclose(s.production, [1., 1.], atol=2e-9)
    assert s.n_secours == 1


def test_solution_secondaire_hors_bornes_est_refusee(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(O, "linprog", lambda *a, **k: SimpleNamespace(success=True, x=np.array([-1., 3., 2., .5])))
    monkeypatch.setattr(O, "_linprog_robuste", lambda *a, **k: (np.array([-1., 3., 2., .5]), "injecte"))
    with pytest.raises(RuntimeError, match="hors bornes"):
        redispatch_exact(radial(), [2.], [2., 2.], np.zeros(2, bool), departage="min_charge_max")


def test_rejeu_pl_synthetique_precision_du_bilan_apres_presolve():
    # Le premier essai peut annoncer success tout en ayant un résidu de bilan
    # de 6.12e-9 pu. Un autre solveur du même objectif donne ici 1.1e-15 pu.
    z = json.loads((Path(__file__).parent / "fixtures/departage_precision.json").read_text(encoding="utf-8"))["probleme"]
    c, A, b, E, e, inf, sup = (np.array(z[k]) for k in ("c", "A", "b", "E", "e", "inf", "sup"))
    x, _ = O.resoudre_pl(c, A, b, E, e, inf, sup, "rejeu synthétique de précision")
    assert np.all(x >= inf) and np.all(x <= sup)
    assert np.max(A @ x - b) <= 5e-9
    assert np.max(np.abs(E @ x - e)) <= 5e-9


def test_diagnostic_cible_identique_au_diagnostic_complet(monkeypatch):
    e = attribuer_fixe()
    cascades = simuler_tous_retraits_n1(e.reseau, e.demande, e.puissance_max, p1=1.)
    cible = [asdict(D.diagnostiquer_cascade(c, e.reseau, e.puissance_max)) for c in cascades]
    original = D._variabilite_flux
    def complet(*args, **kwargs):
        kwargs.pop("indices", None)
        return original(*args, **kwargs)
    monkeypatch.setattr(D, "_variabilite_flux", complet)
    attendu = [asdict(D.diagnostiquer_cascade(c, e.reseau, e.puissance_max)) for c in cascades]
    assert cible == attendu


def test_rejeu_minredispatch_statut_unknown_et_objectif_equivalent():
    z = json.loads((Path(__file__).parent / "fixtures/departage_cout_echelle.json").read_text(encoding="utf-8"))["probleme"]
    c, A, b, E, e, inf, sup = (np.array(z[k], dtype=float) for k in ("c", "A", "b", "E", "e", "inf", "sup"))
    sup = np.where(np.isnan(sup), np.inf, sup)
    x, _ = O.resoudre_pl(c, A, b, E, e, inf, sup, "rejeu min_redispatch")
    assert np.all(x >= inf) and np.all(x <= sup)
    assert np.max(A @ x - b) <= 5e-9
    assert np.max(np.abs(E @ x - e)) <= 5e-9
    # L'objectif L1 original est conservé ; le coût optimal n'est pas divisé
    # par cent dans ce contrôle indépendant des variables de décision.
    assert c @ x == pytest.approx(2.24842231, abs=2e-7)
