"""Oracles B4 exclusivement synthétiques ; nombres de validation non approuvés."""
from dataclasses import replace
import json

import numpy as np
import pytest

import cascade_entropy.attribution_n1 as b4
from cascade_entropy.n1 import evaluer_retrait_n1
from cascade_entropy.topologies_n1 import (
    Topologie, anneau_cordes, arbre_temoin, geometrique, gilbert, grille,
)


def constante(v):
    return b4.Valeurs("constante", [v])


def explicites(v):
    return b4.Valeurs("explicites", v)


def chaine():
    return Topologie("chaine_validation", 3, np.array([[0, 1], [1, 2]]))


def arguments():
    return dict(
        placement=b4.Placement([0], [1, 2], 0),
        demande=explicites([1.0, 2.0]), puissance_max=constante(4.0),
        reserves_minimales=constante(0.5), reactances=explicites([0.2, 0.3]),
        capacites=b4.Capacites("independantes", constante(10.0)),
        base_mva=100.0, unite_puissance="pu", unite_reactance="pu",
    )


def etat(**changements):
    return b4.attribuer(chaine(), **{**arguments(), **changements})


def test_chaine_oracle_analytique_et_reserves():
    e = etat()
    # Bilan de sous-arbre : e0 transporte 1+2 ; e1 transporte 2.
    np.testing.assert_allclose(e.resultat.flux, [3, 2], atol=1e-10)
    np.testing.assert_allclose(e.injections, [3, -1, -2], atol=1e-10)
    np.testing.assert_allclose(e.angles, [0, -0.6, -1.2], atol=1e-10)
    np.testing.assert_allclose(e.susceptances, [5, 10 / 3])
    np.testing.assert_allclose(e.resultat.taux_de_charge, [0.3, 0.2])
    np.testing.assert_allclose(e.resultat.production, [3])
    np.testing.assert_allclose(e.resultat.charge_servie, [1, 2])
    np.testing.assert_allclose(e.reserves_disponibles, [1])
    assert e.reserves_satisfaites.tolist() == [True]
    assert e.resultat.delestage_total == pytest.approx(0, abs=1e-10)
    assert e.avertissements == ()
    assert e.resultat_reference is None


def test_triangle_oracle_partage_dc_et_noeud_neutre():
    t = Topologie("triangle_validation", 3, np.array([[0, 1], [0, 2], [1, 2]]))
    a = arguments()
    a.update(placement=b4.Placement([0], [2], 0), demande=constante(1),
             reactances=constante(1))
    e = b4.attribuer(t, **a)
    # Deux chemins parallèles de réactance 1 et 2 : partage 2/3, 1/3.
    np.testing.assert_allclose(e.resultat.flux, [1 / 3, 2 / 3, 1 / 3], atol=1e-10)
    np.testing.assert_allclose(e.injections, [1, 0, -1], atol=1e-10)
    np.testing.assert_allclose(e.angles, [0, -1 / 3, -2 / 3], atol=1e-10)


def test_conversion_mw_pu_et_base_commune():
    pu = etat()
    mw = etat(demande=explicites([100, 200]), puissance_max=constante(400),
              reserves_minimales=constante(50), unite_puissance="MW",
              capacites=b4.Capacites("independantes", constante(1000)))
    for nom in ("flux", "production", "charge_servie", "delestage", "taux_de_charge"):
        np.testing.assert_array_equal(getattr(pu.resultat, nom), getattr(mw.resultat, nom))
    np.testing.assert_array_equal(mw.reseau.reactances, [0.2, 0.3])
    np.testing.assert_array_equal(mw.reserves_minimales, pu.reserves_minimales)
    assert mw.configuration["base_mva"] == 100
    assert mw.configuration["unite_puissance"] == "MW"


@pytest.mark.parametrize("limites,pmax,delestage", [(10, 2, 1), (1, 4, 2), (10, 0, 3)])
def test_delestage_generation_ou_transport_signale_sans_exclusion(limites, pmax, delestage):
    e = etat(puissance_max=constante(pmax), reserves_minimales=constante(0),
             capacites=b4.Capacites("independantes", constante(limites)))
    assert e.resultat.delestage_total == pytest.approx(delestage, abs=1e-9)
    assert "delestage_initial" in e.avertissements
    assert e.resultat.charge_servie.sum() == pytest.approx(3 - delestage)
    assert e.resultat.production.sum() == pytest.approx(3 - delestage)


def test_reserve_minimale_est_diagnostic_pas_une_reduction_cachee_de_pmax():
    e = etat(reserves_minimales=constante(2))
    assert e.resultat.delestage_total == pytest.approx(0)
    np.testing.assert_allclose(e.resultat.production, [3])
    np.testing.assert_allclose(e.reserves_disponibles, [1])
    assert e.reserves_satisfaites.tolist() == [False]
    assert "reserves_minimales_non_satisfaites" in e.avertissements


def test_demande_nulle_fraction_indefinie_et_plancher_absolu():
    e = etat(demande=constante(0), capacites=b4.Capacites(
        "flux_reference", constante(10), marge=1.5, plancher=0.01))
    np.testing.assert_array_equal(e.reseau.limites, [0.01, 0.01])
    np.testing.assert_array_equal(e.resultat.flux, [0, 0])
    np.testing.assert_array_equal(e.resultat.production, [0])
    assert "demande_totale_nulle_fraction_indefinie" in e.avertissements
    with pytest.raises(ValueError, match="nulle"):
        _ = e.resultat.fraction_delestee


def test_ilots_intacts_sans_generateur_et_noeud_isole():
    t = Topologie("deconnecte_validation", 5, np.array([[0, 1], [2, 3]]))
    a = arguments()
    a.update(placement=b4.Placement([0], [1, 3], 0), reactances=constante(1))
    e = b4.attribuer(t, **a)
    np.testing.assert_allclose(e.resultat.charge_servie, [1, 0])
    np.testing.assert_allclose(e.resultat.flux, [1, 0])
    np.testing.assert_allclose(e.angles, [0, -1, 0, 0, 0])
    assert e.resultat.delestage_total == pytest.approx(2)
    assert e.resultat.statut_ilots == ["resolu", "sans_generateur", "sans_generateur"]
    assert "topologie_non_connexe" in e.avertissements


def test_capacites_depuis_flux_deux_dispatchs_explicites():
    e = etat(capacites=b4.Capacites("flux_reference", constante(10), marge=1.5, plancher=0.1))
    np.testing.assert_allclose(e.resultat_reference.flux, [3, 2])
    np.testing.assert_allclose(e.reseau.limites, [4.5, 3])
    np.testing.assert_allclose(e.resultat.flux, [3, 2])
    np.testing.assert_allclose(e.resultat.taux_de_charge, [2 / 3, 2 / 3])
    assert "capacites_dependantes_du_dispatch_de_reference" in e.avertissements


def test_reference_contrainte_ne_devient_pas_un_point_fixe_implicite():
    # Toute la demande à l'extrémité impose F0=F1 : l'oracle est indépendant
    # du choix de charge servie dans un optimum dégénéré du solveur HiGHS.
    e = etat(demande=explicites([0, 3]), capacites=b4.Capacites(
        "flux_reference", constante(1), marge=2, plancher=0.1))
    assert e.resultat_reference.charge_servie.sum() == pytest.approx(1)
    assert e.resultat.charge_servie.sum() == pytest.approx(2)
    assert e.resultat_reference.delestage_total == pytest.approx(2)
    assert e.resultat.delestage_total == pytest.approx(1)
    np.testing.assert_allclose(e.reseau.limites,
                               np.maximum(2 * abs(e.resultat_reference.flux), 0.1))
    assert "delestage_reference" in e.avertissements
    assert "flux_initial_distinct_du_flux_reference" in e.avertissements


def test_capacites_depuis_flux_unites_mw_plancher():
    pu = etat(demande=constante(0), capacites=b4.Capacites(
        "flux_reference", constante(10), marge=2, plancher=0.1))
    mw = etat(demande=constante(0), unite_puissance="MW", puissance_max=constante(400),
              reserves_minimales=constante(50), capacites=b4.Capacites(
                  "flux_reference", constante(1000), marge=2, plancher=10))
    np.testing.assert_array_equal(pu.reseau.limites, mw.reseau.limites)


def test_ordre_des_roles_preserve_association_des_puissances():
    e = etat(placement=b4.Placement([0], [2, 1], 0), demande=explicites([2, 1]))
    np.testing.assert_array_equal(e.reseau.charges, [2, 1])
    np.testing.assert_allclose(e.resultat.flux, [3, 2])


def test_non_mutation_et_compatibilite_retrait_exact():
    t, a = chaine(), arguments()
    lignes = t.lignes.copy()
    e = b4.attribuer(t, **a)
    retrait = evaluer_retrait_n1(e.reseau, e.demande, e.puissance_max, 1)
    assert retrait.delestage_total == pytest.approx(2)
    assert retrait.flux[1] == 0
    assert np.isnan(retrait.taux_de_charge[1])
    np.testing.assert_array_equal(t.lignes, lignes)
    np.testing.assert_array_equal(e.reseau.lignes, lignes)
    np.testing.assert_array_equal(a["demande"].parametres, [1, 2])
    np.testing.assert_array_equal(a["reactances"].parametres, [0.2, 0.3])
    np.testing.assert_array_equal(e.resultat.flux, [3, 2])
    # Les entrées restent indépendantes du résultat et de son manifeste.
    a["demande"].parametres[0] = 90
    t.lignes[0, 0] = 2
    assert e.demande[0] == 1
    assert e.configuration["demande"]["parametres"] == [1, 2]
    np.testing.assert_array_equal(e.reseau.lignes, lignes)


def test_reproductibilite_tirages_et_etat_initial_sans_rng_global():
    t = grille(3, 3)
    p = b4.placement_aleatoire(t, n_generateurs=2, n_charges=4, graine=19)
    bis = b4.placement_aleatoire(t, n_generateurs=2, n_charges=4, graine=19)
    np.testing.assert_array_equal(p.generateurs, bis.generateurs)
    np.testing.assert_array_equal(p.charges, bis.charges)
    assert not set(p.generateurs) & set(p.charges)
    assert p.reference == p.generateurs[0]
    a = arguments()
    a.update(placement=p, demande=b4.Valeurs("uniforme", [0.1, 0.3], 101),
             puissance_max=constante(3), reactances=b4.Valeurs("uniforme", [0.2, 0.8], 102),
             capacites=b4.Capacites("independantes", b4.Valeurs("uniforme", [2, 5], 103)))
    rng_global_avant = np.random.get_state()
    e, f = b4.attribuer(t, **a), b4.attribuer(t, **a)
    rng_global_apres = np.random.get_state()
    for ancien, nouveau in zip(rng_global_avant, rng_global_apres):
        np.testing.assert_equal(ancien, nouveau)
    for nom in ("flux", "production", "charge_servie", "delestage", "taux_de_charge"):
        np.testing.assert_array_equal(getattr(e.resultat, nom), getattr(f.resultat, nom))
    assert e.configuration == f.configuration
    np.testing.assert_array_equal(e.reseau.reactances, f.reseau.reactances)
    assert e.configuration["placement"]["provenance"]["graine"] == 19
    assert e.configuration["reactances"]["graine"] == 102
    diff = replace(a["reactances"], graine=104).realiser(t.n_lignes)
    assert not np.array_equal(diff, e.reseau.reactances)


@pytest.mark.parametrize("mode", ["independantes", "flux_reference"])
def test_manifeste_rejouable_et_sans_ecrasement(tmp_path, mode):
    cap = b4.Capacites(mode, b4.Valeurs("uniforme", [5, 10], 89),
                       **({"marge": 1.2, "plancher": 0.1} if mode == "flux_reference" else {}))
    e = etat(capacites=cap)
    p = tmp_path / "electrique.json"
    b4.sauver_configuration(e, p)
    f = b4.charger_configuration(p)
    assert f.configuration == e.configuration
    assert f.environnement == e.environnement
    np.testing.assert_array_equal(f.reseau.limites, e.reseau.limites)
    np.testing.assert_array_equal(f.resultat.flux, e.resultat.flux)
    np.testing.assert_array_equal(f.resultat.production, e.resultat.production)
    with pytest.raises(FileExistsError):
        b4.sauver_configuration(e, p)
    contenu = json.loads(p.read_text(encoding="utf-8"))
    contenu["format"] = "autre"
    p.write_text(json.dumps(contenu), encoding="utf-8")
    with pytest.raises(ValueError, match="Format"):
        b4.charger_configuration(p)


@pytest.mark.parametrize("t", [
    anneau_cordes(), grille(), gilbert(graine=42), geometrique(graine=42), arbre_temoin(),
])
def test_cinq_familles_50_noeuds_conservation_independante(t):
    a = arguments()
    a.update(placement=b4.Placement([0], list(range(1, 50)), 0),
             demande=constante(0.01), puissance_max=constante(1),
             reserves_minimales=constante(0.1), reactances=constante(0.5))
    e = b4.attribuer(t, **a)
    assert e.resultat.delestage_total == pytest.approx(0, abs=1e-9)
    assert e.resultat.production.sum() == pytest.approx(0.49)
    # Accumulation nodale indépendante des helpers d'incidence et de flux.
    bilan = np.zeros(t.n_noeuds)
    for (u, v), f in zip(t.lignes, e.resultat.flux):
        bilan[u] += f
        bilan[v] -= f
    np.testing.assert_allclose(bilan, [0.49] + [-0.01] * 49, atol=1e-9)
    np.testing.assert_array_equal(e.reseau.lignes, t.lignes)


@pytest.mark.parametrize("champ,valeur", [
    ("base_mva", 0), ("base_mva", -1), ("base_mva", float("inf")),
    ("base_mva", True), ("base_mva", "100"),
    ("unite_puissance", "kW"), ("unite_reactance", "ohm"),
    ("departage", "exterieur_dabord"),
    ("demande", explicites([1])), ("demande", explicites([[1, 2]])),
    ("demande", explicites([1, float("nan")])), ("demande", explicites([-1, 2])),
    ("demande", explicites([1 + 2j, 2])), ("demande", explicites([])),
    ("puissance_max", explicites([1, 2])), ("puissance_max", constante(-1)),
    ("reserves_minimales", constante(5)), ("reserves_minimales", explicites([0, 1])),
    ("reactances", constante(0)), ("reactances", constante(-1)),
    ("reactances", constante(float("inf"))), ("reactances", constante(1e-320)),
    ("reactances", explicites([1])),
    ("capacites", b4.Capacites("independantes", constante(0))),
    ("capacites", b4.Capacites("independantes", constante(float("nan")))),
    ("capacites", b4.Capacites("inconnue", constante(1))),
    ("capacites", b4.Capacites("independantes", constante(1), marge=2)),
    ("capacites", b4.Capacites("flux_reference", constante(1))),
    ("capacites", b4.Capacites("flux_reference", constante(1), marge=0, plancher=1)),
    ("capacites", b4.Capacites("flux_reference", constante(1), marge=2, plancher=0)),
    ("capacites", b4.Capacites("flux_reference", constante(1), marge=True, plancher=1)),
])
def test_entrees_invalides(champ, valeur):
    with pytest.raises(ValueError):
        etat(**{champ: valeur})


@pytest.mark.parametrize("p", [
    b4.Placement([0.5], [1, 2], 0), b4.Placement([True], [1, 2], 0),
    b4.Placement([0], [1, 1], 0), b4.Placement([0, 0], [1, 2], 0),
    b4.Placement([0], [0, 2], 0), b4.Placement([0], [-1, 2], 0),
    b4.Placement([0], [1, 3], 0), b4.Placement([], [1, 2], 0),
    b4.Placement([0], [], 0), b4.Placement([0], [1, 2], 1),
    b4.Placement([0], [1, 2], True), b4.Placement([[0]], [1, 2], 0),
    b4.Placement([0], [1, 2], 3),
])
def test_indices_invalides(p):
    with pytest.raises(ValueError):
        etat(placement=p)


@pytest.mark.parametrize("v", [
    b4.Valeurs("uniforme", [2, 1], 0), b4.Valeurs("uniforme", [1], 0),
    b4.Valeurs("uniforme", [1, 2]), b4.Valeurs("uniforme", [1, 2], True),
    b4.Valeurs("uniforme", [1, 2], -1), b4.Valeurs("uniforme", [1, 2], 1.2),
    b4.Valeurs("constante", [1], 0), b4.Valeurs("inconnu", [1]),
])
def test_strategies_invalides(v):
    with pytest.raises(ValueError):
        v.realiser(2)


@pytest.mark.parametrize("ng,nc,graine", [(0, 1, 0), (1, 0, 0), (2, 2, 0),
                                           (True, 1, 0), (1, 1, -1), (1, 1, None)])
def test_placement_aleatoire_invalide(ng, nc, graine):
    with pytest.raises(ValueError):
        b4.placement_aleatoire(chaine(), n_generateurs=ng, n_charges=nc, graine=graine)


def test_erreur_solveur_jamais_convertie_en_succes(monkeypatch):
    def echouer(*args, **kwargs):
        raise RuntimeError("échec synthétique du solveur")
    monkeypatch.setattr(b4, "redispatch_exact", echouer)
    with pytest.raises(RuntimeError, match="échec synthétique"):
        etat()


@pytest.mark.parametrize("champ,remplacement", [
    ("flux", np.array([3, 2.1])), ("flux", np.array([np.nan, 2])),
    ("flux", np.array([3])), ("production", np.array([5.])),
    ("delestage", np.array([0.1, 0.])),
    ("taux_de_charge", np.array([0.1, 0.2])),
    ("statut_ilots", ["echec"]),
])
def test_sortie_incoherente_refusee(monkeypatch, champ, remplacement):
    original = b4.redispatch_exact
    def corrompre(*args, **kwargs):
        sol = original(*args, **kwargs)
        setattr(sol, champ, remplacement)
        return sol
    monkeypatch.setattr(b4, "redispatch_exact", corrompre)
    with pytest.raises(RuntimeError, match="incohérent"):
        etat()


def test_kirchhoff_nodal_ne_suffit_pas_loi_dc_controlee(monkeypatch):
    # Ajouter un courant de cycle conserve chaque injection, mais viole la loi
    # des tensions : ce mutant discrimine le contrôle DC du seul bilan nodal.
    original = b4.redispatch_exact
    def courant_cyclique(*args, **kwargs):
        sol = original(*args, **kwargs)
        sol.flux += np.array([0.1, -0.1, 0.1])
        sol.taux_de_charge = abs(sol.flux) / args[0].limites
        return sol
    monkeypatch.setattr(b4, "redispatch_exact", courant_cyclique)
    t = Topologie("triangle_validation", 3, np.array([[0, 1], [0, 2], [1, 2]]))
    a = arguments()
    a["reactances"] = constante(1)
    with pytest.raises(RuntimeError, match="loi DC"):
        b4.attribuer(t, **a)


def test_secours_solveur_visible(monkeypatch):
    original = b4.redispatch_exact
    def secours(*args, **kwargs):
        sol = original(*args, **kwargs)
        sol.n_secours = 1
        return sol
    monkeypatch.setattr(b4, "redispatch_exact", secours)
    assert "secours_solveur_initial" in etat().avertissements


def test_tout_delestage_positif_meme_petit_est_signale():
    t = Topologie("charge_isolee_validation", 3, np.array([[0, 1]]))
    a = arguments()
    a.update(placement=b4.Placement([0], [2], 0), demande=constante(1e-10),
             reactances=constante(1))
    e = b4.attribuer(t, **a)
    assert e.resultat.delestage_total == pytest.approx(1e-10, rel=1e-10, abs=0)
    assert "delestage_initial" in e.avertissements


def test_tolerance_absolue_ne_cache_pas_un_bilan_faux_a_petite_echelle(monkeypatch):
    original = b4.redispatch_exact
    def service_sans_production(*args, **kwargs):
        sol = original(*args, **kwargs)
        sol.production[:] = 0
        sol.charge_servie[:] = 1e-10
        sol.delestage[:] = 0
        sol.flux[:] = [2e-10, 1e-10]
        sol.taux_de_charge[:] = abs(sol.flux) / args[0].limites
        return sol
    monkeypatch.setattr(b4, "redispatch_exact", service_sans_production)
    with pytest.raises(RuntimeError, match="conservation nodale"):
        etat(demande=constante(1e-10), puissance_max=constante(0),
             reserves_minimales=constante(0))


def test_identifiants_reordonnes_apres_creation_refuses():
    t = chaine()
    t.lignes = t.lignes[::-1].copy()
    with pytest.raises(ValueError, match="identifiants"):
        b4.attribuer(t, **arguments())


def test_topologie_sans_lignes_refusee_explicitement():
    t = Topologie("vide_validation", 3, np.empty((0, 2), dtype=int))
    with pytest.raises(ValueError, match="taille"):
        b4.attribuer(t, **arguments())


def test_demande_totale_debordante_refusee():
    with pytest.raises(ValueError, match="totale"):
        etat(demande=constante(1e308))


def test_capacites_calculees_debordantes_refusees():
    with pytest.raises(ValueError, match="Capacités dérivées"):
        etat(capacites=b4.Capacites("flux_reference", constante(10), marge=1e308, plancher=1))


def test_valeur_booleenne_refusee():
    with pytest.raises(ValueError, match="booléenne"):
        etat(demande=constante(True))
