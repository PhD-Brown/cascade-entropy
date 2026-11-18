"""Contrats permanents B4/B5–B6/C ; paramètres synthétiques et provisoires."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cascade_entropy import attribution_n1 as b4
from cascade_entropy import centralites_lignes as c
from cascade_entropy import integration_n1 as integration
from cascade_entropy import persistance_n1 as p
from cascade_entropy import topologies_n1 as topo
from cascade_entropy.cascade_n1 import simuler_cascade_n1


OPTIONS = dict(degre="somme", longueurs="reactances",
    normalisation_geodesique="moyenne", normalisation_vecteur="l2",
    paires_courant="uniformes", normalisation_courant="moyenne",
    convention_resistance="criticite")


def constante(valeur):
    return b4.Valeurs("constante", [valeur])


def chaine(mode="independantes", degrade=False):
    t = topo.arbre_temoin(3, 1)
    return b4.attribuer(t, placement=b4.Placement([0], [2, 1], 0),
        demande=b4.Valeurs("explicites", [3, 0] if degrade else [2, 1]),
        puissance_max=constante(4), reserves_minimales=constante(0.5),
        reactances=b4.Valeurs("explicites", [0.2, 0.3]),
        capacites=b4.Capacites(mode, constante(1 if degrade else 10),
            **({"marge": 2, "plancher": 0.1} if mode == "flux_reference" else {})),
        base_mva=100, unite_puissance="pu", unite_reactance="pu")


def sauver(etat, cible, arguments, **options):
    kwargs = dict(statut="validation", arguments_generation=arguments,
        revision_code="revision-test-synthetique", arbre_modifie=True,
        parametres_experience={"provisoire": True, "options_c": OPTIONS, "graine_pilote": 81})
    kwargs.update(options)
    return integration.sauver_etat_initial(etat, cible, **kwargs)


def cycle_disque(etat, cible, arguments):
    identifiant = sauver(etat, cible, arguments)
    relu = integration.charger_etat_initial(cible)
    assert identifiant == relu.realisation.id_reseau
    p.verifier_reproduction(relu.realisation)
    return relu


def verifier_bilan_independant(e, sol):
    """Accumulation par extrémités, sans matrice d'incidence du modèle."""
    r = e.reseau
    bilan = np.zeros(r.n_noeuds)
    for (u, v), f in zip(r.lignes, sol.flux):
        bilan[u] += f
        bilan[v] -= f
    injection = np.zeros(r.n_noeuds)
    injection[r.generateurs] = sol.production
    injection[r.charges] = -sol.charge_servie
    assert_allclose(bilan, injection, atol=1e-12)
    assert_allclose(sol.charge_servie + sol.delestage, e.demande, atol=1e-12)


def test_chaine_oracle_analytique_et_retraits_independants(tmp_path):
    e = chaine()
    relu = cycle_disque(e, tmp_path / "chaine", dict(n_noeuds=3, embranchement=1))
    f = relu.etat_electrique
    scores = c.calculer_scores(relu.etat_initial, **OPTIONS)
    assert_allclose(f.resultat.flux, [3, 2], atol=1e-14)
    assert_allclose(f.angles, [0, -0.6, -1.2], atol=1e-14)
    assert_array_equal(f.reseau.charges, [2, 1])
    assert_array_equal(f.demande, [2, 1])
    assert_allclose(scores.scores["flux_initial_absolu"], [3, 2], atol=1e-14)
    assert_allclose(scores.scores["taux_charge_initial"], [0.3, 0.2], atol=1e-14)
    assert_allclose(scores.scores["degre"], [3, 3])
    assert_allclose(scores.scores["intermediarite_geodesique"], [2/3, 2/3])
    assert_allclose(scores.scores["intermediarite_courant"], [2/3, 2/3])
    assert_allclose(scores.scores["resistance_locale"], [1, 1], atol=1e-14)
    # Chaîne pondérée : vecteur de Perron = [a/lambda,1,b/lambda]/sqrt(2).
    a, b = 5, 10/3
    assert_allclose(scores.scores["vecteur_propre"],
                    np.array([a*a, b*b]) / (2*np.hypot(a, b)), atol=1e-14)
    for i, attendu in ((1, 2), (0, 3), (1, 2)):
        resultat = simuler_cascade_n1(f.reseau, f.demande, f.puissance_max, i, p1=1)
        assert resultat.delestage_total == pytest.approx(attendu)
        assert resultat.fraction_delestee == pytest.approx(attendu/3)
        assert resultat.delestage_initial == pytest.approx(0)
        for sol in resultat.etats:
            verifier_bilan_independant(f, sol)
            assert_array_equal(sol.flux[sol.hors_service], 0)
            assert np.isnan(sol.taux_de_charge[sol.hors_service]).all()
    verifier_bilan_independant(f, f.resultat)


CAS = [
    (topo.anneau_cordes, dict(n_noeuds=6, n_cordes=1, graine=31)),
    (topo.grille, dict(n_lignes=2, n_colonnes=3)),
    (topo.gilbert, dict(n_noeuds=6, p=0.5, graine=31, max_essais=100)),
    (topo.geometrique, dict(n_noeuds=6, rayon=0.65, graine=31, max_essais=100)),
    (topo.arbre_temoin, dict(n_noeuds=6, embranchement=2)),
]


def six_noeuds(t, mode, unite):
    facteur = 100 if unite == "MW" else 1
    return b4.attribuer(t, placement=b4.Placement([3, 0], [5, 2, 1], 3),
        demande=b4.Valeurs("uniforme", [0.1*facteur, 0.3*facteur], 101),
        puissance_max=b4.Valeurs("explicites", np.array([0.4, 0.7])*facteur),
        reserves_minimales=constante(0.05*facteur),
        reactances=b4.Valeurs("uniforme", [0.2, 0.8], 102),
        capacites=b4.Capacites(mode, b4.Valeurs("uniforme", [2*facteur, 3*facteur], 103),
            **({"marge": 1.4, "plancher": 0.15*facteur} if mode == "flux_reference" else {})),
        base_mva=100, unite_puissance=unite, unite_reactance="pu")


@pytest.mark.parametrize("generateur,arguments", CAS, ids=[f.__name__ for f, _ in CAS])
@pytest.mark.parametrize("mode", ["independantes", "flux_reference"])
@pytest.mark.parametrize("unite", ["pu", "MW"])
def test_cinq_familles_deux_capacites_unites_roles_et_rejeu(tmp_path, generateur, arguments, mode, unite):
    e = six_noeuds(generateur(**arguments), mode, unite)
    relu = cycle_disque(e, tmp_path / "r", arguments)
    f = relu.etat_electrique
    assert f.configuration == e.configuration
    assert f.environnement == e.environnement
    assert f.avertissements == e.avertissements
    assert_array_equal(f.reseau.generateurs, [3, 0])
    assert_array_equal(f.reseau.charges, [5, 2, 1])
    for nom in ("lignes", "reactances", "limites"):
        a, b = getattr(f.reseau, nom), getattr(e.reseau, nom)
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    for nom in ("demande", "puissance_max", "reserves_minimales", "reserves_disponibles",
                "reserves_satisfaites", "angles", "injections"):
        a, b = getattr(f, nom), getattr(e, nom)
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    assert_array_equal(relu.etat_initial.flux, e.resultat.flux)
    assert_array_equal(relu.etat_initial.limites, e.reseau.limites)
    if mode == "flux_reference":
        assert_array_equal(f.resultat_reference.flux, e.resultat_reference.flux)
    electricite = relu.realisation.metadata["electrique"]
    assert electricite["base_puissance"] == {"valeur": 100, "unite": "MVA"}
    champs = electricite["champs"]
    assert_array_equal(p.tableau_champ(champs["rang_generateur"]), [1, -1, -1, 0, -1, -1])
    assert_array_equal(p.tableau_champ(champs["demande"])[[5, 2, 1]], e.demande)
    for nom in ("demande", "puissance_max", "reactances", "limites", "final_flux"):
        assert champs[nom]["unite"] == "pu"
    assert champs["angles"]["unite"] == "rad"
    assert champs["final_taux_de_charge"]["unite"] == "1"
    scores = c.calculer_scores(relu.etat_initial, **OPTIONS)
    assert len(scores.scores) == 7
    assert_array_equal(scores.identifiants_lignes, np.arange(e.reseau.n_lignes))
    assert_array_equal(scores.lignes, e.reseau.lignes)
    assert_array_equal(scores.scores["flux_initial_absolu"], abs(e.resultat.flux))
    assert_array_equal(scores.scores["taux_charge_initial"], e.resultat.taux_de_charge)
    assert_allclose(scores.scores["resistance_locale"].sum(), e.reseau.n_noeuds-1, atol=1e-12)
    verifier_bilan_independant(f, f.resultat)
    b4.sauver_configuration(f, tmp_path / "b4.json")
    rejeu = b4.charger_configuration(tmp_path / "b4.json")
    assert_array_equal(rejeu.resultat.flux, f.resultat.flux)
    assert_array_equal(rejeu.reseau.limites, f.reseau.limites)


@pytest.mark.parametrize("mode", ["independantes", "flux_reference"])
def test_equivalence_physique_mw_pu_et_poids_gc_non_tries(tmp_path, mode):
    t = topo.grille(2, 3)
    entrees = [cycle_disque(six_noeuds(t, mode, unite), tmp_path / unite,
                            dict(n_lignes=2, n_colonnes=3)) for unite in ("pu", "MW")]
    options = dict(OPTIONS, paires_courant="generateur_charge",
        generateurs=entrees[0].etat_electrique.reseau.generateurs,
        charges=entrees[0].etat_electrique.reseau.charges, poids_paires=[[1, 2, 3], [4, 5, 7]])
    a, b = [c.calculer_scores(x.etat_initial, **options) for x in entrees]
    for nom in a.scores:
        assert_allclose(a.scores[nom], b.scores[nom], rtol=1e-12, atol=1e-14)
    faux = c.calculer_scores(entrees[0].etat_initial, **dict(options,
        generateurs=sorted(options["generateurs"]), charges=sorted(options["charges"])))
    assert not np.allclose(faux.scores["intermediarite_courant"], a.scores["intermediarite_courant"])


def test_intact_final_distinct_de_reference_et_delestage_initial(tmp_path):
    e = chaine("flux_reference", degrade=True)
    relu = cycle_disque(e, tmp_path / "degrade", dict(n_noeuds=3, embranchement=1))
    f = relu.etat_electrique
    assert_allclose(f.resultat_reference.flux, [1, 1])
    assert_allclose(f.resultat.flux, [2, 2])
    assert f.resultat.delestage_total == pytest.approx(1)
    assert "delestage_initial" in f.avertissements
    scores = c.calculer_scores(relu.etat_initial, **OPTIONS)
    assert_allclose(scores.scores["flux_initial_absolu"], [2, 2])
    assert_allclose(scores.scores["taux_charge_initial"], [1, 1])
    sol = simuler_cascade_n1(f.reseau, f.demande, f.puissance_max, 1, p1=0)
    assert sol.delestage_initial == pytest.approx(1)
    assert sol.delestage_total == pytest.approx(3)
    assert sol.delestage_incremental == pytest.approx(2)
    assert sol.fraction_delestee == pytest.approx(1)


def test_maille_redispatch_propagation_non_mutation_et_absence_fuite(tmp_path):
    t = topo.gilbert(3, 1, 41)
    e = b4.attribuer(t, placement=b4.Placement([0], [2, 1], 0),
        demande=constante(1), puissance_max=constante(3), reserves_minimales=constante(0),
        reactances=constante(1), capacites=b4.Capacites("independantes",
            b4.Valeurs("explicites", [1.2, 1.2, 2])),
        base_mva=100, unite_puissance="pu", unite_reactance="pu")
    avant = b4._json_copie(asdict(e))
    relu = cycle_disque(e, tmp_path / "maille", dict(n_noeuds=3, p=1, graine=41))
    assert b4._json_copie(asdict(e)) == avant
    f = relu.etat_electrique
    scores = c.calculer_scores(relu.etat_initial, **OPTIONS)
    for p1, attendu in ((0, 0.8), (1, 2)):
        resultat = simuler_cascade_n1(f.reseau, f.demande, f.puissance_max, 0, p1=p1)
        assert resultat.delestage_total == pytest.approx(attendu)
        for sol in resultat.etats:
            verifier_bilan_independant(f, sol)
        with pytest.raises(TypeError, match="EtatInitialIntact"):
            c.calculer_scores(resultat, **OPTIONS)
        with pytest.raises(TypeError, match="EtatElectrique"):
            sauver(resultat, tmp_path / "post", dict(n_noeuds=3, p=1, graine=41))
    assert b4._json_copie(asdict(e)) == avant
    apres = c.calculer_scores(relu.etat_initial, **OPTIONS)
    for nom in scores.scores:
        assert_array_equal(apres.scores[nom], scores.scores[nom])
    # Les deux autres objets restent mutables ; C conserve son snapshot protégé.
    f.resultat.flux[:] = 0
    e.reseau.limites[:] = 99
    for nom, valeurs in c.calculer_scores(relu.etat_initial, **OPTIONS).scores.items():
        assert_array_equal(valeurs, scores.scores[nom])
    with pytest.raises(ValueError):
        relu.etat_initial.flux[0] = 0


def test_identites_electriques_et_non_ecrasement(tmp_path):
    a = cycle_disque(chaine(), tmp_path / "a", dict(n_noeuds=3, embranchement=1))
    b = cycle_disque(chaine("flux_reference"), tmp_path / "b", dict(n_noeuds=3, embranchement=1))
    assert a.realisation.id_reseau == b.realisation.id_reseau
    assert a.etat_initial.identifiant != b.etat_initial.identifiant
    avant = (tmp_path / "a/metadata.json").read_bytes()
    with pytest.raises(FileExistsError):
        sauver(chaine(), tmp_path / "a", dict(n_noeuds=3, embranchement=1))
    assert (tmp_path / "a/metadata.json").read_bytes() == avant


@pytest.mark.parametrize("mutation", [
    lambda e: e.reseau.lignes.__setitem__(slice(None), e.reseau.lignes[::-1]),
    lambda e: e.reseau.charges.__setitem__(slice(None), e.reseau.charges[::-1]),
    lambda e: e.resultat.hors_service.__setitem__(0, True),
    lambda e: e.resultat.flux.__setitem__(0, 0),
    lambda e: e.configuration.update(unite_puissance="kW"),
    lambda e: e.configuration.update(base_mva=0),
    lambda e: e.configuration["placement"].update(charges=[2, 2]),
    lambda e: e.configuration["demande"].update(parametres=[9, 1]),
    lambda e: e.configuration["capacites"].update(mode="inconnu"),
    lambda e: setattr(e, "reserves_satisfaites", np.array([1])),
    lambda e: setattr(e.resultat, "n_secours", True),
    lambda e: setattr(e, "environnement", {}),
])
def test_snapshot_b4_mutile_refuse_avant_ecriture(tmp_path, mutation):
    e = chaine()
    mutation(e)
    with pytest.raises((ValueError, RuntimeError)):
        sauver(e, tmp_path / "r", dict(n_noeuds=3, embranchement=1))
    assert not (tmp_path / "r").exists()


def test_source_non_b4_et_metadata_reservee_refusees(tmp_path):
    with pytest.raises(TypeError):
        sauver(object(), tmp_path / "r", {})
    with pytest.raises(ValueError, match="réservée"):
        sauver(chaine(), tmp_path / "r", dict(n_noeuds=3, embranchement=1),
               parametres_experience={integration.CLE: {}})
    with pytest.raises(ValueError):
        sauver(chaine(), tmp_path / "r", dict(n_noeuds=3, embranchement=1),
               parametres_experience={"nan": float("nan")})
    assert not (tmp_path / "r").exists()


def test_lecture_ne_rejoue_pas_dispatch(tmp_path, monkeypatch):
    sauver(chaine(), tmp_path / "r", dict(n_noeuds=3, embranchement=1))
    def echouer(*args, **kwargs):
        raise AssertionError("La lecture ne doit pas optimiser l'état enregistré.")
    monkeypatch.setattr(b4, "redispatch_exact", echouer)
    import cascade_entropy.dispatch as dispatch
    monkeypatch.setattr(dispatch, "resoudre", echouer)
    assert integration.charger_etat_initial(tmp_path / "r").etat_initial.flux.tolist() == [3, 2]


def test_intact_deconnecte_refuse_sans_modifier_b4(tmp_path):
    t = topo.Topologie("deconnecte_validation", 4, np.array([[0, 1], [2, 3]]))
    e = b4.attribuer(t, placement=b4.Placement([0], [1, 3], 0),
        demande=constante(1), puissance_max=constante(3), reserves_minimales=constante(0),
        reactances=constante(1), capacites=b4.Capacites("independantes", constante(10)),
        base_mva=100, unite_puissance="pu", unite_reactance="pu")
    assert "topologie_non_connexe" in e.avertissements
    with pytest.raises(ValueError, match="connexe"):
        sauver(e, tmp_path / "r", {})
    assert not (tmp_path / "r").exists()


def test_dispatch_reference_ou_diagnostic_delestage_absent_refuse(tmp_path):
    e = chaine("flux_reference", degrade=True)
    e.resultat_reference = None
    with pytest.raises(ValueError, match="référence"):
        sauver(e, tmp_path / "r", dict(n_noeuds=3, embranchement=1))
    e = chaine("flux_reference", degrade=True)
    e.avertissements = ()
    with pytest.raises(ValueError, match="Délestage"):
        sauver(e, tmp_path / "r", dict(n_noeuds=3, embranchement=1))
    assert not (tmp_path / "r").exists()


def test_demande_nulle_preservee_mais_pas_acceptee_comme_booleen(tmp_path):
    e = chaine()
    cfg = e.configuration
    t = topo.Topologie(**cfg["topologie"])
    e = b4.attribuer(t, placement=b4.Placement([0], [2, 1], 0),
        demande=constante(0), puissance_max=constante(4), reserves_minimales=constante(0),
        reactances=constante(1), capacites=b4.Capacites("independantes", constante(10)),
        base_mva=100, unite_puissance="pu", unite_reactance="pu")
    relu = cycle_disque(e, tmp_path / "demande_zero", dict(n_noeuds=3, embranchement=1))
    assert "demande_totale_nulle_fraction_indefinie" in relu.etat_electrique.avertissements
    assert_array_equal(c.calculer_scores(relu.etat_initial, **OPTIONS).scores["flux_initial_absolu"], 0)
    with pytest.raises(ValueError, match="nulle"):
        _ = relu.etat_electrique.resultat.fraction_delestee
    e.demande = np.zeros(2, dtype=bool)
    with pytest.raises(ValueError, match="booléennes"):
        sauver(e, tmp_path / "booleen", dict(n_noeuds=3, embranchement=1))
    assert not (tmp_path / "booleen").exists()


def corrompre(cible, modifier):
    chemin = cible / "metadata.json"
    d = json.loads(chemin.read_text("utf-8"))
    modifier(d)
    d["sha256_metadata"] = hashlib.sha256(json.dumps(
        {k: v for k, v in d.items() if k != "sha256_metadata"}, sort_keys=True,
        ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    chemin.write_text(json.dumps(d), encoding="utf-8")


@pytest.mark.parametrize("mutation", [
    lambda d: d["parametres_experience"][integration.CLE].update(format="inconnu"),
    lambda d: d["parametres_experience"].pop(integration.CLE),
    lambda d: d["electrique"]["champs"]["final_flux"].update(unite="MW"),
    lambda d: d["electrique"]["base_puissance"].update(valeur=200),
    lambda d: d["electrique"]["champs"]["rang_charge"]["valeurs"].reverse(),
    lambda d: d["electrique"]["champs"]["demande"]["valeurs"].__setitem__(0, 1.0),
    lambda d: d["electrique"]["champs"]["limites"]["valeurs"].reverse(),
    lambda d: d["parametres_experience"][integration.CLE]["final"].update(n_secours=-1),
    lambda d: d["parametres_experience"][integration.CLE].update(avertissements="aucun"),
    lambda d: d["parametres_experience"][integration.CLE].update(environnement_b4={}),
])
def test_lecture_semantique_meme_apres_checksum_recalcule(tmp_path, mutation):
    # Limites distinctes, pour rendre leur permutation discriminante.
    e = chaine("flux_reference")
    sauver(e, tmp_path / "r", dict(n_noeuds=3, embranchement=1))
    corrompre(tmp_path / "r", mutation)
    with pytest.raises((ValueError, RuntimeError)):
        integration.charger_etat_initial(tmp_path / "r")


def test_frontiere_analyse_sans_import_modeles_processus_frais():
    code = '''
import importlib.abc
import sys
class Bloqueur(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"cascade_entropy.integration_n1", "cascade_entropy.attribution_n1",
                        "cascade_entropy.reseau", "cascade_entropy.dispatch",
                        "cascade_entropy.persistance_n1", "cascade_entropy.cascade_n1"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Bloqueur())
import cascade_entropy.centralites_lignes
import cascade_entropy.entropie
import cascade_entropy.indicateurs
'''
    resultat = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert resultat.returncode == 0, resultat.stderr
