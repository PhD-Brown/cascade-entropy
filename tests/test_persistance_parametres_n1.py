"""Équivalence numérique ciblée des paramètres réels, sans affaiblir le format."""
import hashlib
import json

import numpy as np
import pytest

from cascade_entropy import persistance_n1 as p
from cascade_entropy import topologies_n1 as topo


def sauver(t, cible, arguments):
    return p.sauver_realisation(t, cible, statut="validation",
        arguments_generation=arguments, revision_code="test-synthetique",
        arbre_modifie=True, parametres_experience={})


@pytest.mark.parametrize("generateur,arguments", [
    (topo.gilbert, dict(n_noeuds=3, p=1, graine=41)),
    (topo.geometrique, dict(n_noeuds=3, rayon=2, graine=41)),
])
def test_regression_parametre_entier_accepte_par_generateur(tmp_path, generateur, arguments):
    t = generateur(**arguments)
    sauver(t, tmp_path / "r", arguments)
    r = p.charger_realisation(tmp_path / "r")
    assert r.metadata["reproduction"]["arguments"].items() >= arguments.items()
    p.verifier_reproduction(r)


@pytest.mark.parametrize("generateur,arguments", [
    (topo.anneau_cordes, dict(n_noeuds=6, n_cordes=0, graine=0)),
    (topo.grille, dict(n_lignes=2, n_colonnes=2)),
    (topo.gilbert, dict(n_noeuds=2, p=1.0, graine=0, max_essais=1)),
    (topo.geometrique, dict(n_noeuds=2, rayon=2.0, graine=0, max_essais=1)),
    (topo.arbre_temoin, dict(n_noeuds=2, embranchement=1)),
])
def test_cinq_familles_parametres_et_identite_preserves(tmp_path, generateur, arguments):
    t = generateur(**arguments)
    identifiant = sauver(t, tmp_path / "r", arguments)
    r = p.charger_realisation(tmp_path / "r")
    assert r.id_reseau == identifiant
    assert r.topologie.parametres == t.parametres
    for nom, valeur in arguments.items():
        assert type(r.metadata["reproduction"]["arguments"][nom]) is type(valeur)
    p.verifier_reproduction(r)


@pytest.mark.parametrize("famille,champ,valeur", [
    ("gilbert", "p", 1), ("geometrique", "rayon", 2),
    ("geometrique", "rayon", 2**53),
    ("geometrique", "rayon", float(np.finfo(float).max)),
])
def test_representations_reelles_legitimes_sans_changer_identite(tmp_path, famille, champ, valeur):
    arguments = dict(n_noeuds=3, graine=41, **{champ: valeur})
    t = getattr(topo, famille)(**arguments)
    autre = dict(arguments, **{champ: float(valeur)})
    a = sauver(t, tmp_path / "a", arguments)
    b = sauver(t, tmp_path / "b", autre)
    assert a == b  # La définition topologique n'a pas été réécrite.
    assert (tmp_path / "a/topologie.json").read_bytes() == (tmp_path / "b/topologie.json").read_bytes()
    for cible in (tmp_path / "a", tmp_path / "b"):
        p.verifier_reproduction(p.charger_realisation(cible))


def reecrire_manifeste(cible, modifier):
    chemin = cible / "metadata.json"
    d = json.loads(chemin.read_text("utf-8"))
    modifier(d)
    contenu = {k: v for k, v in d.items() if k != "sha256_metadata"}
    d["sha256_metadata"] = hashlib.sha256(json.dumps(contenu, sort_keys=True,
        ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    chemin.write_text(json.dumps(d), encoding="utf-8")


def test_lecture_representation_equivalente_mais_checksum_toujours_strict(tmp_path):
    args = dict(n_noeuds=3, p=1.0, graine=41)
    sauver(topo.gilbert(**args), tmp_path / "r", args)
    chemin = tmp_path / "r/metadata.json"
    d = json.loads(chemin.read_text("utf-8"))
    d["reproduction"]["arguments"]["p"] = 1
    chemin.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="Empreinte"):
        p.charger_realisation(tmp_path / "r")
    reecrire_manifeste(tmp_path / "r", lambda d: None)
    r = p.charger_realisation(tmp_path / "r")
    assert type(r.metadata["reproduction"]["arguments"]["p"]) is int
    p.verifier_reproduction(r)


@pytest.mark.parametrize("champ,valeur", [
    ("n_noeuds", 3.0), ("n_noeuds", True), ("graine", 41.0),
    ("graine", True), ("max_essais", 500.0), ("p", True), ("p", "1"),
    ("p", 0), ("p", np.nextafter(1.0, 2.0)), ("p", float("nan")),
    ("p", float("inf")),
])
def test_types_et_valeurs_invalides_non_normalises(tmp_path, champ, valeur):
    args = dict(n_noeuds=3, p=1.0, graine=41)
    t = topo.gilbert(**args)
    with pytest.raises(ValueError):
        sauver(t, tmp_path / "r", dict(args, **{champ: valeur}))
    assert not (tmp_path / "r").exists()


def test_entier_reel_non_representable_exactement_refuse(tmp_path):
    args = dict(n_noeuds=3, rayon=2**53 + 1, graine=41)
    t = topo.geometrique(**args)  # Le générateur historique arrondit ce rayon.
    with pytest.raises(ValueError, match="exactement"):
        sauver(t, tmp_path / "r", args)
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize("valeur", [0, -1, True, "2", float("inf"), float("nan"), 2**1024])
def test_rayon_invalide_ou_non_representable_refuse(tmp_path, valeur):
    args = dict(n_noeuds=3, rayon=2.0, graine=41)
    with pytest.raises(ValueError):
        sauver(topo.geometrique(**args), tmp_path / "r", dict(args, rayon=valeur))
    assert not (tmp_path / "r").exists()


def test_probabilite_voisine_de_un_non_arrondie(tmp_path):
    valeur = float(np.nextafter(1.0, 0.0))
    args = dict(n_noeuds=3, p=valeur, graine=41)
    sauver(topo.gilbert(**args), tmp_path / "r", args)
    r = p.charger_realisation(tmp_path / "r")
    assert r.metadata["reproduction"]["arguments"]["p"] == valeur
    assert r.topologie.parametres["p"] != 1.0
    p.verifier_reproduction(r)


def test_parametres_topologiques_types_distincts_gardent_leur_identite(tmp_path):
    args = dict(n_noeuds=3, p=1.0, graine=41)
    t = topo.gilbert(**args)
    a = sauver(t, tmp_path / "a", args)
    t.parametres["p"] = 1
    b = sauver(t, tmp_path / "b", args)
    assert a != b  # L'équivalence de validation ne fusionne pas les identités JSON.
    p.verifier_reproduction(p.charger_realisation(tmp_path / "b"))


def test_tableaux_et_identifiants_ne_deviennent_pas_numeriquement_laches(tmp_path):
    with pytest.raises(ValueError):
        p.tableau_champ(dict(valeurs=[1], dtype="<f8", axes=["noeud"], unite="pu", convention="test"))
    args = dict(n_noeuds=3, p=1.0, graine=41)
    sauver(topo.gilbert(**args), tmp_path / "r", args)
    reecrire_manifeste(tmp_path / "r", lambda d: d["noeuds"].__setitem__(0, 0.0))
    with pytest.raises(ValueError, match="Identifiants"):
        p.charger_realisation(tmp_path / "r")
