"""Contrats B5/B6 sur fichiers temporaires et réseaux exclusivement synthétiques."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cascade_entropy import persistance_n1 as p
from cascade_entropy import topologies_n1 as topo


CAS = [
    (topo.anneau_cordes, {"n_noeuds": 12, "n_cordes": 4, "graine": 31}),
    (topo.grille, {"n_lignes": 3, "n_colonnes": 4}),
    (topo.gilbert, {"n_noeuds": 12, "p": 0.24, "graine": 31, "max_essais": 100}),
    (topo.geometrique, {"n_noeuds": 12, "rayon": 0.40, "graine": 31, "max_essais": 100}),
    (topo.arbre_temoin, {"n_noeuds": 12, "embranchement": 3}),
]


def options(arguments=None, **changements):
    d = dict(statut="demonstration", arguments_generation=arguments or {},
             revision_code="28eeb9b-test-synthetique", arbre_modifie=True,
             parametres_experience={"graine_cascade": 81, "notes": ["test", None]})
    d.update(changements)
    return d


def electrique(t):
    """Valeurs distinctes par indice pour détecter un décalage d'association."""
    return {
        "base_puissance": {"valeur": 100.0, "unite": "MVA"},
        "conventions": {"origine": "fixture synthétique, sans validation physique",
                        "orientation": "u vers v ; incidence +1 à u, -1 à v",
                        "injections": "production positive, consommation négative"},
        "champs": {
            "reactances": p.champ_numerique(
                np.linspace(0.1, 0.9, t.n_lignes, dtype=np.float64), axes=["ligne"],
                unite="pu", convention="x positif, susceptance b=1/x"),
            "demande": p.champ_numerique(
                np.arange(t.n_noeuds, dtype=np.float32) / np.float32(7), axes=["noeud"],
                unite="pu", convention="demande positive, distincte de l'injection signée"),
            "producteurs": p.champ_numerique(
                np.arange(t.n_noeuds) == 0, axes=["noeud"],
                unite="1", convention="masque des nœuds producteurs"),
            "reference": p.champ_numerique(
                np.int32(0), axes=[], unite="1", convention="identifiant du nœud d'angle nul"),
            "capacites": p.champ_numerique(
                np.arange(1, t.n_lignes + 1, dtype=np.uint64), axes=["ligne"],
                unite="pu", convention="capacité de chaque ligne initiale"),
        },
    }


@pytest.mark.parametrize("generateur,arguments", CAS, ids=[f.__name__ for f, _ in CAS])
def test_cinq_types_identiques_et_regenerables(tmp_path, generateur, arguments):
    t = generateur(**arguments)
    proprietes = electrique(t)
    cible = tmp_path / "realisation"
    identifiant = p.sauver_realisation(t, cible, **options(arguments, electrique=proprietes))
    r = p.charger_realisation(cible)
    assert r.id_reseau == identifiant
    assert r.metadata["statut"] == "demonstration"
    assert r.metadata["electrique"] == proprietes
    assert r.metadata["parametres_experience"] == options()["parametres_experience"]
    assert r.metadata["provenance"]["revision_code"] == options()["revision_code"]
    assert all(r.metadata["provenance"]["versions"].values())
    assert r.metadata["noeuds"] == list(range(t.n_noeuds))
    assert r.metadata["lignes"] == list(range(t.n_lignes))
    assert r.topologie.famille == t.famille
    assert r.topologie.parametres == t.parametres
    assert r.topologie.graine == t.graine
    assert r.topologie.essais == t.essais
    np.testing.assert_array_equal(r.topologie.lignes, t.lignes)
    if t.positions is None:
        assert r.topologie.positions is None
    else:
        np.testing.assert_array_equal(r.topologie.positions, t.positions)
    assert topo.statistiques(r.topologie) == topo.statistiques(t)
    # Le fichier reste directement lisible par le lecteur historique.
    np.testing.assert_array_equal(topo.charger_json(cible / "topologie.json").lignes, t.lignes)
    p.verifier_reproduction(r)
    for nom, champ in proprietes["champs"].items():
        a = p.tableau_champ(champ)
        relu = p.tableau_champ(r.metadata["electrique"]["champs"][nom])
        assert relu.dtype == a.dtype
        assert relu.shape == a.shape
        assert relu.tobytes() == a.tobytes()
    # Modifier les entrées après l'écriture ne change aucun octet stocké.
    t.lignes[:] = 0
    proprietes["champs"]["reactances"]["valeurs"][0] = 999.0
    assert p.charger_realisation(cible).metadata == r.metadata


def test_id_stable_independant_du_dossier_et_du_statut(tmp_path):
    t = topo.anneau_cordes(12, 4, 31)
    args = dict(n_noeuds=12, n_cordes=4, graine=31)
    a = p.sauver_realisation(t, tmp_path / "a", **options(args))
    b = p.sauver_realisation(t, tmp_path / "b", **options(args, statut="validation"))
    c = p.sauver_realisation(topo.anneau_cordes(12, 4, 32), tmp_path / "c",
                            **options(dict(args, graine=32)))
    assert a == b and a != c
    # Les défauts effectifs, et non seulement les arguments saisis, sont conservés.
    p.sauver_realisation(topo.grille(), tmp_path / "defaut", **options())
    r = p.charger_realisation(tmp_path / "defaut")
    assert r.metadata["reproduction"]["arguments"] == {"n_lignes": 5, "n_colonnes": 10}


def test_graines_essais_et_conditionnement_sont_reproduits(tmp_path):
    # Fixture discriminante : plus d'un tirage est nécessaire à la connexité.
    args = dict(n_noeuds=12, p=0.15, graine=3, max_essais=100)
    t = topo.gilbert(**args)
    assert t.essais > 1
    p.sauver_realisation(t, tmp_path / "r", **options(args))
    r = p.charger_realisation(tmp_path / "r")
    assert r.topologie.essais == t.essais
    assert r.metadata["reproduction"]["arguments"]["max_essais"] == 100
    assert r.metadata["reproduction"]["alea"] == "numpy.random.default_rng/PCG64"
    p.verifier_reproduction(r)


@pytest.mark.parametrize("existant", ["dossier", "fichier"])
def test_aucun_ecrasement(tmp_path, existant):
    cible = tmp_path / "r"
    if existant == "dossier":
        cible.mkdir()
        temoin = cible / "temoin.bin"
    else:
        temoin = cible
    temoin.write_bytes(b"a conserver exactement")
    with pytest.raises(FileExistsError):
        p.sauver_realisation(topo.grille(), cible, **options())
    assert temoin.read_bytes() == b"a conserver exactement"


def test_deux_ecritures_concurrentes_un_seul_gagnant(tmp_path):
    def ecrire(_):
        try:
            return p.sauver_realisation(topo.grille(), tmp_path / "r", **options())
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        resultats = list(pool.map(ecrire, range(2)))
    assert resultats.count(None) == 1
    assert p.charger_realisation(tmp_path / "r").id_reseau in resultats


@pytest.mark.parametrize("changements", [
    {"statut": "final"}, {"revision_code": ""}, {"arbre_modifie": 1},
    {"parametres_experience": []}, {"parametres_experience": {"nan": float("nan")}},
    {"parametres_experience": {1: "clé non texte"}},
    {"arguments_generation": {"inconnu": 1}},
    {"arguments_generation": {"n_lignes": 2}},
    {"statut": "campagne_scientifique"}, {"electrique": {}},
])
def test_entrees_invalides_avant_creation(tmp_path, changements):
    with pytest.raises(ValueError):
        p.sauver_realisation(topo.grille(), tmp_path / "r", **options(**changements))
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize("valeurs", [np.array([np.nan]), np.array([np.inf]),
    np.array([1 + 2j]), np.array([object()]), np.array(["1"])])
def test_champ_refuse_valeurs_non_numeriques_ou_non_finies(valeurs):
    with pytest.raises(ValueError):
        p.champ_numerique(valeurs, axes=["ligne"], unite="pu", convention="test")


@pytest.mark.parametrize("modification", [
    lambda d: d.update(base_puissance={"valeur": 0, "unite": "MVA"}),
    lambda d: d.update(base_puissance={"valeur": 100, "unite": "MW"}),
    lambda d: d.update(conventions={}),
    lambda d: d["champs"]["reactances"].update(unite=""),
    lambda d: d["champs"]["reactances"].update(axes=["inconnu"]),
    lambda d: d["champs"]["reactances"].update(axes=["noeud"]),
    lambda d: d["champs"]["reactances"]["valeurs"].pop(),
    lambda d: d["champs"]["reference"].update(valeurs=0.5),
    lambda d: d["champs"]["reference"].update(dtype="int8", valeurs=256),
    lambda d: d["champs"]["reference"].update(dtype="object"),
])
def test_conteneur_electrique_invalide_refuse(tmp_path, modification):
    t = topo.grille()
    d = electrique(t)
    modification(d)
    with pytest.raises(ValueError):
        p.sauver_realisation(t, tmp_path / "r", **options(electrique=d))
    assert not (tmp_path / "r").exists()


def test_scalaire_et_signe_zero_preserves():
    a = np.array([-0.0, np.nextafter(0.0, 1.0), np.finfo(float).max])
    d = p.champ_numerique(a, axes=["ligne"], unite="1", convention="bornes IEEE 754")
    assert p.tableau_champ(d).tobytes() == a.tobytes()


@pytest.mark.parametrize("a", [np.array(9, dtype=">i4"),
    np.array([[2**63 + 1, 2**64 - 1]], dtype=np.uint64),
    np.array([[1.25], [-0.0]], dtype=">f8"), np.array([True, False]),
    np.array([0.1, 1.0], dtype=np.float32)])
def test_champ_type_forme_et_octets_avec_oracle_numpy(a):
    d = p.champ_numerique(a, axes=["noeud"] * a.ndim, unite="1", convention="test de type")
    # Un aller-retour JSON réel, comparé au tableau d'origine, sans tolérance.
    relu = p.tableau_champ(json.loads(json.dumps(d)))
    assert relu.shape == a.shape
    assert relu.dtype == a.dtype
    assert relu.tobytes() == a.tobytes()


def test_provenance_sources_et_statut_declare_sans_simulation(tmp_path):
    t = topo.grille()
    p.sauver_realisation(t, tmp_path / "r", **options(
        statut="campagne_scientifique", electrique=electrique(t)))
    d = p.charger_realisation(tmp_path / "r").metadata
    assert d["statut"] == "campagne_scientifique"
    for nom, valeur in d["provenance"]["sources_sha256"].items():
        assert valeur == hashlib.sha256(Path(p.__file__).with_name(nom).read_bytes()).hexdigest()


def test_reproduction_refuse_topologie_differente_meme_statistiques(tmp_path):
    t = topo.arbre_temoin(12, 3)
    # Échanger les étiquettes de deux feuilles conserve tous les résumés structurels.
    original = topo.statistiques(t)
    permutation = np.arange(12)
    permutation[[4, 7]] = permutation[[7, 4]]
    autre = topo.Topologie(t.famille, 12, permutation[t.lignes], parametres=t.parametres)
    assert topo.statistiques(autre) == original
    assert not np.array_equal(t.lignes, autre.lignes)
    with pytest.raises(ValueError, match="régénération"):
        p.sauver_realisation(autre, tmp_path / "r",
                            **options({"n_noeuds": 12, "embranchement": 3}))
    assert not (tmp_path / "r").exists()


def empreinte(d):
    """Oracle de checksum indépendant, pour atteindre les contrôles sémantiques."""
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def reecrire_metadata(cible, modifier, recalculer=True):
    chemin = cible / "metadata.json"
    d = json.loads(chemin.read_text("utf-8"))
    modifier(d)
    if recalculer:
        d["sha256_metadata"] = empreinte({k: v for k, v in d.items() if k != "sha256_metadata"})
    chemin.write_text(json.dumps(d), encoding="utf-8")


@pytest.mark.parametrize("modification", [
    lambda d: d.update(format="cascade-entropy-realisation-v99"),
    lambda d: d.update(id_reseau="autre"),
    lambda d: d["noeuds"].reverse(),
    lambda d: d["noeuds"].__setitem__(0, False),
    lambda d: d["lignes"].reverse(),
    lambda d: d.pop("parametres_experience"),
    lambda d: d["reproduction"].update(generateur="arbitraire"),
    lambda d: d["reproduction"]["arguments"].update(n_lignes=3),
    lambda d: d["provenance"].update(versions={}),
])
def test_manifeste_incoherent_meme_avec_checksum_recalcule(tmp_path, modification):
    cible = tmp_path / "r"
    p.sauver_realisation(topo.grille(), cible, **options())
    reecrire_metadata(cible, modification)
    with pytest.raises(ValueError):
        p.charger_realisation(cible)


@pytest.mark.parametrize("fichier", ["metadata.json", "topologie.json"])
def test_corruption_detectee(tmp_path, fichier):
    cible = tmp_path / "r"
    p.sauver_realisation(topo.grille(), cible, **options())
    chemin = cible / fichier
    chemin.write_bytes(chemin.read_bytes() + b" ")
    if fichier == "metadata.json":
        # L'empreinte du manifeste porte sur les valeurs, pas sa mise en page.
        reecrire_metadata(cible, lambda d: d.update(statut="validation"), recalculer=False)
    with pytest.raises(ValueError, match="Empreinte"):
        p.charger_realisation(cible)


def test_corruption_numerique_detectee(tmp_path):
    cible = tmp_path / "r"
    t = topo.grille()
    p.sauver_realisation(t, cible, **options(electrique=electrique(t)))
    reecrire_metadata(cible, lambda d: d["electrique"]["champs"]["demande"]["valeurs"].reverse(),
                      recalculer=False)
    with pytest.raises(ValueError, match="Empreinte"):
        p.charger_realisation(cible)


@pytest.mark.parametrize("modification", [
    lambda d: d["lignes"].reverse(),
    lambda d: d["lignes"][0].reverse(),
    lambda d: d["statistiques"].update(n_ponts=999),
    lambda d: d.update(format="inconnu"),
    lambda d: d.pop("graine"),
])
def test_topologie_invalide_apres_recalcul_des_empreintes(tmp_path, modification):
    cible = tmp_path / "r"
    p.sauver_realisation(topo.grille(), cible, **options())
    chemin = cible / "topologie.json"
    d = json.loads(chemin.read_text("utf-8"))
    modification(d)
    chemin.write_text(json.dumps(d), encoding="utf-8")
    reecrire_metadata(cible, lambda m: m.update(
        sha256_topologie=hashlib.sha256(chemin.read_bytes()).hexdigest(),
        id_reseau="n1-" + empreinte(d)))
    with pytest.raises(ValueError):
        p.charger_realisation(cible)


@pytest.mark.parametrize("contenu", [b"{", b"[]", b"{\"format\":NaN}",
    b'{"format":"x","format":"y"}', b"\xff"])
def test_json_invalide_explicite(tmp_path, contenu):
    (tmp_path / "metadata.json").write_bytes(contenu)
    with pytest.raises(ValueError):
        p.charger_realisation(tmp_path)


def test_fichiers_absents_et_chemin_parent_fichier(tmp_path):
    with pytest.raises(FileNotFoundError):
        p.charger_realisation(tmp_path / "absent")
    parent = tmp_path / "fichier"
    parent.write_bytes(b"intact")
    with pytest.raises(OSError):
        p.sauver_realisation(topo.grille(), parent / "r", **options())
    assert parent.read_bytes() == b"intact"


def test_echec_ecriture_ne_publie_pas_une_realisation(tmp_path, monkeypatch):
    cible = tmp_path / "r"
    def refuser(*args, **kwargs):
        raise PermissionError("échec E/S synthétique")
    monkeypatch.setattr(Path, "replace", refuser)
    with pytest.raises(PermissionError, match="synthétique"):
        p.sauver_realisation(topo.grille(), cible, **options())
    assert (cible / "topologie.json").exists()
    assert not (cible / "metadata.json").exists()
    with pytest.raises(FileNotFoundError):
        p.charger_realisation(cible)
    with pytest.raises(FileExistsError):
        p.sauver_realisation(topo.grille(), cible, **options())
