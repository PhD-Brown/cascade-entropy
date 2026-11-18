"""Contrats disque et concordance entre l'intact B4, C et la cascade."""
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cascade_entropy import attribution_n1 as b4
from cascade_entropy import integration_n1 as integration
from cascade_entropy import persistance_n1 as p
from cascade_entropy import topologies_n1 as topo
from cascade_entropy.cascade_n1 import simuler_cascade_n1


def attribuer(mode, regle="highs"):
    options = {"marge": 1.5, "plancher": 0.01} if mode == "flux_reference" else {}
    precedente = {"production_precedente": np.array([1., 2.])} if regle == "min_redispatch" else {}
    return b4.attribuer(topo.arbre_temoin(3, 1), placement=b4.Placement([0, 2], [1], 0),
        demande=b4.Valeurs("constante", [3.]), puissance_max=b4.Valeurs("constante", [4.]),
        reserves_minimales=b4.Valeurs("constante", [0.]), reactances=b4.Valeurs("constante", [.2]),
        capacites=b4.Capacites(mode, b4.Valeurs("constante", [10.]), **options),
        base_mva=100., unite_puissance="pu", unite_reactance="pu", departage=regle, **precedente)


def sauver(etat, dossier):
    return integration.sauver_etat_initial(etat, dossier, statut="validation",
        arguments_generation={"n_noeuds": 3, "embranchement": 1},
        revision_code="revision-test-synthetique", arbre_modifie=False,
        parametres_experience={"provisoire": True})


def figer_provenance(monkeypatch, tmp_path, etat):
    """Fige les seules entrées dépendant de la machine ou du checkout."""
    etat.environnement = {nom: "test" for nom in ("python", "numpy", "scipy")}
    monkeypatch.setattr(p, "platform", SimpleNamespace(
        python_version=lambda: "test", platform=lambda: "test"))
    monkeypatch.setattr(p, "version", lambda nom: "test")
    monkeypatch.setattr(p.np, "__version__", "test")
    monkeypatch.setattr(p, "__file__", str(tmp_path / "persistance_n1.py"))
    for nom in ("persistance_n1.py", "topologies_n1.py", "reseau.py", "_validation.py"):
        (tmp_path / nom).write_bytes(nom.encode("ascii"))


@pytest.mark.parametrize("mode, metadata_crlf, metadata_lf", [
    ("independantes", "0c940c4f1d4454c31a36f61b84b2dd1d7c581e9716c215a0609729b4006f63a1",
     "1c10d0ebc117e778af2734011f1ef43c2e64f7d0d04d9c7fe0a37686486a3556"),
    ("flux_reference", "9c97e1245cf87bf27f743de03acdbc9acb66798b5614117e1b5036026acb05b9",
     "bfda2053cc0b0d28116d789e2d5d931cca782b414a4413ce8d9aa81d69b20da1"),
])
def test_highs_v1_octet_pour_octet(tmp_path, monkeypatch, mode, metadata_crlf, metadata_lf):
    # SHA-256 des fichiers produits par a2f2d90 avec la même provenance figée.
    if os.linesep == "\r\n":
        empreintes = ("05d2467d61f8e0bf7f111c8f6b25102bd9dcb0dcb1fec57ed6a3ab3d650b34d0", metadata_crlf)
    else:
        empreintes = ("ffd49a8ed481f70028c4f3754cdf379a4c84b7301dbdaa766a811f166b27c8ce", metadata_lf)
    etat = attribuer(mode)
    figer_provenance(monkeypatch, tmp_path, etat)
    dossier = tmp_path / "b4"
    sauver(etat, dossier)
    assert sorted(f.name for f in dossier.iterdir()) == ["metadata.json", "topologie.json"]
    assert tuple(sha256((dossier / nom).read_bytes()).hexdigest()
                 for nom in ("topologie.json", "metadata.json")) == empreintes
    assert integration.charger_etat_initial(dossier).etat_electrique.configuration["departage"] == "highs"


@pytest.mark.parametrize("mode", ["independantes", "flux_reference"])
@pytest.mark.parametrize("regle", ["highs", "min_charge_max"])
def test_concordance_disque_cascade(tmp_path, mode, regle):
    etat = attribuer(mode, regle)
    dossier = tmp_path / "b4"
    sauver(etat, dossier)
    relu = integration.charger_etat_initial(dossier)
    e = relu.etat_electrique
    assert e.configuration == etat.configuration
    assert_array_equal(e.resultat.flux, etat.resultat.flux)
    assert_array_equal(e.reseau.limites, etat.reseau.limites)
    if mode == "flux_reference":
        assert_array_equal(e.resultat_reference.flux, etat.resultat_reference.flux)
    cascade = simuler_cascade_n1(e.reseau, e.demande, e.puissance_max, 0,
                                p1=1., departage=regle, limites=e.reseau.limites)
    assert_allclose(cascade.etat_reference.flux, e.resultat.flux, rtol=0, atol=1e-12)
    contrat = relu.realisation.metadata["parametres_experience"][integration.CLE]
    if regle == "min_charge_max":
        assert contrat["format"] == "cascade-entropy-integration-b4-c-v2"
        sources = Path(b4.__file__).parent
        fichiers = ("dispatch.py", "n1.py", "_optimisation_dispatch_n1.py", "attribution_n1.py")
        assert contrat["sources_dispatch_sha256"] == {
            nom: sha256((sources / nom).read_bytes()).hexdigest() for nom in fichiers}


def test_min_redispatch_reserve_a_la_cascade(tmp_path):
    dossier = tmp_path / "b4"
    with pytest.raises(ValueError, match="min_redispatch.*cascade.*departage_intact"):
        sauver(attribuer("independantes", "min_redispatch"), dossier)
    assert not dossier.exists()


@pytest.mark.parametrize("modifier, message", [
    (lambda c: c.pop("sources_dispatch_sha256"), "sources_dispatch_sha256"),
    (lambda c: c["sources_dispatch_sha256"].pop("dispatch.py"), "sources_dispatch_sha256"),
    (lambda c: c["sources_dispatch_sha256"].update({"dispatch.py": "g" * 64}), "SHA-256"),
    (lambda c: c.update(format="cascade-entropy-integration-b4-c-v3"), "Version d'intégration"),
])
def test_contrat_v2_altere_refuse(tmp_path, modifier, message):
    dossier = tmp_path / "b4"
    sauver(attribuer("independantes", "min_charge_max"), dossier)
    chemin = dossier / "metadata.json"
    meta = json.loads(chemin.read_text(encoding="utf-8"))
    modifier(meta["parametres_experience"][integration.CLE])
    meta.pop("sha256_metadata")
    meta["sha256_metadata"] = p._empreinte(meta)
    chemin.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        integration.charger_etat_initial(dossier)
