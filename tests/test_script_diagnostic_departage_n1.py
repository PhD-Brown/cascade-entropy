"""Contrôles des sorties techniques et de la reprise, sans campagne scientifique."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def script():
    chemin = Path(__file__).parents[1] / "scripts/diagnostic_departage_n1.py"
    spec = importlib.util.spec_from_file_location("diagnostic_departage", chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_exclusif_reprise_et_tracabilite(tmp_path, script, monkeypatch):
    # Aucune réduction de taille dans le protocole livré : le test réduit seul
    # le banc à un graphe synthétique de 28 nœuds pour vérifier les fichiers.
    monkeypatch.setattr(script, "reseaux", lambda audit=False: [("temoin", script.T.arbre_temoin(28))])
    args = ["--run-id", "test", "--sortie", str(tmp_path)]
    assert script.main(args) == 0
    dossier = tmp_path / "test"
    fichiers = {p.name: p.read_bytes() for p in dossier.iterdir()}
    m = json.loads(fichiers["metadata.json"])
    assert m["parametres"]["p1"] == 1.
    assert all(m["environnement"][v] for v in ("numpy", "scipy", "highs"))
    assert m["sha256_sources"]["scripts/diagnostic_departage_n1.py"]
    with pytest.raises(FileExistsError):
        script.main(args)
    assert script.main(args + ["--resume"]) == 0
    assert {p.name: p.read_bytes() for p in dossier.iterdir()} == fichiers
    with pytest.raises(ValueError, match="Reprise refusée"):
        script.main(args + ["--resume", "--audit-reference"])
    # Simuler un arrêt après les résultats par règle, avant le marqueur final.
    (dossier / "complete.json").unlink()
    assert script.main(args + ["--resume"]) == 0
    assert (dossier / "comparatif.csv").read_bytes() == fichiers["comparatif.csv"]


@pytest.mark.parametrize("nom", ["../sortie", "a/b", "a\\b", "", "."])
def test_nom_run_invalide(tmp_path, script, nom):
    with pytest.raises(SystemExit):
        script.main(["--run-id", nom, "--sortie", str(tmp_path)])


def test_ecriture_exclusive_atomique(tmp_path, script):
    chemin = tmp_path / "valeur.json"
    script._sauver(chemin, {"a": 1})
    with pytest.raises(FileExistsError):
        script._sauver(chemin, {"a": 2})
    assert json.loads(chemin.read_text(encoding="utf-8")) == {"a": 1}
    assert list(tmp_path.iterdir()) == [chemin]
