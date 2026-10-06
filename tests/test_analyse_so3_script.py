"""Script `09_analyse_so3.py` (SO3, étape 1) sur un petit run synthétique.

Le run est fabriqué dans un dossier temporaire : aucune simulation, aucun
fichier du dépôt n'est lu ni écrit. On vérifie les sorties, le traitement des
cas absents, incomplets ou contenant des NaN, la charge relative dérivée, la
reprise (`--resume`) et l'indépendance du résultat au nombre de processus.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

RACINE = Path(__file__).resolve().parents[1]
JOURS = 6000
CONFIG = {"tailles": [46], "G": [0.5, 1.0, 1.5, 2.0], "jours": JOURS, "transitoire": 500}
PETITS = ["--melanges", "3", "--iaaft", "3", "--iterations-iaaft", "20",
          "--echelles", "1", "2", "5", "--dimensions", "3", "4", "--dimension-figure", "3"]


@pytest.fixture(scope="module")
def script09():
    sys.path.insert(0, str(RACINE))
    spec = importlib.util.spec_from_file_location(
        "script09", RACINE / "scripts" / "09_analyse_so3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cas(chemin: Path, graine: int, jours: int = JOURS, nan_effectif: bool = False) -> None:
    rng = np.random.default_rng(graine)
    t = np.arange(jours)
    effectif = 50 + 5 * rng.standard_normal(jours) + 2 * np.sin(2 * np.pi * t / 1500)
    if nan_effectif:
        effectif[1000] = np.nan  # après le transitoire de 500 jours
    lignes = np.where(rng.random(jours) < 0.05, rng.integers(1, 9, jours), 0)
    np.savez(chemin, jour=t, nombre_effectif=effectif,
             taux_maximal=np.clip(0.8 + 0.1 * rng.standard_normal(jours), 0, 1),
             demande_totale=np.full(jours, 900.0) * (1 + 0.1 * rng.random(jours)),
             capacite_totale_generateurs=np.full(jours, 1000.0),
             fraction_delestee=np.where(lignes > 0, 0.01 * lignes, 0.0),
             n_lignes_tombees=lignes)


@pytest.fixture
def run(script09, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    run_dir = runs / "essai"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"config": CONFIG}), "utf-8")
    _cas(run_dir / "cas_N46_G0p5000.npz", 1)
    _cas(run_dir / "cas_N46_G1p0000.npz", 2)
    _cas(run_dir / "cas_N46_G1p5000.npz", 3, jours=JOURS - 100)   # incomplet
    _cas(run_dir / "cas_N46_G2p0000.npz", 4, nan_effectif=True)   # NaN dans un observable
    monkeypatch.setattr(script09, "DOSSIER_RUNS_07", runs)
    monkeypatch.setattr(script09, "DOSSIER_SORTIE", tmp_path / "sortie")
    monkeypatch.setattr(script09, "DOSSIER_FIG", tmp_path / "figures")
    return tmp_path


def _lire(chemin: Path) -> list[dict]:
    with chemin.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_sorties_completes(script09, run):
    script09.main(["--run", "essai", *PETITS])
    sortie = run / "sortie" / "essai"
    meta = json.loads((sortie / "metadata.json").read_text("utf-8"))
    # 2 cas complets × 3 observables + G = 2 sans nombre_effectif (NaN) = 8 tâches.
    assert len(meta["taches"]) == 8
    assert "N46_G1p5000" in meta["cas_sautes"]
    assert "N46_G2p0000:nombre_effectif" in meta["cas_sautes"]
    assert meta["parametres"]["melanges"] == 3 and meta["methode_sampen"] == "arbre"
    assert set(meta["sha256_entrees"]) == {"cas_N46_G0p5000.npz", "cas_N46_G1p0000.npz",
                                           "cas_N46_G2p0000.npz"}

    entropie = _lire(sortie / "entropie.csv")
    # 8 tâches × (MSE + 2 dimensions de PE) × 3 échelles.
    assert len(entropie) == 8 * 3 * 3
    assert {l["mesure"] for l in entropie} == {"mse", "pe_d3", "pe_d4"}
    resume = _lire(sortie / "resume.csv")
    assert len(resume) == 8
    assert all(float(l["ecart_spectral_iaaft_max"]) < 0.05 for l in resume)

    egalites = _lire(sortie / "egalites.csv")
    # 3 cas lus × (2 séries de diagnostic seul + 3 observables).
    assert len(egalites) == 15
    delestage = [l for l in egalites if l["serie"] == "fraction_delestee"]
    assert all(l["entropie_calculee"] == "False" for l in delestage)
    assert all(float(l["fraction_zeros"]) > 0.9 for l in delestage)

    figures = run / "figures" / "essai"
    for nom in ("mse_nombre_effectif.png", "mse_taux_maximal.png", "mse_charge_relative.png",
                "z_mse_melange.png", "z_mse_iaaft.png", "z_pe_d3_melange.png", "z_pe_d3_iaaft.png"):
        assert (figures / nom).exists(), nom


def test_charge_relative_derivee(script09, run):
    series = script09.lire_cas(run / "runs" / "essai" / "cas_N46_G0p5000.npz", JOURS, 500)
    with np.load(run / "runs" / "essai" / "cas_N46_G0p5000.npz") as d:
        attendu = d["demande_totale"][500:] / d["capacite_totale_generateurs"][500:]
    np.testing.assert_allclose(series["charge_relative"], attendu)
    assert series["nombre_effectif"].size == JOURS - 500


def test_reprise_reutilise_les_taches(script09, run, capsys):
    script09.main(["--run", "essai", *PETITS, "--sans-figures"])
    premier = (run / "sortie" / "essai" / "entropie.csv").read_text("utf-8")
    capsys.readouterr()
    script09.main(["--run", "essai", *PETITS, "--sans-figures", "--resume"])
    assert "8 tâches réutilisées, 0 à calculer" in capsys.readouterr().out
    assert (run / "sortie" / "essai" / "entropie.csv").read_text("utf-8") == premier


def test_reprise_recalcule_si_les_parametres_changent(script09, run, capsys):
    script09.main(["--run", "essai", *PETITS, "--sans-figures"])
    capsys.readouterr()
    autres = [a if a != "20" else "15" for a in PETITS]  # --iterations-iaaft 15
    script09.main(["--run", "essai", *autres, "--sans-figures", "--resume"])
    assert "0 tâches réutilisées, 8 à calculer" in capsys.readouterr().out


def test_resultat_independant_du_nombre_de_processus(script09, run):
    script09.main(["--run", "essai", *PETITS, "--sans-figures", "--sortie", "un"])
    script09.main(["--run", "essai", *PETITS, "--sans-figures", "--sortie", "deux",
                   "--workers", "2"])
    a = _lire(run / "sortie" / "un" / "entropie.csv")
    b = _lire(run / "sortie" / "deux" / "entropie.csv")
    assert a == b


@pytest.mark.parametrize("mauvais", [
    ["--melanges", "1"],
    ["--iaaft", "1"],
    ["--r", "0"],
    ["--echelles", "2", "1"],
    ["--dimensions", "3", "3"],
    ["--dimension-figure", "6"],
    ["--workers", "0"],
    ["--transitoire", "5000"],
])
def test_arguments_invalides(script09, run, mauvais):
    with pytest.raises(SystemExit):
        script09.main(["--run", "essai", *PETITS, *mauvais])


def test_run_absent(script09, run):
    with pytest.raises(SystemExit):
        script09.main(["--run", "inexistant"])
