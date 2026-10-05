"""Script `08_analyse_so2.py` (SO2, mission 3) sur un petit run synthétique.

Le run est fabriqué dans un dossier temporaire : aucune simulation, aucun
fichier du dépôt n'est lu ni écrit. On vérifie les sorties, le traitement des
cas absents ou incomplets, la reproductibilité et les périodes prédites.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

RACINE = Path(__file__).resolve().parents[1]
JOURS = 9000
CONFIG = {"tailles": [46], "G": [0.5, 1.0, 2.0], "jours": JOURS, "transitoire": 1000,
          "mu": 1.05, "lambda_annuel": 1.018, "k": 0.02, "g": 0.9}


@pytest.fixture(scope="module")
def script08():
    sys.path.insert(0, str(RACINE))
    spec = importlib.util.spec_from_file_location(
        "script08", RACINE / "scripts" / "08_analyse_so2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cas_synthetique(chemin: Path, graine: int, jours: int = JOURS) -> None:
    rng = np.random.default_rng(graine)
    t = np.arange(jours)
    lignes = rng.poisson(0.2 * (1 + np.sin(2 * np.pi * t / 998)))
    fraction = np.where(rng.random(jours) < 0.05, rng.random(jours) * 0.3, 0.0)
    fraction[lignes > 0] += 0.01
    demande = np.full(jours, 1000.0)
    np.savez(chemin, jour=t, demande_moyenne=demande, delestage_total=fraction * demande,
             fraction_delestee=fraction, n_lignes_tombees=lignes,
             n_mises_a_niveau=(t % 30 == 0).astype(int), etat_puissance_max=np.ones(12))


@pytest.fixture
def run(script08, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    run_dir = runs / "essai"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"config": CONFIG}), "utf-8")
    _cas_synthetique(run_dir / "cas_N46_G0p5000.npz", 1)
    _cas_synthetique(run_dir / "cas_N46_G1p0000.npz", 2, jours=JOURS - 500)  # incomplet
    # G = 2 : fichier absent.
    monkeypatch.setattr(script08, "DOSSIER_RUNS_07", runs)
    monkeypatch.setattr(script08, "DOSSIER_SORTIE", tmp_path / "sortie")
    monkeypatch.setattr(script08, "DOSSIER_FIG", tmp_path / "figures")
    return tmp_path


def _lire(chemin: Path) -> list[dict]:
    with chemin.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_sorties_et_cas_sautes(script08, run):
    script08.main(["--run", "essai", "--melanges", "5"])
    sortie = run / "sortie" / "essai"
    cas = _lire(sortie / "cas.csv")
    assert len(cas) == 1 and cas[0]["G"] == "0.5"
    hurst = _lire(sortie / "hurst.csv")
    assert len(hurst) == 2 * 4
    assert {h["plage"] for h in hurst} == {"court", "long", "avant_cycle", "apres_cycle"}
    meta = json.loads((sortie / "metadata.json").read_text("utf-8"))
    assert meta["cas_analyses"] == ["N46_G0p5000"]
    assert "incomplet" in meta["cas_sautes"]["N46_G1p0000"]
    assert meta["cas_sautes"]["N46_G2p0000"] == "fichier absent"
    assert set(meta["sha256_entrees"]) == {"cas_N46_G0p5000.npz"}
    figures = {p.name for p in (run / "figures" / "essai").iterdir()}
    assert {"fig04_hurst.png", "fig07_servie_lignes.png", "fig09_grands_blackouts.png",
            "fig08_lignes_N46.png", "fig_cycle.png", "tableI_queue_N46.png",
            "fig03_rs_N46_G0p5000.png"} <= figures


def test_cycle_et_generation_retrouves(script08, run):
    script08.main(["--run", "essai", "--melanges", "3", "--sans-figures-rs"])
    cas = _lire(run / "sortie" / "essai" / "cas.csv")[0]
    assert float(cas["periode_lignes_mesuree"]) == pytest.approx(998, rel=0.05)
    assert float(cas["intervalle_mises_a_niveau_mesure"]) == 30.0


def test_reproductible(script08, run):
    script08.main(["--run", "essai", "--melanges", "4", "--sans-figures-rs"])
    a = (run / "sortie" / "essai" / "hurst.csv").read_text("utf-8")
    script08.main(["--run", "essai", "--melanges", "4", "--sans-figures-rs",
                   "--sortie", "bis"])
    b = (run / "sortie" / "bis" / "hurst.csv").read_text("utf-8")
    assert a == b


def test_periodes_predites(script08):
    assert script08.periode_cycle_lignes(CONFIG) == pytest.approx(
        math.log(1.05) / (math.log(1.018) / 365))
    attendu = (0.02 / 12) / ((1 + 0.5 * 0.9) * math.log(1.018 ** (1 / 365)))
    assert script08.periode_cycle_generation(CONFIG, 0.5, 12) == pytest.approx(attendu)
    plages = script08.plages_hurst(CONFIG)
    assert plages["avant_cycle"][1] == plages["apres_cycle"][0]


@pytest.mark.parametrize("options", [["--melanges", "0"], ["--transitoire", "8000"],
                                     ["--r2-min", "1.5"], ["--classes", "2"]])
def test_arguments_invalides(script08, run, options):
    with pytest.raises(SystemExit):
        script08.main(["--run", "essai", *options])


def test_run_introuvable(script08, run):
    with pytest.raises(SystemExit, match="introuvable"):
        script08.main(["--run", "inexistant"])
