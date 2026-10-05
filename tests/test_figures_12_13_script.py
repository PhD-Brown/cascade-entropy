"""Script `05_figures_12_13.py` (Fig. 12–13 de Carreras 2002).

Les runs de `05_reproduction_carreras.py` sont fabriqués dans un dossier
temporaire (checkpoints au format réel) : aucune simulation n'est lancée et
aucun fichier du dépôt n'est lu ni écrit.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cascade_entropy.carreras import configuration_arbre, seuil_transport

RACINE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def script09():
    sys.path.insert(0, str(RACINE))
    spec = importlib.util.spec_from_file_location(
        "script09", RACINE / "scripts" / "05_figures_12_13.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_05(dossier: Path, run_id: str, cas: dict, n_realisations: int,
            taille_bloc: int, graine: int = 0) -> None:
    """Fabrique un run de 05 : metadata.json et checkpoints par bloc."""
    run_dir = dossier / run_id
    (run_dir / "checkpoints").mkdir(parents=True)
    config = {"n_realisations": n_realisations, "gamma": 1.9, "p0": 1e-4,
              "n_regions": 3, "tailles": sorted({n for n, _ in cas})}
    (run_dir / "metadata.json").write_text(json.dumps({"config": config}), "utf-8")
    rng = np.random.default_rng(graine)
    for (n, ratio), n_total in cas.items():
        for b, debut in enumerate(range(0, n_total, taille_bloc)):
            k = min(taille_bloc, n_total - debut)
            x = np.where(rng.random(k) < 0.3, (rng.pareto(1.0, k) + 1) * 1e-3, 0.0)
            np.savez(run_dir / "checkpoints" / f"N{n}_r{ratio:.6f}_b{b:04d}.npz",
                     fraction_delestee_nominale=np.minimum(x, 1.0),
                     fraction_delestee_realisee=np.minimum(x, 1.0) / 1.2,
                     n_lignes_tombees=rng.integers(0, 3, k),
                     __n__=np.array(k), __n_noeuds__=np.array(n),
                     __ratio__=np.array(ratio), __index_bloc__=np.array(b))


@pytest.fixture
def runs(script09, tmp_path, monkeypatch):
    r_t = {n: seuil_transport(configuration_arbre(n)) for n in (94, 382)}
    dossier = tmp_path / "runs05"
    _run_05(dossier, "fig12", {(382, 0.45 * r_t[382]): 3000,
                               (382, 0.55 * r_t[382]): 3000}, 3000, 1000, graine=1)
    _run_05(dossier, "fig13", {(94, 0.55 * r_t[94]): 3000,
                               (94, 0.70 * r_t[94]): 2500}, 3000, 1000, graine=2)
    monkeypatch.setattr(script09, "DOSSIER_RUNS_05", dossier)
    monkeypatch.setattr(script09, "DOSSIER_SORTIE", tmp_path / "sortie")
    monkeypatch.setattr(script09, "DOSSIER_FIG", tmp_path / "figures")
    return tmp_path


def test_plan_ratios_et_reutilisation(script09, capsys):
    script09.main(["plan", "--n", "100", "--chunk-size", "50", "--workers", "1"])
    sortie = capsys.readouterr().out
    lignes = [l for l in sortie.splitlines()
              if l.startswith("python scripts\\05_reproduction_carreras.py scan")]
    r_t382 = seuil_transport(configuration_arbre(382))
    assert f"{0.45 * r_t382:.6f}" in lignes[0] and "--run-id fig12-N382" in lignes[0]
    # Le 382 à ρ = 0.55 est déjà dans la Fig. 12 : pas de run fig13-N382.
    assert [l.split("--run-id ")[1] for l in lignes[1:]] == [
        "fig13-N46", "fig13-N94", "fig13-N190"]
    assert all("--departage exterieur_dabord" in l for l in lignes)


def test_figures_sorties_et_cas_incomplet(script09, runs):
    script09.main(["figures", "--fig12", "fig12", "--fig13", "fig13",
                   "--prediction", "2000", "--sortie", "essai"])
    lignes = list(csv.DictReader((runs / "sortie" / "essai" / "queues.csv")
                                 .open(encoding="utf-8")))
    assert len(lignes) == 3                       # le cas à 2500 / 3000 est sauté
    meta = json.loads((runs / "sortie" / "essai" / "metadata.json").read_text("utf-8"))
    assert any("incomplet" in v for v in meta["cas_sautes"].values())
    assert len(meta["cas_fig12"]) == 2 and len(meta["cas_fig13"]) == 2
    sous = [l for l in lignes if l["regime"] == "sous-critique"]
    assert len(sous) == 1 and float(sous[0]["rho"]) == pytest.approx(0.45)
    assert np.isfinite(float(sous[0]["prediction_freq_blackout"]))
    figures = {p.name for p in (runs / "figures" / "essai").iterdir()}
    assert figures == {"fig12_pdf_N382.png", "fig13_tailles.png"}


def test_prediction_sans_avarie_egale_limite_de_generation(script09):
    """p0 = 0 : délestage = max(0, D − P_C), reconstruit avec les mêmes tirages."""
    cfg = configuration_arbre(94)
    ratio = 0.65   # la demande peut dépasser P_C si le facteur moyen > 1/0.65
    config = {"gamma": 1.9, "p0": 0.0, "n_regions": 3}
    obtenu = script09.prediction_sous_critique(94, ratio, config, 5000,
                                               np.random.default_rng(3))
    rng = np.random.default_rng(3)
    from cascade_entropy.carreras import groupes_regions
    comptes = np.bincount(groupes_regions(cfg.reseau, 3), minlength=3)
    f = rng.uniform(0.1, 1.9, size=(5000, 3))
    d = ratio * cfg.p_c / cfg.n_charges * f @ comptes
    np.testing.assert_allclose(obtenu, np.maximum(d - cfg.p_c, 0) / (ratio * cfg.p_c))
    assert 0 < np.mean(obtenu > 0) < 0.2


@pytest.mark.parametrize("options", [
    ["figures"],                                   # aucun run
    ["figures", "--fig12", "fig12", "--classes", "2"],
    ["figures", "--fig12", "fig12", "--rho-fig13", "0"],
    ["figures", "--fig12", "inexistant"],
    ["plan", "--rho-fig12", "-0.1"],
])
def test_arguments_invalides(script09, runs, options):
    with pytest.raises(SystemExit):
        script09.main(options)


def test_normalisation_realisee(script09, runs):
    """Option : délestage / demande du tirage ; la valeur par défaut reste nominale."""
    script09.main(["figures", "--fig12", "fig12", "--prediction", "500",
                   "--normalisation", "realisee", "--sortie", "real"])
    meta = json.loads((runs / "sortie" / "real" / "metadata.json").read_text("utf-8"))
    assert meta["parametres"]["observable"] == "fraction_delestee_realisee"
    p = script09.prediction_sous_critique(94, 0.65, {"gamma": 1.9, "p0": 1e-3,
                                                     "n_regions": 3},
                                          3000, np.random.default_rng(1), "realisee")
    assert np.all((p >= 0) & (p <= 1))
    with pytest.raises(ValueError):
        script09.prediction_sous_critique(94, 0.65, {"gamma": 1.9, "p0": 0.0,
                                                     "n_regions": 3},
                                          10, np.random.default_rng(1), "autre")
