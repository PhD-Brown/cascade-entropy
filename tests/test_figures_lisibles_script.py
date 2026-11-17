"""Script `12_figures_lisibles.py` (figures lisibles du rapport d'étape).

Les tests ne lisent et n'écrivent que dans un dossier temporaire : un faux run
de 07 (deux tailles, trois G, séries synthétiques) et un faux entropie.csv
sont construits, et les dossiers du module sont redirigés avec `monkeypatch`.
Aucun test existant n'est touché.
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
TAILLES = [46, 94]
VALEURS_G = [0.1, 1.0, 2.0]
JOURS = 600
TRANSITOIRE = 100


@pytest.fixture(scope="module")
def script12():
    sys.path.insert(0, str(RACINE))
    nom = "script12_figures_lisibles"
    spec = importlib.util.spec_from_file_location(nom, RACINE / "scripts" / "12_figures_lisibles.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[nom] = module
    spec.loader.exec_module(module)
    return module


def _faux_run(script12, dossier: Path, nom: str = "faux-run") -> Path:
    run_dir = dossier / nom
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"config": {
        "tailles": TAILLES, "G": VALEURS_G, "jours": JOURS, "transitoire": TRANSITOIRE}}),
        encoding="utf-8")
    rng = np.random.default_rng(1)
    for n in TAILLES:
        for G in VALEURS_G:
            x = 10.0 + G + rng.normal(0, 0.1, JOURS)
            np.savez(run_dir / f"cas_{script12.nom_cas(n, G)}.npz",
                     jour=np.arange(JOURS), nombre_effectif=x)
    return run_dir


def _faux_entropie(chemin: Path, observables=("nombre_effectif", "taux_maximal",
                                              "charge_relative")) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n_noeuds", "G", "observable", "mesure", "echelle",
                                          "valeur", "z_melange", "z_iaaft"])
        w.writeheader()
        for n in TAILLES:
            for G in VALEURS_G:
                for o in observables:
                    for mesure in ("mse", "pe_d4"):
                        for s in (1, 2, 5, 10):
                            w.writerow({"n_noeuds": n, "G": G, "observable": o, "mesure": mesure,
                                        "echelle": s, "valeur": 1.0,
                                        "z_melange": s * G, "z_iaaft": -G})
    return chemin


@pytest.fixture
def env(script12, tmp_path, monkeypatch):
    """Faux dépôt : run de 07, entropie.csv de 09, sorties dans tmp_path."""
    runs = tmp_path / "data" / "07_dynamique_lente" / "runs"
    run_dir = _faux_run(script12, runs)
    so3 = tmp_path / "data" / "09_analyse_so3"
    _faux_entropie(so3 / run_dir.name / "entropie.csv")
    monkeypatch.setattr(script12, "DOSSIER_RUNS_07", runs)
    monkeypatch.setattr(script12, "DOSSIER_SO3", so3)
    monkeypatch.setattr(script12, "DOSSIER_SORTIE", tmp_path / "data" / "12_figures_lisibles")
    monkeypatch.setattr(script12, "DOSSIER_FIG", tmp_path / "figures" / "12_figures_lisibles")
    return {"tmp": tmp_path, "runs": runs, "run_dir": run_dir, "so3": so3}


# --- Utilitaires ------------------------------------------------------------

def test_nom_cas_identique_aux_scripts_07_et_09(script12):
    assert script12.nom_cas(94, 1) == "N94_G1p0000"
    assert script12.nom_cas(46, 0.25) == "N46_G0p2500"


def test_nettoyer_id_refuse_vide(script12):
    with pytest.raises(ValueError):
        script12.nettoyer_id("  /// ")


def test_moyenne_glissante_valeurs_et_bords(script12):
    x = np.arange(10, dtype=float)
    m = script12.moyenne_glissante(x, 3)
    assert m.shape == x.shape
    assert np.isnan(m[0]) and np.isnan(m[-1])
    np.testing.assert_allclose(m[1:-1], x[1:-1])


def test_moyenne_glissante_largeur_invalide(script12):
    with pytest.raises(SystemExit):
        script12.moyenne_glissante(np.ones(5), 0)
    with pytest.raises(SystemExit):
        script12.moyenne_glissante(np.ones(5), 6)


def test_extraire_fenetre_commence_apres_le_transitoire(script12):
    x = np.arange(100, dtype=float)
    f = script12.extraire_fenetre(x, transitoire=20, debut=5, duree=10)
    np.testing.assert_array_equal(f, np.arange(25, 35))


@pytest.mark.parametrize("debut, duree", [(0, 1), (-1, 10), (0, 1000)])
def test_extraire_fenetre_refuse_les_fenetres_invalides(script12, debut, duree):
    with pytest.raises(SystemExit):
        script12.extraire_fenetre(np.ones(100), transitoire=10, debut=debut, duree=duree)


def test_extraire_fenetre_refuse_les_nan(script12):
    x = np.ones(50)
    x[30] = np.nan
    with pytest.raises(SystemExit):
        script12.extraire_fenetre(x, transitoire=10, debut=0, duree=30)


def test_statistiques(script12):
    s = script12.statistiques(np.array([1.0, 2.0, 3.0]))
    assert s == {"moyenne": 2.0, "ecart_type": pytest.approx(np.std([1, 2, 3])),
                 "min": 1.0, "max": 3.0, "n_jours": 3}


def test_notation_francaise(script12):
    assert script12.texte_G(0.75) == "0{,}75"
    assert script12.texte_G(2) == "2"
    assert script12.virgule(1.5) == "1,5"


def test_trouver_G(script12):
    assert script12.trouver_G(2, VALEURS_G) == 2.0
    with pytest.raises(SystemExit):
        script12.trouver_G(0.5, VALEURS_G)


# --- Recherche du run -------------------------------------------------------

def test_choisir_run_unique(script12, env):
    assert script12.choisir_run(None) == env["run_dir"]


def test_choisir_run_plusieurs_exige_run(script12, env):
    _faux_run(script12, env["runs"], "autre-run")
    with pytest.raises(SystemExit, match="Plusieurs runs"):
        script12.choisir_run(None)
    assert script12.choisir_run("autre-run").name == "autre-run"


def test_choisir_run_absent(script12, env, tmp_path):
    with pytest.raises(SystemExit):
        script12.choisir_run("inexistant")
    with pytest.raises(SystemExit, match="Aucun run"):
        script12.choisir_run(None, tmp_path / "vide")


def test_lire_config_incomplete(script12, tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"config": {"tailles": [46]}}))
    with pytest.raises(SystemExit, match="incomplet"):
        script12.lire_config(tmp_path)


# --- entropie.csv -----------------------------------------------------------

def test_lire_entropie_filtre_la_mesure(script12, env):
    lignes = script12.lire_entropie(env["so3"] / env["run_dir"].name / "entropie.csv")
    assert lignes and all(isinstance(l["z_melange"], float) for l in lignes)
    assert len(lignes) == len(TAILLES) * len(VALEURS_G) * 3 * 4


def test_lire_entropie_colonnes_manquantes(script12, tmp_path):
    chemin = tmp_path / "entropie.csv"
    chemin.write_text("n_noeuds,G\n46,0.1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Colonnes absentes"):
        script12.lire_entropie(chemin)


def test_lire_entropie_observable_inconnu(script12, tmp_path):
    chemin = _faux_entropie(tmp_path / "e.csv", observables=("inconnu",))
    with pytest.raises(SystemExit, match="sans libellé"):
        script12.lire_entropie(chemin)


def test_libelles_sans_noms_de_code(script12):
    for texte in (*script12.LIBELLES.values(), *script12.LIBELLES_COURTS.values()):
        assert "_" not in texte.replace("_{", "")


# --- Script complet ---------------------------------------------------------

def test_main_produit_figures_et_metadonnees(script12, env):
    meta = script12.main(["--run-id", "essai", "--taille", "94", "--G", "0.1", "2",
                          "--duree", "300", "--moyenne", "20"])
    fig_dir = env["tmp"] / "figures" / "12_figures_lisibles" / "essai"
    for nom in ("rapport_series_nflux", "rapport_so3_nflux", "rapport_so3_complet_melange",
                "rapport_so3_complet_iaaft"):
        assert (fig_dir / f"{nom}.png").exists() and (fig_dir / f"{nom}.pdf").exists()
    ecrit = json.loads((env["tmp"] / "data" / "12_figures_lisibles" / "essai" / "metadata.json")
                       .read_text("utf-8"))
    assert ecrit["run_source"] == "faux-run"
    assert set(ecrit["statistiques_extraits"]) == {"G=0.1", "G=2"}
    assert ecrit["statistiques_extraits"]["G=2"]["n_jours"] == 300
    assert "entropie.csv" in ecrit["entrees"]
    assert meta["figures"]


def test_main_fenetre_identique_pour_les_deux_regimes(script12, env):
    meta = script12.main(["--run-id", "fenetre", "--figures", "series", "--duree", "200"])
    stats = meta["statistiques_extraits"]
    assert {s["n_jours"] for s in stats.values()} == {200}
    # Séries synthétiques : moyenne 10 + G (vérifie qu'on lit bien le bon cas).
    assert stats["G=0.1"]["moyenne"] == pytest.approx(10.1, abs=0.05)
    assert stats["G=2"]["moyenne"] == pytest.approx(12.0, abs=0.05)


def test_main_refuse_d_ecraser(script12, env):
    script12.main(["--run-id", "double", "--figures", "so3"])
    with pytest.raises(SystemExit, match="existe déjà"):
        script12.main(["--run-id", "double", "--figures", "so3"])
    script12.main(["--run-id", "double", "--figures", "so3", "--ecraser"])


def test_main_exige_run_id(script12, env):
    with pytest.raises(SystemExit, match="obligatoire"):
        script12.main([])


def test_main_refuse_deux_G_egaux(script12, env):
    with pytest.raises(SystemExit, match="différentes"):
        script12.main(["--run-id", "x", "--G", "1", "1"])


def test_main_taille_absente(script12, env):
    with pytest.raises(SystemExit, match="Taille"):
        script12.main(["--run-id", "x", "--taille", "190", "--figures", "series"])


def test_main_lister_n_ecrit_rien(script12, env, capsys):
    assert script12.main(["--lister"]) == {}
    assert "faux-run" in capsys.readouterr().out
    assert not (env["tmp"] / "data" / "12_figures_lisibles").exists()
