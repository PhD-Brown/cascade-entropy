"""
Réplication contrôlée de Carreras et al. (2002) — effet de taille finie.

Version 2 : exécution reproductible, non destructive et reprenable.

Principes
---------
La physique du modèle reste dans `cascade_entropy/carreras.py`.
Ce script ne change PAS les paramètres scientifiques de la réplication :
- P_G = 2623.9 par générateur ;
- P_C = somme_j P_j^max = 31486.8 ;
- limites de lignes de la Table I, sans recalibration par taille ;
- fluctuations régionales corrélées ;
- N_F=3 par défaut comme hypothèse contrôlée ;
- p0=1e-4, gamma=1.9 ;
- p1=1 par défaut comme hypothèse de réplication.

Cette version professionnalise l'ORCHESTRATION :
- chaque run est isolé dans son propre dossier : aucune donnée n'est écrasée ;
- le scan et la production sont découpés en petits blocs ;
- chaque bloc terminé est sauvegardé immédiatement ;
- un run interrompu peut être repris avec `--resume RUN_ID` ;
- Ctrl+C arrête proprement le processus principal et termine les workers ;
- les graines aléatoires sont déterministes par bloc ;
- le scan accepte `--ratios` pour cibler quelques valeurs précises ;
- un journal texte et un `status.json` permettent de suivre l'état du run ;
- les figures et métadonnées sont propres à chaque run ;
- la taille conditionnelle des cascades de lignes est suivie explicitement.

Arborescence
------------
data/05_reproduction_carreras/
└── runs/
    └── <run_id>/
        ├── metadata.json
        ├── status.json
        ├── run.log
        ├── scan.csv ou resume_production.csv
        ├── checkpoints/
        │   └── *.npz
        └── production_<N>_noeuds.npz  (production seulement)

figures/05_reproduction_carreras/
└── runs/
    └── <run_id>/
        ├── scan_controle.png
        ├── scan_cascades_lignes.png
        └── ccdf_taille_finie.png       (production seulement)

Exemples
--------
Audit :
    python scripts/05_reproduction_carreras.py audit

Scan grossier ciblé :
    python scripts/05_reproduction_carreras.py scan \
        --tailles 190 382 \
        --ratios 0.78 0.80 0.81 \
        --n 1000 \
        --chunk-size 100 \
        --workers 16

Scan régulier :
    python scripts/05_reproduction_carreras.py scan \
        --tailles 46 94 \
        --ratio-min 0.60 --ratio-max 0.90 --pas 0.01 \
        --n 2500 --chunk-size 250 --workers 16

Production :
    python scripts/05_reproduction_carreras.py production \
        --tailles 46 94 \
        --ratio 0.84 \
        --n 60000 \
        --chunk-size 1000 \
        --workers 16

Reprendre un run interrompu :
    python scripts/05_reproduction_carreras.py scan \
        --resume <RUN_ID> \
        --workers 16

Lister les runs :
    python scripts/05_reproduction_carreras.py runs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Évite la sur-parallélisation cachée : nous parallélisons déjà par processus.
# setdefault respecte une valeur explicitement fournie par l'utilisateur.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import matplotlib
import numpy as np
import powerlaw
from scipy.optimize import OptimizeWarning

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cascade_entropy.carreras import (  # noqa: E402
    GAMMA_TABLE,
    P0_TABLE,
    P1_REFERENCE,
    P_G_TABLE,
    P_L_TABLE,
    configuration_arbre,
    demande_regionale,
    groupes_regions,
    sigma_relatif_papier,
    sigma_relatif_uniforme,
)
from cascade_entropy.cascade import journee  # noqa: E402
from cascade_entropy.chemins import DOSSIER_DATA, DOSSIER_FIGURES  # noqa: E402
from cascade_entropy.dispatch import demande_uniforme, resoudre  # noqa: E402
from cascade_entropy.reseau import matrice_de_flux  # noqa: E402


SCRIPT_VERSION = "2.0.0"

DOSSIER_BASE = DOSSIER_DATA / "05_reproduction_carreras"
DOSSIER_RUNS = DOSSIER_BASE / "runs"
DOSSIER_FIGURES_BASE = DOSSIER_FIGURES / "05_reproduction_carreras"
DOSSIER_FIGURES_RUNS = DOSSIER_FIGURES_BASE / "runs"

GRAINE_SCAN = 20261003
GRAINE_PRODUCTION = 20261004

WORKERS_DEFAUT = min(16, max(1, (os.cpu_count() or 2) - 1))

ETAT_EN_COURS = "running"
ETAT_TERMINE = "completed"
ETAT_INTERROMPU = "interrupted"
ETAT_ERREUR = "failed"


# ===========================================================================
# Utilitaires généraux
# ===========================================================================

def maintenant_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def maintenant_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def nettoyer_id(texte: str) -> str:
    texte = re.sub(r"[^A-Za-z0-9_.-]+", "-", texte.strip())
    texte = texte.strip("-_.")
    if not texte:
        raise ValueError("run_id invalide.")
    return texte


def ratio_token(ratio: float) -> str:
    """Représentation stable et lisible d'un ratio dans un nom de fichier."""
    return f"{float(ratio):.6f}".replace("-", "m").replace(".", "p")


def tailles_token(tailles: list[int] | tuple[int, ...]) -> str:
    return "-".join(str(int(n)) for n in tailles)


def atomic_write_text(path: Path, texte: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(texte, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def atomic_write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    """
    Sauvegarde atomique d'un checkpoint.

    On écrit d'abord dans un fichier temporaire, puis `os.replace()` effectue
    un remplacement atomique sur le même volume.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def lire_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def journaliser(run_dir: Path | None, message: str = "") -> None:
    print(message, flush=True)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(f"[{maintenant_iso()}] {message}\n")


def graine_stable(*composants: Any) -> int:
    """
    Produit une graine uint64 stable entre machines et exécutions.

    Contrairement à `hash()`, SHA-256 n'est pas salé par processus.
    """
    texte = "|".join(str(x) for x in composants)
    digest = hashlib.sha256(texte.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def decouper(n: int, chunk: int) -> list[int]:
    blocs: list[int] = []
    reste = int(n)
    while reste > 0:
        courant = min(int(chunk), reste)
        blocs.append(courant)
        reste -= courant
    return blocs


def _initialiser_worker() -> None:
    """
    Les workers ignorent Ctrl+C : seul le processus principal gère l'interruption.

    Cela évite les dizaines de tracebacks `KeyboardInterrupt` sous Windows.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


def _terminer_executor(executor: ProcessPoolExecutor) -> None:
    """
    Arrêt immédiat sous Python 3.11.

    `ProcessPoolExecutor` ne fournit pas encore `terminate_workers()` en 3.11.
    Nous annulons donc les tâches en attente puis terminons explicitement les
    processus actifs. `_processes` est une API privée : elle est utilisée ici
    uniquement pour garantir un Ctrl+C réellement immédiat sous Windows.

    Important : on capture la liste des processus AVANT `shutdown()`, car
    Python peut mettre `_processes` à None pendant la fermeture de l'executor.
    """
    processus_obj = getattr(executor, "_processes", None)
    processus = (
        list(processus_obj.values())
        if processus_obj
        else []
    )

    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    for p in processus:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:
            pass

    for p in processus:
        try:
            p.join(timeout=2.0)
        except Exception:
            pass

    for p in processus:
        try:
            if p.is_alive():
                p.kill()
        except Exception:
            pass


# ===========================================================================
# Gestion des runs
# ===========================================================================

def creer_run_id(commande: str, args: argparse.Namespace) -> str:
    if getattr(args, "run_id", None):
        return nettoyer_id(args.run_id)

    tailles = tailles_token(args.tailles)
    if commande == "scan":
        if getattr(args, "ratios", None):
            rdesc = "r-" + "-".join(ratio_token(r) for r in args.ratios)
        else:
            rdesc = (
                f"r{ratio_token(args.ratio_min)}-"
                f"{ratio_token(args.ratio_max)}-"
                f"s{ratio_token(args.pas)}"
            )
        base = f"scan_N{tailles}_n{args.n}_{rdesc}"
    else:
        base = (
            f"production_N{tailles}_n{args.n}_"
            f"r{ratio_token(args.ratio)}"
        )

    return nettoyer_id(f"{base}_{maintenant_id()}")


def chemins_run(run_id: str) -> tuple[Path, Path]:
    return DOSSIER_RUNS / run_id, DOSSIER_FIGURES_RUNS / run_id


def construire_config(commande: str, args: argparse.Namespace) -> dict:
    ratios = None
    if commande == "scan":
        ratios = [float(r) for r in resoudre_ratios_scan(args)]

    return {
        "script_version": SCRIPT_VERSION,
        "commande": commande,
        "tailles": [int(x) for x in args.tailles],
        "n_realisations": int(args.n),
        "chunk_size": int(args.chunk_size),
        "ratios_scan": ratios,
        "ratio_production": (
            float(args.ratio) if commande == "production" else None
        ),
        "gamma": float(args.gamma),
        "p0": float(args.p0),
        "p1": float(args.p1),
        "n_regions": int(args.n_regions),
        "min_tail": int(getattr(args, "min_tail", 100)),
        "P_G_par_generateur": float(P_G_TABLE),
        "N_G": 12,
        "P_C": float(12 * P_G_TABLE),
        "P_L_table": float(P_L_TABLE),
        "graine_scan": GRAINE_SCAN,
        "graine_production": GRAINE_PRODUCTION,
        "scientific_notes": {
            "p1_status":
                "hypothese de reproduction; la valeur n'est pas redonnee "
                "explicitement dans le paragraphe de la Fig. 12",
            "n_regions_status":
                "choix explicite; N_F est defini dans l'article mais sa valeur "
                "numerique n'est pas fournie dans le passage disponible",
            "partition_regions":
                "trois branches principales du niveau 1; racine centrale "
                "assignee deterministement a la region 0"
                if args.n_regions == 3 else
                "toutes les charges dans une seule region",
            "loi_facteur_regional":
                "uniforme sur [2-gamma, gamma]; hypothese explicite, "
                "les bornes seules sont publiees",
            "limites":
                "Table I exactes, aucune recalibration par taille",
            "observable_principale":
                "fraction_delestee_nominale = "
                "P_shed / (ratio_PD_PC * P_C)",
        },
    }


def ouvrir_nouveau_run(
    commande: str,
    args: argparse.Namespace,
) -> tuple[str, Path, Path, dict]:
    run_id = creer_run_id(commande, args)
    run_dir, fig_dir = chemins_run(run_id)

    if run_dir.exists():
        raise FileExistsError(
            f"Le run '{run_id}' existe déjà. "
            "Utiliser --resume pour le reprendre ou choisir un autre --run-id."
        )

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    fig_dir.mkdir(parents=True, exist_ok=False)

    config = construire_config(commande, args)
    metadata = {
        "run_id": run_id,
        "created_at": maintenant_iso(),
        "config": config,
    }
    atomic_write_json(run_dir / "metadata.json", metadata)
    atomic_write_json(
        run_dir / "status.json",
        {
            "run_id": run_id,
            "state": ETAT_EN_COURS,
            "started_at": maintenant_iso(),
            "updated_at": maintenant_iso(),
            "completed_blocks": 0,
            "total_blocks": None,
            "message": "Run créé.",
        },
    )
    return run_id, run_dir, fig_dir, config


def ouvrir_resume(
    commande: str,
    args: argparse.Namespace,
) -> tuple[str, Path, Path, dict]:
    run_id = nettoyer_id(args.resume)
    run_dir, fig_dir = chemins_run(run_id)

    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Run introuvable : {run_id}\nAttendu : {meta_path}"
        )

    metadata = lire_json(meta_path)
    config = metadata["config"]

    if config.get("commande") != commande:
        raise ValueError(
            f"Le run '{run_id}' est un run '{config.get('commande')}', "
            f"pas '{commande}'."
        )

    # Les paramètres scientifiques et numériques sont repris du run original.
    args.tailles = list(config["tailles"])
    args.n = int(config["n_realisations"])
    args.chunk_size = int(config["chunk_size"])
    args.gamma = float(config["gamma"])
    args.p0 = float(config["p0"])
    args.p1 = float(config["p1"])
    args.n_regions = int(config["n_regions"])

    if commande == "scan":
        args.ratios = list(config["ratios_scan"])
        args.min_tail = int(config.get("min_tail", 100))
    else:
        args.ratio = float(config["ratio_production"])

    fig_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    return run_id, run_dir, fig_dir, config


def preparer_run(
    commande: str,
    args: argparse.Namespace,
) -> tuple[str, Path, Path, dict]:
    if getattr(args, "resume", None):
        return ouvrir_resume(commande, args)
    return ouvrir_nouveau_run(commande, args)


def maj_status(
    run_dir: Path,
    *,
    state: str | None = None,
    completed_blocks: int | None = None,
    total_blocks: int | None = None,
    message: str | None = None,
) -> None:
    path = run_dir / "status.json"
    status = lire_json(path) if path.exists() else {}
    if state is not None:
        status["state"] = state
    if completed_blocks is not None:
        status["completed_blocks"] = int(completed_blocks)
    if total_blocks is not None:
        status["total_blocks"] = int(total_blocks)
    if message is not None:
        status["message"] = str(message)
    status["updated_at"] = maintenant_iso()
    if state in (ETAT_TERMINE, ETAT_INTERROMPU, ETAT_ERREUR):
        status["finished_at"] = maintenant_iso()
    atomic_write_json(path, status)


def lister_runs() -> None:
    DOSSIER_RUNS.mkdir(parents=True, exist_ok=True)
    runs = []

    for d in sorted(DOSSIER_RUNS.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        status_path = d / "status.json"
        if not meta_path.exists():
            continue
        try:
            meta = lire_json(meta_path)
            status = lire_json(status_path) if status_path.exists() else {}
            cfg = meta["config"]
            runs.append({
                "run_id": meta.get("run_id", d.name),
                "commande": cfg.get("commande"),
                "tailles": ",".join(str(x) for x in cfg.get("tailles", [])),
                "n": cfg.get("n_realisations"),
                "etat": status.get("state", "?"),
                "progression": (
                    f"{status.get('completed_blocks', '?')}/"
                    f"{status.get('total_blocks', '?')}"
                ),
                "updated": status.get("updated_at", ""),
            })
        except Exception:
            continue

    if not runs:
        print("Aucun run enregistré.")
        return

    print(
        f"{'RUN_ID':<68} {'CMD':<11} {'N':<16} {'n':>8} "
        f"{'ETAT':<12} {'BLOCS':>12}"
    )
    print("-" * 135)
    for r in runs:
        print(
            f"{r['run_id']:<68} {str(r['commande']):<11} "
            f"{r['tailles']:<16} {str(r['n']):>8} "
            f"{r['etat']:<12} {r['progression']:>12}"
        )


# ===========================================================================
# Simulation
# ===========================================================================

def simuler_bloc(
    n_noeuds: int,
    ratio: float,
    n: int,
    graine: int,
    *,
    gamma: float,
    n_regions: int,
    p0: float,
    p1: float,
) -> dict[str, np.ndarray]:
    """
    Simule `n` réalisations i.i.d. avec fluctuations régionales.

    `journee()` reçoit g=0 car la fluctuation est déjà appliquée par
    `demande_regionale()`.
    """
    cfg = configuration_arbre(n_noeuds)
    rng = np.random.default_rng(graine)

    fraction_nominale = np.empty(n)
    fraction_realisee = np.empty(n)
    delestage_total = np.empty(n)
    demande_totale = np.empty(n)
    n_lignes_tombees = np.empty(n, dtype=int)
    n_avaries_p0 = np.empty(n, dtype=int)
    taux_maximal_initial = np.empty(n)
    facteur_demande_totale = np.empty(n)

    pd_nominal = ratio * cfg.p_c

    for i in range(n):
        demande, _facteurs, _groupes = demande_regionale(
            cfg,
            ratio,
            rng,
            gamma=gamma,
            n_regions=n_regions,
        )

        resultat = journee(
            cfg.reseau,
            demande,
            cfg.puissance_max,
            p0=p0,
            p1=p1,
            g=0.0,
            rng=rng,
            limites=cfg.limites,
        )

        pd_reel = float(resultat.demande.sum())
        shed = float(resultat.delestage_total)

        delestage_total[i] = shed
        demande_totale[i] = pd_reel
        fraction_nominale[i] = shed / pd_nominal
        fraction_realisee[i] = shed / pd_reel if pd_reel > 0 else 0.0
        n_lignes_tombees[i] = int(resultat.lignes_tombees.size)
        n_avaries_p0[i] = int(resultat.avaries_accidentelles.size)
        taux_maximal_initial[i] = float(resultat.taux_maximal_initial)
        facteur_demande_totale[i] = pd_reel / pd_nominal

    return {
        "fraction_delestee_nominale": fraction_nominale,
        "fraction_delestee_realisee": fraction_realisee,
        "delestage_total": delestage_total,
        "demande_totale": demande_totale,
        "n_lignes_tombees": n_lignes_tombees,
        "n_avaries_p0": n_avaries_p0,
        "taux_maximal_initial": taux_maximal_initial,
        "facteur_demande_totale": facteur_demande_totale,
    }


def concatener_series(blocs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not blocs:
        raise ValueError("Aucun bloc.")
    cles = list(blocs[0].keys())
    return {
        cle: np.concatenate([b[cle] for b in blocs])
        for cle in cles
    }


# ===========================================================================
# Ajustement statistique
# ===========================================================================

def fit_vide(n_pos: int, n_unique: int) -> dict:
    return {
        "n_pos": n_pos,
        "n_unique": n_unique,
        "alpha": np.nan,
        "xmin": np.nan,
        "D": np.nan,
        "n_tail": 0,
        "fraction_tail": np.nan,
        "tail_decades": np.nan,
        "R_lognormale": np.nan,
        "p_lognormale": np.nan,
        "R_exponentielle": np.nan,
        "p_exponentielle": np.nan,
    }


def ajuster_queue(valeurs: np.ndarray, *, discrete: bool = False) -> dict:
    valeurs = np.asarray(valeurs)
    seuil = 0 if discrete else 1e-12
    positifs = valeurs[np.isfinite(valeurs) & (valeurs > seuil)]

    n_pos = int(positifs.size)
    n_unique = int(np.unique(positifs).size)
    if n_pos < 50 or n_unique < 3:
        return fit_vide(n_pos, n_unique)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)

            fit = powerlaw.Fit(positifs, discrete=discrete, verbose=False)
            alpha = float(fit.power_law.alpha)
            xmin = float(fit.power_law.xmin)
            D = float(fit.power_law.D)

            if not all(np.isfinite(v) for v in (alpha, xmin, D)):
                return fit_vide(n_pos, n_unique)

            queue = positifs[positifs >= xmin]
            if queue.size == 0:
                return fit_vide(n_pos, n_unique)

            n_tail = int(queue.size)
            etendue = (
                float(np.log10(queue.max() / xmin))
                if queue.max() > xmin > 0
                else 0.0
            )

            try:
                R_ln, p_ln = fit.distribution_compare(
                    "power_law",
                    "lognormal",
                )
            except Exception:
                R_ln, p_ln = np.nan, np.nan

            try:
                R_exp, p_exp = fit.distribution_compare(
                    "power_law",
                    "exponential",
                )
            except Exception:
                R_exp, p_exp = np.nan, np.nan

        return {
            "n_pos": n_pos,
            "n_unique": n_unique,
            "alpha": alpha,
            "xmin": xmin,
            "D": D,
            "n_tail": n_tail,
            "fraction_tail": n_tail / n_pos,
            "tail_decades": etendue,
            "R_lognormale": float(R_ln),
            "p_lognormale": float(p_ln),
            "R_exponentielle": float(R_exp),
            "p_exponentielle": float(p_exp),
        }
    except Exception:
        return fit_vide(n_pos, n_unique)


def resume(n_noeuds: int, ratio: float, series: dict[str, np.ndarray]) -> dict:
    frac = series["fraction_delestee_nominale"]
    lignes = series["n_lignes_tombees"]
    facteur = series["facteur_demande_totale"]

    fit_shed = ajuster_queue(frac, discrete=False)
    fit_lignes = ajuster_queue(lignes, discrete=True)

    avec_lignes = lignes > 0
    multi = lignes > 1

    lignes_cond_moy = (
        float(lignes[avec_lignes].mean())
        if np.any(avec_lignes)
        else np.nan
    )
    lignes_cond_mediane = (
        float(np.median(lignes[avec_lignes]))
        if np.any(avec_lignes)
        else np.nan
    )
    lignes_cond_q95 = (
        float(np.quantile(lignes[avec_lignes], 0.95))
        if np.any(avec_lignes)
        else np.nan
    )
    frac_multi_cond = (
        float(np.mean(multi[avec_lignes]))
        if np.any(avec_lignes)
        else np.nan
    )

    row = {
        "n_noeuds": n_noeuds,
        "ratio_PD_PC": ratio,
        "n_realisations": int(frac.size),
        "freq_blackout": float(np.mean(frac > 1e-12)),
        "freq_lignes_tombees": float(np.mean(avec_lignes)),
        "freq_cascade_multiligne": float(np.mean(multi)),
        "fraction_multiligne_conditionnelle": frac_multi_cond,
        "fraction_delestee_moyenne": float(frac.mean()),
        "fraction_delestee_max": float(frac.max()),
        "lignes_tombees_moyenne": float(lignes.mean()),
        "lignes_tombees_moyenne_conditionnelle": lignes_cond_moy,
        "lignes_tombees_mediane_conditionnelle": lignes_cond_mediane,
        "lignes_tombees_q95_conditionnelle": lignes_cond_q95,
        "lignes_tombees_max": int(lignes.max()),
        "Mmax_initial_moyen": float(
            series["taux_maximal_initial"].mean()
        ),
        "facteur_demande_moyen": float(facteur.mean()),
        "facteur_demande_sigma": float(facteur.std(ddof=1)),
    }

    for prefixe, stats in (("shed", fit_shed), ("lignes", fit_lignes)):
        for cle, valeur in stats.items():
            row[f"{prefixe}_{cle}"] = valeur

    return row


# ===========================================================================
# Workers
# ===========================================================================

def worker_bloc(
    n_noeuds: int,
    ratio: float,
    n: int,
    graine: int,
    index_bloc: int,
    gamma: float,
    n_regions: int,
    p0: float,
    p1: float,
) -> tuple[int, float, int, dict[str, np.ndarray]]:
    series = simuler_bloc(
        n_noeuds,
        ratio,
        n,
        graine,
        gamma=gamma,
        n_regions=n_regions,
        p0=p0,
        p1=p1,
    )
    return n_noeuds, ratio, index_bloc, series


# ===========================================================================
# Audit
# ===========================================================================

def trouver_transitions_deterministes(
    n_noeuds: int,
) -> tuple[float, float]:
    cfg = configuration_arbre(n_noeuds)
    A = matrice_de_flux(cfg.reseau)
    ratios = np.linspace(0.2, 1.8, 1601)

    r_delestage = np.nan
    r_transport = np.nan

    for ratio in ratios:
        sol = resoudre(
            cfg.reseau,
            demande_uniforme(cfg.reseau, ratio * cfg.p_c),
            limites=cfg.limites,
            puissance_max=cfg.puissance_max,
            A=A,
        )
        if (
            np.isnan(r_delestage)
            and sol.delestage_total > 1e-7 * cfg.p_c
        ):
            r_delestage = float(ratio)
        if (
            np.isnan(r_transport)
            and sol.taux_maximal >= 0.99
        ):
            r_transport = float(ratio)
        if np.isfinite(r_delestage) and np.isfinite(r_transport):
            break

    return r_delestage, r_transport


def commande_audit(args: argparse.Namespace) -> None:
    print("=== Audit Carreras 2002 ===")
    print(f"P_G (par générateur) : {P_G_TABLE:.4f}")
    print("N_G                  : 12")
    print(f"P_C = 12 P_G         : {12 * P_G_TABLE:.4f}")
    print(f"P_L Table I          : {P_L_TABLE:.4f}")
    print(f"gamma                : {args.gamma:.4f}")
    print(f"p0                   : {args.p0:g}")
    print(f"p1 (hypothèse)       : {args.p1:g}")
    print(f"N_F                  : {args.n_regions} (choix explicite)")

    for n_noeuds in args.tailles:
        cfg = configuration_arbre(n_noeuds)
        groupes = groupes_regions(cfg.reseau, args.n_regions)
        comptes = np.bincount(
            groupes,
            minlength=args.n_regions,
        )
        sigma_u = sigma_relatif_uniforme(
            groupes,
            args.gamma,
        )
        sigma_p = sigma_relatif_papier(
            args.n_regions,
            args.gamma,
        )
        r_gen, r_ligne = trouver_transitions_deterministes(
            n_noeuds,
        )

        print(f"\n--- {n_noeuds} nœuds ---")
        print(f"lignes               : {cfg.reseau.n_lignes}")
        print(f"charges              : {cfg.n_charges}")
        print(f"générateurs          : {cfg.n_generateurs}")
        print(f"P_C                  : {cfg.p_c:.4f}")
        print(f"P_D/P_C si P_L=-74   : {cfg.ratio_table:.6f}")
        print(f"régions (charges)    : {comptes.tolist()}")
        print(f"sigma rel. uniforme  : {sigma_u:.6f}")
        print(
            f"sigma rel. formule papier : {sigma_p:.6f}"
        )
        print(
            f"transition génération (audit) : ~{r_gen:.3f}"
        )
        print(
            f"transition transport  (audit) : ~{r_ligne:.3f}"
        )
        print(
            "limites distinctes   : "
            f"{np.unique(cfg.limites)[::-1].tolist()}"
        )


# ===========================================================================
# Checkpoints
# ===========================================================================

def checkpoint_path(
    run_dir: Path,
    n_noeuds: int,
    ratio: float,
    index_bloc: int,
) -> Path:
    return (
        run_dir
        / "checkpoints"
        / (
            f"N{int(n_noeuds)}_"
            f"r{ratio_token(ratio)}_"
            f"b{int(index_bloc):04d}.npz"
        )
    )


def charger_checkpoint(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            k: np.array(data[k])
            for k in data.files
            if not k.startswith("__")
        }


def checkpoint_valide(
    path: Path,
    *,
    n_attendu: int,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if "__n__" not in data.files:
                return False
            return int(data["__n__"]) == int(n_attendu)
    except Exception:
        return False


def sauver_checkpoint(
    path: Path,
    series: dict[str, np.ndarray],
    *,
    n_noeuds: int,
    ratio: float,
    index_bloc: int,
    n_bloc: int,
    graine: int,
) -> None:
    atomic_save_npz(
        path,
        **series,
        __n__=np.array(n_bloc),
        __n_noeuds__=np.array(n_noeuds),
        __ratio__=np.array(ratio),
        __index_bloc__=np.array(index_bloc),
        __graine__=np.array(graine, dtype=np.uint64),
    )


def blocs_complets_cas(
    run_dir: Path,
    n_noeuds: int,
    ratio: float,
    tailles_blocs: list[int],
) -> bool:
    return all(
        checkpoint_valide(
            checkpoint_path(
                run_dir,
                n_noeuds,
                ratio,
                i,
            ),
            n_attendu=n_bloc,
        )
        for i, n_bloc in enumerate(tailles_blocs)
    )


def charger_series_cas(
    run_dir: Path,
    n_noeuds: int,
    ratio: float,
    tailles_blocs: list[int],
) -> dict[str, np.ndarray]:
    blocs = [
        charger_checkpoint(
            checkpoint_path(
                run_dir,
                n_noeuds,
                ratio,
                i,
            )
        )
        for i in range(len(tailles_blocs))
    ]
    return concatener_series(blocs)


def compter_checkpoints_valides(
    run_dir: Path,
    cas: list[tuple[int, float]],
    tailles_blocs: list[int],
) -> int:
    total = 0
    for n_noeuds, ratio in cas:
        for i, n_bloc in enumerate(tailles_blocs):
            path = checkpoint_path(
                run_dir,
                n_noeuds,
                ratio,
                i,
            )
            if checkpoint_valide(path, n_attendu=n_bloc):
                total += 1
    return total


# ===========================================================================
# Figures
# ===========================================================================

def ccdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    x = np.sort(x)
    y = np.arange(x.size, 0, -1) / x.size
    return x, y


def tracer_scan(
    rows: list[dict],
    tailles: list[int],
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.6, 9.0),
        sharex=True,
    )

    for n_noeuds in tailles:
        sub = [
            r
            for r in rows
            if int(r["n_noeuds"]) == int(n_noeuds)
        ]
        sub.sort(key=lambda r: float(r["ratio_PD_PC"]))
        if not sub:
            continue
        ratios = np.array(
            [r["ratio_PD_PC"] for r in sub],
            dtype=float,
        )
        axes[0].plot(
            ratios,
            [r["freq_blackout"] for r in sub],
            "o-",
            label=f"{n_noeuds}",
        )
        axes[1].plot(
            ratios,
            [r["shed_D"] for r in sub],
            "o-",
            label=f"{n_noeuds}",
        )
        axes[2].plot(
            ratios,
            [r["shed_tail_decades"] for r in sub],
            "o-",
            label=f"{n_noeuds}",
        )

    axes[0].set_ylabel("fréquence blackout")
    axes[1].set_ylabel("KS $D$")
    axes[2].set_ylabel("queue (décades)")
    axes[2].set_xlabel(r"$P_D/P_C$")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(title="nœuds", frameon=False)

    fig.suptitle(
        "Scan contrôlé — paramètres exacts de la Table I"
    )
    fig.tight_layout()
    fig.savefig(
        fig_dir / "scan_controle.png",
        dpi=180,
    )
    plt.close(fig)


def tracer_scan_cascades(
    rows: list[dict],
    tailles: list[int],
    fig_dir: Path,
) -> None:
    """
    Figure dédiée à l'émergence de grandes cascades de lignes.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.6, 6.4),
        sharex=True,
    )

    for n_noeuds in tailles:
        sub = [
            r
            for r in rows
            if int(r["n_noeuds"]) == int(n_noeuds)
        ]
        sub.sort(key=lambda r: float(r["ratio_PD_PC"]))
        if not sub:
            continue
        ratios = np.array(
            [r["ratio_PD_PC"] for r in sub],
            dtype=float,
        )
        axes[0].plot(
            ratios,
            [r["freq_lignes_tombees"] for r in sub],
            "o-",
            label=f"{n_noeuds}",
        )
        axes[1].plot(
            ratios,
            [
                r["lignes_tombees_moyenne_conditionnelle"]
                for r in sub
            ],
            "o-",
            label=f"{n_noeuds}",
        )

    axes[0].set_ylabel("fréq. panne de ligne")
    axes[1].set_ylabel(
        r"$E[N_{\rm outages}\mid N_{\rm outages}>0]$"
    )
    axes[1].set_xlabel(r"$P_D/P_C$")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(title="nœuds", frameon=False)

    fig.suptitle(
        "Émergence des cascades de lignes"
    )
    fig.tight_layout()
    fig.savefig(
        fig_dir / "scan_cascades_lignes.png",
        dpi=180,
    )
    plt.close(fig)


def tracer_ccdf(
    productions: dict[int, dict],
    ratio: float,
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.1))

    for n_noeuds, payload in productions.items():
        series = payload["series"]
        row = payload["row"]
        x, y = ccdf(
            series["fraction_delestee_nominale"]
        )
        ligne, = ax.loglog(
            x,
            y,
            ".",
            markersize=2.5,
            alpha=0.6,
            label=f"{n_noeuds} nœuds",
        )

        alpha = row["shed_alpha"]
        xmin = row["shed_xmin"]
        if (
            np.isfinite(alpha)
            and np.isfinite(xmin)
            and xmin > 0
        ):
            tail = x[x >= xmin]
            if tail.size:
                p_xmin = np.mean(x >= xmin)
                y_fit = (
                    p_xmin
                    * (tail / xmin) ** (-(alpha - 1.0))
                )
                ax.loglog(
                    tail,
                    y_fit,
                    linewidth=1.5,
                    color=ligne.get_color(),
                )

    ax.set_xlabel(
        r"$P_{\rm shed}/\bar P_D$"
    )
    ax.set_ylabel(r"$P(X\geq x)$")
    ax.set_title(
        fr"Effet de taille finie — $P_D/P_C={ratio:.3f}$"
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        fig_dir / "ccdf_taille_finie.png",
        dpi=180,
    )
    plt.close(fig)


# ===========================================================================
# Choix du ratio
# ===========================================================================

def choisir_ratio_commun(
    rows: list[dict],
    tailles: list[int],
    *,
    min_tail: int,
) -> list[dict]:
    """
    Classe les ratios communs.

    Critère primaire :
        minimiser le pire D_KS parmi les tailles,
    sous la contrainte :
        n_tail >= min_tail pour chaque taille.

    Le classement reste un outil de sélection du protocole, pas une preuve
    de loi de puissance.
    """
    ratios = sorted(
        {float(r["ratio_PD_PC"]) for r in rows}
    )
    candidats = []

    for ratio in ratios:
        rr = [
            r
            for r in rows
            if np.isclose(
                float(r["ratio_PD_PC"]),
                ratio,
            )
        ]
        tailles_presentes = {
            int(r["n_noeuds"])
            for r in rr
        }
        if tailles_presentes != set(int(x) for x in tailles):
            continue

        if not all(
            int(r["shed_n_tail"]) >= int(min_tail)
            and np.isfinite(float(r["shed_D"]))
            for r in rr
        ):
            continue

        candidats.append({
            "ratio": ratio,
            "max_D": max(
                float(r["shed_D"]) for r in rr
            ),
            "min_tail_decades": min(
                float(r["shed_tail_decades"])
                for r in rr
            ),
            "freq_min": min(
                float(r["freq_blackout"])
                for r in rr
            ),
            "freq_max": max(
                float(r["freq_blackout"])
                for r in rr
            ),
            "max_ecart_freq": (
                max(float(r["freq_blackout"]) for r in rr)
                - min(float(r["freq_blackout"]) for r in rr)
            ),
        })

    candidats.sort(
        key=lambda d: (
            d["max_D"],
            -d["min_tail_decades"],
            d["max_ecart_freq"],
        )
    )
    return candidats


# ===========================================================================
# Scan professionnel : blocs + checkpoints + reprise
# ===========================================================================

def resoudre_ratios_scan(
    args: argparse.Namespace,
) -> list[float]:
    if getattr(args, "ratios", None):
        ratios = sorted(
            {round(float(r), 12) for r in args.ratios}
        )
        return ratios

    arr = np.arange(
        args.ratio_min,
        args.ratio_max + args.pas / 2,
        args.pas,
    )
    return [
        round(float(r), 12)
        for r in arr
    ]


def graine_scan_bloc(
    n_noeuds: int,
    index_bloc: int,
) -> int:
    """
    Le ratio n'entre PAS dans la graine du scan.

    Cela conserve l'idée de common random numbers entre ratios d'une même taille :
    les blocs homologues commencent avec le même flux aléatoire.
    """
    return graine_stable(
        GRAINE_SCAN,
        "scan",
        int(n_noeuds),
        int(index_bloc),
    )


def reconstruire_rows_scan(
    run_dir: Path,
    tailles: list[int],
    ratios: list[float],
    tailles_blocs: list[int],
) -> list[dict]:
    rows = []
    for n_noeuds in tailles:
        for ratio in ratios:
            if not blocs_complets_cas(
                run_dir,
                n_noeuds,
                ratio,
                tailles_blocs,
            ):
                continue
            series = charger_series_cas(
                run_dir,
                n_noeuds,
                ratio,
                tailles_blocs,
            )
            rows.append(
                resume(n_noeuds, ratio, series)
            )

    rows.sort(
        key=lambda r: (
            int(r["n_noeuds"]),
            float(r["ratio_PD_PC"]),
        )
    )
    return rows


def commande_scan(args: argparse.Namespace) -> None:
    run_id, run_dir, fig_dir, _config = preparer_run(
        "scan",
        args,
    )
    ratios = resoudre_ratios_scan(args)
    tailles_blocs = decouper(
        args.n,
        args.chunk_size,
    )
    cas = [
        (int(n_noeuds), float(ratio))
        for n_noeuds in args.tailles
        for ratio in ratios
    ]

    total_blocks = (
        len(cas)
        * len(tailles_blocs)
    )
    completed_before = compter_checkpoints_valides(
        run_dir,
        cas,
        tailles_blocs,
    )

    maj_status(
        run_dir,
        state=ETAT_EN_COURS,
        completed_blocks=completed_before,
        total_blocks=total_blocks,
        message=(
            "Reprise du scan."
            if args.resume
            else "Scan démarré."
        ),
    )

    journaliser(
        run_dir,
        (
            f"Run : {run_id}\n"
            f"Scan contrôlé : {len(cas)} cas, "
            f"{args.n} réalisations/cas, "
            f"blocs de {args.chunk_size}, "
            f"{args.workers} processus."
        ),
    )
    if completed_before:
        journaliser(
            run_dir,
            (
                f"Reprise : {completed_before}/{total_blocks} "
                "blocs déjà valides."
            ),
        )

    taches = []
    for n_noeuds, ratio in cas:
        for index_bloc, n_bloc in enumerate(
            tailles_blocs
        ):
            path = checkpoint_path(
                run_dir,
                n_noeuds,
                ratio,
                index_bloc,
            )
            if checkpoint_valide(
                path,
                n_attendu=n_bloc,
            ):
                continue

            graine = graine_scan_bloc(
                n_noeuds,
                index_bloc,
            )
            taches.append((
                n_noeuds,
                ratio,
                n_bloc,
                graine,
                index_bloc,
                args.gamma,
                args.n_regions,
                args.p0,
                args.p1,
            ))

    # Les cas déjà complets sont immédiatement réécrits dans scan.csv.
    rows = reconstruire_rows_scan(
        run_dir,
        list(args.tailles),
        ratios,
        tailles_blocs,
    )
    if rows:
        atomic_write_csv(
            run_dir / "scan.csv",
            rows,
        )
        tracer_scan(
            rows,
            list(args.tailles),
            fig_dir,
        )
        tracer_scan_cascades(
            rows,
            list(args.tailles),
            fig_dir,
        )

    if not taches:
        journaliser(
            run_dir,
            "Tous les blocs sont déjà présents : aucune simulation à lancer.",
        )
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialiser_worker,
        )
        futurs = {
            executor.submit(
                worker_bloc,
                *t,
            ): t
            for t in taches
        }

        completed = completed_before

        try:
            for fut in as_completed(futurs):
                t = futurs[fut]
                (
                    n_noeuds,
                    ratio,
                    n_bloc,
                    graine,
                    index_bloc,
                    *_,
                ) = t

                (
                    n_ret,
                    ratio_ret,
                    idx_ret,
                    series,
                ) = fut.result()

                path = checkpoint_path(
                    run_dir,
                    n_ret,
                    ratio_ret,
                    idx_ret,
                )
                sauver_checkpoint(
                    path,
                    series,
                    n_noeuds=n_ret,
                    ratio=ratio_ret,
                    index_bloc=idx_ret,
                    n_bloc=n_bloc,
                    graine=graine,
                )

                completed += 1
                maj_status(
                    run_dir,
                    completed_blocks=completed,
                    total_blocks=total_blocks,
                    message=(
                        f"Bloc {completed}/{total_blocks} sauvegardé."
                    ),
                )

                # Progression détaillée du cas courant.
                faits_cas = sum(
                    checkpoint_valide(
                        checkpoint_path(
                            run_dir,
                            n_ret,
                            ratio_ret,
                            i,
                        ),
                        n_attendu=nb,
                    )
                    for i, nb in enumerate(
                        tailles_blocs
                    )
                )
                n_fait_cas = sum(
                    tailles_blocs[i]
                    for i in range(len(tailles_blocs))
                    if checkpoint_valide(
                        checkpoint_path(
                            run_dir,
                            n_ret,
                            ratio_ret,
                            i,
                        ),
                        n_attendu=tailles_blocs[i],
                    )
                )

                journaliser(
                    run_dir,
                    (
                        f"[{completed:04d}/{total_blocks:04d}] "
                        f"{n_ret:3d} nœuds | "
                        f"{ratio_ret:.3f} | "
                        f"bloc {faits_cas}/{len(tailles_blocs)} | "
                        f"{n_fait_cas}/{args.n}"
                    ),
                )

                # Dès qu'un cas est complet, on le résume et on sauvegarde le CSV.
                if blocs_complets_cas(
                    run_dir,
                    n_ret,
                    ratio_ret,
                    tailles_blocs,
                ):
                    series_cas = charger_series_cas(
                        run_dir,
                        n_ret,
                        ratio_ret,
                        tailles_blocs,
                    )
                    row = resume(
                        n_ret,
                        ratio_ret,
                        series_cas,
                    )
                    journaliser(
                        run_dir,
                        (
                            f"  ↳ CAS COMPLET | "
                            f"blackout={row['freq_blackout']:.1%} | "
                            f"D={row['shed_D']:.4f} | "
                            f"n_tail={int(row['shed_n_tail'])} | "
                            f"Δ={row['shed_tail_decades']:.3f} | "
                            f"E[Nout|>0]="
                            f"{row['lignes_tombees_moyenne_conditionnelle']:.2f}"
                        ),
                    )

                    rows = reconstruire_rows_scan(
                        run_dir,
                        list(args.tailles),
                        ratios,
                        tailles_blocs,
                    )
                    atomic_write_csv(
                        run_dir / "scan.csv",
                        rows,
                    )
                    tracer_scan(
                        rows,
                        list(args.tailles),
                        fig_dir,
                    )
                    tracer_scan_cascades(
                        rows,
                        list(args.tailles),
                        fig_dir,
                    )

            executor.shutdown(wait=True)

        except KeyboardInterrupt:
            journaliser(
                run_dir,
                "\nInterruption demandée — arrêt immédiat des workers...",
            )
            _terminer_executor(executor)

            rows = reconstruire_rows_scan(
                run_dir,
                list(args.tailles),
                ratios,
                tailles_blocs,
            )
            if rows:
                atomic_write_csv(
                    run_dir / "scan.csv",
                    rows,
                )
                tracer_scan(
                    rows,
                    list(args.tailles),
                    fig_dir,
                )
                tracer_scan_cascades(
                    rows,
                    list(args.tailles),
                    fig_dir,
                )

            completed = compter_checkpoints_valides(
                run_dir,
                cas,
                tailles_blocs,
            )
            maj_status(
                run_dir,
                state=ETAT_INTERROMPU,
                completed_blocks=completed,
                total_blocks=total_blocks,
                message=(
                    "Run interrompu proprement par l'utilisateur."
                ),
            )
            journaliser(
                run_dir,
                (
                    f"Checkpoints préservés : "
                    f"{completed}/{total_blocks} blocs.\n"
                    f"Pour reprendre :\n"
                    f"  python scripts/05_reproduction_carreras.py "
                    f"scan --resume {run_id} --workers {args.workers}"
                ),
            )
            return

        except Exception as exc:
            _terminer_executor(executor)
            completed = compter_checkpoints_valides(
                run_dir,
                cas,
                tailles_blocs,
            )
            maj_status(
                run_dir,
                state=ETAT_ERREUR,
                completed_blocks=completed,
                total_blocks=total_blocks,
                message=f"{type(exc).__name__}: {exc}",
            )
            raise

    # Résultats finaux
    rows = reconstruire_rows_scan(
        run_dir,
        list(args.tailles),
        ratios,
        tailles_blocs,
    )
    atomic_write_csv(
        run_dir / "scan.csv",
        rows,
    )
    tracer_scan(
        rows,
        list(args.tailles),
        fig_dir,
    )
    tracer_scan_cascades(
        rows,
        list(args.tailles),
        fig_dir,
    )

    candidats = choisir_ratio_commun(
        rows,
        list(args.tailles),
        min_tail=args.min_tail,
    )

    atomic_write_json(
        run_dir / "candidats.json",
        candidats,
    )

    journaliser(
        run_dir,
        (
            "\nCandidats COMMUNS "
            f"(n_tail >= {args.min_tail}, min du pire KS) :"
        ),
    )
    if not candidats:
        journaliser(
            run_dir,
            "  Aucun candidat satisfaisant le seuil.",
        )
    else:
        for c in candidats[:10]:
            journaliser(
                run_dir,
                (
                    f"  ratio={c['ratio']:.3f} | "
                    f"max(D)={c['max_D']:.4f} | "
                    f"min(Δ)={c['min_tail_decades']:.3f} | "
                    f"fréq blackout="
                    f"{c['freq_min']:.1%}–{c['freq_max']:.1%}"
                ),
            )

    maj_status(
        run_dir,
        state=ETAT_TERMINE,
        completed_blocks=total_blocks,
        total_blocks=total_blocks,
        message="Scan terminé.",
    )

    journaliser(
        run_dir,
        (
            f"\nRun terminé : {run_id}\n"
            f"CSV      : {run_dir / 'scan.csv'}\n"
            f"Figures  : {fig_dir}\n"
            f"Métadonnées : {run_dir / 'metadata.json'}"
        ),
    )


# ===========================================================================
# Production professionnelle
# ===========================================================================

def graine_production_bloc(
    n_noeuds: int,
    ratio: float,
    index_bloc: int,
) -> int:
    return graine_stable(
        GRAINE_PRODUCTION,
        "production",
        int(n_noeuds),
        f"{float(ratio):.12f}",
        int(index_bloc),
    )


def commande_production(
    args: argparse.Namespace,
) -> None:
    run_id, run_dir, fig_dir, _config = preparer_run(
        "production",
        args,
    )

    ratio = float(args.ratio)
    tailles_blocs = decouper(
        args.n,
        args.chunk_size,
    )
    cas = [
        (int(n), ratio)
        for n in args.tailles
    ]
    total_blocks = (
        len(cas)
        * len(tailles_blocs)
    )
    completed_before = compter_checkpoints_valides(
        run_dir,
        cas,
        tailles_blocs,
    )

    maj_status(
        run_dir,
        state=ETAT_EN_COURS,
        completed_blocks=completed_before,
        total_blocks=total_blocks,
        message=(
            "Reprise de la production."
            if args.resume
            else "Production démarrée."
        ),
    )

    journaliser(
        run_dir,
        (
            f"Run : {run_id}\n"
            f"Production contrôlée : ratio commun={ratio:.4f}, "
            f"{args.n} réalisations/taille, "
            f"blocs de {args.chunk_size}, "
            f"{args.workers} processus."
        ),
    )

    taches = []
    for n_noeuds, _ in cas:
        for index_bloc, n_bloc in enumerate(
            tailles_blocs
        ):
            path = checkpoint_path(
                run_dir,
                n_noeuds,
                ratio,
                index_bloc,
            )
            if checkpoint_valide(
                path,
                n_attendu=n_bloc,
            ):
                continue
            graine = graine_production_bloc(
                n_noeuds,
                ratio,
                index_bloc,
            )
            taches.append((
                n_noeuds,
                ratio,
                n_bloc,
                graine,
                index_bloc,
                args.gamma,
                args.n_regions,
                args.p0,
                args.p1,
            ))

    if taches:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialiser_worker,
        )
        futurs = {
            executor.submit(
                worker_bloc,
                *t,
            ): t
            for t in taches
        }
        completed = completed_before

        try:
            for fut in as_completed(futurs):
                t = futurs[fut]
                (
                    n_noeuds,
                    ratio_t,
                    n_bloc,
                    graine,
                    index_bloc,
                    *_,
                ) = t
                n_ret, ratio_ret, idx_ret, series = fut.result()

                sauver_checkpoint(
                    checkpoint_path(
                        run_dir,
                        n_ret,
                        ratio_ret,
                        idx_ret,
                    ),
                    series,
                    n_noeuds=n_ret,
                    ratio=ratio_ret,
                    index_bloc=idx_ret,
                    n_bloc=n_bloc,
                    graine=graine,
                )

                completed += 1
                maj_status(
                    run_dir,
                    completed_blocks=completed,
                    total_blocks=total_blocks,
                    message=(
                        f"Bloc {completed}/{total_blocks} sauvegardé."
                    ),
                )
                journaliser(
                    run_dir,
                    (
                        f"[{completed:04d}/{total_blocks:04d}] "
                        f"{n_ret} nœuds | "
                        f"bloc {idx_ret + 1}/{len(tailles_blocs)}"
                    ),
                )

            executor.shutdown(wait=True)

        except KeyboardInterrupt:
            journaliser(
                run_dir,
                "\nInterruption demandée — arrêt immédiat des workers...",
            )
            _terminer_executor(executor)
            completed = compter_checkpoints_valides(
                run_dir,
                cas,
                tailles_blocs,
            )
            maj_status(
                run_dir,
                state=ETAT_INTERROMPU,
                completed_blocks=completed,
                total_blocks=total_blocks,
                message=(
                    "Production interrompue proprement par l'utilisateur."
                ),
            )
            journaliser(
                run_dir,
                (
                    f"Checkpoints préservés : "
                    f"{completed}/{total_blocks} blocs.\n"
                    f"Pour reprendre :\n"
                    f"  python scripts/05_reproduction_carreras.py "
                    f"production --resume {run_id} "
                    f"--workers {args.workers}"
                ),
            )
            return

        except Exception as exc:
            _terminer_executor(executor)
            completed = compter_checkpoints_valides(
                run_dir,
                cas,
                tailles_blocs,
            )
            maj_status(
                run_dir,
                state=ETAT_ERREUR,
                completed_blocks=completed,
                total_blocks=total_blocks,
                message=f"{type(exc).__name__}: {exc}",
            )
            raise

    productions: dict[int, dict] = {}
    rows = []

    for n_noeuds in args.tailles:
        if not blocs_complets_cas(
            run_dir,
            n_noeuds,
            ratio,
            tailles_blocs,
        ):
            raise RuntimeError(
                f"Production incomplète pour {n_noeuds} nœuds."
            )

        series = charger_series_cas(
            run_dir,
            n_noeuds,
            ratio,
            tailles_blocs,
        )
        row = resume(
            n_noeuds,
            ratio,
            series,
        )
        rows.append(row)
        productions[n_noeuds] = {
            "series": series,
            "row": row,
        }

        atomic_save_npz(
            run_dir / f"production_{n_noeuds}_noeuds.npz",
            **series,
            n_noeuds=np.array(n_noeuds),
            ratio_PD_PC=np.array(ratio),
            P_C=np.array(
                configuration_arbre(n_noeuds).p_c
            ),
            gamma=np.array(args.gamma),
            p0=np.array(args.p0),
            p1=np.array(args.p1),
            n_regions=np.array(args.n_regions),
        )

        journaliser(
            run_dir,
            (
                f"\n=== {n_noeuds} nœuds ===\n"
                f"blackouts          : {row['freq_blackout']:.2%}\n"
                f"panne ligne        : {row['freq_lignes_tombees']:.2%}\n"
                f"E[Nout|>0]         : "
                f"{row['lignes_tombees_moyenne_conditionnelle']:.3f}\n"
                f"Nout max           : {int(row['lignes_tombees_max'])}\n"
                f"sigma demande obs. : {row['facteur_demande_sigma']:.5f}\n"
                f"alpha              : {row['shed_alpha']:.4f}\n"
                f"xmin               : {row['shed_xmin']:.6g}\n"
                f"D (KS)             : {row['shed_D']:.5f}\n"
                f"n_tail             : {int(row['shed_n_tail'])}\n"
                f"Δ queue            : {row['shed_tail_decades']:.3f} décades\n"
                f"vs lognormale      : "
                f"R={row['shed_R_lognormale']:+.2f}, "
                f"p={row['shed_p_lognormale']:.3g}"
            ),
        )

    rows.sort(key=lambda r: int(r["n_noeuds"]))
    atomic_write_csv(
        run_dir / "resume_production.csv",
        rows,
    )
    tracer_ccdf(
        productions,
        ratio,
        fig_dir,
    )

    deltas = [
        (
            int(r["n_noeuds"]),
            float(r["shed_tail_decades"]),
        )
        for r in rows
    ]
    croissant = all(
        b[1] > a[1]
        for a, b in zip(
            deltas,
            deltas[1:],
        )
    )

    journaliser(
        run_dir,
        "\n=== Test descriptif de taille finie ===",
    )
    for n_noeuds, delta in deltas:
        journaliser(
            run_dir,
            f"Δ_{n_noeuds} = {delta:.3f} décades",
        )
    journaliser(
        run_dir,
        (
            "Ordre strictement croissant : "
            + ("OUI" if croissant else "NON")
        ),
    )

    maj_status(
        run_dir,
        state=ETAT_TERMINE,
        completed_blocks=total_blocks,
        total_blocks=total_blocks,
        message="Production terminée.",
    )

    journaliser(
        run_dir,
        (
            f"\nRun terminé : {run_id}\n"
            f"Résumé   : {run_dir / 'resume_production.csv'}\n"
            f"Données  : {run_dir}\n"
            f"Figures  : {fig_dir}"
        ),
    )


# ===========================================================================
# CLI
# ===========================================================================

def ajouter_communs(
    p: argparse.ArgumentParser,
) -> None:
    p.add_argument(
        "--tailles",
        nargs="+",
        type=int,
        default=[46, 94],
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=GAMMA_TABLE,
    )
    p.add_argument(
        "--p0",
        type=float,
        default=P0_TABLE,
    )
    p.add_argument(
        "--p1",
        type=float,
        default=P1_REFERENCE,
    )
    p.add_argument(
        "--n-regions",
        type=int,
        choices=[1, 3],
        default=3,
    )


def ajouter_run_args(
    p: argparse.ArgumentParser,
) -> None:
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Identifiant explicite d'un nouveau run. "
            "S'il est omis, un identifiant horodaté est créé."
        ),
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Reprend un run existant. Les paramètres scientifiques "
            "et numériques sont relus depuis metadata.json."
        ),
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(
        dest="commande",
        required=True,
    )

    audit = sub.add_parser(
        "audit",
        help="Vérifie Table I, P_C, régions et transitions.",
    )
    ajouter_communs(audit)
    audit.set_defaults(func=commande_audit)

    runs = sub.add_parser(
        "runs",
        help="Liste les runs enregistrés et leur état.",
    )
    runs.set_defaults(func=lambda _args: lister_runs())

    scan = sub.add_parser(
        "scan",
        help="Scan contrôlé, bloc par bloc, reprenable.",
    )
    ajouter_communs(scan)
    ajouter_run_args(scan)
    scan.add_argument(
        "--n",
        type=int,
        default=2500,
    )
    scan.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help=(
            "Réalisations par bloc. Petit = progression plus fine "
            "et meilleure reprise; défaut : 250."
        ),
    )
    scan.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Liste explicite de ratios, ex. "
            "--ratios 0.78 0.80 0.81. "
            "Prioritaire sur --ratio-min/--ratio-max/--pas."
        ),
    )
    scan.add_argument(
        "--ratio-min",
        type=float,
        default=0.60,
    )
    scan.add_argument(
        "--ratio-max",
        type=float,
        default=0.90,
    )
    scan.add_argument(
        "--pas",
        type=float,
        default=0.01,
    )
    scan.add_argument(
        "--min-tail",
        type=int,
        default=100,
        help=(
            "Nombre minimal de points dans la queue pour classer "
            "un ratio comme candidat commun."
        ),
    )
    scan.add_argument(
        "--workers",
        type=int,
        default=WORKERS_DEFAUT,
    )
    scan.set_defaults(func=commande_scan)

    prod = sub.add_parser(
        "production",
        help="Production longue, bloc par bloc, reprenable.",
    )
    ajouter_communs(prod)
    ajouter_run_args(prod)
    prod.add_argument(
        "--ratio",
        type=float,
        default=None,
        help=(
            "Ratio commun. Requis pour un nouveau run; relu des "
            "métadonnées avec --resume."
        ),
    )
    prod.add_argument(
        "--n",
        type=int,
        default=60000,
    )
    prod.add_argument(
        "--workers",
        type=int,
        default=WORKERS_DEFAUT,
    )
    prod.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help=(
            "Réalisations par bloc de production; défaut : 1000."
        ),
    )
    prod.set_defaults(func=commande_production)

    return p


def valider_args(
    p: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.commande == "runs":
        return

    for n in args.tailles:
        configuration_arbre(n)

    if not (1 <= args.gamma <= 2):
        p.error("--gamma doit être dans [1,2].")
    if not (
        0 <= args.p0 <= 1
        and 0 <= args.p1 <= 1
    ):
        p.error("--p0 et --p1 doivent être dans [0,1].")

    if args.commande == "audit":
        return

    if args.run_id and args.resume:
        p.error(
            "--run-id et --resume sont incompatibles."
        )
    if args.workers < 1:
        p.error("--workers doit être >= 1.")
    if args.n < 100:
        p.error("--n doit être >= 100.")
    if args.chunk_size < 10:
        p.error("--chunk-size doit être >= 10.")

    if args.commande == "scan":
        if args.min_tail < 1:
            p.error("--min-tail doit être >= 1.")
        if args.ratios:
            if any(r <= 0 for r in args.ratios):
                p.error(
                    "Tous les --ratios doivent être > 0."
                )
        else:
            if (
                args.ratio_min <= 0
                or args.ratio_max <= args.ratio_min
                or args.pas <= 0
            ):
                p.error(
                    "Intervalle de scan invalide."
                )

    if args.commande == "production":
        if not args.resume and (
            args.ratio is None
            or args.ratio <= 0
        ):
            p.error(
                "--ratio > 0 est requis pour un nouveau run de production."
            )


def main() -> None:
    p = parser()
    args = p.parse_args()
    valider_args(p, args)

    try:
        args.func(args)
    except KeyboardInterrupt:
        # Filet de sécurité pour une interruption hors des boucles parallèles.
        print(
            "\nInterruption demandée par l'utilisateur.",
            file=sys.stderr,
        )
        raise SystemExit(130)


if __name__ == "__main__":
    main()
