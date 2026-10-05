"""
SO2 — dynamique lente auto-organisée de Carreras et al. (2004).

Ce script produit les longues séries journalières de blackouts sur lesquelles
porteront le Hurst (SO2) et les entropies (SO3). Chaque CAS est un couple
(taille d'arbre, G) ; chaque cas est une simulation indépendante de plusieurs
dizaines de milliers de jours, découpée en blocs et reprenable.

Modèle (référence : Carreras, Lynch, Dobson & Newman, Chaos 14, 643, 2004)
-------------------------------------------------------------------------
- Réseau et limites de lignes : Table I de Carreras et al. (2002), validés
  par SO1 (tag `so1-v1`).
- Croissance de la demande : 1,8 %/an (λ ≈ 1.00005 par jour).
- Mises à niveau des générateurs : Éq. (3)-(4), seuil (ΔP/P)_c = G · g
  (Éq. 5, `evolution.marge_pour_G`).
- Renforcement des lignes tombées par surcharge un jour de blackout : μ.
- Fluctuations : facteur uniforme sur [1 − g, 1 + g], régional (N_F = 3,
  comme SO1) par défaut ; « noeud » disponible.
- Départage du délestage : « exterieur_dabord » par défaut (règle validée en
  SO1 et indépendante de la plateforme) ; « highs » disponible en contrôle.

Les valeurs que le papier ne donne pas (g, μ, k, p1, demande initiale, N_F)
sont des options explicites, toutes enregistrées dans metadata.json.

Sorties
-------
data/07_dynamique_lente/runs/<RUN_ID>/
    metadata.json      configuration complète (et version du script)
    status.json        état d'avancement du run
    run.log            journal horodaté
    resume.csv         une ligne par cas : fréquences, tailles, transitoire, H
    cas_N<n>_G<G>.npz  séries journalières + état de reprise (checkpoint)
    echecs/            programme linéaire fautif si toutes les stratégies du
                       solveur échouent (audit ; le cas s'arrête, les autres
                       continuent, status.json passe à « partial »)
figures/07_dynamique_lente/runs/<RUN_ID>/
    dynamique_N<n>_G<G>.png/.pdf  analogue de la Fig. 2 de Carreras 2004

Les exposants de Hurst de resume.csv sont des ESTIMATIONS PRÉLIMINAIRES
(R/S simple sur le régime stationnaire, deux plages d'échelles). L'analyse
complète (intervalles nuls, contrôles par mélange, H(G)) fera l'objet d'un
script séparé.

Exemples
--------
Essai rapide (quelques minutes) :
    python scripts/07_dynamique_lente.py run --tailles 46 --G 1.0 \
        --jours 30000 --transitoire 10000 --run-id essai-46

Balayage en G (production) :
    python scripts/07_dynamique_lente.py run --tailles 46 94 \
        --G 0.25 0.5 0.75 1.0 1.5 2.0 --jours 120000 --workers 8 \
        --run-id so2-G-scan

Robustesse du solveur (1.1.0) : un échec numérique de HiGHS est rattrapé par
les stratégies de secours de `dispatch.STRATEGIES_SOLVEUR` ; chaque jour
concerné est compté (série `n_secours_solveur`, colonne
`jours_secours_solveur` de resume.csv). Les checkpoints 1.0.0 restent lisibles.

Reprendre un run interrompu (Ctrl+C, fermeture, panne) :
    python scripts/07_dynamique_lente.py run --resume so2-G-scan --workers 8

Lister les runs :
    python scripts/07_dynamique_lente.py runs
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Une seule unité de calcul par processus : la parallélisation se fait par cas.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cascade_entropy.carreras import configuration_arbre  # noqa: E402
from cascade_entropy.chemins import DOSSIER_DATA, DOSSIER_FIGURES  # noqa: E402
from cascade_entropy.dispatch import (  # noqa: E402
    DEPARTAGE_EXTERIEUR_DABORD,
    DEPARTAGES,
    EchecDispatch,
)
from cascade_entropy.evolution import (  # noqa: E402
    FLUCTUATIONS,
    EtatEvolution,
    Parametres,
    marge_pour_G,
    puissance_max_pour_marge,
    series,
    simuler,
)
from cascade_entropy.indicateurs import (  # noqa: E402
    comptes_par_fenetre,
    hurst,
    troncature_mser,
)

SCRIPT_VERSION = "1.1.0"
TAILLES_SUPPORTEES = (46, 94, 190, 382)

DOSSIER_RUNS = DOSSIER_DATA / "07_dynamique_lente" / "runs"
DOSSIER_FIG_RUNS = DOSSIER_FIGURES / "07_dynamique_lente" / "runs"

# Seuil relatif sous lequel un délestage est du bruit numérique, pas un blackout
# (même convention que evolution.TOLERANCE_DELESTAGE).
SEUIL_BLACKOUT = 1e-9

# Fenêtre de comptage de la Fig. 2 de Carreras 2004 (« per 300 days »).
FENETRE = 300

# Plages d'échelles R/S (jours). Courte : « a few days and a few years » ;
# longue : « 600 < t < 10^5 » (Fig. 4), bornée par le quart de la série.
ECHELLES_COURTES = (10, 365)
ECHELLE_LONGUE_MIN = 600

# Séries sauvegardées dans chaque checkpoint.
CLES_SERIES = (
    "jour", "demande_moyenne", "demande_totale", "delestage_total",
    "fraction_delestee", "n_lignes_tombees", "n_avaries_surcharge",
    "n_avaries_p0", "taux_maximal", "nombre_effectif", "n_lignes_en_service",
    "capacite_totale_generateurs", "marge_moyenne", "n_mises_a_niveau",
    "n_lignes_renforcees", "n_secours_solveur",
)

# Séries ajoutées après la version 1.0.0 : absentes des anciens checkpoints.
# Valeur de remplissage exacte pour les jours déjà simulés : avant 1.1.0, aucune
# stratégie de secours n'existait (un échec du solveur arrêtait le run), donc
# chaque jour sauvegardé a été résolu par l'appel historique.
SERIES_AJOUTEES = {"n_secours_solveur": 0}


# ===========================================================================
# Utilitaires
# ===========================================================================

def maintenant_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def nettoyer_id(texte: str) -> str:
    texte = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(texte).strip()).strip("-_.")
    if not texte:
        raise ValueError("run_id invalide.")
    return texte


def graine_stable(*composants: Any) -> int:
    """Graine stable entre machines (SHA-256, jamais `hash()`)."""
    texte = "|".join(str(x) for x in composants)
    return int.from_bytes(hashlib.sha256(texte.encode()).digest()[:8], "little")


def jeton_G(G: float) -> str:
    return f"{float(G):.4f}".replace(".", "p")


def nom_cas(n_noeuds: int, G: float) -> str:
    return f"N{int(n_noeuds)}_G{jeton_G(G)}"


def atomic_write_text(path: Path, texte: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(texte, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False,
                                       sort_keys=True) + "\n")


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    """Checkpoint atomique : fichier temporaire puis `os.replace()`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def journaliser(run_dir: Path | None, message: str = "") -> None:
    print(message, flush=True)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(f"[{maintenant_iso()}] {message}\n")


def _initialiser_worker() -> None:
    """Les workers ignorent Ctrl+C : seul le processus principal l'intercepte."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


def _terminer_executor(executor: ProcessPoolExecutor) -> None:
    """
    Arrêt immédiat des workers (Ctrl+C), y compris sous Windows.

    `shutdown(cancel_futures=True)` n'arrête pas un cas déjà en cours : on
    termine donc explicitement les processus. Le checkpoint est écrit de façon
    atomique (fichier temporaire puis remplacement) : tuer un worker pendant
    l'écriture laisse au pire un `.tmp`, jamais un checkpoint corrompu.
    `_processes` est une API privée, utilisée uniquement ici.
    """
    processus = list((getattr(executor, "_processes", None) or {}).values())
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    for proc in processus:
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
    for proc in processus:
        try:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.kill()
        except Exception:
            pass


# ===========================================================================
# Configuration
# ===========================================================================

def construire_config(args: argparse.Namespace) -> dict:
    return {
        "script_version": SCRIPT_VERSION,
        "tailles": [int(n) for n in args.tailles],
        "G": [float(G) for G in args.G],
        "jours": int(args.jours),
        "bloc": int(args.bloc),
        "transitoire": int(args.transitoire),
        "g": float(args.g),
        "fluctuation": str(args.fluctuation),
        "n_regions": int(args.n_regions),
        "p0": float(args.p0),
        "p1": float(args.p1),
        "mu": float(args.mu),
        "k": float(args.k),
        "lambda_annuel": float(args.lambda_annuel),
        "ratio_initial": float(args.ratio_initial),
        "departage": str(args.departage),
        "graine": int(args.graine),
        "notes_scientifiques": {
            "G": "Eq. (5) Carreras 2004 : marge_seuil = G * g (gamma_tilde = g)",
            "lambda": "1.8 %/an publie ; facteur quotidien = lambda_annuel**(1/365)",
            "transitoire": "Carreras 2004 : ~20 000 jours ecartes ; MSER rapporte "
                           "a titre de verification",
            "g": "non publie dans Carreras 2004 ; 0.9 = gamma 1.9 de SO1",
            "fluctuation": "Carreras 2004 annexe : par charge ou par region",
            "mu": "publie dans [1.01, 1.1]",
            "k": "publie : « a few percent »",
            "p1": "publie dans [0.1, 1]",
            "ratio_initial": "demande initiale / P_C de la Table I ; sans effet "
                             "sur le regime stationnaire, seulement sur le transitoire",
            "departage": "exterieur_dabord : regle validee en SO1 (Fig. 9-10)",
        },
    }


def parametres_cas(config: dict, n_noeuds: int, G: float):
    """Réseau, paramètres et état initial d'un cas, à partir de la configuration."""
    cfg = configuration_arbre(n_noeuds)
    demande = config["ratio_initial"] * cfg.p_c
    marge = marge_pour_G(G, config["g"])
    parametres = Parametres(
        mode="auto_organise",
        demande_initiale=demande,
        # Marge initiale = seuil : le mécanisme est actif dès les premiers jours.
        puissance_max=puissance_max_pour_marge(cfg.reseau, demande, marge),
        g=config["g"], p0=config["p0"], p1=config["p1"],
        limites=cfg.limites, lambda_=config["lambda_annuel"], k=config["k"],
        marge_seuil=marge, mu=config["mu"], departage=config["departage"],
        fluctuation=config["fluctuation"], n_regions=config["n_regions"],
    )
    return cfg.reseau, parametres


# ===========================================================================
# Simulation d'un cas (exécutée dans un worker)
# ===========================================================================

def _lire_series(d) -> dict[str, np.ndarray]:
    """Lit les séries d'un checkpoint, en complétant celles d'une version antérieure."""
    n = d["jour"].size
    donnees = {}
    for cle in CLES_SERIES:
        if cle in d.files:
            donnees[cle] = d[cle]
        elif cle in SERIES_AJOUTEES:
            donnees[cle] = np.full(n, SERIES_AJOUTEES[cle], dtype=int)
        else:
            raise KeyError(f"Série '{cle}' absente du checkpoint.")
    return donnees


def _charger_checkpoint(chemin: Path):
    if not chemin.exists():
        return None
    with np.load(chemin, allow_pickle=False) as d:
        donnees = _lire_series(d)
        etat = EtatEvolution(
            jour=int(d["etat_jour"]),
            limites=d["etat_limites"].astype(float),
            puissance_max=d["etat_puissance_max"].astype(float),
        )
        rng_state = json.loads(str(d["rng_state"]))
    return donnees, etat, rng_state


def simuler_cas(run_dir: str, config: dict, n_noeuds: int, G: float) -> dict:
    """
    Simule un cas jusqu'à `config["jours"]`, bloc par bloc.

    Après chaque bloc : checkpoint atomique (séries + état du réseau + état du
    générateur aléatoire). Une reprise repart du dernier checkpoint et produit
    exactement la même série qu'un run ininterrompu.
    """
    run_dir = Path(run_dir)
    chemin = run_dir / f"cas_{nom_cas(n_noeuds, G)}.npz"
    reseau, parametres = parametres_cas(config, n_noeuds, G)

    rng = np.random.default_rng(graine_stable("SO2", config["graine"], n_noeuds,
                                              f"{float(G):.6f}"))
    charge = _charger_checkpoint(chemin)
    if charge is None:
        donnees = {cle: np.empty(0) for cle in CLES_SERIES}
        etat = None
        jour = 0
    else:
        donnees, etat, rng_state = charge
        rng.bit_generator.state = rng_state
        jour = etat.jour

    debut = time.time()
    while jour < config["jours"]:
        n = min(config["bloc"], config["jours"] - jour)
        try:
            jours, etat = simuler(reseau, n, parametres, rng, etat=etat,
                                  retourner_etat=True)
        except EchecDispatch as e:
            # Aucune stratégie du solveur n'a réussi : on sauvegarde le
            # programme linéaire fautif pour l'audit, puis on signale l'échec
            # du cas seul (RuntimeError : EchecDispatch ne traverse pas la
            # frontière entre processus). Le dernier checkpoint reste intact.
            dossier = run_dir / "echecs"
            dossier.mkdir(parents=True, exist_ok=True)
            fichier = dossier / f"cas_{nom_cas(n_noeuds, G)}_bloc_{jour}.npz"
            np.savez_compressed(fichier, **e.probleme)
            raise RuntimeError(
                f"{nom_cas(n_noeuds, G)} : échec du solveur dans le bloc "
                f"commençant au jour {jour}. Programme sauvegardé : {fichier}. "
                f"Détail : {e}") from None
        s = series(jours)
        donnees = {cle: np.concatenate([donnees[cle], s[cle]]) for cle in CLES_SERIES}
        jour = etat.jour
        atomic_save_npz(
            chemin, **donnees,
            etat_jour=np.array(etat.jour),
            etat_limites=etat.limites,
            etat_puissance_max=etat.puissance_max,
            rng_state=np.array(json.dumps(rng.bit_generator.state)),
            n_noeuds=np.array(n_noeuds), G=np.array(G),
        )
    return {"n_noeuds": n_noeuds, "G": G, "jours": jour,
            "secondes": time.time() - debut}


# ===========================================================================
# Analyse et figures (processus principal)
# ===========================================================================

def _hurst_plage(serie: np.ndarray, emin: int, emax: int) -> float:
    emax = min(int(emax), serie.size // 4)
    if emax <= emin * 2 or np.std(serie) == 0:
        return float("nan")
    echelles = np.unique(np.logspace(np.log10(emin), np.log10(emax), 15).astype(int))
    try:
        return float(hurst(serie, echelles))
    except ValueError:
        return float("nan")


def analyser_cas(run_dir: Path, fig_dir: Path, config: dict, n_noeuds: int,
                 G: float) -> dict:
    with np.load(run_dir / f"cas_{nom_cas(n_noeuds, G)}.npz") as d:
        s = _lire_series(d)

    blackout = s["delestage_total"] > SEUIL_BLACKOUT * s["demande_moyenne"]
    t0 = int(config["transitoire"])
    # Deux diagnostics : la fréquence des blackouts et celle des renforcements
    # de lignes. La seconde révèle le transitoire pendant lequel les lignes de
    # la Table I ont encore de la marge (aucun renforcement), que la première
    # peut manquer. On retient le plus tardif des deux.
    mser_blackouts = troncature_mser(blackout.astype(int), fenetre=FENETRE)
    mser_renforts = troncature_mser((s["n_lignes_renforcees"] > 0).astype(int),
                                    fenetre=FENETRE)
    mser = max(mser_blackouts, mser_renforts)
    st = slice(t0, None)
    b = blackout[st]
    frac = s["fraction_delestee"][st]
    lignes = s["n_lignes_tombees"][st]
    n_b = int(b.sum())

    ligne = {
        "n_noeuds": n_noeuds,
        "G": G,
        "marge_seuil": marge_pour_G(G, config["g"]),
        "jours": int(s["jour"].size),
        "transitoire_utilise": t0,
        "transitoire_mser": mser,
        "transitoire_mser_blackouts": mser_blackouts,
        "transitoire_mser_renforts": mser_renforts,
        "alerte_transitoire": bool(mser > t0),
        "jours_stationnaires": int(b.size),
        "freq_blackout": float(b.mean()) if b.size else float("nan"),
        "blackouts_par_300j": float(b.mean() * FENETRE) if b.size else float("nan"),
        "puissance_servie_relative": float(np.mean(1 - frac)) if b.size else float("nan"),
        "fraction_delestee_moy_blackout": float(frac[b].mean()) if n_b else float("nan"),
        "lignes_par_blackout": float(lignes[b].mean()) if n_b else float("nan"),
        "freq_plus_15_lignes_sur_freq_blackout": (
            float(np.mean(lignes[b] > 15)) if n_b else float("nan")),
        # Jours (sur toute la série) dont au moins un dispatch a demandé une
        # stratégie de secours du solveur : doit rester rarissime.
        "jours_secours_solveur": int(np.count_nonzero(s["n_secours_solveur"])),
        "H_delestage_court": _hurst_plage(frac, *ECHELLES_COURTES),
        "H_delestage_long": _hurst_plage(frac, ECHELLE_LONGUE_MIN, 10 ** 5),
        "H_lignes_court": _hurst_plage(lignes.astype(float), *ECHELLES_COURTES),
        "H_lignes_long": _hurst_plage(lignes.astype(float), ECHELLE_LONGUE_MIN, 10 ** 5),
    }
    figure_cas(fig_dir, s, blackout, n_noeuds, G, t0, mser)
    return ligne


def figure_cas(fig_dir: Path, s: dict, blackout: np.ndarray, n_noeuds: int,
               G: float, t0: int, mser: int) -> None:
    """Analogue de la Fig. 2 de Carreras 2004 : transitoire puis régime stationnaire."""
    comptes = comptes_par_fenetre(blackout.astype(int), FENETRE)
    t_fen = (np.arange(comptes.size) + 0.5) * FENETRE
    n = comptes.size * FENETRE
    servie = 1 - s["fraction_delestee"][:n].reshape(-1, FENETRE).mean(axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True)
    axes[0].plot(t_fen, comptes, linewidth=0.8)
    axes[0].set_ylabel(f"blackouts / {FENETRE} j")
    # Réponse d'ingénierie : lignes renforcées par fenêtre. L'équilibre
    # dynamique de Carreras (R_D = R_R) se lit comme un niveau stationnaire.
    renforts = s["n_lignes_renforcees"][:n].reshape(-1, FENETRE).sum(axis=1)
    axes[1].plot(t_fen, renforts, linewidth=0.8)
    axes[1].set_ylabel(f"lignes renforcées / {FENETRE} j")
    axes[2].plot(t_fen, servie, linewidth=0.8)
    axes[2].set_ylabel("puissance servie / demande")
    axes[2].set_xlabel("jour")
    for ax in axes:
        ax.axvline(t0, linestyle="--", linewidth=0.9, color="0.3",
                   label="transitoire écarté")
        ax.axvline(mser, linestyle=":", linewidth=0.9, color="0.5",
                   label="MSER (max des deux critères)")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle(f"Dynamique lente — arbre {n_noeuds} nœuds, G = {G:g}")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext, opts in (("png", {"dpi": 170}), ("pdf", {})):
        chemin = fig_dir / f"dynamique_{nom_cas(n_noeuds, G)}.{ext}"
        try:
            fig.savefig(chemin, bbox_inches="tight", **opts)
        except PermissionError:
            print(f"  ATTENTION : {chemin.name} est ouvert ailleurs, non mis à jour.")
    plt.close(fig)


def ecrire_resume(run_dir: Path, lignes: list[dict]) -> None:
    if not lignes:
        return
    tmp = run_dir / "resume.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        w.writeheader()
        w.writerows(lignes)
    os.replace(tmp, run_dir / "resume.csv")


# ===========================================================================
# Commandes
# ===========================================================================

def commande_run(args: argparse.Namespace) -> None:
    if args.resume:
        run_id = nettoyer_id(args.resume)
        run_dir = DOSSIER_RUNS / run_id
        if not (run_dir / "metadata.json").exists():
            raise SystemExit(f"Run introuvable : {run_dir}")
        config = json.loads((run_dir / "metadata.json").read_text("utf-8"))["config"]
        journaliser(run_dir, f"Reprise du run {run_id} (paramètres relus).")
    else:
        run_id = nettoyer_id(args.run_id or datetime.now().strftime("so2-%Y%m%d-%H%M%S"))
        run_dir = DOSSIER_RUNS / run_id
        if run_dir.exists():
            raise SystemExit(f"Le run '{run_id}' existe déjà : utiliser --resume "
                             "ou un autre --run-id.")
        config = construire_config(args)
        atomic_write_json(run_dir / "metadata.json", {
            "run_id": run_id, "created_at": maintenant_iso(), "config": config})
        journaliser(run_dir, f"Run : {run_id}")

    fig_dir = DOSSIER_FIG_RUNS / run_id
    cas = [(int(n), float(G)) for n in config["tailles"] for G in config["G"]]
    journaliser(run_dir, f"{len(cas)} cas × {config['jours']} jours, blocs de "
                         f"{config['bloc']}, {args.workers} processus, "
                         f"départage {config['departage']}, fluctuation "
                         f"{config['fluctuation']}.")
    atomic_write_json(run_dir / "status.json", {
        "state": "running", "updated_at": maintenant_iso(), "cas": len(cas)})

    executor = ProcessPoolExecutor(max_workers=args.workers,
                                   initializer=_initialiser_worker)
    echecs: dict[tuple[int, float], str] = {}
    try:
        futurs = {executor.submit(simuler_cas, str(run_dir), config, n, G): (n, G)
                  for n, G in cas}
        for i, futur in enumerate(as_completed(futurs), 1):
            n, G = futurs[futur]
            try:
                r = futur.result()
            except Exception as e:  # noqa: BLE001 — un cas fautif n'arrête pas les autres
                echecs[(n, G)] = f"{type(e).__name__}: {e}"
                journaliser(run_dir, f"[{i}/{len(cas)}] {nom_cas(n, G)} ÉCHEC : "
                                     f"{echecs[(n, G)]}")
                continue
            journaliser(run_dir, f"[{i}/{len(cas)}] {nom_cas(r['n_noeuds'], r['G'])} "
                                 f"terminé : {r['jours']} jours en "
                                 f"{r['secondes'] / 60:.1f} min")
    except KeyboardInterrupt:
        journaliser(run_dir, "Interruption : les checkpoints sont conservés. "
                             f"Reprendre avec : run --resume {run_id}")
        _terminer_executor(executor)
        atomic_write_json(run_dir / "status.json", {
            "state": "interrupted", "updated_at": maintenant_iso()})
        sys.exit(130)
    except BaseException:
        # Erreur inattendue du processus principal : ne pas laisser les
        # workers tourner en arrière-plan.
        _terminer_executor(executor)
        raise
    executor.shutdown(wait=True)

    reussis = [(n, G) for n, G in cas if (n, G) not in echecs]
    lignes = [analyser_cas(run_dir, fig_dir, config, n, G) for n, G in reussis]
    ecrire_resume(run_dir, lignes)
    for l in lignes:
        alerte = "  ⚠ MSER > transitoire" if l["alerte_transitoire"] else ""
        journaliser(run_dir, (
            f"{nom_cas(l['n_noeuds'], l['G'])} | blackouts/300 j "
            f"{l['blackouts_par_300j']:.1f} | servie {l['puissance_servie_relative']:.4f} "
            f"| lignes/blackout {l['lignes_par_blackout']:.2f} | H délestage "
            f"{l['H_delestage_court']:.2f}/{l['H_delestage_long']:.2f} | H lignes "
            f"{l['H_lignes_court']:.2f}/{l['H_lignes_long']:.2f} | MSER "
            f"{l['transitoire_mser']}{alerte}"))
    if echecs:
        atomic_write_json(run_dir / "status.json", {
            "state": "partial", "updated_at": maintenant_iso(),
            "echecs": {nom_cas(n, G): m for (n, G), m in echecs.items()}})
        journaliser(run_dir, (
            f"\n{len(echecs)} cas en échec (détail dans run.log et status.json) : "
            + ", ".join(nom_cas(n, G) for n, G in echecs)
            + f"\nLeurs checkpoints sont intacts. Après correction : "
              f"run --resume {run_id}"))
    else:
        atomic_write_json(run_dir / "status.json", {
            "state": "completed", "updated_at": maintenant_iso()})
    journaliser(run_dir, f"\nRésumé  : {run_dir / 'resume.csv'}\nFigures : {fig_dir}")


def commande_runs(_args: argparse.Namespace) -> None:
    DOSSIER_RUNS.mkdir(parents=True, exist_ok=True)
    for d in sorted(DOSSIER_RUNS.iterdir(), reverse=True):
        meta = d / "metadata.json"
        if not meta.exists():
            continue
        cfg = json.loads(meta.read_text("utf-8"))["config"]
        statut = d / "status.json"
        etat = json.loads(statut.read_text("utf-8")).get("state") if statut.exists() else "?"
        print(f"{d.name:30s} {etat:12s} tailles={cfg['tailles']} G={cfg['G']} "
              f"jours={cfg['jours']} départage={cfg['departage']}")


# ===========================================================================
# Interface
# ===========================================================================

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="commande", required=True)

    run = sub.add_parser("run", help="Lance ou reprend un run.")
    run.add_argument("--tailles", nargs="+", type=int, default=[46])
    run.add_argument("--G", nargs="+", type=float, default=[1.0])
    run.add_argument("--jours", type=int, default=120_000)
    run.add_argument("--bloc", type=int, default=5_000,
                     help="Jours entre deux checkpoints (défaut 5000).")
    run.add_argument("--transitoire", type=int, default=20_000,
                     help="Jours écartés avant l'analyse (Carreras 2004 : 20 000).")
    run.add_argument("--g", type=float, default=0.9,
                     help="Amplitude des fluctuations, facteur dans [1-g, 1+g].")
    run.add_argument("--fluctuation", choices=list(FLUCTUATIONS), default="regionale")
    run.add_argument("--n-regions", type=int, choices=[1, 3], default=3)
    run.add_argument("--p0", type=float, default=1e-4)
    run.add_argument("--p1", type=float, default=1.0)
    run.add_argument("--mu", type=float, default=1.05)
    run.add_argument("--k", type=float, default=0.02)
    run.add_argument("--lambda-annuel", type=float, default=1.018)
    run.add_argument("--ratio-initial", type=float, default=0.7,
                     help="Demande initiale / P_C de la Table I.")
    run.add_argument("--departage", choices=list(DEPARTAGES),
                     default=DEPARTAGE_EXTERIEUR_DABORD)
    run.add_argument("--graine", type=int, default=20261005)
    run.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    run.add_argument("--run-id", type=str, default=None)
    run.add_argument("--resume", type=str, default=None,
                     help="Reprend un run ; ses paramètres sont relus.")
    run.set_defaults(func=commande_run)

    runs = sub.add_parser("runs", help="Liste les runs.")
    runs.set_defaults(func=commande_runs)
    return p


def valider(p: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.commande != "run":
        return
    if args.run_id and args.resume:
        p.error("--run-id et --resume sont incompatibles.")
    if args.workers < 1:
        p.error("--workers doit être >= 1.")
    if args.resume:
        return
    if any(n not in TAILLES_SUPPORTEES for n in args.tailles):
        p.error(f"--tailles doit être parmi {TAILLES_SUPPORTEES}.")
    if any(G < 0 for G in args.G):
        p.error("--G doit être >= 0.")
    if not 0 < args.g <= 1:
        p.error("--g doit être dans ]0, 1].")
    if not (0 <= args.p0 <= 1 and 0 <= args.p1 <= 1):
        p.error("--p0 et --p1 doivent être dans [0, 1].")
    if args.mu < 1:
        p.error("--mu doit être >= 1.")
    if args.k <= 0 or args.lambda_annuel <= 0 or args.ratio_initial <= 0:
        p.error("--k, --lambda-annuel et --ratio-initial doivent être > 0.")
    if args.bloc < 100:
        p.error("--bloc doit être >= 100.")
    if args.jours < args.transitoire + 4 * FENETRE * 10:
        p.error("--jours doit dépasser --transitoire d'au moins 12 000 jours.")


def main() -> None:
    p = parser()
    args = p.parse_args()
    valider(p, args)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrompu. Les checkpoints déjà écrits sont conservés ; "
              "relancer avec run --resume <RUN_ID>.", flush=True)
        sys.exit(130)


if __name__ == "__main__":
    main()
