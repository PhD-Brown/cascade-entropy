"""
Réplication contrôlée de Carreras et al. (2002) — effet de taille finie.

Cette expérience corrige les deux confusions de la première campagne :
1) P_G=2623.9 est traité comme la capacité MAXIMALE PAR GÉNÉRATEUR, donc
   P_C = 12 * P_G = 31486.8 ;
2) les limites des lignes sont celles de la Table I, SANS recalibration
   indépendante par taille de réseau.

Elle introduit aussi des fluctuations spatiales corrélées : toutes les charges
d'une même région partagent le même facteur aléatoire. N_F=3 est le choix de
réplication principal (trois branches racines), conservé identique pour toutes
les tailles.

Important : le papier définit N_F mais ne donne pas sa valeur numérique dans le
passage disponible. N_F=3 est donc explicitement une hypothèse contrôlée, et
`--n-regions 1` permet un contrôle de sensibilité.

Usage
-----
0) Audit des paramètres :
    python scripts/05_reproduction_carreras.py audit

1) Petit scan commun aux tailles :
    python scripts/05_reproduction_carreras.py scan --workers 16

2) Après inspection du scan, production à UN ratio commun :
    python scripts/05_reproduction_carreras.py production \
        --ratio 0.77 --n 60000 --workers 16 --chunk-size 2500

3) Plus tard, même protocole en changeant seulement la taille :
    python scripts/05_reproduction_carreras.py scan \
        --tailles 46 94 190 382 --workers 16

Dépendances :
    numpy, scipy, matplotlib, powerlaw, package cascade_entropy installé en editable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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


DOSSIER_SORTIE = DOSSIER_DATA / "05_reproduction_carreras"
DOSSIER_FIGURES_SORTIE = DOSSIER_FIGURES / "05_reproduction_carreras"

GRAINE_SCAN = 20261003
GRAINE_PRODUCTION = 20261004

WORKERS_DEFAUT = min(16, max(1, (os.cpu_count() or 2) - 1))


# ---------------------------------------------------------------------------
# Simulation d'une réalisation
# ---------------------------------------------------------------------------

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
    Simule n réalisations i.i.d. avec fluctuations régionales.

    `journee()` reçoit g=0 car la fluctuation a DÉJÀ été appliquée par
    `demande_regionale()`. Cela évite tout double tirage indépendant.
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

        # La demande est déjà fluctuée : g=0 dans journee().
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
    return {
        cle: np.concatenate([b[cle] for b in blocs])
        for cle in blocs[0]
    }


# ---------------------------------------------------------------------------
# Ajustement de queue
# ---------------------------------------------------------------------------

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
                if queue.max() > xmin > 0 else 0.0
            )

            try:
                R_ln, p_ln = fit.distribution_compare("power_law", "lognormal")
            except Exception:
                R_ln, p_ln = np.nan, np.nan

            try:
                R_exp, p_exp = fit.distribution_compare("power_law", "exponential")
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

    row = {
        "n_noeuds": n_noeuds,
        "ratio_PD_PC": ratio,
        "n_realisations": int(frac.size),
        "freq_blackout": float(np.mean(frac > 1e-12)),
        "freq_lignes_tombees": float(np.mean(lignes > 0)),
        "fraction_delestee_moyenne": float(frac.mean()),
        "fraction_delestee_max": float(frac.max()),
        "lignes_tombees_moyenne": float(lignes.mean()),
        "lignes_tombees_max": int(lignes.max()),
        "Mmax_initial_moyen": float(series["taux_maximal_initial"].mean()),
        "facteur_demande_moyen": float(facteur.mean()),
        "facteur_demande_sigma": float(facteur.std(ddof=1)),
    }

    for prefixe, stats in (("shed", fit_shed), ("lignes", fit_lignes)):
        for cle, valeur in stats.items():
            row[f"{prefixe}_{cle}"] = valeur

    return row


# ---------------------------------------------------------------------------
# Workers Windows-safe
# ---------------------------------------------------------------------------

def worker_scan(
    n_noeuds: int,
    ratio: float,
    n: int,
    graine: int,
    gamma: float,
    n_regions: int,
    p0: float,
    p1: float,
) -> dict:
    series = simuler_bloc(
        n_noeuds, ratio, n, graine,
        gamma=gamma, n_regions=n_regions, p0=p0, p1=p1,
    )
    return resume(n_noeuds, ratio, series)


def worker_production(
    n_noeuds: int,
    ratio: float,
    n: int,
    graine: int,
    index_bloc: int,
    gamma: float,
    n_regions: int,
    p0: float,
    p1: float,
):
    series = simuler_bloc(
        n_noeuds, ratio, n, graine,
        gamma=gamma, n_regions=n_regions, p0=p0, p1=p1,
    )
    return n_noeuds, index_bloc, series


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def trouver_transitions_deterministes(n_noeuds: int) -> tuple[float, float]:
    """
    Audit qualitatif sans fluctuations ni pannes.

    Retourne le premier ratio où apparaît du délestage et le premier ratio où
    une ligne atteint 99 % de sa limite. Ce n'est pas utilisé pour recalibrer.
    """
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
        if np.isnan(r_delestage) and sol.delestage_total > 1e-7 * cfg.p_c:
            r_delestage = float(ratio)
        if np.isnan(r_transport) and sol.taux_maximal >= 0.99:
            r_transport = float(ratio)
        if np.isfinite(r_delestage) and np.isfinite(r_transport):
            break

    return r_delestage, r_transport


def commande_audit(args):
    print("=== Audit Carreras 2002 ===")
    print(f"P_G (par générateur) : {P_G_TABLE:.4f}")
    print(f"N_G                  : 12")
    print(f"P_C = 12 P_G         : {12 * P_G_TABLE:.4f}")
    print(f"P_L Table I          : {P_L_TABLE:.4f}")
    print(f"gamma                : {args.gamma:.4f}")
    print(f"p0                   : {args.p0:g}")
    print(f"p1 (hypothèse)       : {args.p1:g}")
    print(f"N_F                  : {args.n_regions} (choix explicite)")

    for n_noeuds in args.tailles:
        cfg = configuration_arbre(n_noeuds)
        groupes = groupes_regions(cfg.reseau, args.n_regions)
        comptes = np.bincount(groupes, minlength=args.n_regions)
        sigma_u = sigma_relatif_uniforme(groupes, args.gamma)
        sigma_p = sigma_relatif_papier(args.n_regions, args.gamma)
        r_gen, r_ligne = trouver_transitions_deterministes(n_noeuds)

        print(f"\n--- {n_noeuds} nœuds ---")
        print(f"lignes               : {cfg.reseau.n_lignes}")
        print(f"charges              : {cfg.n_charges}")
        print(f"générateurs          : {cfg.n_generateurs}")
        print(f"P_C                  : {cfg.p_c:.4f}")
        print(f"P_D/P_C si P_L=-74   : {cfg.ratio_table:.6f}")
        print(f"régions (charges)    : {comptes.tolist()}")
        print(f"sigma rel. uniforme  : {sigma_u:.6f}")
        print(f"sigma rel. formule papier : {sigma_p:.6f}")
        print(f"transition génération (audit) : ~{r_gen:.3f}")
        print(f"transition transport  (audit) : ~{r_ligne:.3f}")
        print(f"limites distinctes   : {np.unique(cfg.limites)[::-1].tolist()}")


# ---------------------------------------------------------------------------
# I/O et figures
# ---------------------------------------------------------------------------

def ecrire_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def ecrire_metadata(path: Path, *, args, commande: str, ratio=None) -> None:
    meta = {
        "commande": commande,
        "tailles": list(args.tailles),
        "ratio_PD_PC_production": ratio,
        "n_realisations": getattr(args, "n", None),
        "workers": getattr(args, "workers", None),
        "chunk_size": getattr(args, "chunk_size", None),
        "P_G_par_generateur": P_G_TABLE,
        "N_G": 12,
        "P_C": 12 * P_G_TABLE,
        "P_L_table": P_L_TABLE,
        "gamma": args.gamma,
        "p0": args.p0,
        "p1": args.p1,
        "p1_status": (
            "hypothese de reproduction; la valeur n'est pas redonnee "
            "explicitement dans le paragraphe de la Fig. 12"
        ),
        "n_regions": args.n_regions,
        "n_regions_status": (
            "choix explicite; N_F est defini dans l'article mais sa valeur "
            "numerique n'est pas fournie dans le passage disponible"
        ),
        "partition_regions": (
            "trois branches principales du niveau 1; racine centrale assignee "
            "deterministement a la region 0"
            if args.n_regions == 3 else
            "toutes les charges dans une seule region"
        ),
        "loi_facteur_regional": (
            "uniforme sur [2-gamma, gamma]; hypothese explicite, "
            "les bornes seules sont publiees"
        ),
        "limites": "Table I exactes, aucune recalibration par taille",
        "observable_principale": (
            "fraction_delestee_nominale = P_shed / (ratio_PD_PC * P_C)"
        ),
        "graine_scan": GRAINE_SCAN,
        "graine_production": GRAINE_PRODUCTION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ccdf(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    x = np.sort(x)
    y = np.arange(x.size, 0, -1) / x.size
    return x, y


def tracer_scan(rows: list[dict], tailles: list[int]) -> None:
    DOSSIER_FIGURES_SORTIE.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 9.0), sharex=True)

    for n_noeuds in tailles:
        sub = [r for r in rows if int(r["n_noeuds"]) == n_noeuds]
        ratios = np.array([r["ratio_PD_PC"] for r in sub])
        axes[0].plot(ratios, [r["freq_blackout"] for r in sub], "o-", label=f"{n_noeuds}")
        axes[1].plot(ratios, [r["shed_D"] for r in sub], "o-", label=f"{n_noeuds}")
        axes[2].plot(ratios, [r["shed_tail_decades"] for r in sub], "o-", label=f"{n_noeuds}")

    axes[0].set_ylabel("fréquence blackout")
    axes[1].set_ylabel("KS $D$")
    axes[2].set_ylabel("queue (décades)")
    axes[2].set_xlabel(r"$P_D/P_C$")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(title="nœuds", frameon=False)
    fig.suptitle("Scan contrôlé — paramètres exacts de la Table I")
    fig.tight_layout()
    fig.savefig(DOSSIER_FIGURES_SORTIE / "05_scan_controle.png", dpi=180)
    plt.close(fig)


def tracer_ccdf(productions: dict[int, dict], ratio: float) -> None:
    DOSSIER_FIGURES_SORTIE.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.1))

    for n_noeuds, payload in productions.items():
        series = payload["series"]
        row = payload["row"]
        x, y = ccdf(series["fraction_delestee_nominale"])
        ligne, = ax.loglog(x, y, ".", markersize=2.5, alpha=0.6,
                           label=f"{n_noeuds} nœuds")

        alpha = row["shed_alpha"]
        xmin = row["shed_xmin"]
        if np.isfinite(alpha) and np.isfinite(xmin) and xmin > 0:
            tail = x[x >= xmin]
            if tail.size:
                p_xmin = np.mean(x >= xmin)
                y_fit = p_xmin * (tail / xmin) ** (-(alpha - 1.0))
                ax.loglog(tail, y_fit, linewidth=1.5, color=ligne.get_color())

    ax.set_xlabel(r"$P_{\rm shed}/P_D$")
    ax.set_ylabel(r"$P(X\geq x)$")
    ax.set_title(fr"Effet de taille finie — $P_D/P_C={ratio:.3f}$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(DOSSIER_FIGURES_SORTIE / "05_ccdf_taille_finie.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def choisir_ratio_commun(rows: list[dict], tailles: list[int]) -> list[dict]:
    """
    Classe les ratios COMMUNS.

    Score primaire = pire distance KS parmi les tailles, sous condition que
    chaque taille fournisse au moins 100 points dans sa queue. On n'annonce pas
    ce classement comme une preuve de criticalité : c'est un outil de choix
    pré-enregistré pour la production.
    """
    ratios = sorted({float(r["ratio_PD_PC"]) for r in rows})
    candidats = []

    for ratio in ratios:
        rr = [r for r in rows if np.isclose(r["ratio_PD_PC"], ratio)]
        if len(rr) != len(tailles):
            continue
        if not all(int(r["shed_n_tail"]) >= 100 and np.isfinite(r["shed_D"]) for r in rr):
            continue

        candidats.append({
            "ratio": ratio,
            "max_D": max(float(r["shed_D"]) for r in rr),
            "min_tail_decades": min(float(r["shed_tail_decades"]) for r in rr),
            "freq_min": min(float(r["freq_blackout"]) for r in rr),
            "freq_max": max(float(r["freq_blackout"]) for r in rr),
        })

    candidats.sort(key=lambda d: (d["max_D"], -d["min_tail_decades"]))
    return candidats


def commande_scan(args):
    ratios = np.arange(args.ratio_min, args.ratio_max + args.pas / 2, args.pas)
    taches = []

    for n_noeuds in args.tailles:
        # Même graine pour tous les ratios d'une taille (common random numbers).
        graine = GRAINE_SCAN + n_noeuds
        for ratio in ratios:
            taches.append((n_noeuds, float(ratio), args.n, graine,
                           args.gamma, args.n_regions, args.p0, args.p1))

    print(
        f"Scan contrôlé : {len(taches)} cas, {args.n} réalisations/cas, "
        f"{args.workers} processus."
    )
    rows = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futurs = {
            ex.submit(worker_scan, *t): (t[0], t[1])
            for t in taches
        }
        for i, fut in enumerate(as_completed(futurs), 1):
            n_noeuds, ratio = futurs[fut]
            row = fut.result()
            rows.append(row)
            print(
                f"[{i:02d}/{len(taches):02d}] {n_noeuds:3d} nœuds | "
                f"{ratio:.3f} | blackout={row['freq_blackout']:6.1%} | "
                f"D={row['shed_D']:.4f} | n_tail={int(row['shed_n_tail']):5d} | "
                f"Δ={row['shed_tail_decades']:.3f}"
            )

    rows.sort(key=lambda r: (r["n_noeuds"], r["ratio_PD_PC"]))
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    ecrire_csv(DOSSIER_SORTIE / "scan.csv", rows)
    ecrire_metadata(DOSSIER_SORTIE / "metadata_scan.json", args=args, commande="scan")
    tracer_scan(rows, list(args.tailles))

    candidats = choisir_ratio_commun(rows, list(args.tailles))
    print("\nCandidats COMMUNS (critère pré-enregistré : min du pire KS) :")
    if not candidats:
        print("  Aucun ratio avec >=100 points dans la queue pour toutes les tailles.")
    else:
        for c in candidats[:5]:
            print(
                f"  ratio={c['ratio']:.3f} | max(D)={c['max_D']:.4f} | "
                f"min(Δ)={c['min_tail_decades']:.3f} | "
                f"fréq blackout={c['freq_min']:.1%}–{c['freq_max']:.1%}"
            )

    print(f"\nCSV    : {DOSSIER_SORTIE / 'scan.csv'}")
    print(f"Figure : {DOSSIER_FIGURES_SORTIE / '05_scan_controle.png'}")


# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------

def decouper(n: int, chunk: int):
    blocs = []
    reste = n
    while reste:
        courant = min(chunk, reste)
        blocs.append(courant)
        reste -= courant
    return blocs


def commande_production(args):
    tailles_blocs = decouper(args.n, args.chunk_size)
    n_taches = len(args.tailles) * len(tailles_blocs)

    seed = np.random.SeedSequence(GRAINE_PRODUCTION)
    enfants = seed.spawn(n_taches)

    taches = []
    k = 0
    for n_noeuds in args.tailles:
        for index, n_bloc in enumerate(tailles_blocs):
            graine = int(enfants[k].generate_state(1, dtype=np.uint64)[0])
            k += 1
            taches.append((
                n_noeuds, args.ratio, n_bloc, graine, index,
                args.gamma, args.n_regions, args.p0, args.p1,
            ))

    print(
        f"Production contrôlée : ratio COMMUN={args.ratio:.4f}, "
        f"{args.n} réalisations/taille, {args.workers} processus."
    )

    blocs_par_taille = {n: {} for n in args.tailles}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futurs = {
            ex.submit(worker_production, *t): (t[0], t[4], t[2])
            for t in taches
        }
        for i, fut in enumerate(as_completed(futurs), 1):
            n_noeuds, index, n_bloc = futurs[fut]
            n_ret, idx_ret, series = fut.result()
            blocs_par_taille[n_ret][idx_ret] = series
            print(
                f"[{i:02d}/{n_taches:02d}] {n_noeuds} nœuds | "
                f"bloc {index+1}/{len(tailles_blocs)} ({n_bloc})"
            )

    productions = {}
    rows = []
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

    for n_noeuds in args.tailles:
        series = concatener_series(
            [blocs_par_taille[n_noeuds][i] for i in range(len(tailles_blocs))]
        )
        row = resume(n_noeuds, args.ratio, series)
        rows.append(row)
        productions[n_noeuds] = {"series": series, "row": row}

        np.savez_compressed(
            DOSSIER_SORTIE / f"production_{n_noeuds}_noeuds.npz",
            **series,
            n_noeuds=n_noeuds,
            ratio_PD_PC=args.ratio,
            P_C=configuration_arbre(n_noeuds).p_c,
            gamma=args.gamma,
            p0=args.p0,
            p1=args.p1,
            n_regions=args.n_regions,
        )

        print(f"\n=== {n_noeuds} nœuds ===")
        print(f"blackouts          : {row['freq_blackout']:.2%}")
        print(f"sigma demande obs. : {row['facteur_demande_sigma']:.5f}")
        print(f"alpha              : {row['shed_alpha']:.4f}")
        print(f"xmin               : {row['shed_xmin']:.6g}")
        print(f"D (KS)             : {row['shed_D']:.5f}")
        print(f"n_tail             : {int(row['shed_n_tail'])}")
        print(f"Δ queue            : {row['shed_tail_decades']:.3f} décades")
        print(
            f"vs lognormale      : R={row['shed_R_lognormale']:+.2f}, "
            f"p={row['shed_p_lognormale']:.3g}"
        )

    ecrire_csv(DOSSIER_SORTIE / "resume_production.csv", rows)
    ecrire_metadata(
        DOSSIER_SORTIE / "metadata_production.json",
        args=args, commande="production", ratio=args.ratio,
    )
    tracer_ccdf(productions, args.ratio)

    deltas = [(int(r["n_noeuds"]), float(r["shed_tail_decades"])) for r in rows]
    croissant = all(b[1] > a[1] for a, b in zip(deltas, deltas[1:]))

    print("\n=== Test descriptif de taille finie ===")
    for n_noeuds, delta in deltas:
        print(f"Δ_{n_noeuds} = {delta:.3f} décades")
    print(
        "Ordre strictement croissant : "
        + ("OUI" if croissant else "NON")
    )
    if len(deltas) >= 2:
        print(
            "Critère 46→94 : "
            + ("Δ_94 > Δ_46" if deltas[1][1] > deltas[0][1]
               else "Δ_94 <= Δ_46")
        )

    print(f"\nRésumé : {DOSSIER_SORTIE / 'resume_production.csv'}")
    print(f"Figure : {DOSSIER_FIGURES_SORTIE / '05_ccdf_taille_finie.png'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def ajouter_communs(p):
    p.add_argument("--tailles", nargs="+", type=int, default=[46, 94])
    p.add_argument("--gamma", type=float, default=GAMMA_TABLE)
    p.add_argument("--p0", type=float, default=P0_TABLE)
    p.add_argument("--p1", type=float, default=P1_REFERENCE)
    p.add_argument("--n-regions", type=int, choices=[1, 3], default=3)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="commande", required=True)

    audit = sub.add_parser("audit", help="Vérifie Table I, P_C, régions et transitions.")
    ajouter_communs(audit)
    audit.set_defaults(func=commande_audit)

    scan = sub.add_parser("scan", help="Petit scan commun avec paramètres contrôlés.")
    ajouter_communs(scan)
    scan.add_argument("--n", type=int, default=2500)
    scan.add_argument("--ratio-min", type=float, default=0.60)
    scan.add_argument("--ratio-max", type=float, default=0.90)
    scan.add_argument("--pas", type=float, default=0.01)
    scan.add_argument("--workers", type=int, default=WORKERS_DEFAUT)
    scan.set_defaults(func=commande_scan)

    prod = sub.add_parser("production", help="60 000 cas à un ratio COMMUN.")
    ajouter_communs(prod)
    prod.add_argument("--ratio", type=float, required=True)
    prod.add_argument("--n", type=int, default=60000)
    prod.add_argument("--workers", type=int, default=WORKERS_DEFAUT)
    prod.add_argument("--chunk-size", type=int, default=2500)
    prod.set_defaults(func=commande_production)

    return p


def main():
    p = parser()
    args = p.parse_args()

    for n in args.tailles:
        configuration_arbre(n)  # validation précoce

    if not (1 <= args.gamma <= 2):
        p.error("--gamma doit être dans [1,2].")
    if not (0 <= args.p0 <= 1 and 0 <= args.p1 <= 1):
        p.error("--p0 et --p1 doivent être dans [0,1].")

    if hasattr(args, "workers") and args.workers < 1:
        p.error("--workers doit être >=1.")
    if hasattr(args, "n") and args.n < 100:
        p.error("--n doit être >=100.")
    if args.commande == "scan":
        if args.ratio_min <= 0 or args.ratio_max <= args.ratio_min or args.pas <= 0:
            p.error("Intervalle de scan invalide.")
    if args.commande == "production":
        if args.ratio <= 0 or args.chunk_size < 100:
            p.error("Paramètres de production invalides.")

    args.func(args)


if __name__ == "__main__":
    main()
