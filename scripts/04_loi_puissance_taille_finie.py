"""
Étude de taille finie de la queue des tailles de blackout.

Objectif
--------
Tester, sur deux arbres (46 et 94 nœuds), si une région compatible avec une
loi de puissance apparaît près du régime critique, et vérifier si cette région
s'élargit avec la taille du réseau comme dans Carreras et al. (2002, 2004).

Le script sépare deux étapes :

1) SCAN exploratoire
   - balayage fin de P_D / P_C ;
   - échantillon modéré par ratio ;
   - estimation de alpha, xmin, distance KS, nombre de points dans la queue ;
   - comparaison loi de puissance vs lognormale et exponentielle ;
   - aucune conclusion confirmatoire à cette étape.

2) PRODUCTION confirmatoire
   - 60 000 réalisations i.i.d. pour chaque taille de réseau ;
   - graine différente de celle du scan ;
   - ratio choisi AVANT le lancement à partir du scan ;
   - comparaison 46 vs 94 nœuds sur la fraction de charge délestée
     et le nombre de lignes tombées.

Robustesse
----------
Certaines distributions discrètes (notamment le nombre de lignes tombées à
faible charge) peuvent ne contenir qu'une ou deux valeurs positives distinctes.
Dans ce cas, `powerlaw` ne peut pas estimer correctement xmin et alpha.
Le script détecte maintenant ces cas et retourne NaN au lieu d'interrompre
l'expérience.

Parallélisation
---------------
Le scan est parallélisé par couple (taille du réseau, P_D/P_C).
La production est parallélisée par blocs de réalisations. Le code est compatible
avec Windows grâce au garde `if __name__ == "__main__"`.

Important
---------
- Ici, une "réalisation" correspond au mode indépendant de Carreras et al. (2002),
  pas à une journée d'une dynamique auto-organisée.
- Le paramètre g du code suit la convention interne du projet :
      g = 0.9 ici <-> g = 1.9 dans Carreras et al. (2002).
- p0 = 1e-4, valeur publiée dans l'expérience de Carreras et al. (2002).
- p1 = 1.0 par défaut pour reproduire le cas où toute ligne surchargée tombe.
- Le délestage analysé est NORMALISÉ par la demande totale de la réalisation :
      S = P_shed / P_D
  conformément à la grandeur tracée dans les articles.
- La routine actuelle `cascade.journee()` applique une fluctuation indépendante
  à chaque charge. Carreras regroupe certaines charges par régions dans une partie
  de ses expériences. Le présent script teste donc rigoureusement LE MODÈLE ACTUEL
  avec les paramètres publiés, mais ne prétend pas reproduire exactement cette
  corrélation spatiale des fluctuations.

Usage depuis la racine du dépôt
-------------------------------

Étape 1 — scan :
    python scripts/04_loi_puissance_taille_finie.py scan

Scan rapide :
    python scripts/04_loi_puissance_taille_finie.py scan --n 1000

Contrôler le nombre de processus :
    python scripts/04_loi_puissance_taille_finie.py scan --workers 6

Étape 2 — production, après inspection du scan :
    python scripts/04_loi_puissance_taille_finie.py production \
        --ratio-46 0.80 --ratio-94 0.85

Production avec 8 processus et blocs de 5000 :
    python scripts/04_loi_puissance_taille_finie.py production \
        --ratio-46 0.80 --ratio-94 0.85 --workers 8 --chunk-size 5000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import powerlaw
from scipy.optimize import OptimizeWarning

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cascade_entropy import figures  # noqa: E402
from cascade_entropy.cascade import journee  # noqa: E402
from cascade_entropy.chemins import DOSSIER_DATA, DOSSIER_FIGURES  # noqa: E402
from cascade_entropy.dispatch import demande_uniforme, resoudre  # noqa: E402
from cascade_entropy.reseau import arbre, limites_par_niveau, matrice_de_flux  # noqa: E402


# ---------------------------------------------------------------------------
# Paramètres de référence
# ---------------------------------------------------------------------------

P_C = 2623.9
CIBLE_SECONDE = 1.45

# Convention interne du projet : [1-g, 1+g].
# Correspond à g = 1.9 dans la convention de Carreras et al. (2002).
G = 0.9

# Valeur utilisée dans Carreras et al. (2002), section sur les fluctuations.
P0 = 1e-4

# Cas limite étudié dans l'article : toute ligne surchargée déclenche.
P1 = 1.0

TAILLES = {
    46: 4,   # arbre(4) -> 46 nœuds
    94: 5,   # arbre(5) -> 94 nœuds
}

GRAINE_SCAN = 20261002
GRAINE_PRODUCTION = 20261003

DOSSIER_SORTIE = DOSSIER_DATA / "04_loi_puissance_taille_finie"
DOSSIER_FIGURES_SORTIE = DOSSIER_FIGURES / "04_loi_puissance_taille_finie"

# Laisse généralement un peu de ressources à Windows/Jupyter tout en exploitant
# plusieurs cœurs. L'utilisateur peut remplacer cette valeur avec --workers.
WORKERS_DEFAUT = min(8, max(1, (os.cpu_count() or 2) - 1))


@dataclass
class Systeme:
    n_noeuds: int
    reseau: object
    limites: np.ndarray
    puissance_max: np.ndarray
    facteur_calibration: float


# ---------------------------------------------------------------------------
# Préparation du réseau
# ---------------------------------------------------------------------------

def preparer_systeme(n_noeuds: int) -> Systeme:
    """
    Construit l'arbre et conserve la calibration déjà utilisée dans les carnets
    02, 03 et 05 : l'échelle des limites est ajustée pour que la seconde
    transition déterministe soit à P_D/P_C ~= 1.45.
    """
    if n_noeuds not in TAILLES:
        raise ValueError(f"Taille non supportée : {n_noeuds}. Choisir {tuple(TAILLES)}.")

    reseau = arbre(TAILLES[n_noeuds])
    if reseau.n_noeuds != n_noeuds:
        raise RuntimeError(
            f"arbre({TAILLES[n_noeuds]}) a produit {reseau.n_noeuds} nœuds, "
            f"pas {n_noeuds}."
        )

    puissance_max = np.full(
        reseau.generateurs.size,
        P_C / reseau.generateurs.size,
        dtype=float,
    )

    limites_base = limites_par_niveau(reseau)
    A = matrice_de_flux(reseau)

    solution_cible = resoudre(
        reseau,
        demande_uniforme(reseau, CIBLE_SECONDE * P_C),
        limites=limites_base,
        puissance_max=puissance_max,
        A=A,
    )

    facteur = float(solution_cible.taux_maximal)
    limites = limites_base * facteur

    return Systeme(
        n_noeuds=n_noeuds,
        reseau=reseau,
        limites=limites,
        puissance_max=puissance_max,
        facteur_calibration=facteur,
    )


# ---------------------------------------------------------------------------
# Simulation i.i.d.
# ---------------------------------------------------------------------------

def simuler_realisation_iid(
    systeme: Systeme,
    ratio: float,
    n: int,
    graine: int,
) -> dict[str, np.ndarray]:
    """Effectue n réalisations indépendantes à charge moyenne fixée."""
    rng = np.random.default_rng(graine)
    demande_moyenne = demande_uniforme(systeme.reseau, ratio * P_C)

    fraction_delestee = np.empty(n, dtype=float)
    delestage_total = np.empty(n, dtype=float)
    demande_totale = np.empty(n, dtype=float)
    n_lignes_tombees = np.empty(n, dtype=int)
    taux_maximal_initial = np.empty(n, dtype=float)

    for i in range(n):
        resultat = journee(
            systeme.reseau,
            demande_moyenne,
            systeme.puissance_max,
            p0=P0,
            p1=P1,
            g=G,
            rng=rng,
            limites=systeme.limites,
        )

        p_demande = float(resultat.demande.sum())
        p_shed = float(resultat.delestage_total)

        demande_totale[i] = p_demande
        delestage_total[i] = p_shed
        fraction_delestee[i] = p_shed / p_demande if p_demande > 0 else 0.0
        n_lignes_tombees[i] = int(resultat.lignes_tombees.size)
        taux_maximal_initial[i] = float(resultat.taux_maximal_initial)

    return {
        "fraction_delestee": fraction_delestee,
        "delestage_total": delestage_total,
        "demande_totale": demande_totale,
        "n_lignes_tombees": n_lignes_tombees,
        "taux_maximal_initial": taux_maximal_initial,
    }


def _concatener_series(blocs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatène des blocs de simulation dans leur ordre déterministe."""
    if not blocs:
        raise ValueError("Aucun bloc de simulation.")
    cles = blocs[0].keys()
    return {
        cle: np.concatenate([bloc[cle] for bloc in blocs])
        for cle in cles
    }


# ---------------------------------------------------------------------------
# Ajustement de queue
# ---------------------------------------------------------------------------

def _resultat_fit_vide(n_pos: int, n_unique: int) -> dict[str, float | int]:
    return {
        "n_pos": int(n_pos),
        "n_unique": int(n_unique),
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


def ajuster_queue(
    valeurs: np.ndarray,
    *,
    discret: bool,
) -> dict[str, float | int]:
    """
    Ajuste une loi de puissance avec `powerlaw`, sans jamais faire planter le scan.

    Cas volontairement déclarés "non ajustables" :
    - moins de 50 observations positives ;
    - moins de 3 valeurs positives distinctes ;
    - échec numérique de `powerlaw` ;
    - xmin/alpha/D non finis.

    Les comparaisons de distributions sont relatives : elles ne constituent pas,
    à elles seules, un test absolu de goodness-of-fit.
    """
    valeurs = np.asarray(valeurs)
    seuil_zero = 0 if discret else 1e-12
    positifs = valeurs[np.isfinite(valeurs) & (valeurs > seuil_zero)]

    n_pos = int(positifs.size)
    n_unique = int(np.unique(positifs).size)

    # C'est précisément le cas qui faisait planter le scan sur le 94 nœuds :
    # assez de valeurs positives, mais pratiquement toutes identiques.
    if n_pos < 50 or n_unique < 3:
        return _resultat_fit_vide(n_pos, n_unique)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)

            fit = powerlaw.Fit(
                positifs,
                discrete=discret,
                verbose=False,
            )

            alpha = float(fit.power_law.alpha)
            xmin = float(fit.power_law.xmin)
            D = float(fit.power_law.D)

            if not (np.isfinite(alpha) and np.isfinite(xmin) and np.isfinite(D)):
                return _resultat_fit_vide(n_pos, n_unique)

            queue = positifs[positifs >= xmin]
            n_tail = int(queue.size)
            if n_tail == 0:
                return _resultat_fit_vide(n_pos, n_unique)

            fraction_tail = n_tail / n_pos
            tail_decades = (
                float(np.log10(queue.max() / xmin))
                if queue.max() > xmin > 0
                else 0.0
            )

            # Chaque comparaison est protégée séparément : une distribution
            # alternative peut parfois être impossible à ajuster sur une queue
            # discrète pauvre, sans invalider l'ajustement principal.
            try:
                R_ln, p_ln = fit.distribution_compare("power_law", "lognormal")
                R_ln, p_ln = float(R_ln), float(p_ln)
            except (ValueError, FloatingPointError, ZeroDivisionError, OverflowError):
                R_ln, p_ln = np.nan, np.nan

            try:
                R_exp, p_exp = fit.distribution_compare("power_law", "exponential")
                R_exp, p_exp = float(R_exp), float(p_exp)
            except (ValueError, FloatingPointError, ZeroDivisionError, OverflowError):
                R_exp, p_exp = np.nan, np.nan

        return {
            "n_pos": n_pos,
            "n_unique": n_unique,
            "alpha": alpha,
            "xmin": xmin,
            "D": D,
            "n_tail": n_tail,
            "fraction_tail": float(fraction_tail),
            "tail_decades": tail_decades,
            "R_lognormale": R_ln,
            "p_lognormale": p_ln,
            "R_exponentielle": R_exp,
            "p_exponentielle": p_exp,
        }

    except (
        ValueError,
        FloatingPointError,
        ZeroDivisionError,
        OverflowError,
        ArithmeticError,
    ):
        return _resultat_fit_vide(n_pos, n_unique)


def ligne_resume(
    n_noeuds: int,
    ratio: float,
    series: dict[str, np.ndarray],
) -> dict[str, float | int]:
    """Construit une ligne de résumé pour un ratio et une taille de réseau."""
    frac = series["fraction_delestee"]
    lignes = series["n_lignes_tombees"]

    fit_frac = ajuster_queue(frac, discret=False)
    fit_lignes = ajuster_queue(lignes, discret=True)

    row: dict[str, float | int] = {
        "n_noeuds": n_noeuds,
        "ratio_PD_PC": float(ratio),
        "n_realisations": int(frac.size),
        "freq_blackout": float(np.mean(frac > 1e-12)),
        "freq_lignes_tombees": float(np.mean(lignes > 0)),
        "fraction_delestee_moyenne": float(frac.mean()),
        "fraction_delestee_max": float(frac.max()),
        "lignes_tombees_moyenne": float(lignes.mean()),
        "lignes_tombees_max": int(lignes.max()),
        "Mmax_initial_moyen": float(series["taux_maximal_initial"].mean()),
        "Mmax_initial_q95": float(np.quantile(series["taux_maximal_initial"], 0.95)),
    }

    for prefixe, stats in (("shed", fit_frac), ("lignes", fit_lignes)):
        for cle, valeur in stats.items():
            row[f"{prefixe}_{cle}"] = valeur

    return row


# ---------------------------------------------------------------------------
# Workers multiprocessus — fonctions au niveau module pour Windows
# ---------------------------------------------------------------------------

def _worker_scan(
    n_noeuds: int,
    ratio: float,
    n: int,
    graine: int,
) -> dict[str, float | int]:
    """
    Une tâche de scan complète : prépare le réseau, simule et résume.

    Le résultat est petit (une ligne de tableau), ce qui évite de transférer
    toutes les séries entre processus pendant le scan.
    """
    systeme = preparer_systeme(n_noeuds)
    series = simuler_realisation_iid(systeme, ratio, n, graine)
    return ligne_resume(n_noeuds, ratio, series)


def _worker_bloc_production(
    n_noeuds: int,
    ratio: float,
    n: int,
    graine: int,
    index_bloc: int,
) -> tuple[int, int, dict[str, np.ndarray]]:
    """
    Simule un bloc indépendant de la production.

    Retourne (taille réseau, index du bloc, séries). L'index permet de concaténer
    les blocs dans un ordre déterministe indépendamment de l'ordre de terminaison.
    """
    systeme = preparer_systeme(n_noeuds)
    series = simuler_realisation_iid(systeme, ratio, n, graine)
    return n_noeuds, index_bloc, series


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def ecrire_csv(chemin: Path, lignes: list[dict]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    if not lignes:
        raise ValueError("Aucune ligne à écrire.")
    with chemin.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        writer.writeheader()
        writer.writerows(lignes)


def sauver_npz(
    chemin: Path,
    series: dict[str, np.ndarray],
    *,
    n_noeuds: int,
    ratio: float,
) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        chemin,
        **series,
        n_noeuds=np.array(n_noeuds),
        ratio_PD_PC=np.array(ratio),
        P_C=np.array(P_C),
        g=np.array(G),
        p0=np.array(P0),
        p1=np.array(P1),
    )


def ecrire_metadata(
    chemin: Path,
    *,
    commande: str,
    n: int,
    workers: int,
    ratios: dict[str, float] | None = None,
    chunk_size: int | None = None,
) -> None:
    metadata = {
        "commande": commande,
        "n_realisations_par_cas": n,
        "workers": workers,
        "chunk_size": chunk_size,
        "P_C": P_C,
        "g_interne": G,
        "g_carreras_equivalent": 1.9,
        "p0": P0,
        "p1": P1,
        "cible_seconde_transition": CIBLE_SECONDE,
        "tailles": list(TAILLES),
        "graine_scan": GRAINE_SCAN,
        "graine_production": GRAINE_PRODUCTION,
        "ratios_production": ratios,
        "observable_principale": "fraction_delestee = delestage_total / demande_totale",
        "note_fluctuations":
            "cascade.journee applique actuellement une fluctuation indépendante "
            "à chaque charge; la corrélation régionale de certaines expériences "
            "de Carreras n'est pas reproduite ici.",
        "interpretation_tests":
            "Les rapports de vraisemblance comparent des modèles relativement; "
            "ils ne sont pas un test absolu de goodness-of-fit.",
        "parallelisation":
            "Scan: un processus par couple taille/ratio. Production: blocs "
            "indépendants avec graines enfant issues de SeedSequence.",
    }
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def ccdf(valeurs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valeurs = np.asarray(valeurs, dtype=float)
    valeurs = valeurs[np.isfinite(valeurs) & (valeurs > 0)]
    x = np.sort(valeurs)
    y = np.arange(x.size, 0, -1, dtype=float) / x.size
    return x, y


def tracer_scan(rows: list[dict]) -> None:
    DOSSIER_FIGURES_SORTIE.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.2), sharex=True)

    for n_noeuds in TAILLES:
        sub = [r for r in rows if int(r["n_noeuds"]) == n_noeuds]
        ratios = np.array([r["ratio_PD_PC"] for r in sub], dtype=float)

        axes[0].plot(
            ratios,
            [r["freq_blackout"] for r in sub],
            "o-",
            label=f"{n_noeuds} nœuds",
        )
        axes[1].plot(
            ratios,
            [r["shed_D"] for r in sub],
            "o-",
            label=f"{n_noeuds} nœuds",
        )

    axes[0].set_ylabel("fréquence de délestage")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].set_xlabel(r"$P_D/P_C$")
    axes[1].set_ylabel("distance KS de la queue")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("Scan du régime critique — arbres 46 et 94 nœuds")
    fig.tight_layout()
    fig.savefig(
        DOSSIER_FIGURES_SORTIE / "04_scan_regime_critique.png",
        dpi=180,
    )
    plt.close(fig)


def tracer_comparaison_ccdf(
    productions: dict[int, tuple[float, dict[str, np.ndarray], dict]],
) -> None:
    DOSSIER_FIGURES_SORTIE.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for n_noeuds, (ratio, series, row) in productions.items():
        x, y = ccdf(series["fraction_delestee"])
        if x.size == 0:
            continue

        ligne, = ax.loglog(
            x,
            y,
            marker=".",
            linestyle="none",
            markersize=3,
            alpha=0.65,
            label=f"{n_noeuds} nœuds, $P_D/P_C={ratio:.3f}$",
        )

        alpha = float(row["shed_alpha"])
        xmin = float(row["shed_xmin"])

        if np.isfinite(alpha) and np.isfinite(xmin) and xmin > 0:
            tail = x[x >= xmin]
            if tail.size:
                p_xmin = np.mean(x >= xmin)
                y_fit = p_xmin * (tail / xmin) ** (-(alpha - 1.0))
                ax.loglog(
                    tail,
                    y_fit,
                    linewidth=1.5,
                    color=ligne.get_color(),
                )

    ax.set_xlabel(r"fraction délestée $P_{\rm shed}/P_D$")
    ax.set_ylabel(r"$P(X \geq x)$")
    ax.set_title("Effet de taille finie sur la queue des blackouts")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(
        DOSSIER_FIGURES_SORTIE / "04_ccdf_46_vs_94.png",
        dpi=180,
    )
    plt.close(fig)


def tracer_comparaison_lignes(
    productions: dict[int, tuple[float, dict[str, np.ndarray], dict]],
) -> None:
    DOSSIER_FIGURES_SORTIE.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for n_noeuds, (ratio, series, _row) in productions.items():
        x, y = ccdf(series["n_lignes_tombees"].astype(float))
        if x.size == 0:
            continue
        ax.loglog(
            x,
            y,
            marker="o",
            linestyle="-",
            markersize=3,
            linewidth=1,
            label=f"{n_noeuds} nœuds, $P_D/P_C={ratio:.3f}$",
        )

    ax.set_xlabel("nombre de lignes tombées")
    ax.set_ylabel(r"$P(N_{\rm outages} \geq n)$")
    ax.set_title("Distribution cumulative des lignes tombées")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(
        DOSSIER_FIGURES_SORTIE / "04_ccdf_lignes_46_vs_94.png",
        dpi=180,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Étape 1 : scan parallèle
# ---------------------------------------------------------------------------

def commande_scan(args: argparse.Namespace) -> None:
    figures.appliquer_style()
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

    ratios = np.arange(
        args.ratio_min,
        args.ratio_max + 0.5 * args.pas,
        args.pas,
    )

    # Affichage des calibrations une seule fois dans le processus principal.
    for n_noeuds in TAILLES:
        systeme = preparer_systeme(n_noeuds)
        print(
            f"\n=== {n_noeuds} nœuds | {systeme.reseau.n_lignes} lignes | "
            f"{systeme.reseau.generateurs.size} générateurs ==="
        )
        print(f"facteur de calibration des limites : {systeme.facteur_calibration:.6g}")

    taches = []
    for n_noeuds in TAILLES:
        # Même graine pour tous les ratios d'une taille donnée :
        # common random numbers => comparaison du scan moins bruitée.
        graine = GRAINE_SCAN + n_noeuds
        for ratio in ratios:
            taches.append((n_noeuds, float(ratio), args.n, graine))

    rows: list[dict] = []
    total = len(taches)

    print(f"\nScan parallèle : {total} cas avec {args.workers} processus.\n")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futurs = {
            executor.submit(_worker_scan, *tache): (tache[0], tache[1])
            for tache in taches
        }

        terminees = 0
        for futur in as_completed(futurs):
            n_noeuds, ratio = futurs[futur]
            row = futur.result()
            rows.append(row)
            terminees += 1

            print(
                f"[{terminees:02d}/{total:02d}] "
                f"{n_noeuds:3d} nœuds | {ratio:5.3f} | "
                f"blackout={row['freq_blackout']:6.1%} | "
                f"n_tail={int(row['shed_n_tail']):5d} | "
                f"D={row['shed_D']:.4f} | "
                f"alpha={row['shed_alpha']:.3f} | "
                f"R_ln={row['shed_R_lognormale']:+.2f}"
            )

    rows.sort(key=lambda r: (int(r["n_noeuds"]), float(r["ratio_PD_PC"])))

    chemin = DOSSIER_SORTIE / "scan.csv"
    ecrire_csv(chemin, rows)
    ecrire_metadata(
        DOSSIER_SORTIE / "metadata_scan.json",
        commande="scan",
        n=args.n,
        workers=args.workers,
    )
    tracer_scan(rows)

    print("\nScan terminé.")
    print(f"Tableau : {chemin}")
    print(f"Figure  : {DOSSIER_FIGURES_SORTIE / '04_scan_regime_critique.png'}")

    print("\nCandidats exploratoires (plus petit D parmi les queues avec >= 100 points) :")
    for n_noeuds in TAILLES:
        candidats = [
            r for r in rows
            if int(r["n_noeuds"]) == n_noeuds
            and int(r["shed_n_tail"]) >= 100
            and np.isfinite(r["shed_D"])
        ]
        candidats.sort(key=lambda r: float(r["shed_D"]))

        print(f"  {n_noeuds} nœuds :")
        if not candidats:
            print("    aucun candidat avec >= 100 points dans la queue")
            continue

        for r in candidats[:3]:
            print(
                f"    P_D/P_C={r['ratio_PD_PC']:.3f} | "
                f"D={r['shed_D']:.4f} | n_tail={int(r['shed_n_tail'])} | "
                f"{r['shed_tail_decades']:.2f} décades | "
                f"R_ln={r['shed_R_lognormale']:+.2f}"
            )


# ---------------------------------------------------------------------------
# Étape 2 : production parallèle par blocs
# ---------------------------------------------------------------------------

def _decouper_n(n: int, chunk_size: int) -> list[int]:
    """Décompose n en blocs, le dernier pouvant être plus petit."""
    blocs = []
    restant = n
    while restant > 0:
        courant = min(chunk_size, restant)
        blocs.append(courant)
        restant -= courant
    return blocs


def commande_production(args: argparse.Namespace) -> None:
    figures.appliquer_style()
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

    ratios = {
        46: float(args.ratio_46),
        94: float(args.ratio_94),
    }

    tailles_blocs = _decouper_n(args.n, args.chunk_size)

    # SeedSequence produit des flux indépendants et reproductibles entre blocs.
    seed_maitre = np.random.SeedSequence(GRAINE_PRODUCTION)
    n_taches = len(TAILLES) * len(tailles_blocs)
    graines_enfants = seed_maitre.spawn(n_taches)

    taches = []
    indice_seed = 0
    for n_noeuds in TAILLES:
        for index_bloc, n_bloc in enumerate(tailles_blocs):
            graine = int(graines_enfants[indice_seed].generate_state(1, dtype=np.uint64)[0])
            indice_seed += 1
            taches.append(
                (
                    n_noeuds,
                    ratios[n_noeuds],
                    n_bloc,
                    graine,
                    index_bloc,
                )
            )

    print(
        f"\nProduction parallèle : {args.n} réalisations par réseau, "
        f"{len(tailles_blocs)} blocs par réseau, {args.workers} processus.\n"
    )

    blocs_par_reseau: dict[int, dict[int, dict[str, np.ndarray]]] = {
        n_noeuds: {} for n_noeuds in TAILLES
    }

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futurs = {
            executor.submit(_worker_bloc_production, *tache):
                (tache[0], tache[4], tache[2])
            for tache in taches
        }

        terminees = 0
        for futur in as_completed(futurs):
            n_noeuds, index_bloc, n_bloc = futurs[futur]
            n_retour, index_retour, series = futur.result()
            blocs_par_reseau[n_retour][index_retour] = series

            terminees += 1
            print(
                f"[{terminees:02d}/{len(taches):02d}] "
                f"{n_noeuds} nœuds | bloc {index_bloc + 1}/{len(tailles_blocs)} "
                f"({n_bloc} réalisations)"
            )

    rows: list[dict] = []
    productions: dict[int, tuple[float, dict[str, np.ndarray], dict]] = {}

    for n_noeuds in TAILLES:
        blocs_ordonnes = [
            blocs_par_reseau[n_noeuds][i]
            for i in range(len(tailles_blocs))
        ]
        series = _concatener_series(blocs_ordonnes)
        ratio = ratios[n_noeuds]
        row = ligne_resume(n_noeuds, ratio, series)

        rows.append(row)
        productions[n_noeuds] = (ratio, series, row)

        sauver_npz(
            DOSSIER_SORTIE / f"production_{n_noeuds}_noeuds.npz",
            series,
            n_noeuds=n_noeuds,
            ratio=ratio,
        )

        print(
            f"\n=== Résumé : {n_noeuds} nœuds | "
            f"P_D/P_C={ratio:.4f} | n={args.n} ==="
        )
        print(f"blackouts              : {row['freq_blackout']:.2%}")
        print(f"lignes tombées > 0     : {row['freq_lignes_tombees']:.2%}")
        print(f"fraction délestée max  : {row['fraction_delestee_max']:.5f}")
        print(f"lignes tombées max     : {int(row['lignes_tombees_max'])}")
        print("--- queue du délestage normalisé ---")
        print(f"alpha                   : {row['shed_alpha']:.4f}")
        print(f"xmin                    : {row['shed_xmin']:.6g}")
        print(f"D (KS)                  : {row['shed_D']:.5f}")
        print(f"n_tail                  : {int(row['shed_n_tail'])}")
        print(f"valeurs distinctes > 0 : {int(row['shed_n_unique'])}")
        print(f"étendue de queue        : {row['shed_tail_decades']:.3f} décades")
        print(
            f"vs lognormale           : R={row['shed_R_lognormale']:+.3f}, "
            f"p={row['shed_p_lognormale']:.4g}"
        )
        print(
            f"vs exponentielle        : R={row['shed_R_exponentielle']:+.3f}, "
            f"p={row['shed_p_exponentielle']:.4g}"
        )

    ecrire_csv(DOSSIER_SORTIE / "resume_production.csv", rows)
    ecrire_metadata(
        DOSSIER_SORTIE / "metadata_production.json",
        commande="production",
        n=args.n,
        workers=args.workers,
        ratios={str(k): v for k, v in ratios.items()},
        chunk_size=args.chunk_size,
    )

    tracer_comparaison_ccdf(productions)
    tracer_comparaison_lignes(productions)

    print("\nProduction terminée.")
    print(f"Résumé : {DOSSIER_SORTIE / 'resume_production.csv'}")
    print(f"Données : {DOSSIER_SORTIE}")
    print(f"Figures : {DOSSIER_FIGURES_SORTIE}")

    r46 = productions[46][2]
    r94 = productions[94][2]

    print("\nComparaison de taille finie (descriptive) :")
    print(
        f"  étendue queue délestage : "
        f"46 nœuds = {r46['shed_tail_decades']:.3f} décades ; "
        f"94 nœuds = {r94['shed_tail_decades']:.3f} décades"
    )
    print(
        f"  n_tail                  : "
        f"46 nœuds = {int(r46['shed_n_tail'])} ; "
        f"94 nœuds = {int(r94['shed_n_tail'])}"
    )
    print(
        "  Une queue plus étendue sur 94 nœuds irait dans le sens de l'effet "
        "de taille finie rapporté par Carreras, sans constituer à elle seule "
        "une preuve de loi de puissance."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def construire_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sous = parser.add_subparsers(dest="commande", required=True)

    scan = sous.add_parser(
        "scan",
        help="Balayage exploratoire fin de P_D/P_C sur 46 et 94 nœuds.",
    )
    scan.add_argument(
        "--n",
        type=int,
        default=2500,
        help="Réalisations par ratio et par taille (défaut : 2500).",
    )
    scan.add_argument("--ratio-min", type=float, default=0.55)
    scan.add_argument("--ratio-max", type=float, default=1.05)
    scan.add_argument("--pas", type=float, default=0.025)
    scan.add_argument(
        "--workers",
        type=int,
        default=WORKERS_DEFAUT,
        help=f"Nombre de processus parallèles (défaut : {WORKERS_DEFAUT}).",
    )
    scan.set_defaults(fonction=commande_scan)

    prod = sous.add_parser(
        "production",
        help="Simulation confirmatoire longue aux ratios choisis après le scan.",
    )
    prod.add_argument(
        "--ratio-46",
        type=float,
        required=True,
        help="P_D/P_C retenu pour l'arbre de 46 nœuds.",
    )
    prod.add_argument(
        "--ratio-94",
        type=float,
        required=True,
        help="P_D/P_C retenu pour l'arbre de 94 nœuds.",
    )
    prod.add_argument(
        "--n",
        type=int,
        default=60_000,
        help="Réalisations par taille (défaut : 60000).",
    )
    prod.add_argument(
        "--workers",
        type=int,
        default=WORKERS_DEFAUT,
        help=f"Nombre de processus parallèles (défaut : {WORKERS_DEFAUT}).",
    )
    prod.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Taille d'un bloc de production (défaut : 5000).",
    )
    prod.set_defaults(fonction=commande_production)

    return parser


def main() -> None:
    parser = construire_parser()
    args = parser.parse_args()

    if args.n < 100:
        parser.error("--n doit être >= 100.")

    if args.workers < 1:
        parser.error("--workers doit être >= 1.")

    if args.commande == "scan":
        if args.ratio_min <= 0 or args.ratio_max <= args.ratio_min:
            parser.error("Intervalle de ratios invalide.")
        if args.pas <= 0:
            parser.error("--pas doit être > 0.")

    if args.commande == "production":
        if args.ratio_46 <= 0 or args.ratio_94 <= 0:
            parser.error("Les ratios de production doivent être > 0.")
        if args.chunk_size < 100:
            parser.error("--chunk-size doit être >= 100.")

    args.fonction(args)


if __name__ == "__main__":
    main()
