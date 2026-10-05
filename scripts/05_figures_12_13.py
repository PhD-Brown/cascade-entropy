"""
SO1 : Fig. 12 et 13 de Carreras et al. (2002), distributions du délestage.

Fig. 12 : densité (PDF) du délestage normalisé, arbre de 382 nœuds, pour
plusieurs niveaux de demande, avec fluctuations régionales (γ = 1.9,
p0 = 1e-4, 60 000 tirages par niveau). Publié (Sec. V) :
  - bien sous le point critique : densité piquée aux petits délestages, queue
    en P^-2, conséquence analytique d'une avarie isolée dans un arbre ;
  - près du point critique : queue algébrique d'indice proche de −1 ;
  - bien au-dessus : densité piquée aux grands délestages.
Fig. 13 : densité près du point critique pour plusieurs tailles d'arbre ; la
région algébrique s'allonge avec la taille.

Ce script NE SIMULE PAS les réalisations : il lit les checkpoints de
`05_reproduction_carreras.py scan`, déjà validé en SO1 (reprise, graines,
départage). Il a deux commandes :

  plan     écrit les commandes `05 ... scan` à lancer. Les niveaux de demande
           sont donnés en ρ = r / r_T(N) (tableau SO1, section C) : à ρ égal,
           la probabilité qu'une cascade démarre est la même pour toutes les
           tailles, ce qui rend la Fig. 13 comparable.
  figures  lit les runs et produit les figures, les ajustements et les CSV.

Prédiction dérivée (sans ajustement) pour les niveaux sous-critiques
--------------------------------------------------------------------
Si γ ρ < 0.99, aucune région ne peut saturer une ligne : il n'y a pas de
cascade, et le dispatch se réduit au bilan par îlot
(`carreras.delestage_bilan_ilots`) : délestage = Σ max(0, D_îlot − P_îlot),
les îlots venant des avaries p0. Cette formule est vérifiée contre le dispatch
dans `tests/test_carreras_fig12.py`. Le script en tire, par Monte-Carlo sans
programme linéaire, la densité attendue et la superpose à la mesure. La
pente −2 de Carreras en découle : chaque ligne extérieure a la même
probabilité de tomber, le nombre de lignes double quand la taille du
sous-arbre est divisée par deux (`carreras.delestage_avarie_unique`).

Ajustements de queue (même méthode que 08_analyse_so2.py)
---------------------------------------------------------
Plus longue plage contiguë, en échelle logarithmique, où la relation log-log
est linéaire (R² ≥ 0.98) et décroissante, sur la densité (pdf_*) et sur la
fréquence cumulée P(X ≥ x) (cumul_*, pente < −0.1). L'« étendue » est le
rapport x_fin / x_debut de cette plage, comme dans la Table I de Carreras
2004. C'est NOTRE opérationnalisation : le papier ne décrit pas sa méthode.

Sorties
-------
data/05_reproduction_carreras/figures_12_13/<SORTIE>/
    metadata.json   runs lus, paramètres, cas analysés ou sautés
    queues.csv      une ligne par cas (N, r, ρ) : fréquences, pentes, étendues,
                    prédiction sous-critique
figures/05_reproduction_carreras/figures_12_13/<SORTIE>/
    fig12_pdf_N<n>.png   densité et fréquence cumulée, tous les niveaux
    fig13_tailles.png    densité et fréquence cumulée à ρ fixé, toutes tailles

Exemples (PowerShell)
---------------------
    python scripts\\05_figures_12_13.py plan
    python scripts\\05_figures_12_13.py figures --fig12 fig12-N382 `
        --fig13 fig13-N46 fig13-N94 fig13-N190
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cascade_entropy.analyse_series import (  # noqa: E402
    ccdf_logarithmique,
    pdf_logarithmique,
    plage_loi_puissance,
)
from cascade_entropy.carreras import (  # noqa: E402
    configuration_arbre,
    delestage_bilan_ilots,
    groupes_regions,
    seuil_transport,
)
from cascade_entropy.chemins import DOSSIER_DATA, DOSSIER_FIGURES  # noqa: E402
from cascade_entropy.dispatch import SEUIL_SATURATION  # noqa: E402

SCRIPT_VERSION = "1.0.0"

DOSSIER_RUNS_05 = DOSSIER_DATA / "05_reproduction_carreras" / "runs"
# Même famille que 05_reproduction_carreras.py, dont il lit les runs.
DOSSIER_SORTIE = DOSSIER_DATA / "05_reproduction_carreras" / "figures_12_13"
DOSSIER_FIG = DOSSIER_FIGURES / "05_reproduction_carreras" / "figures_12_13"

TAILLES = (46, 94, 190, 382)
# Normalisation du délestage. « nominale » : P_shed / (r P_C), l'observable
# historique de SO1 (défaut). « realisee » : P_shed / P_D du tirage, toujours
# dans [0, 1], plus proche du « load shed normalized to the power demand ».
OBSERVABLES = {"nominale": "fraction_delestee_nominale",
               "realisee": "fraction_delestee_realisee"}
XLABELS = {"nominale": "délestage / demande nominale",
           "realisee": "délestage / demande du tirage"}
SEUIL_BLACKOUT = 1e-12
EFFECTIF_MIN_CUMUL = 20
PENTE_MAX_CUMUL = -0.1
# Tolérance pour reconnaître un ρ cible à partir du ratio écrit dans un run.
TOLERANCE_RHO = 1e-3

RHO_FIG12 = (0.45, 0.526, 0.55, 0.70)
RHO_FIG13 = 0.55


# ===========================================================================
# Utilitaires
# ===========================================================================

def maintenant_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def nettoyer_id(texte: str) -> str:
    texte = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(texte).strip()).strip("-_.")
    if not texte:
        raise ValueError("Identifiant vide ou invalide.")
    return texte


def ecrire_json(chemin: Path, obj: Any) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tmp = chemin.with_name(chemin.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, chemin)


def ecrire_csv(chemin: Path, lignes: list[dict]) -> None:
    if not lignes:
        return
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tmp = chemin.with_name(chemin.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        w.writeheader()
        w.writerows(lignes)
    os.replace(tmp, chemin)


def sauvegarder(fig, chemin: Path) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(chemin, dpi=160, bbox_inches="tight")
    except PermissionError:
        print(f"  ATTENTION : {chemin.name} est ouvert ailleurs, non mis à jour.")
    plt.close(fig)


def seuils() -> dict[int, float]:
    """r_T(N) pour les quatre arbres (tableau SO1, section B)."""
    return {n: seuil_transport(configuration_arbre(n)) for n in TAILLES}


# ===========================================================================
# Commande plan
# ===========================================================================

def commandes_plan(args: argparse.Namespace) -> list[str]:
    """Lignes de commande `05 ... scan` pour les Fig. 12 et 13."""
    r_t = seuils()
    commun = (f"--n {args.n} --chunk-size {args.chunk_size} --workers {args.workers} "
              f"--departage {args.departage}")
    lignes = []
    ratios12 = " ".join(f"{rho * r_t[args.taille_fig12]:.6f}" for rho in args.rho_fig12)
    lignes.append(
        f"python scripts\\05_reproduction_carreras.py scan --tailles {args.taille_fig12} "
        f"--ratios {ratios12} {commun} --run-id fig12-N{args.taille_fig12}")
    for n in args.tailles_fig13:
        # Le cas de la Fig. 12 au même ρ est réutilisé, pas recalculé.
        if n == args.taille_fig12 and any(abs(r - args.rho_fig13) < TOLERANCE_RHO
                                           for r in args.rho_fig12):
            continue
        lignes.append(
            f"python scripts\\05_reproduction_carreras.py scan --tailles {n} "
            f"--ratios {args.rho_fig13 * r_t[n]:.6f} {commun} --run-id fig13-N{n}")
    return lignes


def commande_plan(args: argparse.Namespace) -> None:
    valider_rhos(args.rho_fig12 + [args.rho_fig13])
    r_t = seuils()
    print("Seuils de transport r_T(N) : "
          + ", ".join(f"{n} : {r:.6f}" for n, r in r_t.items()))
    print(f"Fig. 12 ({args.taille_fig12} nœuds) : ρ = "
          + ", ".join(f"{rho:g} (r = {rho * r_t[args.taille_fig12]:.4f}, "
                      f"γρ = {1.9 * rho:.3f})" for rho in args.rho_fig12))
    print(f"Fig. 13 : ρ = {args.rho_fig13:g} pour " + ", ".join(map(str, args.tailles_fig13)))
    print("\nCommandes à lancer (PowerShell) :\n")
    for ligne in commandes_plan(args):
        print(ligne)
    print("\nPuis :\n")
    fig13 = " ".join(f"fig13-N{n}" for n in args.tailles_fig13 if n != args.taille_fig12)
    print(f"python scripts\\05_figures_12_13.py figures --fig12 fig12-N{args.taille_fig12}"
          + (f" --fig13 {fig13}" if fig13 else ""))


def valider_rhos(rhos: list[float]) -> None:
    for rho in rhos:
        if not np.isfinite(rho) or not 0 < rho <= 1.5:
            raise SystemExit(f"ρ doit être dans ]0, 1.5], pas {rho}.")


# ===========================================================================
# Lecture des runs de 05
# ===========================================================================

def charger_run(run_id: str, observable: str = OBSERVABLES["nominale"]
                ) -> tuple[dict, dict[tuple[int, float], dict], dict]:
    """
    Cas complets d'un run de `05_reproduction_carreras.py`.

    Retourne (config du run, {(N, ratio): séries}, {nom: raison} des cas sautés).
    Un cas est complet quand ses blocs totalisent `n_realisations` tirages.
    """
    run_dir = DOSSIER_RUNS_05 / nettoyer_id(run_id)
    meta = run_dir / "metadata.json"
    if not meta.exists():
        raise SystemExit(f"Run introuvable : {meta}")
    config = json.loads(meta.read_text("utf-8"))["config"]
    attendu = int(config["n_realisations"])

    blocs: dict[tuple[int, float], list[tuple[int, dict]]] = {}
    for chemin in sorted((run_dir / "checkpoints").glob("*.npz")):
        with np.load(chemin, allow_pickle=False) as d:
            cle = (int(d["__n_noeuds__"]), float(d["__ratio__"]))
            blocs.setdefault(cle, []).append((int(d["__index_bloc__"]), {
                "fraction": np.array(d[observable], dtype=float),
                "n_lignes_tombees": np.array(d["n_lignes_tombees"]),
            }))

    cas, sautes = {}, {}
    for (n, ratio), liste in blocs.items():
        liste.sort(key=lambda t: t[0])
        series = {k: np.concatenate([b[k] for _, b in liste]) for k in liste[0][1]}
        taille = series["fraction"].size
        if taille != attendu:
            sautes[f"{run_id}:N{n}_r{ratio:.6f}"] = f"incomplet ({taille} / {attendu})"
            continue
        cas[(n, ratio)] = series
    return config, cas, sautes


# ===========================================================================
# Analyse d'un cas
# ===========================================================================

def prediction_sous_critique(n_noeuds: int, ratio: float, config: dict, n: int,
                             rng: np.random.Generator,
                             normalisation: str = "nominale") -> np.ndarray:
    """
    Délestage normalisé attendu sans cascade : avaries p0 + limite de génération.

    Tirages vectorisés de la demande régionale ; le bilan par îlot n'est
    évalué au cas par cas que pour les tirages avec au moins une avarie (≈ 4 %
    à p0 = 1e-4 sur le 382). Valide seulement si γ ρ < 0.99.
    """
    cfg = configuration_arbre(n_noeuds)
    gamma, p0 = float(config["gamma"]), float(config["p0"])
    n_regions = int(config["n_regions"])
    groupes = groupes_regions(cfg.reseau, n_regions=n_regions)
    base = ratio * cfg.p_c / cfg.n_charges
    facteurs = rng.uniform(2.0 - gamma, gamma, size=(n, n_regions))
    charges_par_region = np.bincount(groupes, minlength=n_regions)
    demande_totale = base * facteurs @ charges_par_region
    delestage = np.maximum(demande_totale - cfg.p_c, 0.0)
    morts = rng.random((n, cfg.reseau.n_lignes)) < p0
    for i in np.flatnonzero(morts.any(axis=1)):
        delestage[i] = delestage_bilan_ilots(cfg, base * facteurs[i, groupes], morts[i])
    if normalisation == "realisee":
        return delestage / demande_totale
    if normalisation != "nominale":
        raise ValueError("normalisation doit être 'nominale' ou 'realisee'.")
    return delestage / (ratio * cfg.p_c)


def ajuster(valeurs: np.ndarray, classes: int, r2_min: float) -> dict:
    """Densité, fréquence cumulée et leurs plages de loi de puissance."""
    vide = plage_loi_puissance(np.ones(3), np.ones(3), np.zeros(3))
    positives = valeurs[valeurs > SEUIL_BLACKOUT]
    if positives.size < 2 or np.unique(positives).size < 2:
        return {"pdf": (np.empty(0),) * 3 + (vide,), "cumul": (np.empty(0),) * 3 + (vide,)}
    c, d, e = pdf_logarithmique(positives, classes)
    q_pdf = plage_loi_puissance(c, d, e, r2_min=r2_min, pente_max=0.0)
    g, f, n = ccdf_logarithmique(positives, classes + 5)
    q_cumul = plage_loi_puissance(g, f, n, r2_min=r2_min, effectif_min=EFFECTIF_MIN_CUMUL,
                                  pente_max=PENTE_MAX_CUMUL)
    return {"pdf": (c, d, e, q_pdf), "cumul": (g, f, n, q_cumul)}


def analyser_cas(n_noeuds: int, ratio: float, series: dict, config: dict,
                 r_t: dict[int, float], args: argparse.Namespace) -> dict:
    rho = ratio / r_t[n_noeuds]
    gamma = float(config["gamma"])
    x = series["fraction"]
    lignes = series["n_lignes_tombees"]
    fits = ajuster(x, args.classes, args.r2_min)
    sous_critique = gamma * rho < SEUIL_SATURATION
    ligne = {
        "n_noeuds": n_noeuds, "ratio_PD_PC": ratio, "rho": rho, "gamma_rho": gamma * rho,
        "regime": "sous-critique" if sous_critique else "cascades possibles",
        "n_realisations": int(x.size),
        "freq_blackout": float(np.mean(x > SEUIL_BLACKOUT)),
        "freq_cascade_multiligne": float(np.mean(lignes > 1)),
        "delestage_moyen": float(x.mean()),
        "delestage_moyen_si_blackout": (float(x[x > SEUIL_BLACKOUT].mean())
                                        if np.any(x > SEUIL_BLACKOUT) else float("nan")),
    }
    for nom in ("pdf", "cumul"):
        q = fits[nom][3]
        for cle in ("pente", "etendue", "x_debut", "x_fin", "r2", "n_classes"):
            ligne[f"{nom}_{cle}"] = q[cle]
    prediction = None
    if sous_critique and args.prediction > 0:
        rng = np.random.default_rng(args.graine + n_noeuds)
        prediction = prediction_sous_critique(n_noeuds, ratio, config, args.prediction, rng,
                                              args.normalisation)
        ligne["prediction_freq_blackout"] = float(np.mean(prediction > SEUIL_BLACKOUT))
        ligne["prediction_delestage_moyen"] = float(prediction.mean())
    else:
        ligne["prediction_freq_blackout"] = float("nan")
        ligne["prediction_delestage_moyen"] = float("nan")
    return {"ligne": ligne, "fits": fits, "prediction": prediction}


# ===========================================================================
# Figures
# ===========================================================================

def _guide(ax, x0: float, y0: float, pente: float, texte: str) -> None:
    """Droite de référence (non ajustée) de pente donnée, sur une décade."""
    xs = np.array([x0, x0 * 10])
    ax.loglog(xs, y0 * (xs / x0) ** pente, color="0.25", linewidth=0.9, linestyle="--")
    ax.text(xs[1], y0 * 10 ** pente, texte, fontsize=7, color="0.25")


def figure_distributions(chemin: Path, resultats: list[dict], titre: str,
                         etiquette, classes: int,
                         xlabel: str = "délestage / demande nominale") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    couleurs = plt.cm.viridis(np.linspace(0, 0.9, max(len(resultats), 1)))
    for res, c in zip(resultats, couleurs):
        lab = etiquette(res["ligne"])
        for ax, nom in zip(axes, ("pdf", "cumul")):
            x, y, _, q = res["fits"][nom]
            if x.size == 0:
                continue
            texte = f"{lab}, pente {q['pente']:.2f}" if q["n_classes"] else f"{lab}"
            ax.loglog(x, y, "o", markersize=3, color=c, label=texte)
            if q["n_classes"]:
                xs = np.array([q["x_debut"], q["x_fin"]])
                y0 = y[np.argmin(np.abs(x - xs[0]))]
                ax.loglog(xs, y0 * (xs / xs[0]) ** q["pente"], "-", color=c, linewidth=1.4)
        if res["prediction"] is not None:
            p = res["prediction"][res["prediction"] > SEUIL_BLACKOUT]
            if p.size >= 2 and np.unique(p).size >= 2:
                cp, dp, _ = pdf_logarithmique(p, classes)
                # Même normalisation que la mesure : densité conditionnelle au blackout.
                axes[0].loglog(cp, dp, "-", color=c, linewidth=0.9, alpha=0.7)
                gp, fp, _ = ccdf_logarithmique(p, classes + 5)
                axes[1].loglog(gp, fp, "-", color=c, linewidth=0.9, alpha=0.7)
    # Guides publiés : −2 (sous-critique) et −1 (critique), sur la densité.
    ymax = max((np.max(r["fits"]["pdf"][1]) for r in resultats if r["fits"]["pdf"][1].size),
               default=1.0)
    _guide(axes[0], 1e-3, ymax, -2.0, "pente −2 (publié, sous-critique)")
    _guide(axes[0], 1e-2, ymax / 30, -1.0, "pente −1 (publié, critique)")
    axes[0].set_ylabel("densité (conditionnelle à un délestage > 0)")
    axes[1].set_ylabel("fréquence cumulée P(X ≥ x)")
    axes[0].set_title("densité ; traits fins : prédiction sans cascade")
    axes[1].set_title("fréquence cumulée")
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.2, which="both")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(titre)
    fig.tight_layout()
    sauvegarder(fig, chemin)


# ===========================================================================
# Commande figures
# ===========================================================================

def commande_figures(args: argparse.Namespace) -> None:
    if args.classes < 3:
        raise SystemExit("--classes doit être ≥ 3.")
    if not 0 < args.r2_min <= 1:
        raise SystemExit("--r2-min doit être dans ]0, 1].")
    if args.prediction < 0:
        raise SystemExit("--prediction doit être ≥ 0.")
    valider_rhos([args.rho_fig13])
    if not args.fig12 and not args.fig13:
        raise SystemExit("Donner au moins --fig12 ou --fig13.")

    r_t = seuils()
    sortie = nettoyer_id(args.sortie)
    dossier, fig_dir = DOSSIER_SORTIE / sortie, DOSSIER_FIG / sortie

    tous: dict[tuple[int, float], tuple[dict, dict, str]] = {}
    sautes: dict[str, str] = {}
    configs = {}
    for run_id in list(args.fig12 or []) + list(args.fig13 or []):
        config, cas, s = charger_run(run_id, OBSERVABLES[args.normalisation])
        configs[run_id] = config
        sautes.update(s)
        for cle, series in cas.items():
            tous.setdefault(cle, (series, config, run_id))

    resultats: dict[tuple[int, float], dict] = {}
    for (n, ratio), (series, config, run_id) in sorted(tous.items()):
        res = analyser_cas(n, ratio, series, config, r_t, args)
        res["ligne"]["run"] = run_id
        resultats[(n, ratio)] = res
        l = res["ligne"]
        print(f"  N{n} r = {ratio:.4f} (ρ = {l['rho']:.3f}, {l['regime']}) | "
              f"blackouts {l['freq_blackout']:.3f}"
              + (f" (prédit {l['prediction_freq_blackout']:.3f})"
                 if np.isfinite(l['prediction_freq_blackout']) else "")
              + f" | pente densité {l['pdf_pente']:.2f} (étendue {l['pdf_etendue']:.1f})"
              f" | pente cumulée {l['cumul_pente']:.2f} (étendue {l['cumul_etendue']:.1f})")

    if not resultats:
        raise SystemExit("Aucun cas complet dans les runs donnés.")

    # Fig. 12 : tous les niveaux de la taille choisie, issus des runs --fig12.
    runs12 = set(args.fig12 or [])
    fig12 = [r for (n, _), r in sorted(resultats.items())
             if n == args.taille_fig12 and r["ligne"]["run"] in runs12]
    if fig12:
        figure_distributions(
            fig_dir / f"fig12_pdf_N{args.taille_fig12}.png", fig12,
            f"Analogue de la Fig. 12 : {args.taille_fig12} nœuds, plusieurs niveaux de demande",
            lambda l: f"ρ = {l['rho']:.3f} (r = {l['ratio_PD_PC']:.3f})", args.classes,
            XLABELS[args.normalisation])

    # Fig. 13 : un cas par taille au ρ demandé, quel que soit le run d'origine.
    fig13 = []
    for n in TAILLES:
        proches = [r for (m, _), r in resultats.items()
                   if m == n and abs(r["ligne"]["rho"] - args.rho_fig13) < TOLERANCE_RHO]
        if proches:
            fig13.append(proches[0])
    if len(fig13) >= 2:
        figure_distributions(
            fig_dir / "fig13_tailles.png", fig13,
            f"Analogue de la Fig. 13 : tailles d'arbre à ρ = {args.rho_fig13:g}",
            lambda l: f"{l['n_noeuds']} nœuds", args.classes, XLABELS[args.normalisation])

    ecrire_csv(dossier / "queues.csv", [r["ligne"] for r in resultats.values()])
    ecrire_json(dossier / "metadata.json", {
        "script": "05_figures_12_13.py", "script_version": SCRIPT_VERSION,
        "created_at": maintenant_iso(), "runs": configs,
        "parametres": {"classes": args.classes, "r2_min": args.r2_min,
                       "prediction": args.prediction, "graine": args.graine,
                       "rho_fig13": args.rho_fig13, "taille_fig12": args.taille_fig12,
                       "normalisation": args.normalisation,
                       "observable": OBSERVABLES[args.normalisation],
                       "seuils_transport": r_t},
        "cas_fig12": [f"N{r['ligne']['n_noeuds']}_r{r['ligne']['ratio_PD_PC']:.6f}" for r in fig12],
        "cas_fig13": [f"N{r['ligne']['n_noeuds']}_r{r['ligne']['ratio_PD_PC']:.6f}" for r in fig13],
        "cas_sautes": sautes,
    })
    print(f"\nRésultats : {dossier}\nFigures   : {fig_dir}")
    if sautes:
        print(f"{len(sautes)} cas sautés : " + ", ".join(sautes))


# ===========================================================================
# Interface
# ===========================================================================

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="commande", required=True)

    plan = sub.add_parser("plan", help="Écrit les commandes de scan à lancer.")
    plan.add_argument("--rho-fig12", nargs="+", type=float, default=list(RHO_FIG12))
    plan.add_argument("--rho-fig13", type=float, default=RHO_FIG13)
    plan.add_argument("--taille-fig12", type=int, choices=TAILLES, default=382)
    plan.add_argument("--tailles-fig13", nargs="+", type=int, choices=TAILLES,
                      default=list(TAILLES))
    plan.add_argument("--n", type=int, default=60_000,
                      help="tirages par niveau (60 000 dans Carreras 2002)")
    plan.add_argument("--chunk-size", type=int, default=1000)
    plan.add_argument("--workers", type=int, default=16)
    plan.add_argument("--departage", default="exterieur_dabord",
                      choices=["highs", "exterieur_dabord"])
    plan.set_defaults(func=commande_plan)

    fig = sub.add_parser("figures", help="Lit les runs et trace les Fig. 12 et 13.")
    fig.add_argument("--fig12", nargs="+", default=None, help="run(s) de la Fig. 12")
    fig.add_argument("--fig13", nargs="+", default=None,
                     help="run(s) supplémentaires pour la Fig. 13")
    fig.add_argument("--taille-fig12", type=int, choices=TAILLES, default=382)
    fig.add_argument("--rho-fig13", type=float, default=RHO_FIG13)
    fig.add_argument("--sortie", default="fig12-13")
    fig.add_argument("--classes", type=int, default=30)
    fig.add_argument("--r2-min", type=float, default=0.98)
    fig.add_argument("--prediction", type=int, default=200_000,
                     help="tirages Monte-Carlo de la prédiction sous-critique (0 : aucune)")
    fig.add_argument("--graine", type=int, default=20261005)
    fig.add_argument("--normalisation", choices=sorted(OBSERVABLES), default="nominale",
                     help="délestage / demande nominale (défaut, SO1) ou / demande du tirage")
    fig.set_defaults(func=commande_figures)
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
