"""
SO3, étape 1 : entropies des séries de la dynamique lente, avec deux références.

Ce script lit un run produit par `07_dynamique_lente.py` (aucune simulation ici)
et calcule, pour chaque cas (taille d'arbre, G) et chaque observable retenu :

- l'entropie de permutation multiéchelle (PE de dimension d sur la série
  moyennée par blocs de s jours), pour d dans `--dimensions` ;
- l'entropie multiéchelle (MSE : SampEn de la série moyennée par blocs, avec
  la tolérance fixe r·σ de la série d'origine, Costa et al. 2002) ;

et compare chaque valeur à deux familles de séries de substitution
(`cascade_entropy.substituts`) :

- `--melanges` MÉLANGES (jours permutés) : même distribution, aucun ordre.
  Écart au mélange = l'entropie voit une structure temporelle ;
- `--iaaft` substituts IAAFT : même distribution ET même spectre, phases
  aléatoires. Écart à l'IAAFT = l'entropie voit une structure que le spectre
  (donc Hurst et toute corrélation linéaire) n'explique pas. C'est la question
  « information complémentaire à Hurst » de la problématique.

Observables (décision de septembre, docs/journal.md) :
    nombre_effectif   e^H de la répartition des flux au premier dispatch
    taux_maximal      charge de la ligne la plus chargée (indicateur de référence)
    charge_relative   demande totale / capacité totale des générateurs
                      (indicateur de référence)
Le délestage et le nombre de lignes tombées sont surtout faits de zéros
exacts : ils ne reçoivent PAS de PE ni de SampEn (ces mesures y compteraient
les égalités). Leur diagnostic d'égalités est néanmoins écrit dans
`egalites.csv`, avec celui des trois observables, pour que la règle du journal
(« mesurer d'abord la fraction de valeurs identiques ») soit tracée.

Les séries sont analysées après le transitoire du run (`--transitoire` pour le
remplacer), sans retrait de tendance. Un observable qui contient des NaN est
sauté pour ce cas, avec un message : aucune valeur n'est complétée.

Chaque couple (cas, observable) est une tâche indépendante, au résultat écrit
dans `taches/` dès qu'elle finit. `--resume` réutilise les tâches dont les
paramètres et l'empreinte SHA-256 du fichier d'entrée sont inchangés. Les
graines dépendent de (graine, cas, observable, référence), jamais de l'ordre
d'exécution : le résultat est le même avec 1 ou 20 processus.

Ce script appartient au fil B : il n'importe aucun module du réseau et ne
communique avec le fil A que par les fichiers .npz du run.

Sorties
-------
data/09_analyse_so3/<SORTIE>/
    metadata.json     paramètres, run source, empreintes SHA-256 des entrées
    egalites.csv      une ligne par (cas, série) : zéros, égalités, valeurs répétées
    resume.csv        une ligne par (cas, observable) : moments, autocorrélation,
                      rapport de variance des blocs de 1000 jours, écart
                      spectral des IAAFT, |z| maximal de la MSE
    entropie.csv      une ligne par (cas, observable, mesure, échelle) : valeur,
                      références mélange et IAAFT (moyenne, écart-type,
                      intervalle à 95 %), écarts réduits z_melange et z_iaaft
    taches/*.npz      résultats bruts par tâche (reprise)
figures/09_analyse_so3/<SORTIE>/
    mse_<observable>.png         MSE de l'original et bandes des deux références
    z_<mesure>_<reference>.png   écart réduit selon l'échelle, par G et par taille

Exemples (PowerShell)
---------------------
    python scripts\\09_analyse_so3.py --run so2-G-scan --workers 20
    python scripts\\09_analyse_so3.py --run so2-G-scan --workers 20 --resume
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cascade_entropy.analyse_entropie import (  # noqa: E402
    DIMENSIONS_DEFAUT,
    ECHELLES_DEFAUT,
    analyser_avec_references,
    diagnostic_egalites,
    ignorer_interruption,
    resume_substituts,
)
from cascade_entropy.chemins import DOSSIER_DATA, DOSSIER_FIGURES  # noqa: E402

SCRIPT_VERSION = "1.0.1"

DOSSIER_RUNS_07 = DOSSIER_DATA / "07_dynamique_lente" / "runs"
DOSSIER_SORTIE = DOSSIER_DATA / "09_analyse_so3"
DOSSIER_FIG = DOSSIER_FIGURES / "09_analyse_so3"

OBSERVABLES = {
    "nombre_effectif": "nombre effectif de lignes (e^H des flux)",
    "taux_maximal": "taux de charge maximal",
    "charge_relative": "demande / capacité de génération",
}
# Séries dont on mesure seulement les égalités (riches en zéros exacts).
SERIES_DIAGNOSTIC_SEUL = ("fraction_delestee", "n_lignes_tombees")
CLES_LUES = ("jour", "nombre_effectif", "taux_maximal", "demande_totale",
             "capacite_totale_generateurs", "fraction_delestee", "n_lignes_tombees")
TAILLE_BLOCS_VARIANCE = 1000
SEUIL_Z = 2.0


# ===========================================================================
# Utilitaires (mêmes conventions que 08_analyse_so2.py)
# ===========================================================================

def maintenant_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def nettoyer_id(texte: str) -> str:
    """Identifiant sûr pour un nom de dossier (même règle que 07 et 08)."""
    texte = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(texte).strip()).strip("-_.")
    if not texte:
        raise ValueError("Identifiant vide ou invalide.")
    return texte


def graine_stable(*composants: Any) -> int:
    """Graine reproductible entre machines (SHA-256, jamais `hash()`)."""
    texte = "|".join(str(x) for x in composants)
    return int.from_bytes(hashlib.sha256(texte.encode()).digest()[:8], "little")


def jeton_G(G: float) -> str:
    return f"{float(G):.4f}".replace(".", "p")


def nom_cas(n_noeuds: int, G: float) -> str:
    return f"N{int(n_noeuds)}_G{jeton_G(G)}"


def sha256_fichier(chemin: Path) -> str:
    h = hashlib.sha256()
    with chemin.open("rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


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
    """Enregistre une figure ; une figure ouverte ailleurs (Windows) n'arrête pas le script."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(chemin, dpi=160, bbox_inches="tight")
    except PermissionError:
        print(f"  ATTENTION : {chemin.name} est ouvert ailleurs, non mis à jour.")
    plt.close(fig)


def _terminer_executor(executor: ProcessPoolExecutor) -> None:
    """Arrêt immédiat des workers (même méthode que 07_dynamique_lente.py)."""
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


# ===========================================================================
# Lecture du run
# ===========================================================================

def charger_run(run_id: str) -> tuple[Path, dict]:
    run_dir = DOSSIER_RUNS_07 / nettoyer_id(run_id)
    meta = run_dir / "metadata.json"
    if not meta.exists():
        raise SystemExit(f"Run introuvable : {meta}")
    config = json.loads(meta.read_text("utf-8"))["config"]
    for cle in ("tailles", "G", "jours", "transitoire"):
        if cle not in config:
            raise SystemExit(f"metadata.json incomplet : clé '{cle}' absente.")
    return run_dir, config


def lire_cas(chemin: Path, jours_attendus: int, transitoire: int) -> dict[str, np.ndarray] | str:
    """
    Séries d'un cas après le transitoire, ou un message si le cas est inutilisable.

    La charge relative est dérivée ici : demande totale / capacité totale des
    générateurs, jour par jour.
    """
    if not chemin.exists():
        return "fichier absent"
    with np.load(chemin, allow_pickle=False) as d:
        manquantes = [c for c in CLES_LUES if c not in d.files]
        if manquantes:
            return f"séries absentes : {', '.join(manquantes)}"
        brutes = {c: np.asarray(d[c]) for c in CLES_LUES}
    if brutes["jour"].size < jours_attendus:
        return f"incomplet ({brutes['jour'].size} / {jours_attendus} jours)"
    series = {c: brutes[c][transitoire:].astype(float) for c in CLES_LUES if c != "jour"}
    capacite = series.pop("capacite_totale_generateurs")
    demande = series.pop("demande_totale")
    with np.errstate(divide="ignore", invalid="ignore"):
        series["charge_relative"] = np.where(capacite > 0, demande / capacite, np.nan)
    return series


# ===========================================================================
# Une tâche : (cas, observable)
# ===========================================================================

def parametres_mesure(args: argparse.Namespace) -> dict:
    """Paramètres qui déterminent le résultat d'une tâche (et sa reprise)."""
    return {"echelles": [int(e) for e in args.echelles],
            "dimensions": [int(d) for d in args.dimensions],
            "m": int(args.m), "r": float(args.r),
            "melanges": int(args.melanges), "iaaft": int(args.iaaft),
            "iterations_iaaft": int(args.iterations_iaaft),
            "graine": int(args.graine), "script_version": SCRIPT_VERSION}


def empreinte_tache(parametres: dict, sha_entree: str, transitoire: int) -> str:
    texte = json.dumps({"p": parametres, "sha": sha_entree, "t": transitoire}, sort_keys=True)
    return hashlib.sha256(texte.encode()).hexdigest()


def arguments_tache(cas: str, observable: str, parametres: dict) -> dict:
    """
    Arguments de `analyse_entropie.analyser_avec_references` pour une tâche.

    Les graines dérivent de (graine, cas, observable, référence) : elles ne
    dépendent pas de l'ordre d'exécution ni du nombre de processus.
    """
    g = parametres["graine"]
    return {"echelles": parametres["echelles"], "dimensions": parametres["dimensions"],
            "m": parametres["m"], "r": parametres["r"],
            "n_melanges": parametres["melanges"], "n_iaaft": parametres["iaaft"],
            "iterations_iaaft": parametres["iterations_iaaft"],
            "graine_melange": graine_stable(g, cas, observable, "melange"),
            "graine_iaaft": graine_stable(g, cas, observable, "iaaft"),
            "taille_blocs": TAILLE_BLOCS_VARIANCE}


def sauver_tache(chemin: Path, resultat: dict, empreinte: str) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tmp = chemin.with_name(chemin.stem + ".tmp.npz")
    np.savez(tmp, empreinte=np.array(empreinte),
             **{k: np.asarray(v) for k, v in resultat.items()})
    os.replace(tmp, chemin)


def lire_tache(chemin: Path, empreinte: str) -> dict | None:
    """Résultat d'une tâche déjà calculée avec les mêmes paramètres, sinon None."""
    if not chemin.exists():
        return None
    with np.load(chemin, allow_pickle=False) as d:
        if str(d["empreinte"]) != empreinte:
            return None
        res = {k: d[k] for k in d.files if k != "empreinte"}
    for k in ("moyenne", "ecart_type", "autocorr_1", "autocorr_10",
              "rapport_variance_blocs", "secondes"):
        res[k] = float(res[k])
    res["n_points"] = int(res["n_points"])
    return res


# ===========================================================================
# Tableaux de sortie
# ===========================================================================

def lignes_entropie(n: int, G: float, observable: str, res: dict, parametres: dict) -> list[dict]:
    lignes = []
    mesures = [("mse", res["mse"], res["mse_melange"], res["mse_iaaft"])]
    for i, d in enumerate(parametres["dimensions"]):
        mesures.append((f"pe_d{d}", res["pe"][i], res["pe_melange"][:, i], res["pe_iaaft"][:, i]))
    for nom, valeurs, mel, iaa in mesures:
        rm, ri = resume_substituts(valeurs, mel), resume_substituts(valeurs, iaa)
        for j, s in enumerate(parametres["echelles"]):
            lignes.append({
                "n_noeuds": n, "G": G, "observable": observable, "mesure": nom,
                "echelle": s, "valeur": float(valeurs[j]),
                "melange_moyenne": float(rm["moyenne"][j]),
                "melange_ecart_type": float(rm["ecart_type"][j]),
                "melange_bas": float(rm["bas"][j]), "melange_haut": float(rm["haut"][j]),
                "z_melange": float(rm["z"][j]),
                "iaaft_moyenne": float(ri["moyenne"][j]),
                "iaaft_ecart_type": float(ri["ecart_type"][j]),
                "iaaft_bas": float(ri["bas"][j]), "iaaft_haut": float(ri["haut"][j]),
                "z_iaaft": float(ri["z"][j]),
            })
    return lignes


def _max_abs(z: np.ndarray, echelles: list[int]) -> tuple[float, float]:
    """|z| maximal (signé) et son échelle ; NaN si aucune valeur finie."""
    z = np.asarray(z, dtype=float)
    if not np.any(np.isfinite(z)):
        return float("nan"), float("nan")
    j = int(np.nanargmax(np.abs(z)))
    return float(z[j]), float(echelles[j])


def ligne_resume(n: int, G: float, observable: str, res: dict, parametres: dict) -> dict:
    ech = parametres["echelles"]
    zm = resume_substituts(res["mse"], res["mse_melange"])["z"]
    zi = resume_substituts(res["mse"], res["mse_iaaft"])["z"]
    zm_max, sm = _max_abs(zm, ech)
    zi_max, si = _max_abs(zi, ech)
    return {
        "n_noeuds": n, "G": G, "observable": observable, "n_points": res["n_points"],
        "moyenne": res["moyenne"], "ecart_type": res["ecart_type"],
        "autocorr_1": res["autocorr_1"], "autocorr_10": res["autocorr_10"],
        f"rapport_variance_blocs_{TAILLE_BLOCS_VARIANCE}": res["rapport_variance_blocs"],
        "ecart_spectral_iaaft_max": float(np.max(res["ecart_spectral_iaaft"])),
        "mse_z_melange_max": zm_max, "mse_echelle_z_melange_max": sm,
        "mse_z_iaaft_max": zi_max, "mse_echelle_z_iaaft_max": si,
        "mse_echelles_hors_iaaft": int(np.sum(np.abs(zi[np.isfinite(zi)]) > SEUIL_Z)),
        "secondes": res["secondes"],
    }


# ===========================================================================
# Figures
# ===========================================================================

def _couleurs(valeurs_G: list[float]) -> dict[float, Any]:
    cmap = plt.get_cmap("viridis")
    n = max(len(valeurs_G) - 1, 1)
    return {G: cmap(i / n) for i, G in enumerate(sorted(valeurs_G))}


def figure_mse(fig_dir: Path, observable: str, resultats: dict, echelles: list[int]) -> None:
    """Une ligne de panneaux par taille, un panneau par G : original et deux bandes."""
    tailles = sorted({n for (n, _, o) in resultats if o == observable})
    Gs = sorted({G for (_, G, o) in resultats if o == observable})
    if not tailles:
        return
    fig, axes = plt.subplots(len(tailles), len(Gs), figsize=(2.6 * len(Gs), 2.4 * len(tailles)),
                             sharex=True, sharey=True, squeeze=False)
    s = np.asarray(echelles, dtype=float)
    for i, n in enumerate(tailles):
        for j, G in enumerate(Gs):
            ax = axes[i][j]
            res = resultats.get((n, G, observable))
            if res is None:
                ax.set_axis_off()
                continue
            for cle, couleur, nom in (("mse_melange", "0.6", "mélanges"),
                                      ("mse_iaaft", "tab:orange", "IAAFT")):
                r = resume_substituts(res["mse"], res[cle])
                ax.fill_between(s, r["bas"], r["haut"], color=couleur, alpha=0.35,
                                linewidth=0, label=f"{nom} (95 %)")
            ax.plot(s, res["mse"], "o-", color="tab:blue", markersize=3, label="original")
            ax.set_xscale("log")
            ax.set_title(f"N = {n}, G = {G:g}", fontsize=8)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel("SampEn", fontsize=8)
            if i == len(tailles) - 1:
                ax.set_xlabel("échelle (jours)", fontsize=8)
    axes[0][0].legend(fontsize=6, loc="upper right")
    fig.suptitle(f"MSE : {OBSERVABLES.get(observable, observable)}", fontsize=10)
    fig.tight_layout()
    sauvegarder(fig, fig_dir / f"mse_{observable}.png")


def figure_z(fig_dir: Path, lignes: list[dict], mesure: str, reference: str) -> None:
    """Écart réduit selon l'échelle : une colonne par observable, une ligne par taille."""
    sel = [l for l in lignes if l["mesure"] == mesure]
    if not sel:
        return
    observables = [o for o in OBSERVABLES if any(l["observable"] == o for l in sel)]
    tailles = sorted({l["n_noeuds"] for l in sel})
    couleurs = _couleurs(sorted({l["G"] for l in sel}))
    fig, axes = plt.subplots(len(tailles), len(observables),
                             figsize=(4.2 * len(observables), 2.8 * len(tailles)),
                             sharex=True, squeeze=False)
    cle = f"z_{reference}"
    for i, n in enumerate(tailles):
        for j, o in enumerate(observables):
            ax = axes[i][j]
            for G, couleur in couleurs.items():
                pts = sorted((l["echelle"], l[cle]) for l in sel
                             if l["n_noeuds"] == n and l["observable"] == o and l["G"] == G)
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=couleur,
                            markersize=3)
            for niveau in (-SEUIL_Z, SEUIL_Z):
                ax.axhline(niveau, color="0.4", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="0.7", linewidth=0.6)
            ax.set_xscale("log")
            ax.set_title(f"{o}, N = {n}", fontsize=8)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(f"z ({reference})", fontsize=8)
            if i == len(tailles) - 1:
                ax.set_xlabel("échelle (jours)", fontsize=8)
    # Légende commune : toutes les valeurs de G, même si une taille en manque.
    poignees = [plt.Line2D([], [], color=c, marker="o", markersize=3, label=f"G = {G:g}")
                for G, c in couleurs.items()]
    fig.legend(handles=poignees, fontsize=7, ncol=len(poignees), loc="lower center",
               bbox_to_anchor=(0.5, -0.04))
    nom_ref = "mélanges" if reference == "melange" else "substituts IAAFT"
    fig.suptitle(f"{mesure} : écart réduit à la référence par {nom_ref} (tirets : ± {SEUIL_Z:g})",
                 fontsize=10)
    fig.tight_layout()
    sauvegarder(fig, fig_dir / f"z_{mesure}_{reference}.png")


# ===========================================================================
# Ligne de commande
# ===========================================================================

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="identifiant du run de 07_dynamique_lente.py")
    p.add_argument("--sortie", default=None,
                   help="identifiant du dossier de sortie (défaut : celui du run)")
    p.add_argument("--observables", nargs="+", default=list(OBSERVABLES),
                   choices=list(OBSERVABLES), help="observables analysés (défaut : les trois)")
    p.add_argument("--melanges", type=int, default=20,
                   help="mélanges par série (référence sans ordre, défaut 20)")
    p.add_argument("--iaaft", type=int, default=20,
                   help="substituts IAAFT par série (référence au spectre conservé, défaut 20)")
    p.add_argument("--iterations-iaaft", type=int, default=100,
                   help="passages maximaux de l'algorithme IAAFT (défaut 100)")
    p.add_argument("--echelles", nargs="+", type=int, default=list(ECHELLES_DEFAUT),
                   help="échelles de granularisation en jours (défaut 1 2 5 10 20 50 100 200)")
    p.add_argument("--dimensions", nargs="+", type=int, default=list(DIMENSIONS_DEFAUT),
                   help="dimensions de la PE (défaut 3 4 5 6)")
    p.add_argument("--m", type=int, default=2, help="longueur des motifs de SampEn (défaut 2)")
    p.add_argument("--r", type=float, default=0.15,
                   help="tolérance de SampEn en fraction de σ (défaut 0.15)")
    p.add_argument("--dimension-figure", type=int, default=4,
                   help="dimension de PE tracée dans les figures z (défaut 4)")
    p.add_argument("--graine", type=int, default=20261005)
    p.add_argument("--transitoire", type=int, default=None,
                   help="jours écartés (défaut : valeur du run)")
    p.add_argument("--workers", type=int, default=1, help="processus parallèles (défaut 1)")
    p.add_argument("--resume", action="store_true",
                   help="réutiliser les tâches déjà calculées avec les mêmes paramètres")
    p.add_argument("--sans-figures", action="store_true", help="ne tracer aucune figure")
    return p


def valider_arguments(args: argparse.Namespace, config: dict) -> int:
    if args.melanges < 2 or args.iaaft < 2:
        raise SystemExit("--melanges et --iaaft doivent être ≥ 2 (écart-type des références).")
    if args.iterations_iaaft < 1:
        raise SystemExit("--iterations-iaaft doit être ≥ 1.")
    if args.m < 1:
        raise SystemExit("--m doit être ≥ 1.")
    if not (np.isfinite(args.r) and args.r > 0):
        raise SystemExit("--r doit être > 0.")
    if args.workers < 1:
        raise SystemExit("--workers doit être ≥ 1.")
    if any(e < 1 for e in args.echelles) or any(b <= a for a, b in zip(args.echelles, args.echelles[1:])):
        raise SystemExit("--echelles doit être une liste d'entiers ≥ 1 strictement croissante.")
    if any(d < 2 for d in args.dimensions) or len(set(args.dimensions)) != len(args.dimensions):
        raise SystemExit("--dimensions doit être une liste d'entiers ≥ 2 sans doublon.")
    if args.dimension_figure not in args.dimensions:
        raise SystemExit("--dimension-figure doit faire partie de --dimensions.")
    transitoire = int(config["transitoire"] if args.transitoire is None else args.transitoire)
    if not 0 <= transitoire < int(config["jours"]) - 4 * 365:
        raise SystemExit("--transitoire doit laisser au moins 4 ans de série.")
    return transitoire


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    run_dir, config = charger_run(args.run)
    transitoire = valider_arguments(args, config)
    sortie = nettoyer_id(args.sortie or args.run)
    dossier = DOSSIER_SORTIE / sortie
    fig_dir = DOSSIER_FIG / sortie
    parametres = parametres_mesure(args)

    cas = [(int(n), float(G)) for n in config["tailles"] for G in config["G"]]
    print(f"SO3 sur {run_dir.name} : {len(cas)} cas × {len(args.observables)} observables, "
          f"transitoire {transitoire} j, {args.melanges} mélanges et {args.iaaft} IAAFT par série, "
          f"{args.workers} processus.")

    egalites, sautes, empreintes = [], {}, {}
    taches: list[tuple[int, float, str, np.ndarray, str]] = []
    for n, G in cas:
        chemin = run_dir / f"cas_{nom_cas(n, G)}.npz"
        series = lire_cas(chemin, int(config["jours"]), transitoire)
        if isinstance(series, str):
            sautes[nom_cas(n, G)] = series
            print(f"  {nom_cas(n, G)} : sauté ({series})")
            continue
        empreintes[chemin.name] = sha256_fichier(chemin)
        for cle in (*SERIES_DIAGNOSTIC_SEUL, *OBSERVABLES):
            x = series[cle]
            fini = np.isfinite(x)
            diag = diagnostic_egalites(x[fini]) if fini.any() else {
                "fraction_zeros": np.nan, "fraction_egalites_consecutives": np.nan,
                "fraction_valeurs_repetees": np.nan}
            egalites.append({"n_noeuds": n, "G": G, "serie": cle,
                             "entropie_calculee": cle in args.observables,
                             "fraction_nan": float(np.mean(~fini)), **diag})
        for o in args.observables:
            x = series[o]
            if not np.all(np.isfinite(x)):
                sautes[f"{nom_cas(n, G)}:{o}"] = f"{int(np.sum(~np.isfinite(x)))} valeurs NaN"
                print(f"  {nom_cas(n, G)} {o} : sauté (valeurs NaN)")
                continue
            taches.append((n, G, o, x, empreinte_tache(parametres, empreintes[chemin.name],
                                                       transitoire)))

    if not taches:
        raise SystemExit("Aucune tâche analysable.")

    dossier_taches = dossier / "taches"
    resultats: dict[tuple[int, float, str], dict] = {}
    a_calculer = []
    for n, G, o, x, emp in taches:
        chemin = dossier_taches / f"{nom_cas(n, G)}__{o}.npz"
        deja = lire_tache(chemin, emp) if args.resume else None
        if deja is not None:
            resultats[(n, G, o)] = deja
        else:
            a_calculer.append((n, G, o, x, emp, chemin))
    if args.resume:
        print(f"  reprise : {len(resultats)} tâches réutilisées, {len(a_calculer)} à calculer.")

    def consigner(i: int, n: int, G: float, o: str, res: dict) -> None:
        zi = resume_substituts(res["mse"], res["mse_iaaft"])["z"]
        zm = resume_substituts(res["mse"], res["mse_melange"])["z"]
        j = len(parametres["echelles"]) - 1
        while j > 0 and not np.isfinite(res["mse"][j]):
            j -= 1
        print(f"  [{i}/{len(a_calculer)}] {nom_cas(n, G)} {o:15s} | MSE à "
              f"{parametres['echelles'][j]} j : {res['mse'][j]:.3f} "
              f"(z mélange {zm[j]:+.1f}, z IAAFT {zi[j]:+.1f}) | "
              f"blocs {res['rapport_variance_blocs']:.1f} | {res['secondes']:.0f} s")

    if args.workers == 1:
        for i, (n, G, o, x, emp, chemin) in enumerate(a_calculer, 1):
            res = analyser_avec_references(x, **arguments_tache(nom_cas(n, G), o, parametres))
            sauver_tache(chemin, res, emp)
            resultats[(n, G, o)] = lire_tache(chemin, emp)
            consigner(i, n, G, o, resultats[(n, G, o)])
    elif a_calculer:
        executor = ProcessPoolExecutor(max_workers=args.workers, initializer=ignorer_interruption)
        try:
            futurs = {executor.submit(analyser_avec_references, x,
                                      **arguments_tache(nom_cas(n, G), o, parametres)):
                      (n, G, o, emp, chemin) for n, G, o, x, emp, chemin in a_calculer}
            for i, futur in enumerate(as_completed(futurs), 1):
                n, G, o, emp, chemin = futurs[futur]
                sauver_tache(chemin, futur.result(), emp)
                resultats[(n, G, o)] = lire_tache(chemin, emp)
                consigner(i, n, G, o, resultats[(n, G, o)])
        except KeyboardInterrupt:
            print("Interruption : les tâches terminées sont conservées ; relancer avec --resume.")
            _terminer_executor(executor)
            sys.exit(130)
        except BaseException:
            _terminer_executor(executor)
            raise
        executor.shutdown(wait=True)

    ordre = sorted(resultats, key=lambda k: (k[0], k[1], list(OBSERVABLES).index(k[2])))
    lignes_e = [l for k in ordre for l in lignes_entropie(*k, resultats[k], parametres)]
    resumes = [ligne_resume(*k, resultats[k], parametres) for k in ordre]
    ecrire_csv(dossier / "egalites.csv", egalites)
    ecrire_csv(dossier / "resume.csv", resumes)
    ecrire_csv(dossier / "entropie.csv", lignes_e)

    if not args.sans_figures:
        for o in args.observables:
            figure_mse(fig_dir, o, resultats, parametres["echelles"])
        for mesure in ("mse", f"pe_d{args.dimension_figure}"):
            for reference in ("melange", "iaaft"):
                figure_z(fig_dir, lignes_e, mesure, reference)

    ecrire_json(dossier / "metadata.json", {
        "script": "09_analyse_so3.py", "script_version": SCRIPT_VERSION,
        "created_at": maintenant_iso(), "run_source": run_dir.name, "config_run": config,
        "transitoire": transitoire, "parametres": parametres,
        "observables": args.observables, "taille_blocs_variance": TAILLE_BLOCS_VARIANCE,
        "seuil_z": SEUIL_Z, "methode_sampen": "arbre",
        "taches": [f"{nom_cas(n, G)}:{o}" for n, G, o in ordre],
        "cas_sautes": sautes, "sha256_entrees": empreintes,
    })
    print(f"\nRésultats : {dossier}\nFigures   : {fig_dir}")
    if sautes:
        print(f"{len(sautes)} éléments sautés : " + ", ".join(sautes))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
