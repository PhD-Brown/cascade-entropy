"""
SO2, mission 3 : analyse des séries de la dynamique lente (Carreras et al. 2004).

Ce script lit un run produit par `07_dynamique_lente.py` (aucune simulation ici)
et calcule, pour chaque cas (taille d'arbre, G), les observables comparables au
papier, avec leurs contrôles :

- Exposant de Hurst par R/S sur quatre plages d'échelles :
    court        10 à 365 jours      (« a few days and a few years », Fig. 3)
    long         600 à 10^5 jours    (plage de la Fig. 4)
    avant_cycle  P/3 à P             (P = ln μ / ln λ_jour : période prédite du
    apres_cycle  P à 10^5            cycle de renforcement des lignes)
  Chaque H est accompagné d'une RÉFÉRENCE PAR MÉLANGE : la même mesure sur
  `--melanges` copies de la série dont les jours ont été permutés au hasard.
  Le mélange garde la distribution mais détruit la mémoire : un H dans
  l'intervalle de mélange n'indique AUCUNE mémoire, quel que soit son écart à
  0.5 (biais de R/S aux petites échelles, Anis et Lloyd 1976).
- Cycle des lignes : pic du périodogramme des lignes tombées (50 à
  20 000 j), comparé à P. Cycle de la génération : intervalle médian entre
  deux jours de mise à niveau des générateurs, comparé à
  (k / N_G) / ((1 + G·g) ln λ_jour). Ce second cycle se lit directement sur
  la série `n_mises_a_niveau` : son pic spectral dans le délestage est trop
  bruité pour être fiable.
- Fig. 7 : puissance servie relative et lignes par blackout selon G.
- Fig. 8 : distribution du nombre de lignes tombées par blackout.
- Fig. 9 : part des blackouts de plus de `--seuil-lignes` lignes (15 dans le papier).
- Table I : queue du délestage normalisé, sur DEUX représentations.
  Le texte de Carreras 2004 (Sec. III) calcule la Table I sur la fréquence
  cumulée relative (fonction de rang) et définit l'« étendue » comme le
  rapport entre le plus grand et le plus petit délestage décrits par la loi
  de puissance ; sa légende dit pourtant « PDF ». On ajuste donc :
    - la fréquence cumulée P(X ≥ x) (colonnes cumul_*, v1.1.0), comparable
      aux indices publiés (≈ −0.55) ;
    - la densité (colonnes queue_*), dont la pente vaut la pente cumulée − 1.
  Dans les deux cas : plus longue plage contiguë, en échelle logarithmique,
  où la relation log-log est linéaire avec R² ≥ 0.98 et de pente négative,
  inférieure à −0.1 pour la fréquence cumulée afin d'écarter son plateau
  initial à ≈ 1 (`analyse_series.plage_loi_puissance`). C'est NOTRE opérationnalisation ;
  une étendue petite signale l'absence de vraie loi de puissance.

Le transitoire écarté est celui du run (`transitoire` de metadata.json), sauf
si `--transitoire` le remplace. Seuls les cas complets sont analysés ; les
autres sont signalés et sautés.

Ce script appartient au fil B : il n'importe aucun module du réseau et ne
communique avec le fil A que par les fichiers .npz du run.

Sorties
-------
data/08_analyse_so2/<SORTIE>/
    metadata.json             paramètres, run source, empreinte SHA-256 des .npz
    cas.csv                   une ligne par cas : fréquences, Fig. 7 et 9,
                              périodes, queue (Table I)
    hurst.csv                 une ligne par (cas, série, plage) : H, médiane et
                              intervalle de mélange, écart réduit z
    lignes_par_blackout.csv   distribution du nombre de lignes par blackout
figures/08_analyse_so2/<SORTIE>/
    fig03_rs_<cas>.png        courbes R/S et enveloppe des mélanges
    fig04_hurst.png           H(G) court et long, délestage et lignes
    fig04_cassure.png         H des lignes avant et après le cycle
    fig07_servie_lignes.png   puissance servie et lignes par blackout
    fig08_lignes_N<n>.png     distribution des lignes par blackout
    fig09_grands_blackouts.png
    fig_cycle.png             période mesurée et prédite
    tableI_queue_N<n>.png     densité du délestage et plage de loi de puissance

Exemples (PowerShell)
---------------------
    python scripts\\08_analyse_so2.py --run so2-G-scan
    python scripts\\08_analyse_so2.py --run so2-controle-mu102 --melanges 30
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
    echelles_log,
    ccdf_logarithmique,
    pdf_logarithmique,
    pentes_depuis_courbe,
    periode_dominante,
    plage_loi_puissance,
    resume_reference,
    rs_vectorise,
)
from cascade_entropy.chemins import DOSSIER_DATA, DOSSIER_FIGURES  # noqa: E402

SCRIPT_VERSION = "1.1.0"

DOSSIER_RUNS_07 = DOSSIER_DATA / "07_dynamique_lente" / "runs"
DOSSIER_SORTIE = DOSSIER_DATA / "08_analyse_so2"
DOSSIER_FIG = DOSSIER_FIGURES / "08_analyse_so2"

# Même convention que 07_dynamique_lente.SEUIL_BLACKOUT.
SEUIL_BLACKOUT = 1e-9
# Fenêtre de la Fig. 2 de Carreras (« per 300 days »).
FENETRE = 300
# Séries analysées : (clé dans le .npz, libellé).
SERIES = (("fraction_delestee", "délestage normalisé"),
          ("n_lignes_tombees", "lignes tombées"))
# Bandes de recherche du périodogramme (jours).
BANDE_LIGNES = (50.0, 20_000.0)
# Valeurs publiées, tracées comme repères (Carreras et al. 2004).
H_COURT_PUBLIE = (0.53, 0.57)          # 0.55 ± 0.02, Fig. 3
H_LIGNES_LONG_PUBLIE = (0.2, 0.4)      # Fig. 4b, texte p. 647
RATIO_GRANDS_PUBLIE = {"faible G": 0.001, "G > 1": 0.007}  # Fig. 9
# Table I (Carreras 2004) : pente de la fréquence cumulée par taille.
PENTE_CUMUL_PUBLIEE = {46: -0.56, 94: -0.51, 190: -0.55, 382: -0.58}
ETENDUE_PUBLIEE = {46: 4, 94: 8, 190: 13, 382: 31}
# Nombre minimal de valeurs au-dessus d'un point de la fréquence cumulée pour
# qu'il entre dans l'ajustement (les derniers points reposent sur 1 à 10 valeurs).
EFFECTIF_MIN_CUMUL = 20
# La fréquence cumulée vaut ≈ 1 sur tout le début de la distribution : ce
# plateau est trivialement « linéaire » et ne dit rien de la queue. On exige
# donc une pente cumulée < −0.1 (une densité qui décroît au moins en x^-1.1).
PENTE_MAX_CUMUL = -0.1


# ===========================================================================
# Utilitaires
# ===========================================================================

def maintenant_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def nettoyer_id(texte: str) -> str:
    """Identifiant de run sûr pour un nom de dossier (même règle que 07)."""
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


# ===========================================================================
# Paramètres dérivés du run
# ===========================================================================

def lambda_jour(config: dict) -> float:
    """Facteur de croissance quotidien de la demande (λ_annuel^(1/365))."""
    return float(config["lambda_annuel"]) ** (1.0 / 365.0)


def periode_cycle_lignes(config: dict) -> float:
    """
    Période prédite du cycle de renforcement des lignes : ln μ / ln λ_jour.

    Une ligne renforcée d'un facteur μ redevient saturée quand la demande a été
    multipliée par μ, soit après ln μ / ln λ_jour jours.
    """
    return math.log(float(config["mu"])) / math.log(lambda_jour(config))


def periode_cycle_generation(config: dict, G: float, n_generateurs: int) -> float:
    """
    Période prédite des mises à niveau de la génération.

    Chaque ajout vaut k / N_G de la demande et relève la marge d'autant ; la
    marge, collée au seuil G·g, baisse de (1 + G·g) ln λ_jour par jour.
    """
    marge = float(G) * float(config["g"])
    return (float(config["k"]) / n_generateurs) / ((1.0 + marge) * math.log(lambda_jour(config)))


def plages_hurst(config: dict) -> dict[str, tuple[float, float]]:
    P = periode_cycle_lignes(config)
    return {
        "court": (10.0, 365.0),
        "long": (600.0, 1e5),
        "avant_cycle": (P / 3.0, P),
        "apres_cycle": (P, 1e5),
    }


# ===========================================================================
# Lecture du run
# ===========================================================================

def charger_run(run_id: str) -> tuple[Path, dict]:
    run_dir = DOSSIER_RUNS_07 / nettoyer_id(run_id)
    meta = run_dir / "metadata.json"
    if not meta.exists():
        raise SystemExit(f"Run introuvable : {meta}")
    config = json.loads(meta.read_text("utf-8"))["config"]
    for cle in ("tailles", "G", "jours", "transitoire", "mu", "lambda_annuel", "k", "g"):
        if cle not in config:
            raise SystemExit(f"metadata.json incomplet : clé '{cle}' absente.")
    return run_dir, config


def lire_cas(chemin: Path, jours_attendus: int) -> dict[str, np.ndarray] | str:
    """Séries d'un cas, ou un message d'explication si le cas est inutilisable."""
    if not chemin.exists():
        return "fichier absent"
    cles = ("jour", "demande_moyenne", "delestage_total", "fraction_delestee",
            "n_lignes_tombees", "n_mises_a_niveau", "etat_puissance_max")
    with np.load(chemin, allow_pickle=False) as d:
        manquantes = [c for c in cles if c not in d.files]
        if manquantes:
            return f"séries absentes : {', '.join(manquantes)}"
        donnees = {c: d[c] for c in cles}
    if donnees["jour"].size < jours_attendus:
        return f"incomplet ({donnees['jour'].size} / {jours_attendus} jours)"
    return donnees


# ===========================================================================
# Analyse d'un cas
# ===========================================================================

def analyser_hurst(serie: np.ndarray, plages: dict, n_melanges: int,
                   rng: np.random.Generator) -> dict:
    """R/S de la série, R/S de ses mélanges et pentes sur chaque plage."""
    echelles = echelles_log(serie.size)
    bornes = np.array(list(plages.values()))
    tailles, valeurs = rs_vectorise(serie, echelles)
    pentes = pentes_depuis_courbe(tailles, valeurs, bornes)
    courbes_melange, pentes_melange = [], []
    for _ in range(n_melanges):
        t_m, v_m = rs_vectorise(rng.permutation(serie), echelles)
        if t_m.shape != tailles.shape or np.any(t_m != tailles):
            # Un mélange peut perdre une échelle (blocs tous constants) : on
            # aligne sur les échelles de la série d'origine.
            v_m = np.interp(tailles, t_m, v_m)
            t_m = tailles
        courbes_melange.append(v_m)
        pentes_melange.append(pentes_depuis_courbe(t_m, v_m, bornes))
    pentes_melange = np.array(pentes_melange)
    med, bas, haut = resume_reference(pentes_melange)
    return {"tailles": tailles, "valeurs": valeurs, "pentes": pentes,
            "courbes_melange": np.array(courbes_melange),
            "pentes_melange": pentes_melange, "mediane": med, "bas": bas, "haut": haut}


def analyser_cas(donnees: dict, config: dict, n_noeuds: int, G: float,
                 transitoire: int, args: argparse.Namespace) -> dict:
    st = slice(transitoire, None)
    demande = donnees["demande_moyenne"][st]
    fraction = donnees["fraction_delestee"][st].astype(float)
    lignes = donnees["n_lignes_tombees"][st].astype(float)
    blackout = donnees["delestage_total"][st] > SEUIL_BLACKOUT * demande
    n_b = int(blackout.sum())
    lignes_b = lignes[blackout]
    n_generateurs = int(donnees["etat_puissance_max"].size)
    plages = plages_hurst(config)

    hurst = {}
    for cle, _ in SERIES:
        rng = np.random.default_rng(graine_stable("SO2-analyse", args.graine,
                                                  n_noeuds, jeton_G(G), cle))
        hurst[cle] = analyser_hurst(donnees[cle][st].astype(float), plages,
                                    args.melanges, rng)

    # La bande est bornée au quart de la série : au moins quatre périodes.
    P_lignes, f_lignes = periode_dominante(
        lignes, BANDE_LIGNES[0], min(BANDE_LIGNES[1], lignes.size / 4))
    jours_maj = np.flatnonzero(donnees["n_mises_a_niveau"][st] > 0)
    P_gen = float(np.median(np.diff(jours_maj))) if jours_maj.size >= 3 else float("nan")
    if np.count_nonzero(fraction > 0) >= 2 and np.unique(fraction[fraction > 0]).size >= 2:
        centres, densite, effectifs = pdf_logarithmique(fraction, args.classes)
        # Une queue décroît : les plages de pente positive sont exclues.
        queue = plage_loi_puissance(centres, densite, effectifs, r2_min=args.r2_min,
                                    pente_max=0.0)
        grille, cumul, au_dessus = ccdf_logarithmique(fraction, args.classes + 5)
        queue_cumul = plage_loi_puissance(grille, cumul, au_dessus, r2_min=args.r2_min,
                                          effectif_min=EFFECTIF_MIN_CUMUL,
                                          pente_max=PENTE_MAX_CUMUL)
    else:
        centres = densite = effectifs = np.empty(0)
        grille = cumul = au_dessus = np.empty(0)
        queue = plage_loi_puissance(np.ones(3), np.ones(3), np.zeros(3))  # tout NaN
        queue_cumul = dict(queue)

    valeurs, effectifs_l = np.unique(lignes_b.astype(int), return_counts=True)
    resume = {
        "n_noeuds": n_noeuds,
        "G": G,
        "jours_analyses": int(fraction.size),
        "transitoire": transitoire,
        "n_blackouts": n_b,
        "blackouts_par_300j": float(blackout.mean() * FENETRE),
        "puissance_servie_relative": float(np.mean(1.0 - fraction)),
        "lignes_par_blackout": float(lignes_b.mean()) if n_b else float("nan"),
        "part_blackouts_sans_ligne": float(np.mean(lignes_b == 0)) if n_b else float("nan"),
        f"part_blackouts_plus_{args.seuil_lignes}_lignes": (
            float(np.mean(lignes_b > args.seuil_lignes)) if n_b else float("nan")),
        "periode_lignes_mesuree": P_lignes,
        "periode_lignes_predite": periode_cycle_lignes(config),
        "part_puissance_cycle_lignes": f_lignes,
        "intervalle_mises_a_niveau_mesure": P_gen,
        "intervalle_mises_a_niveau_predit": periode_cycle_generation(config, G, n_generateurs),
        "n_jours_mise_a_niveau": int(jours_maj.size),
        "queue_pente": queue["pente"],
        "queue_etendue": queue["etendue"],
        "queue_x_debut": queue["x_debut"],
        "queue_x_fin": queue["x_fin"],
        "queue_r2": queue["r2"],
        "queue_n_classes": queue["n_classes"],
        "cumul_pente": queue_cumul["pente"],
        "cumul_etendue": queue_cumul["etendue"],
        "cumul_x_debut": queue_cumul["x_debut"],
        "cumul_x_fin": queue_cumul["x_fin"],
        "cumul_r2": queue_cumul["r2"],
        "cumul_n_points": queue_cumul["n_classes"],
    }
    return {"resume": resume, "hurst": hurst, "plages": plages,
            "distribution": (valeurs, effectifs_l),
            "queue": (centres, densite, effectifs, queue),
            "cumul": (grille, cumul, au_dessus, queue_cumul)}


def lignes_hurst(res: dict) -> list[dict]:
    r = res["resume"]
    sortie = []
    for cle, _ in SERIES:
        h = res["hurst"][cle]
        ecart_type = np.nanstd(h["pentes_melange"], axis=0)
        for i, (nom, (debut, fin)) in enumerate(res["plages"].items()):
            H = h["pentes"][i]
            z = (H - h["mediane"][i]) / ecart_type[i] if ecart_type[i] > 0 else float("nan")
            sortie.append({
                "n_noeuds": r["n_noeuds"], "G": r["G"], "serie": cle, "plage": nom,
                "debut_j": round(debut, 1), "fin_j": round(fin, 1),
                "H": H, "H_melange_mediane": h["mediane"][i],
                "H_melange_bas": h["bas"][i], "H_melange_haut": h["haut"][i],
                "ecart_reduit_z": z,
                "hors_reference": bool(np.isfinite(H) and np.isfinite(h["bas"][i])
                                       and not h["bas"][i] <= H <= h["haut"][i]),
            })
    return sortie


# ===========================================================================
# Figures
# ===========================================================================

def figure_rs(fig_dir: Path, res: dict, args) -> None:
    r = res["resume"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (cle, libelle) in zip(axes, SERIES):
        h = res["hurst"][cle]
        if h["courbes_melange"].size:
            ax.fill_between(h["tailles"], h["courbes_melange"].min(axis=0),
                            h["courbes_melange"].max(axis=0), color="0.8",
                            label=f"{args.melanges} mélanges (min–max)")
        ax.loglog(h["tailles"], h["valeurs"], "o-", markersize=3, label="série")
        for x, style in ((365, ":"), (600, "--"), (r["periode_lignes_predite"], "-.")):
            ax.axvline(x, linestyle=style, color="0.4", linewidth=0.8)
        texte = "\n".join(f"{nom} : H = {h['pentes'][i]:.2f} (mél. {h['mediane'][i]:.2f})"
                          for i, nom in enumerate(res["plages"]))
        ax.text(0.03, 0.97, texte, transform=ax.transAxes, va="top", fontsize=7.5)
        ax.set_title(libelle)
        ax.set_xlabel("échelle n (jours)")
        ax.grid(alpha=0.2, which="both")
    axes[0].set_ylabel("R/S")
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle(f"R/S, {r['n_noeuds']} nœuds, G = {r['G']:g}  "
                 f"(pointillés : 365 j, 600 j, période prédite P)")
    fig.tight_layout()
    sauvegarder(fig, fig_dir / f"fig03_rs_{nom_cas(r['n_noeuds'], r['G'])}.png")


def _par_taille(lignes: list[dict]) -> dict[int, list[dict]]:
    groupes: dict[int, list[dict]] = {}
    for l in lignes:
        groupes.setdefault(int(l["n_noeuds"]), []).append(l)
    for g in groupes.values():
        g.sort(key=lambda l: l["G"])
    return groupes


def figure_hurst(fig_dir: Path, lignes_h: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
    for i, (cle, libelle) in enumerate(SERIES):
        for j, plage in enumerate(("court", "long")):
            ax = axes[i, j]
            choix = [l for l in lignes_h if l["serie"] == cle and l["plage"] == plage]
            for n, groupe in _par_taille(choix).items():
                G = [l["G"] for l in groupe]
                ligne, = ax.plot(G, [l["H"] for l in groupe], "o-", label=f"{n} nœuds")
                ax.fill_between(G, [l["H_melange_bas"] for l in groupe],
                                [l["H_melange_haut"] for l in groupe],
                                color=ligne.get_color(), alpha=0.15)
            if plage == "court":
                ax.axhspan(*H_COURT_PUBLIE, color="k", alpha=0.08, label="publié 0.55 ± 0.02")
            elif cle == "n_lignes_tombees":
                ax.axhspan(*H_LIGNES_LONG_PUBLIE, color="k", alpha=0.08, label="publié 0.2–0.4")
            else:
                ax.axhline(0.5, color="k", linewidth=0.8, linestyle=":",
                           label="publié ≈ 0.5 si G < 1")
            ax.set_title(f"{libelle}, plage {plage}")
            ax.set_xscale("log")
            ax.grid(alpha=0.2)
        axes[i, 0].set_ylabel("H (R/S)")
    for ax in axes[1]:
        ax.set_xlabel("G")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("H(G) — bandes colorées : intervalle à 95 % des mélanges (aucune mémoire)")
    fig.tight_layout()
    sauvegarder(fig, fig_dir / "fig04_hurst.png")


def figure_cassure(fig_dir: Path, lignes_h: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, plage in zip(axes, ("avant_cycle", "apres_cycle")):
        choix = [l for l in lignes_h if l["serie"] == "n_lignes_tombees" and l["plage"] == plage]
        for n, groupe in _par_taille(choix).items():
            G = [l["G"] for l in groupe]
            ligne, = ax.plot(G, [l["H"] for l in groupe], "o-", label=f"{n} nœuds")
            ax.fill_between(G, [l["H_melange_bas"] for l in groupe],
                            [l["H_melange_haut"] for l in groupe],
                            color=ligne.get_color(), alpha=0.15)
        ax.axhline(0.5, color="k", linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_title(f"lignes tombées, {plage.replace('_', ' ')}")
        ax.set_xlabel("G")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("H (R/S)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    sauvegarder(fig, fig_dir / "fig04_cassure.png")


def figure_fig7_fig9(fig_dir: Path, resumes: list[dict], seuil: int) -> None:
    groupes = _par_taille(resumes)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for n, groupe in groupes.items():
        G = [r["G"] for r in groupe]
        axes[0].plot(G, [r["puissance_servie_relative"] for r in groupe], "o-", label=f"{n} nœuds")
        axes[1].plot(G, [r["lignes_par_blackout"] for r in groupe], "o-", label=f"{n} nœuds")
    axes[0].set_ylabel("puissance servie / demande")
    axes[1].set_ylabel("lignes tombées par blackout")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("G")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Analogue de la Fig. 7 (Carreras 2004 : servie faible aux deux extrémités de G)")
    fig.tight_layout()
    sauvegarder(fig, fig_dir / "fig07_servie_lignes.png")

    cle = f"part_blackouts_plus_{seuil}_lignes"
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for n, groupe in groupes.items():
        # Les rapports nuls ne peuvent pas être tracés en échelle log.
        points = [(r["G"], r[cle]) for r in groupe if r[cle] > 0]
        if points:
            ax.plot(*zip(*points), "o-", label=f"{n} nœuds")
    for nom, valeur in RATIO_GRANDS_PUBLIE.items():
        ax.axhline(valeur, linestyle="--", color="0.4", linewidth=0.8)
        # x en coordonnées d'axe (0 à 1), y en données : le texte reste dans le cadre.
        ax.text(0.02, valeur * 1.15, f"publié, {nom}", fontsize=7, color="0.3",
                transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("G")
    ax.set_ylabel(f"blackouts > {seuil} lignes / blackouts")
    ax.grid(alpha=0.2, which="both")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8)
    ax.set_title("Analogue de la Fig. 9")
    fig.tight_layout()
    sauvegarder(fig, fig_dir / "fig09_grands_blackouts.png")


def figure_fig8(fig_dir: Path, resultats: list[dict]) -> None:
    groupes: dict[int, list[dict]] = {}
    for res in resultats:
        groupes.setdefault(res["resume"]["n_noeuds"], []).append(res)
    for n, groupe in groupes.items():
        groupe.sort(key=lambda res: res["resume"]["G"])
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        couleurs = plt.cm.viridis(np.linspace(0, 0.9, len(groupe)))
        for res, c in zip(groupe, couleurs):
            valeurs, effectifs = res["distribution"]
            if effectifs.size:
                ax.plot(valeurs, effectifs / effectifs.sum(), "o-", markersize=3,
                        color=c, label=f"G = {res['resume']['G']:g}")
        ax.set_yscale("log")
        ax.set_xlabel("lignes tombées par blackout")
        ax.set_ylabel("probabilité")
        ax.set_title(f"Analogue de la Fig. 8, {n} nœuds "
                     "(Carreras, 94 nœuds : pic à 4, puis ≈ 17 à G élevé)")
        ax.grid(alpha=0.2, which="both")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        sauvegarder(fig, fig_dir / f"fig08_lignes_N{n}.png")


def figure_cycle(fig_dir: Path, resumes: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for n, groupe in _par_taille(resumes).items():
        G = [r["G"] for r in groupe]
        axes[0].plot(G, [r["periode_lignes_mesuree"] for r in groupe], "o-", label=f"{n} nœuds")
        ligne, = axes[1].plot(G, [r["intervalle_mises_a_niveau_mesure"] for r in groupe],
                              "o-", label=f"{n} nœuds (mesuré)")
        axes[1].plot(G, [r["intervalle_mises_a_niveau_predit"] for r in groupe], "--",
                     color=ligne.get_color(), label=f"{n} nœuds (prédit)")
    if resumes:
        axes[0].axhline(resumes[0]["periode_lignes_predite"], color="k", linestyle="--",
                        linewidth=0.8, label="prédite ln μ / ln λ_jour")
    axes[0].set_title("cycle des lignes (pic du périodogramme)")
    axes[1].set_title("cycle de la génération (intervalle entre mises à niveau)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("G")
        ax.set_ylabel("période (jours)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    sauvegarder(fig, fig_dir / "fig_cycle.png")


def figure_queue(fig_dir: Path, resultats: list[dict]) -> None:
    """Densité et fréquence cumulée du délestage, avec la plage ajustée."""
    groupes: dict[int, list[dict]] = {}
    for res in resultats:
        groupes.setdefault(res["resume"]["n_noeuds"], []).append(res)
    for n, groupe in groupes.items():
        groupe.sort(key=lambda res: res["resume"]["G"])
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
        couleurs = plt.cm.viridis(np.linspace(0, 0.9, len(groupe)))
        for res, c in zip(groupe, couleurs):
            G = res["resume"]["G"]
            for ax, cle in zip(axes, ("queue", "cumul")):
                x, y, _, q = res[cle]
                if x.size == 0:
                    continue
                ax.loglog(x, y, "o", markersize=3, color=c,
                          label=f"G = {G:g}  pente {q['pente']:.2f}"
                                + (f", étendue {q['etendue']:.0f}" if q["n_classes"] else ""))
                if q["n_classes"]:
                    xs = np.array([q["x_debut"], q["x_fin"]])
                    y0 = y[np.argmin(np.abs(x - q["x_debut"]))]
                    ax.loglog(xs, y0 * (xs / xs[0]) ** q["pente"], "-", color=c,
                              linewidth=1.5)
        for ax in axes:
            ax.set_xlabel("délestage / demande")
            ax.grid(alpha=0.2, which="both")
            ax.legend(frameon=False, fontsize=7)
        axes[0].set_ylabel("densité de probabilité")
        axes[0].set_title("densité (pente = pente cumulée − 1)")
        axes[1].set_ylabel("fréquence cumulée P(X ≥ x)")
        publie = PENTE_CUMUL_PUBLIEE.get(n)
        axes[1].set_title("fréquence cumulée"
                          + (f" (Table I : pente {publie}, étendue {ETENDUE_PUBLIEE[n]})"
                             if publie is not None else ""))
        fig.suptitle(f"Queue du délestage, {n} nœuds")
        fig.tight_layout()
        sauvegarder(fig, fig_dir / f"tableI_queue_N{n}.png")


# ===========================================================================
# Programme principal
# ===========================================================================

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="identifiant du run de 07_dynamique_lente.py")
    p.add_argument("--sortie", default=None,
                   help="identifiant du dossier de sortie (défaut : celui du run)")
    p.add_argument("--melanges", type=int, default=50,
                   help="nombre de mélanges par série pour la référence (défaut 50)")
    p.add_argument("--graine", type=int, default=20261005)
    p.add_argument("--transitoire", type=int, default=None,
                   help="jours écartés (défaut : valeur du run)")
    p.add_argument("--seuil-lignes", type=int, default=15,
                   help="seuil des grands blackouts, Fig. 9 (défaut 15)")
    p.add_argument("--classes", type=int, default=25,
                   help="classes logarithmiques de la densité du délestage")
    p.add_argument("--r2-min", type=float, default=0.98,
                   help="R² minimal de la plage de loi de puissance (Table I)")
    p.add_argument("--sans-figures-rs", action="store_true",
                   help="ne pas tracer une figure R/S par cas")
    return p


def valider_arguments(args: argparse.Namespace, config: dict) -> int:
    if args.melanges < 1:
        raise SystemExit("--melanges doit être ≥ 1.")
    if args.seuil_lignes < 0:
        raise SystemExit("--seuil-lignes doit être ≥ 0.")
    if args.classes < 3:
        raise SystemExit("--classes doit être ≥ 3.")
    if not 0 < args.r2_min <= 1:
        raise SystemExit("--r2-min doit être dans ]0, 1].")
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

    cas = [(int(n), float(G)) for n in config["tailles"] for G in config["G"]]
    print(f"Analyse de {run_dir.name} : {len(cas)} cas, transitoire {transitoire} j, "
          f"{args.melanges} mélanges, période prédite des lignes "
          f"{periode_cycle_lignes(config):.0f} j.")

    resultats, sautes, empreintes = [], {}, {}
    for n, G in cas:
        chemin = run_dir / f"cas_{nom_cas(n, G)}.npz"
        donnees = lire_cas(chemin, int(config["jours"]))
        if isinstance(donnees, str):
            sautes[nom_cas(n, G)] = donnees
            print(f"  {nom_cas(n, G)} : sauté ({donnees})")
            continue
        empreintes[chemin.name] = sha256_fichier(chemin)
        res = analyser_cas(donnees, config, n, G, transitoire, args)
        resultats.append(res)
        r = res["resume"]
        hd, hl = res["hurst"]["fraction_delestee"], res["hurst"]["n_lignes_tombees"]
        print(f"  {nom_cas(n, G)} | H court {hd['pentes'][0]:.2f} (mél. {hd['mediane'][0]:.2f})"
              f" | H long délestage {hd['pentes'][1]:.2f} | H lignes avant/après cycle "
              f"{hl['pentes'][2]:.2f}/{hl['pentes'][3]:.2f} | cycle {r['periode_lignes_mesuree']:.0f} j"
              f" | > {args.seuil_lignes} lignes {r[f'part_blackouts_plus_{args.seuil_lignes}_lignes']:.3f}")
        if not args.sans_figures_rs:
            figure_rs(fig_dir, res, args)

    if not resultats:
        raise SystemExit("Aucun cas analysable.")

    resumes = [res["resume"] for res in resultats]
    lignes_h = [l for res in resultats for l in lignes_hurst(res)]
    distribution = [{"n_noeuds": res["resume"]["n_noeuds"], "G": res["resume"]["G"],
                     "n_lignes": int(v), "effectif": int(e),
                     "probabilite": float(e / res["distribution"][1].sum())}
                    for res in resultats for v, e in zip(*res["distribution"])]
    ecrire_csv(dossier / "cas.csv", resumes)
    ecrire_csv(dossier / "hurst.csv", lignes_h)
    ecrire_csv(dossier / "lignes_par_blackout.csv", distribution)

    figure_hurst(fig_dir, lignes_h)
    figure_cassure(fig_dir, lignes_h)
    figure_fig7_fig9(fig_dir, resumes, args.seuil_lignes)
    figure_fig8(fig_dir, resultats)
    figure_cycle(fig_dir, resumes)
    figure_queue(fig_dir, resultats)

    ecrire_json(dossier / "metadata.json", {
        "script": "08_analyse_so2.py", "script_version": SCRIPT_VERSION,
        "created_at": maintenant_iso(), "run_source": run_dir.name,
        "config_run": config, "transitoire": transitoire,
        "parametres": {"melanges": args.melanges, "graine": args.graine,
                       "seuil_lignes": args.seuil_lignes, "classes": args.classes,
                       "r2_min": args.r2_min, "plages_hurst": plages_hurst(config),
                       "bande_lignes": BANDE_LIGNES},
        "cas_analyses": [nom_cas(r["n_noeuds"], r["G"]) for r in resumes],
        "cas_sautes": sautes, "sha256_entrees": empreintes,
    })
    print(f"\nRésultats : {dossier}\nFigures   : {fig_dir}")
    if sautes:
        print(f"{len(sautes)} cas sautés : " + ", ".join(sautes))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
