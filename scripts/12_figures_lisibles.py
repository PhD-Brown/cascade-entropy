"""
Figures lisibles pour le rapport d'étape (PHY-3202, 30 octobre 2026).

Réponse aux points 7 et 10 des commentaires de P. Desrosiers (28 octobre) :

7.  « Montrer les séries temporelles analysées » : deux courts extraits de
    N_flux(t) (le ``nombre_effectif`` du code) pour UNE même taille de réseau
    et DEUX régimes (deux valeurs de G), sur la même fenêtre de jours et avec
    les mêmes axes.
10. « Remplacer les noms de variables comme nombre_effectif et taux_maximal
    par les termes du texte » : les figures de l'écart réduit z de la MSE
    (section 5 et annexe D) sont retracées à partir de ``entropie.csv``, avec
    les termes du rapport et des polices lisibles à la taille imprimée.

Ce script ne calcule AUCUN résultat nouveau et ne lance aucune simulation. Il
lit seulement :

- les fichiers ``cas_<N>_G<...>.npz`` d'un run de ``07_dynamique_lente.py``
  (séries quotidiennes) et son ``metadata.json`` (tailles, G, transitoire) ;
- le ``entropie.csv`` écrit par ``09_analyse_so3.py`` (valeurs de z déjà
  calculées).

Il appartient au fil B : il n'importe aucun module du réseau (fil A) et ne
communique avec lui que par ces fichiers.

Choix de la fenêtre des extraits (fixé AVANT de regarder les courbes) : les
``--duree`` premiers jours qui suivent le transitoire du run, pour les deux
régimes. Aucune fenêtre n'est choisie pour sa ressemblance avec un résultat.
Une moyenne glissante de ``--moyenne`` jours est superposée, pour rendre
visibles les variations lentes que la MSE mesure aux grandes échelles.

Sorties (jamais d'écrasement d'un run existant, sauf ``--ecraser``) :

    figures/12_figures_lisibles/<run-id>/
        rapport_series_nflux.png / .pdf      extraits de N_flux(t), point 7
        rapport_so3_nflux.png / .pdf         z de la MSE de N_flux (section 5)
        rapport_so3_complet_melange.png/.pdf z de la MSE, 3 grandeurs (annexe D)
        rapport_so3_complet_iaaft.png / .pdf
    data/12_figures_lisibles/<run-id>/
        metadata.json    entrées (chemins, SHA-256), paramètres, statistiques
                         des extraits (moyenne, écart-type, min, max)

Usage (PowerShell), depuis la racine du dépôt :

    python scripts\\12_figures_lisibles.py --run-id rapport-etape
    python scripts\\12_figures_lisibles.py --run-id rapport-etape --run so2-G-scan
    python scripts\\12_figures_lisibles.py --run-id essai --taille 94 --G 0.1 2 --duree 2000

Sans ``--run``, le script cherche les runs de ``data/07_dynamique_lente/runs/``
qui contiennent un ``metadata.json`` : s'il n'y en a qu'un, il le prend, sinon
il affiche la liste et s'arrête. Sans ``--entropie``, il prend
``data/09_analyse_so3/<run>/entropie.csv``. ``--lister`` affiche ce qui est
disponible (runs, cas, G) sans rien tracer.

Durée : quelques secondes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

from cascade_entropy.chemins import DOSSIER_DATA, DOSSIER_FIGURES  # noqa: E402

SCRIPT_VERSION = "1.0.0"

DOSSIER_RUNS_07 = DOSSIER_DATA / "07_dynamique_lente" / "runs"
DOSSIER_SO3 = DOSSIER_DATA / "09_analyse_so3"
DOSSIER_SORTIE = DOSSIER_DATA / "12_figures_lisibles"
DOSSIER_FIG = DOSSIER_FIGURES / "12_figures_lisibles"

# Termes du rapport (point 10) : clé du code -> libellé imprimé.
LIBELLES = {
    "nombre_effectif": r"Nombre effectif de lignes $N_{\mathrm{flux}}$",
    "taux_maximal": r"Taux de charge maximal $M_{\max}$",
    "charge_relative": "Charge relative",
}
LIBELLES_COURTS = {
    "nombre_effectif": r"$N_{\mathrm{flux}}$",
    "taux_maximal": r"$M_{\max}$",
    "charge_relative": "charge relative",
}
REFERENCES = {"melange": "séries mélangées", "iaaft": "substituts IAAFT"}
COLONNES_ENTROPIE = ("n_noeuds", "G", "observable", "mesure", "echelle",
                     "z_melange", "z_iaaft")
SEUIL_Z = 2.0
FIGURES_DISPONIBLES = ("series", "so3")

# Largeurs utiles du rapport (8 pt, A4, marges 1,5 cm) en pouces : les figures
# sont dessinées à leur taille d'impression, donc les polices restent lisibles.
LARGEUR_PAGE = 7.08
LARGEUR_COLONNE = 3.42


# ===========================================================================
# Utilitaires
# ===========================================================================

def maintenant_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def nettoyer_id(texte: str) -> str:
    """Identifiant sûr pour un nom de dossier (même règle que les scripts 07 à 11)."""
    texte = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(texte).strip()).strip("-_.")
    if not texte:
        raise ValueError("Identifiant vide ou invalide.")
    return texte


def jeton_G(G: float) -> str:
    return f"{float(G):.4f}".replace(".", "p")


def nom_cas(n_noeuds: int, G: float) -> str:
    """Nom de cas des runs de 07 (ex. ``N94_G1p0000``)."""
    return f"N{int(n_noeuds)}_G{jeton_G(G)}"


def sha256_fichier(chemin: Path) -> str:
    h = hashlib.sha256()
    with chemin.open("rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def sauvegarder(fig, base: Path) -> list[str]:
    """Écrit ``base.png`` (300 ppp) et ``base.pdf`` ; renvoie les noms écrits."""
    base.parent.mkdir(parents=True, exist_ok=True)
    ecrits = []
    for ext, opts in ((".png", {"dpi": 300}), (".pdf", {})):
        chemin = base.with_suffix(ext)
        try:
            fig.savefig(chemin, bbox_inches="tight", **opts)
            ecrits.append(chemin.name)
        except PermissionError:
            print(f"  ATTENTION : {chemin.name} est ouvert ailleurs, non mis à jour.")
    plt.close(fig)
    return ecrits


def texte_G(G: float) -> str:
    """Valeur de G en notation française pour mathtext (virgule décimale)."""
    return f"{float(G):g}".replace(".", "{,}")


def virgule(valeur: float, _pos=None) -> str:
    """Graduation avec virgule décimale (convention du rapport)."""
    return f"{valeur:g}".replace(".", ",")


def couleurs_G(valeurs_G: list[float]) -> dict[float, Any]:
    """Même palette que 09_analyse_so3.py (viridis, G croissant)."""
    cmap = plt.get_cmap("viridis")
    n = max(len(valeurs_G) - 1, 1)
    return {G: cmap(i / n) for i, G in enumerate(sorted(valeurs_G))}


# ===========================================================================
# Recherche et lecture des entrées
# ===========================================================================

def runs_disponibles(dossier: Path = None) -> list[Path]:
    """Dossiers de run de 07 qui contiennent un ``metadata.json``, triés par nom."""
    dossier = DOSSIER_RUNS_07 if dossier is None else Path(dossier)
    if not dossier.is_dir():
        return []
    return sorted(p for p in dossier.iterdir() if p.is_dir() and (p / "metadata.json").exists())


def choisir_run(run: str | None, dossier: Path = None) -> Path:
    """
    Dossier du run de 07 à utiliser.

    Avec ``run`` : ce run, qui doit exister. Sans ``run`` : le seul run
    disponible ; s'il y en a zéro ou plusieurs, arrêt avec la liste.
    """
    dossier = DOSSIER_RUNS_07 if dossier is None else Path(dossier)
    if run is not None:
        chemin = dossier / nettoyer_id(run)
        if not (chemin / "metadata.json").exists():
            raise SystemExit(f"Run introuvable : {chemin / 'metadata.json'}")
        return chemin
    runs = runs_disponibles(dossier)
    if not runs:
        raise SystemExit(f"Aucun run de 07_dynamique_lente dans {dossier}.")
    if len(runs) > 1:
        noms = ", ".join(p.name for p in runs)
        raise SystemExit(f"Plusieurs runs disponibles ({noms}) : préciser --run.")
    return runs[0]


def lire_config(run_dir: Path) -> dict:
    """Configuration du run (clés tailles, G, jours, transitoire), validée."""
    config = json.loads((run_dir / "metadata.json").read_text("utf-8")).get("config", {})
    for cle in ("tailles", "G", "jours", "transitoire"):
        if cle not in config:
            raise SystemExit(f"metadata.json incomplet : clé '{cle}' absente.")
    return config


def trouver_G(valeur: float, liste_G: list[float]) -> float:
    """Valeur de G du run égale (à 1e-9 près) à ``valeur``, sinon arrêt."""
    for G in liste_G:
        if abs(float(G) - float(valeur)) < 1e-9:
            return float(G)
    raise SystemExit(f"G = {valeur:g} absent du run (disponibles : "
                     + ", ".join(f"{float(g):g}" for g in liste_G) + ").")


def lire_serie(run_dir: Path, n: int, G: float, cle: str = "nombre_effectif") -> np.ndarray:
    """Série quotidienne complète ``cle`` d'un cas (transitoire compris)."""
    chemin = run_dir / f"cas_{nom_cas(n, G)}.npz"
    if not chemin.exists():
        raise SystemExit(f"Cas absent : {chemin}")
    with np.load(chemin, allow_pickle=False) as d:
        if cle not in d.files:
            raise SystemExit(f"Série '{cle}' absente de {chemin.name}.")
        x = np.asarray(d[cle], dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise SystemExit(f"Série '{cle}' de {chemin.name} vide ou mal formée.")
    return x


def extraire_fenetre(x: np.ndarray, transitoire: int, debut: int, duree: int) -> np.ndarray:
    """
    Les ``duree`` jours qui commencent ``debut`` jours après le transitoire.

    Arrête si la fenêtre dépasse la série ou contient des valeurs non finies :
    aucune valeur n'est complétée.
    """
    if duree < 2:
        raise SystemExit("--duree doit être d'au moins 2 jours.")
    if debut < 0:
        raise SystemExit("--debut doit être ≥ 0.")
    i0 = int(transitoire) + int(debut)
    i1 = i0 + int(duree)
    if i1 > x.size:
        raise SystemExit(f"Fenêtre [{i0}, {i1}) hors de la série ({x.size} jours).")
    fenetre = x[i0:i1]
    if not np.all(np.isfinite(fenetre)):
        raise SystemExit("La fenêtre contient des valeurs non finies.")
    return fenetre


def moyenne_glissante(x: np.ndarray, largeur: int) -> np.ndarray:
    """
    Moyenne glissante centrée sur ``largeur`` jours (valeurs complètes seulement).

    Renvoie un tableau de même longueur que ``x``, NaN là où la fenêtre déborde.
    """
    x = np.asarray(x, dtype=float)
    if largeur < 1 or largeur > x.size:
        raise SystemExit("--moyenne doit être comprise entre 1 et --duree.")
    sortie = np.full(x.size, np.nan)
    noyau = np.convolve(x, np.ones(largeur) / largeur, mode="valid")
    debut = (largeur - 1) // 2
    sortie[debut:debut + noyau.size] = noyau
    return sortie


def statistiques(x: np.ndarray) -> dict:
    return {"moyenne": float(np.mean(x)), "ecart_type": float(np.std(x)),
            "min": float(np.min(x)), "max": float(np.max(x)), "n_jours": int(x.size)}


def lire_entropie(chemin: Path, mesure: str = "mse") -> list[dict]:
    """Lignes de ``entropie.csv`` pour ``mesure``, colonnes converties et validées."""
    if not chemin.exists():
        raise SystemExit(f"entropie.csv introuvable : {chemin}")
    with chemin.open(encoding="utf-8", newline="") as f:
        lecteur = csv.DictReader(f)
        manquantes = [c for c in COLONNES_ENTROPIE if c not in (lecteur.fieldnames or [])]
        if manquantes:
            raise SystemExit(f"Colonnes absentes de {chemin.name} : {', '.join(manquantes)}")
        lignes = []
        for l in lecteur:
            if l["mesure"] != mesure:
                continue
            lignes.append({"n_noeuds": int(l["n_noeuds"]), "G": float(l["G"]),
                           "observable": l["observable"], "echelle": int(l["echelle"]),
                           "z_melange": float(l["z_melange"]), "z_iaaft": float(l["z_iaaft"])})
    if not lignes:
        raise SystemExit(f"Aucune ligne de mesure '{mesure}' dans {chemin.name}.")
    inconnues = sorted({l["observable"] for l in lignes} - set(LIBELLES))
    if inconnues:
        raise SystemExit(f"Observables sans libellé : {', '.join(inconnues)}")
    return lignes


# ===========================================================================
# Figures
# ===========================================================================

def figure_series(extraits: dict[float, np.ndarray], lisses: dict[float, np.ndarray],
                  n: int, debut: int, largeur_moyenne: int, base: Path) -> list[str]:
    """Deux panneaux superposés, mêmes axes : N_flux(t) pour deux valeurs de G."""
    valeurs = sorted(extraits)
    tous = np.concatenate([extraits[G] for G in valeurs])
    marge = 0.03 * (tous.max() - tous.min() or 1.0)
    ylim = (tous.min() - marge, tous.max() + marge)
    plt.rcParams.update({"font.size": 7.5})
    fig, axes = plt.subplots(len(valeurs), 1, figsize=(LARGEUR_COLONNE, 1.35 * len(valeurs) + 0.4),
                             sharex=True, sharey=True, squeeze=False)
    for ax, G in zip(axes[:, 0], valeurs):
        jours = debut + np.arange(extraits[G].size)
        ax.plot(jours, extraits[G], color="0.65", linewidth=0.35, label="valeur du jour")
        ax.plot(jours, lisses[G], color="C3", linewidth=1.1,
                label=f"moyenne sur {largeur_moyenne} jours")
        ax.set_ylim(*ylim)
        ax.set_title(f"$G = {texte_G(G)}$", fontsize=8, loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(virgule))
        ax.set_ylabel(r"$N_{\mathrm{flux}}$")
    axes[-1, 0].set_xlabel("jours après le transitoire")
    axes[0, 0].legend(fontsize=6.5, loc="upper right", frameon=False, ncol=2)
    fig.suptitle(f"Arbre de {n} nœuds", fontsize=8)
    fig.tight_layout(h_pad=0.4)
    return sauvegarder(fig, base)


def _panneau_z(ax, lignes: list[dict], n: int, observable: str, reference: str,
               couleurs: dict[float, Any]) -> None:
    cle = f"z_{reference}"
    for G, couleur in couleurs.items():
        pts = sorted((l["echelle"], l[cle]) for l in lignes
                     if l["n_noeuds"] == n and l["observable"] == observable and l["G"] == G)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=couleur,
                    markersize=2.5, linewidth=1.0)
    for niveau in (-SEUIL_Z, SEUIL_Z):
        ax.axhline(niveau, color="0.4", linestyle="--", linewidth=0.7)
    ax.axhline(0, color="0.75", linewidth=0.5)
    ax.set_xscale("log")


def _legende_G(fig, couleurs: dict[float, Any], y: float) -> None:
    poignees = [plt.Line2D([], [], color=c, marker="o", markersize=3, label=f"$G = {texte_G(G)}$")
                for G, c in couleurs.items()]
    fig.legend(handles=poignees, ncol=len(poignees), loc="lower center",
               bbox_to_anchor=(0.5, y), frameon=False, fontsize=7.5)


def figure_so3_nflux(lignes: list[dict], base: Path) -> list[str]:
    """Section 5 : z de la MSE de N_flux, mélanges en haut, IAAFT en bas, 3 tailles."""
    tailles = sorted({l["n_noeuds"] for l in lignes if l["observable"] == "nombre_effectif"})
    if not tailles:
        raise SystemExit("Aucune ligne pour nombre_effectif dans entropie.csv.")
    couleurs = couleurs_G(sorted({l["G"] for l in lignes}))
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(2, len(tailles), figsize=(LARGEUR_PAGE, 4.3),
                             sharex=True, squeeze=False)
    for i, reference in enumerate(("melange", "iaaft")):
        for j, n in enumerate(tailles):
            ax = axes[i][j]
            _panneau_z(ax, lignes, n, "nombre_effectif", reference, couleurs)
            if i == 0:
                ax.set_title(f"{n} nœuds", fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"$z$, {REFERENCES[reference]}")
            if i == 1:
                ax.set_xlabel("échelle $\\tau$ (jours)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _legende_G(fig, couleurs, 0.0)
    return sauvegarder(fig, base)


def figure_so3_complet(lignes: list[dict], reference: str, base: Path) -> list[str]:
    """Annexe D : z de la MSE pour les trois grandeurs (colonnes) et trois tailles (lignes)."""
    observables = [o for o in LIBELLES if any(l["observable"] == o for l in lignes)]
    tailles = sorted({l["n_noeuds"] for l in lignes})
    couleurs = couleurs_G(sorted({l["G"] for l in lignes}))
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(len(tailles), len(observables),
                             figsize=(LARGEUR_PAGE, 2.05 * len(tailles) + 0.5),
                             sharex=True, squeeze=False)
    for i, n in enumerate(tailles):
        for j, o in enumerate(observables):
            ax = axes[i][j]
            _panneau_z(ax, lignes, n, o, reference, couleurs)
            if i == 0:
                ax.set_title(LIBELLES[o], fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{n} nœuds\n$z$, {REFERENCES[reference]}")
            if i == len(tailles) - 1:
                ax.set_xlabel("échelle $\\tau$ (jours)")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _legende_G(fig, couleurs, 0.0)
    return sauvegarder(fig, base)


# ===========================================================================
# Ligne de commande
# ===========================================================================

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", help="identifiant du dossier de sortie (obligatoire sauf --lister)")
    p.add_argument("--run", default=None,
                   help="run de 07_dynamique_lente.py (défaut : le seul run disponible)")
    p.add_argument("--entropie", type=Path, default=None,
                   help="entropie.csv de 09_analyse_so3.py (défaut : data/09_analyse_so3/<run>/)")
    p.add_argument("--figures", nargs="+", choices=FIGURES_DISPONIBLES,
                   default=list(FIGURES_DISPONIBLES), help="figures à tracer (défaut : toutes)")
    p.add_argument("--taille", type=int, default=94, help="taille du réseau des extraits (défaut 94)")
    p.add_argument("--G", nargs=2, type=float, default=[0.1, 2.0], metavar=("G1", "G2"),
                   help="deux valeurs de G pour les extraits (défaut 0.1 et 2)")
    p.add_argument("--debut", type=int, default=0,
                   help="début de la fenêtre, en jours après le transitoire (défaut 0)")
    p.add_argument("--duree", type=int, default=2000, help="durée de la fenêtre en jours (défaut 2000)")
    p.add_argument("--moyenne", type=int, default=50,
                   help="largeur de la moyenne glissante superposée, en jours (défaut 50)")
    p.add_argument("--ecraser", action="store_true", help="autoriser l'écrasement d'un run existant")
    p.add_argument("--lister", action="store_true",
                   help="afficher les runs, tailles et G disponibles, sans rien tracer")
    return p


def lister(dossier: Path = None) -> None:
    runs = runs_disponibles(dossier)
    if not runs:
        print(f"Aucun run dans {DOSSIER_RUNS_07 if dossier is None else dossier}.")
        return
    for run_dir in runs:
        config = lire_config(run_dir)
        print(f"{run_dir.name} : tailles {config['tailles']}, "
              f"G {[float(g) for g in config['G']]}, {config['jours']} jours, "
              f"transitoire {config['transitoire']}")


def main(argv: list[str] | None = None) -> dict:
    args = parser().parse_args(argv)
    if args.lister:
        lister()
        return {}
    if not args.run_id:
        raise SystemExit("--run-id est obligatoire.")
    run_id = nettoyer_id(args.run_id)
    dossier = DOSSIER_SORTIE / run_id
    fig_dir = DOSSIER_FIG / run_id
    if (dossier / "metadata.json").exists() and not args.ecraser:
        raise SystemExit(f"Le run {run_id} existe déjà ({dossier}) : choisir un autre "
                         "--run-id ou ajouter --ecraser.")
    if abs(args.G[0] - args.G[1]) < 1e-12:
        raise SystemExit("--G doit contenir deux valeurs différentes.")

    run_dir = choisir_run(args.run)
    config = lire_config(run_dir)
    meta: dict[str, Any] = {
        "script": "12_figures_lisibles.py", "script_version": SCRIPT_VERSION,
        "created_at": maintenant_iso(), "python": platform.python_version(),
        "plateforme": platform.platform(), "run_source": run_dir.name,
        "config_run": config, "parametres": {k: (str(v) if isinstance(v, Path) else v)
                                             for k, v in vars(args).items()},
        "entrees": {}, "figures": [],
    }

    if "series" in args.figures:
        if args.taille not in [int(t) for t in config["tailles"]]:
            raise SystemExit(f"Taille {args.taille} absente du run ({config['tailles']}).")
        extraits, lisses, stats = {}, {}, {}
        for valeur in args.G:
            G = trouver_G(valeur, config["G"])
            chemin = run_dir / f"cas_{nom_cas(args.taille, G)}.npz"
            x = lire_serie(run_dir, args.taille, G)
            extraits[G] = extraire_fenetre(x, int(config["transitoire"]), args.debut, args.duree)
            lisses[G] = moyenne_glissante(extraits[G], args.moyenne)
            stats[f"G={G:g}"] = statistiques(extraits[G])
            meta["entrees"][chemin.name] = sha256_fichier(chemin)
        meta["statistiques_extraits"] = stats
        meta["figures"] += figure_series(extraits, lisses, args.taille, args.debut,
                                         args.moyenne, fig_dir / "rapport_series_nflux")
        for cle, s in stats.items():
            print(f"  N_flux, {args.taille} nœuds, {cle} : moyenne {s['moyenne']:.3f}, "
                  f"écart-type {s['ecart_type']:.3f} sur {s['n_jours']} jours")

    if "so3" in args.figures:
        chemin_e = args.entropie or (DOSSIER_SO3 / run_dir.name / "entropie.csv")
        lignes = lire_entropie(Path(chemin_e))
        meta["entrees"][str(Path(chemin_e).name)] = sha256_fichier(Path(chemin_e))
        meta["figures"] += figure_so3_nflux(lignes, fig_dir / "rapport_so3_nflux")
        for reference in ("melange", "iaaft"):
            meta["figures"] += figure_so3_complet(lignes, reference,
                                                  fig_dir / f"rapport_so3_complet_{reference}")

    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nFigures : {fig_dir}\nMétadonnées : {dossier / 'metadata.json'}")
    return meta


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
