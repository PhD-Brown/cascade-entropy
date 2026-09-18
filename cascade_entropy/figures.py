"""Style graphique et tracés réutilisables.

Centralise l'apparence des figures pour que les notebooks, les scripts et les
rapports aient un rendu identique. Les notebooks appellent `appliquer_style()`
puis les fonctions de tracé d'ici : aucune logique de figure ne doit vivre dans
une cellule.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Palette : deux teintes contrastées pour les comparaisons à deux séries, plus
# des neutres pour les repères et les annotations.
COULEURS = {
    "a": "#1f4e79",        # bleu profond — première série
    "b": "#c55a11",        # orange brûlé — seconde série
    "c": "#2e7d5b",        # vert — troisième série
    "neutre": "#6b6b6b",
    "grille": "#d9d9d9",
    "alerte": "#a4262c",
}

CYCLE = [COULEURS["a"], COULEURS["b"], COULEURS["c"], COULEURS["neutre"]]


def appliquer_style() -> None:
    """Applique le style du projet à toutes les figures suivantes."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.4),
            "figure.dpi": 110,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "axes.prop_cycle": plt.cycler(color=CYCLE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlepad": 10,
            "axes.grid": True,
            "grid.color": COULEURS["grille"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "font.size": 10,
        }
    )


def galerie_series(series: dict[str, np.ndarray], n_points: int = 400, titre: str = ""):
    """Aperçu empilé de plusieurs séries, tronquées aux premiers points.

    Sert à montrer d'un coup d'œil à quoi ressemblent les séries de validation
    avant d'en discuter les mesures.
    """
    fig, axes = plt.subplots(
        len(series), 1, figsize=(7.2, 1.5 * len(series)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, (nom, serie), couleur in zip(axes, series.items(), CYCLE):
        ax.plot(serie[:n_points], color=couleur, linewidth=1.0)
        ax.set_ylabel(nom, fontsize=9)
        ax.set_yticks([])
        ax.grid(False)
    axes[-1].set_xlabel("indice temporel")
    if titre:
        axes[0].set_title(titre)
    fig.tight_layout()
    return fig


def barres_comparatives(valeurs: dict[str, float], ylabel: str, titre: str = "",
                        reference: float | None = None):
    """Comparaison d'une mesure scalaire entre plusieurs séries."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    noms = list(valeurs)
    hauteurs = [valeurs[n] for n in noms]
    ax.bar(noms, hauteurs, color=CYCLE[: len(noms)], width=0.6)
    for x, h in enumerate(hauteurs):
        ax.text(x, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    if reference is not None:
        ax.axhline(reference, color=COULEURS["alerte"], linestyle="--", linewidth=1,
                   label=f"référence = {reference:g}")
        ax.legend()
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(hauteurs) * 1.18)
    ax.grid(axis="x", visible=False)
    if titre:
        ax.set_title(titre)
    fig.tight_layout()
    return fig


def courbes_multiechelle(echelles, courbes: dict[str, np.ndarray],
                         erreurs: dict[str, np.ndarray] | None = None,
                         titre: str = "", annoter_croisement: bool = True):
    """Entropie en fonction de l'échelle, avec repérage du croisement.

    `courbes` associe un nom à un vecteur de valeurs ; `erreurs` fournit
    éventuellement les dispersions correspondantes.
    """
    fig, ax = plt.subplots()
    marqueurs = ["o", "s", "^", "D"]
    noms = list(courbes)

    for (nom, valeurs), marqueur, couleur in zip(courbes.items(), marqueurs, CYCLE):
        erreur = None if erreurs is None else erreurs.get(nom)
        ax.errorbar(echelles, valeurs, yerr=erreur, marker=marqueur,
                    capsize=2.5, color=couleur, label=nom)

    if annoter_croisement and len(noms) == 2:
        a, b = courbes[noms[0]], courbes[noms[1]]
        indices = np.flatnonzero((a[:-1] > b[:-1]) & (a[1:] < b[1:]))
        if indices.size:
            x = echelles[indices[0] + 1]
            ax.axvline(x, color=COULEURS["neutre"], linestyle=":", linewidth=1)
            ax.annotate(f"croisement\néchelle {int(x)}", xy=(x, ax.get_ylim()[1]),
                        xytext=(6, -14), textcoords="offset points",
                        fontsize=9, color=COULEURS["neutre"], va="top")

    ax.set_xlabel("échelle de granularisation")
    ax.set_ylabel("entropie d'échantillon")
    ax.legend()
    if titre:
        ax.set_title(titre)
    fig.tight_layout()
    return fig


def ajustement_rs(tailles, valeurs, pente, ordonnee, titre: str = ""):
    """Droite d'ajustement R/S en échelle logarithmique.

    Permet de juger visuellement si l'exposant de Hurst résume correctement le
    comportement, ou si la courbe change de pente selon l'échelle.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.loglog(tailles, valeurs, "o", color=COULEURS["a"], label="R/S mesuré")
    ajuste = np.exp(ordonnee) * tailles ** pente
    ax.loglog(tailles, ajuste, "-", color=COULEURS["b"],
              label=f"ajustement, pente = {pente:.3f}")
    ax.set_xlabel("taille du segment")
    ax.set_ylabel("R/S")
    ax.legend()
    if titre:
        ax.set_title(titre)
    fig.tight_layout()
    return fig


def distribution_nulle(valeurs: np.ndarray, mesure: float | None = None,
                       xlabel: str = "", titre: str = ""):
    """Histogramme d'une distribution nulle, avec la valeur mesurée en repère.

    Outil de décision : une valeur n'est significative que si elle sort de la
    distribution obtenue sur des séries sans la propriété recherchée.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(valeurs, bins=30, color=COULEURS["a"], alpha=0.75, edgecolor="white")
    ax.axvline(float(np.mean(valeurs)), color=COULEURS["neutre"], linestyle="--",
               linewidth=1, label=f"moyenne nulle = {np.mean(valeurs):.3f}")
    if mesure is not None:
        ax.axvline(mesure, color=COULEURS["alerte"], linewidth=1.6,
                   label=f"valeur mesurée = {mesure:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("effectif")
    ax.legend()
    ax.grid(axis="x", visible=False)
    if titre:
        ax.set_title(titre)
    fig.tight_layout()
    return fig


def reseau_en_arbre(reseau, taux=None, titre: str = ""):
    """Tracé d'un réseau en arbre, les nœuds disposés par niveau.

    Si `taux` est fourni, l'épaisseur et la couleur des lignes reflètent le taux
    de charge, ce qui rend visible la concentration de la puissance.
    """
    from matplotlib.collections import LineCollection

    niveaux = reseau.niveaux
    positions = np.zeros((reseau.n_noeuds, 2))
    for niveau in range(niveaux.max() + 1):
        noeuds = np.flatnonzero(niveaux == niveau)
        xs = np.linspace(-1, 1, noeuds.size + 2)[1:-1] if noeuds.size > 1 else [0.0]
        positions[noeuds, 0] = xs
        positions[noeuds, 1] = -niveau

    segments = [[positions[i], positions[j]] for i, j in reseau.lignes]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    if taux is None:
        collection = LineCollection(segments, colors=COULEURS["neutre"], linewidths=1.0)
        ax.add_collection(collection)
    else:
        collection = LineCollection(segments, cmap="YlOrRd", linewidths=0.8 + 3 * taux)
        collection.set_array(np.asarray(taux))
        collection.set_clim(0, 1)
        ax.add_collection(collection)
        barre = fig.colorbar(collection, ax=ax, fraction=0.03, pad=0.02)
        barre.set_label("taux de charge")

    ax.scatter(positions[reseau.charges, 0], positions[reseau.charges, 1],
               s=18, color=COULEURS["a"], zorder=3, label="charges")
    ax.scatter(positions[reseau.generateurs, 0], positions[reseau.generateurs, 1],
               s=46, marker="s", color=COULEURS["b"], zorder=4, label="générateurs")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-niveaux.max() - 0.4, 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.legend(loc="lower right")
    if titre:
        ax.set_title(titre)
    fig.tight_layout()
    return fig
