"""
Style graphique et tracés réutilisables.

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
    """
    Applique la charte graphique du projet aux figures suivantes.

    Matplotlib conserve ces paramètres dans `plt.rcParams`. Après cet appel, les
    nouvelles figures héritent donc automatiquement des mêmes dimensions,
    couleurs, grilles, polices et styles de lignes.
    """
    # Une seule mise à jour globale garantit que les graphiques produits dans les
    # notebooks, les scripts et les rapports suivent les mêmes conventions.
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
    """
    Aperçu empilé de plusieurs séries, tronquées aux premiers points.

    Sert à montrer d'un coup d'œil à quoi ressemblent les séries de validation
    avant d'en discuter les mesures. Chaque série occupe son propre panneau et
    tous les panneaux partagent le même axe temporel.
    """
    # On crée un panneau par série, avec une hauteur qui augmente avec leur
    # nombre pour conserver une lecture confortable.
    fig, axes = plt.subplots(
        len(series), 1, figsize=(7.2, 1.5 * len(series)), sharex=True
    )
    # Matplotlib renvoie parfois un Axe unique pour une seule série. Cette
    # conversion permet de traiter uniformément un ou plusieurs panneaux.
    axes = np.atleast_1d(axes)
    for ax, (nom, serie), couleur in zip(axes, series.items(), CYCLE):
        # On limite l'aperçu aux premiers points pour montrer la forme locale sans
        # rendre la figure illisible pour une série très longue.
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
    """
    Compare une mesure scalaire entre plusieurs séries.

    Chaque clé du dictionnaire devient une catégorie sur l'axe horizontal et sa
    valeur devient la hauteur d'une barre. Une référence optionnelle permet de
    comparer directement les résultats à un seuil ou à une valeur théorique.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    # L'ordre d'insertion du dictionnaire est conservé dans l'ordre des barres.
    noms = list(valeurs)
    hauteurs = [valeurs[n] for n in noms]
    ax.bar(noms, hauteurs, color=CYCLE[: len(noms)], width=0.6)
    # La valeur exacte est affichée au-dessus de chaque barre.
    for x, h in enumerate(hauteurs):
        ax.text(x, h, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    if reference is not None:
        # Une ligne horizontale rend la comparaison avec la référence immédiate.
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
    """
    Trace l'entropie en fonction de l'échelle, avec repérage du croisement.

    `courbes` associe un nom à un vecteur de valeurs ; `erreurs` fournit
    éventuellement les dispersions correspondantes. Chaque point représente une
    mesure à une échelle de granularisation donnée.
    """
    fig, ax = plt.subplots()
    marqueurs = ["o", "s", "^", "D"]
    noms = list(courbes)

    for (nom, valeurs), marqueur, couleur in zip(courbes.items(), marqueurs, CYCLE):
        # `errorbar` affiche la valeur centrale et, si elle existe, sa dispersion.
        erreur = None if erreurs is None else erreurs.get(nom)
        ax.errorbar(echelles, valeurs, yerr=erreur, marker=marqueur,
                    capsize=2.5, color=couleur, label=nom)

    if annoter_croisement and len(noms) == 2:
        # On recherche le premier passage de "a au-dessus de b" à "a au-dessous
        # de b" entre deux échelles successives.
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
    """
    Trace l'ajustement R/S en échelle logarithmique.

    Permet de juger visuellement si l'exposant de Hurst résume correctement le
    comportement, ou si la courbe change de pente selon l'échelle. Sur le graphe
    log-log, la pente de la droite affichée correspond à l'exposant estimé.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    # Les points montrent les rapports R/S effectivement mesurés.
    ax.loglog(tailles, valeurs, "o", color=COULEURS["a"], label="R/S mesuré")
    # Dans l'espace original, la droite log-log correspond à exp(ordonnee) * n**H.
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
    """
    Trace l'histogramme d'une distribution nulle, avec la valeur mesurée en repère.

    Outil de décision : une valeur n'est significative que si elle sort de la
    distribution obtenue sur des séries sans la propriété recherchée. La ligne
    grise situe la moyenne de référence et la ligne rouge situe la mesure étudiée.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    # L'histogramme montre la variabilité attendue sous l'hypothèse nulle.
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
    """
    Trace un réseau en arbre avec les nœuds disposés par niveau.

    Si `taux` est fourni, l'épaisseur et la couleur des lignes reflètent le taux
    de charge, ce qui rend visible la concentration de la puissance. Les charges
    sont dessinées comme des points et les générateurs comme des carrés.
    """
    # Import local : LineCollection n'est nécessaire que pour cette fonction.
    from matplotlib.collections import LineCollection

    niveaux = reseau.niveaux
    # Chaque nœud reçoit une position. La profondeur est représentée verticalement
    # et les nœuds d'un même niveau sont répartis horizontalement.
    positions = np.zeros((reseau.n_noeuds, 2))
    for niveau in range(niveaux.max() + 1):
        noeuds = np.flatnonzero(niveaux == niveau)
        # Le cas d'un nœud unique est placé au centre ; sinon les nœuds sont
        # espacés régulièrement dans l'intervalle horizontal [-1, 1].
        xs = np.linspace(-1, 1, noeuds.size + 2)[1:-1] if noeuds.size > 1 else [0.0]
        positions[noeuds, 0] = xs
        positions[noeuds, 1] = -niveau

    # Chaque ligne devient un segment reliant les positions de ses deux extrémités.
    segments = [[positions[i], positions[j]] for i, j in reseau.lignes]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    if taux is None:
        # Sans taux de charge, toutes les lignes utilisent la même apparence.
        collection = LineCollection(segments, colors=COULEURS["neutre"], linewidths=1.0)
        ax.add_collection(collection)
    else:
        # Avec des taux, la couleur et l'épaisseur codent la sollicitation de
        # chaque ligne. Les bornes 0 et 1 rendent les figures comparables.
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


def distributions_taux(taux_par_cas: dict[str, np.ndarray], titre: str = ""):
    """
    Trace les distributions des taux de charge des lignes, un cas par courbe.

    C'est la représentation de l'observable retenue dans SO2 : deux réseaux
    peuvent avoir le même taux de charge moyen et des distributions très
    différentes, donc des vulnérabilités différentes. Les histogrammes en tracé
    seul se superposent sans masquer complètement les formes des autres cas.
    """
    fig, ax = plt.subplots()
    # Des bornes communes entre 0 et 1 rendent les distributions comparables.
    bords = np.linspace(0, 1, 26)
    for (nom, taux), couleur in zip(taux_par_cas.items(), CYCLE):
        ax.hist(taux, bins=bords, histtype="step", linewidth=1.8,
                color=couleur, label=nom)
    ax.set_xlabel("taux de charge de la ligne")
    ax.set_ylabel("nombre de lignes")
    ax.legend()
    if titre:
        ax.set_title(titre)
    fig.tight_layout()
    return fig


def transitions(ratios, courbes: dict[str, np.ndarray], seuils: dict[str, float],
                ylabel: str = "", titre: str = ""):
    """
    Trace une grandeur en fonction du niveau de charge et repère les transitions.

    `ratios` fournit l'axe horizontal, `courbes` contient les séries à comparer
    et `seuils` associe une étiquette à chaque niveau critique. Les lignes
    verticales relient visuellement un changement de régime à sa charge.
    """
    fig, ax = plt.subplots()
    for (nom, valeurs), couleur in zip(courbes.items(), CYCLE):
        ax.plot(ratios, valeurs, color=couleur, label=nom)
    for etiquette, seuil in seuils.items():
        ax.axvline(seuil, color=COULEURS["alerte"], linestyle="--", linewidth=1)
        ax.annotate(etiquette, xy=(seuil, ax.get_ylim()[1]), xytext=(4, -12),
                    textcoords="offset points", fontsize=8,
                    color=COULEURS["alerte"], va="top")
    ax.set_xlabel("$P_D / P_C$")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.margins(x=0.02)
    if titre:
        ax.set_title(titre)
    fig.tight_layout()
    return fig
