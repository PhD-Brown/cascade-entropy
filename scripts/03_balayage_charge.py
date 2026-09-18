"""Balayage du niveau de charge : les deux transitions du modèle.

Reproduit qualitativement la figure de référence de Carreras et al. En augmentant
la demande à profil fixe, deux changements de régime apparaissent.

Le premier survient lorsque la demande atteint la capacité totale de production.
Le délestage commence, et la puissance servie plafonne définitivement.

Le second survient plus haut, lorsque les flux atteignent les limites de
transport. Il est moins intuitif : la puissance totale servie ne bouge plus
depuis la première transition, et pourtant les lignes continuent de se charger.
La raison est que le délestage n'est pas uniforme — la répartition de la
puissance entre les charges change, donc les flux aussi.

Usage, depuis la racine du dépôt :
    python scripts/03_balayage_charge.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cascade_entropy import figures  # noqa: E402
from cascade_entropy.dispatch import demande_uniforme, resoudre  # noqa: E402
from cascade_entropy.entropie import nombre_effectif  # noqa: E402
from cascade_entropy.reseau import (  # noqa: E402
    arbre,
    limites_par_niveau,
    matrice_de_flux,
)

GENERATIONS = 4
P_C = 2623.9          # capacité totale de production, valeur publiée
CIBLE_SECONDE = 1.45  # niveau de charge où les limites de transport sont atteintes
RATIOS = np.linspace(0.2, 2.0, 55)

RACINE = Path(__file__).resolve().parents[1]
SORTIE = RACINE / "figures" / "03_balayage_charge.png"


def preparer():
    """Réseau, matrice de flux et capacités mises à l'échelle.

    L'échelle des capacités est le seul paramètre libre de cette reproduction :
    elle est fixée pour que la seconde transition tombe au niveau rapporté dans
    la littérature. Tout le reste découle du modèle.
    """
    reseau = arbre(GENERATIONS)
    A = matrice_de_flux(reseau)
    p_max = np.full(reseau.generateurs.size, P_C / reseau.generateurs.size)
    base = limites_par_niveau(reseau)
    facteur = resoudre(reseau, demande_uniforme(reseau, CIBLE_SECONDE * P_C),
                       limites=base, puissance_max=p_max, A=A).taux_maximal
    return reseau, A, base * facteur, p_max


def balayer(reseau, A, limites, p_max):
    """Grandeurs caractéristiques en fonction du niveau de charge."""
    colonnes = {cle: [] for cle in
                ("servie", "delestage", "taux_max", "taux_moyen", "saturees", "lignes_eff")}
    for ratio in RATIOS:
        solution = resoudre(reseau, demande_uniforme(reseau, ratio * P_C),
                            limites=limites, puissance_max=p_max, A=A)
        colonnes["servie"].append(solution.charge_servie.sum() / P_C)
        colonnes["delestage"].append(solution.delestage_total / (ratio * P_C))
        colonnes["taux_max"].append(solution.taux_maximal)
        colonnes["taux_moyen"].append(solution.taux_de_charge.mean())
        colonnes["saturees"].append(solution.lignes_saturees.size)
        colonnes["lignes_eff"].append(nombre_effectif(solution.flux))
    return {cle: np.array(valeurs) for cle, valeurs in colonnes.items()}


def tracer(resultats, n_lignes: int):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)
    couleurs = figures.COULEURS

    axes[0].plot(RATIOS, resultats["servie"], color=couleurs["a"],
                 label="puissance servie / $P_C$")
    axes[0].plot(RATIOS, resultats["delestage"], color=couleurs["b"],
                 label="fraction délestée")
    axes[0].set_ylabel("puissance")
    axes[0].legend()

    axes[1].plot(RATIOS, resultats["taux_max"], color=couleurs["a"], label="$M_{max}$")
    axes[1].plot(RATIOS, resultats["taux_moyen"], color=couleurs["b"],
                 label="taux de charge moyen")
    axes[1].axhline(1.0, color=couleurs["neutre"], linestyle=":", linewidth=1)
    axes[1].set_ylabel("taux de charge")
    axes[1].legend()

    axes[2].plot(RATIOS, resultats["saturees"], color=couleurs["a"],
                 label="lignes saturées")
    axes[2].plot(RATIOS, resultats["lignes_eff"], color=couleurs["c"],
                 label="nombre effectif de lignes")
    axes[2].axhline(n_lignes, color=couleurs["neutre"], linestyle=":", linewidth=1)
    axes[2].set_ylabel("nombre de lignes")
    axes[2].set_xlabel("$P_D / P_C$")
    axes[2].legend()

    for ax in axes:
        for seuil, etiquette in ((1.0, "limite de production"),
                                 (CIBLE_SECONDE, "limites de transport")):
            ax.axvline(seuil, color=couleurs["alerte"], linestyle="--", linewidth=1)
        ax.margins(x=0.02)

    axes[0].set_title("Deux transitions en fonction du niveau de charge")
    for seuil, etiquette in ((1.0, "production"), (CIBLE_SECONDE, "transport")):
        axes[0].annotate(etiquette, xy=(seuil, 1.0), xytext=(4, -10),
                         textcoords="offset points", fontsize=8,
                         color=couleurs["alerte"])
    fig.tight_layout()
    return fig


def main() -> None:
    reseau, A, limites, p_max = preparer()
    resultats = balayer(reseau, A, limites, p_max)

    SORTIE.parent.mkdir(exist_ok=True)
    tracer(resultats, reseau.n_lignes).savefig(SORTIE)

    premiere = RATIOS[np.argmax(resultats["delestage"] > 1e-6)]
    seconde = RATIOS[np.argmax(resultats["saturees"] > 0)]
    print(f"réseau            : {reseau.n_noeuds} nœuds, {reseau.n_lignes} lignes, "
          f"{reseau.generateurs.size} générateurs")
    print(f"première transition (délestage)      : P_D/P_C = {premiere:.2f}")
    print(f"seconde transition (lignes saturées) : P_D/P_C = {seconde:.2f}")
    print(f"lignes effectives, de {resultats['lignes_eff'][0]:.1f} "
          f"à {resultats['lignes_eff'][-1]:.1f}")
    print(f"figure            : {SORTIE.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
