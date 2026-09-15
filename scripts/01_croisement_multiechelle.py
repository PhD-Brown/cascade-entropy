"""Reproduction du croisement multiéchelle de Costa et al.

Compare l'entropie multiéchelle d'un bruit non corrélé et d'un bruit corrélé à
longue portée. À l'échelle 1, le bruit non corrélé obtient la valeur la plus
élevée ; après granularisation, il perd rapidement son entropie tandis que le
bruit corrélé conserve la sienne sur toutes les échelles.

Cette figure justifie le recours au multiéchelle plutôt qu'à l'entropie brute :
une forte irrégularité à une seule échelle n'est pas une forte complexité. C'est
le point qui rend la mesure pertinente pour distinguer des régimes de réseau
plutôt que de simplement classer des séries par leur bruit apparent.

Usage, depuis la racine du dépôt :
    python scripts/01_croisement_multiechelle.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cascade_entropy.entropie import multiechelle  # noqa: E402
from cascade_entropy.synthetiques import bruit_puissance  # noqa: E402

N = 16384
ECHELLES = 20
N_REALISATIONS = 5
GRAINE = 20260915

RACINE = Path(__file__).resolve().parents[1]
SORTIE = RACINE / "figures" / "01_croisement_multiechelle.png"


def courbe_moyenne(beta: float, rng: np.random.Generator):
    """Entropie multiéchelle moyennée sur plusieurs réalisations."""
    resultats = []
    for _ in range(N_REALISATIONS):
        echelles, valeurs = multiechelle(
            bruit_puissance(N, beta=beta, rng=rng), echelles=ECHELLES
        )
        resultats.append(valeurs)
    empilees = np.vstack(resultats)
    return echelles, empilees.mean(axis=0), empilees.std(axis=0)


def main() -> None:
    rng = np.random.default_rng(GRAINE)

    echelles, moy_blanc, ec_blanc = courbe_moyenne(0.0, rng)
    _, moy_rose, ec_rose = courbe_moyenne(1.0, rng)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(
        echelles, moy_blanc, yerr=ec_blanc,
        marker="o", markersize=4, capsize=2, linewidth=1,
        label="bruit non corrélé",
    )
    ax.errorbar(
        echelles, moy_rose, yerr=ec_rose,
        marker="s", markersize=4, capsize=2, linewidth=1,
        label="bruit corrélé à longue portée (1/f)",
    )
    ax.set_xlabel("échelle de granularisation")
    ax.set_ylabel("entropie d'échantillon")
    ax.set_title(f"Entropie multiéchelle, N = {N}, {N_REALISATIONS} réalisations")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    SORTIE.parent.mkdir(exist_ok=True)
    fig.savefig(SORTIE, dpi=150)

    croisement = next(
        (int(e) for e, a, b in zip(echelles, moy_blanc, moy_rose) if a < b), None
    )
    print(f"échelle 1  : blanc {moy_blanc[0]:.3f}   1/f {moy_rose[0]:.3f}")
    print(f"échelle {int(echelles[-1])} : blanc {moy_blanc[-1]:.3f}   1/f {moy_rose[-1]:.3f}")
    print(f"croisement : échelle {croisement}")
    print(f"figure     : {SORTIE.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
