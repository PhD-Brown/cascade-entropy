"""Exemples N−1 synthétiques ; aucun fichier produit et aucune donnée externe."""
from __future__ import annotations

import numpy as np

from cascade_entropy.cascade_n1 import simuler_cascade_n1
from cascade_entropy.reseau import Reseau


def triangle() -> Reseau:
    return Reseau(
        n_noeuds=3,
        lignes=np.array([[0, 1], [1, 2], [0, 2]]),
        reactances=np.ones(3),
        generateurs=np.array([0]),
        charges=np.array([1, 2]),
        limites=np.array([1.2, 2.0, 1.2]),
        reference=0,
    )


def main() -> None:
    r = triangle()
    for p1 in (0.0, 1.0):
        resultat = simuler_cascade_n1(
            r, demande=np.ones(2), puissance_max=np.array([3.]),
            ligne_initiale=0, p1=p1,
        )
        print(f"Retrait e0, p1={p1:g}")
        print(f"  Delestage avant retrait : {resultat.delestage_initial:.6f} pu")
        for i, etat in enumerate(resultat.etats):
            phase = "apres retrait initial" if i == 0 else f"apres vague {i}"
            print(f"  {phase}: delestage={etat.delestage_total:.6f} pu")
        print("  Vagues secondaires :", [v.tolist() for v in resultat.vagues_secondaires])
        print("  Lignes tombees :", resultat.lignes_tombees.tolist())
        print(f"  Fraction finale delestee : {100 * resultat.fraction_delestee:.2f}%")
        print()


if __name__ == "__main__":
    main()
