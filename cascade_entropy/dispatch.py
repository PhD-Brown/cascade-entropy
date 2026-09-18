"""Dispatch de la production sous contraintes.

Deuxième module du fil A. Étant donné une demande, il décide quel générateur
produit combien et quelle charge est servie, en respectant les capacités de
production et les limites de transport.

Le problème est linéaire. Les variables de décision sont la production de chaque
générateur et la puissance effectivement servie à chaque charge. Le coût vaut une
unité par unité produite et moins W unités par unité servie, avec W grand : servir
la charge rapporte donc cent fois plus que produire ne coûte, ce qui revient à
dire « sers le maximum, et si c'est impossible, coupe le moins possible ».

    minimiser   sum_i G_i  -  W sum_j L_j

    sous        0 <= G_i <= P_i^max          capacités de production
                0 <= L_j <= d_j              on ne sert jamais plus que demandé
                sum_i G_i = sum_j L_j        équilibre global
                |F_l| <= F_l^max             limites de transport

Les flux se déduisent des injections par la matrice de flux du module `reseau`,
ce qui rend les contraintes de transport linéaires en les variables de décision.

Point essentiel pour la suite : le dispatch ne produit jamais de ligne au-dessus
de sa limite. Une ligne peut au mieux être saturée, c'est-à-dire collée contre sa
contrainte. C'est cette saturation, et non un dépassement, qui déclenchera
l'avarie dans le module `cascade`.

Critères de validation : sur un réseau réduit résolu à la main, les flux doivent
coïncider ; à faible charge, la solution ne doit comporter ni délestage ni ligne
saturée ; au-delà de la capacité totale de production, le délestage doit égaler
exactement l'excédent de demande.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .reseau import Reseau, matrice_de_flux

POIDS_DELESTAGE = 100.0
SEUIL_SATURATION = 0.99


@dataclass
class Solution:
    """Résultat d'un dispatch.

    `injections` couvre tous les nœuds, positif en production et négatif en
    consommation ; c'est la grandeur que le module `reseau` attend. `delestage`
    est la part de la demande qui n'a pas pu être servie, charge par charge.
    """

    injections: np.ndarray
    production: np.ndarray
    charge_servie: np.ndarray
    delestage: np.ndarray
    flux: np.ndarray
    taux_de_charge: np.ndarray
    cout: float

    @property
    def delestage_total(self) -> float:
        return float(self.delestage.sum())

    @property
    def taux_maximal(self) -> float:
        """M_max, indicateur témoin principal du projet."""
        return float(self.taux_de_charge.max())

    @property
    def lignes_saturees(self) -> np.ndarray:
        """Indices des lignes à moins de 1 % de leur limite.

        Ce sont elles, et elles seules, qui peuvent tomber lors d'une cascade.
        """
        return np.flatnonzero(self.taux_de_charge >= SEUIL_SATURATION)


def _matrices(reseau: Reseau, A: np.ndarray | None):
    """Construit la correspondance entre variables de décision et flux.

    Les variables sont rangées dans l'ordre [production, charge servie]. La
    matrice de sélection convertit ce vecteur en injections par nœud : identité
    aux nœuds générateurs, opposé aux nœuds de charge.
    """
    if A is None:
        A = matrice_de_flux(reseau)

    n_g, n_c = reseau.generateurs.size, reseau.charges.size
    selection = np.zeros((reseau.n_noeuds, n_g + n_c))
    selection[reseau.generateurs, np.arange(n_g)] = 1.0
    selection[reseau.charges, n_g + np.arange(n_c)] = -1.0

    autres = np.array([i for i in range(reseau.n_noeuds) if i != reseau.reference])
    return A, selection, A @ selection[autres], n_g, n_c


def resoudre(
    reseau: Reseau,
    demande: np.ndarray,
    limites: np.ndarray | None = None,
    puissance_max: np.ndarray | None = None,
    poids_delestage: float = POIDS_DELESTAGE,
    A: np.ndarray | None = None,
) -> Solution:
    """Résout le dispatch pour une demande donnée.

    `demande` est positive et définie sur les nœuds de charge uniquement, dans
    l'ordre de `reseau.charges`. `limites` et `puissance_max` remplacent au besoin
    celles du réseau, ce qui permet de rejouer un dispatch après avarie sans
    reconstruire le réseau. Passer `A` évite de reconstruire la matrice de flux.

    Lève une erreur si le solveur échoue : un problème infaisable signale une
    incohérence du modèle, jamais un état physique valide, puisque tout délester
    est toujours une solution admissible.
    """
    demande = np.asarray(demande, dtype=float)
    if demande.shape != reseau.charges.shape:
        raise ValueError("Une valeur de demande par nœud de charge est requise.")
    if np.any(demande < 0):
        raise ValueError("La demande doit être positive.")

    limites = reseau.limites if limites is None else np.asarray(limites, dtype=float)
    if limites is None:
        raise ValueError("Aucune limite de ligne définie.")
    if puissance_max is None:
        raise ValueError("Les capacités de production doivent être fournies.")
    puissance_max = np.asarray(puissance_max, dtype=float)

    A, selection, flux_par_variable, n_g, n_c = _matrices(reseau, A)

    # Coût : produire coûte 1, servir rapporte W.
    cout = np.concatenate([np.ones(n_g), -poids_delestage * np.ones(n_c)])

    # Transport : le flux reste entre -F_max et +F_max, soit deux inégalités.
    A_ub = np.vstack([flux_par_variable, -flux_par_variable])
    b_ub = np.concatenate([limites, limites])

    # Équilibre global : tout ce qui est produit est servi.
    A_eq = np.concatenate([np.ones(n_g), -np.ones(n_c)])[None, :]
    b_eq = np.zeros(1)

    bornes = [(0.0, p) for p in puissance_max] + [(0.0, d) for d in demande]

    resultat = linprog(cout, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                       bounds=bornes, method="highs")
    if not resultat.success:
        raise RuntimeError(f"Le dispatch a échoué : {resultat.message}")

    production = resultat.x[:n_g]
    charge_servie = resultat.x[n_g:]

    injections = np.zeros(reseau.n_noeuds)
    injections[reseau.generateurs] = production
    injections[reseau.charges] = -charge_servie

    flux = flux_par_variable @ resultat.x

    return Solution(
        injections=injections,
        production=production,
        charge_servie=charge_servie,
        delestage=demande - charge_servie,
        flux=flux,
        taux_de_charge=np.abs(flux) / limites,
        cout=float(resultat.fun),
    )


def demande_uniforme(reseau: Reseau, total: float,
                     poids: np.ndarray | None = None) -> np.ndarray:
    """Répartit une demande totale entre les nœuds de charge.

    Sans `poids`, la répartition est égale. Avec, elle est proportionnelle, ce
    qui permet de garder un profil de charge fixe tout en faisant varier le
    niveau global — l'expérience de balayage du modèle de référence.
    """
    if poids is None:
        poids = np.ones(reseau.charges.size)
    poids = np.asarray(poids, dtype=float)
    if np.any(poids < 0) or poids.sum() == 0:
        raise ValueError("Les poids doivent être positifs et de somme non nulle.")
    return total * poids / poids.sum()
