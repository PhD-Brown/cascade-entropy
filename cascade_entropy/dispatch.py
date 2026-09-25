"""
Dispatch de la production sous contraintes.

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
    """
    Représente la solution optimiser d'un dispatch.

    On garde à la fois les grandeurs physiques du réseau (flux, injections,
    charge servie) et les grandeurs utiles pour la cascade (délestage, taux de
    charge). C'est la structure de sortie qui sera réutilisée ensuite dans les
    modules de contrôle et d'analyse.
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
        """Somme du délestage sur toutes les charges."""
        return float(self.delestage.sum())

    @property
    def taux_maximal(self) -> float:
        """M_max, indicateur témoin principal du projet."""
        return float(self.taux_de_charge.max())

    @property
    def lignes_saturees(self) -> np.ndarray:
        """
        Indices des lignes à moins de 1 % de leur limite.

        Ce sont elles, et elles seules, qui peuvent tomber lors d'une cascade.
        """
        return np.flatnonzero(self.taux_de_charge >= SEUIL_SATURATION)


def _matrices(reseau: Reseau, A: np.ndarray | None):
    """
    Construit la correspondance entre variables de décision et flux.

    Les variables de décision sont rangées dans l'ordre :
    1. production de chaque générateur,
    2. puissance servie à chaque charge.

    La matrice `selection` transforme ce vecteur de décision en injections du
    réseau : production positive sur les nœuds générateurs, consommation négative
    sur les nœuds de charge. Ensuite, on applique la matrice de flux pour obtenir
    les flux sur les lignes.
    """
    if A is None:
        A = matrice_de_flux(reseau)

    n_g, n_c = reseau.generateurs.size, reseau.charges.size
    selection = np.zeros((reseau.n_noeuds, n_g + n_c))
    selection[reseau.generateurs, np.arange(n_g)] = 1.0
    selection[reseau.charges, n_g + np.arange(n_c)] = -1.0

    autres = np.array([i for i in range(reseau.n_noeuds) if i != reseau.reference])
    # On applique A aux injections hors référence pour obtenir les flux de ligne.
    return A, selection, A @ selection[autres], n_g, n_c


def resoudre(
    reseau: Reseau,
    demande: np.ndarray,
    limites: np.ndarray | None = None,
    puissance_max: np.ndarray | None = None,
    poids_delestage: float = POIDS_DELESTAGE,
    A: np.ndarray | None = None,
) -> Solution:
    """
    Résout le dispatch pour une demande donnée.

    La logique générale est la suivante :
    - on choisit la production à chaque générateur,
    - on choisit la puissance effectivement servie à chaque charge,
    - on impose que la production totale soit égale à la charge totale servie,
    - on impose que chaque flux respecte sa limite de transport,
    - on minimise le coût économique de l'opération, avec un lourd pénalisation
      du délestage pour servir le plus possible.

    L'algorithme est un programme linéaire, ce qui est parfaitement adapté ici
    puisque les contraintes sont affines et la fonction objectif est linéaire.
    """
    demande = np.asarray(demande, dtype=float)
    if demande.shape != reseau.charges.shape:
        raise ValueError("Une valeur de demande par nœud de charge est requise.")
    if np.any(demande < 0):
        raise ValueError("La demande doit être positive.")

    # Si aucune limite n'est fournie, on prend celle du réseau par défaut.
    limites = reseau.limites if limites is None else np.asarray(limites, dtype=float)
    if limites is None:
        raise ValueError("Aucune limite de ligne définie.")
    if puissance_max is None:
        raise ValueError("Les capacités de production doivent être fournies.")
    puissance_max = np.asarray(puissance_max, dtype=float)

    A, selection, flux_par_variable, n_g, n_c = _matrices(reseau, A)

    # Coût : produire coûte 1, servir la demande rapporte W, donc le solveur préfère
    # servir autant que possible et ne délester que le strict nécessaire.
    cout = np.concatenate([np.ones(n_g), -poids_delestage * np.ones(n_c)])

    # Les contraintes de transport se traduisent par des bornes sur les flux :
    # -F_max <= F <= F_max, soit deux inégalités de la forme A x <= b.
    A_ub = np.vstack([flux_par_variable, -flux_par_variable])
    b_ub = np.concatenate([limites, limites])

    # Équilibre global : la puissance produite doit compenser exactement la charge servie.
    A_eq = np.concatenate([np.ones(n_g), -np.ones(n_c)])[None, :]
    b_eq = np.zeros(1)

    # Chaque variable de décision est bornée par 0 et par sa limite physique :
    # production <= puissance_max, charge_servie <= demande.
    bornes = [(0.0, p) for p in puissance_max] + [(0.0, d) for d in demande]

    resultat = linprog(cout, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                       bounds=bornes, method="highs")
    if not resultat.success:
        raise RuntimeError(f"Le dispatch a échoué : {resultat.message}")

    production = resultat.x[:n_g]
    charge_servie = resultat.x[n_g:]

    # On reconstruit les injections au format attendu par le module réseau.
    injections = np.zeros(reseau.n_noeuds)
    injections[reseau.generateurs] = production
    injections[reseau.charges] = -charge_servie

    # Le flux est simplement la matrice de transfert appliquée au vecteur de décision.
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
    """
    Répartit une demande totale entre les nœuds de charge.

    Cette aide est utilisée pour créer des scénarios de balayage de charge : on
    garde le même profil de demande, mais on ajuste le niveau global. On obtient
    ainsi une famille de demandes uniformes, faciles à comparer entre elles.
    """
    if poids is None:
        # Cas le plus simple : chaque charge reçoit la même quantité.
        poids = np.ones(reseau.charges.size)
    poids = np.asarray(poids, dtype=float)
    if np.any(poids < 0) or poids.sum() == 0:
        raise ValueError("Les poids doivent être positifs et de somme non nulle.")
    # La demande est d'abord distribuée selon les poids, puis normalisée pour
    # faire exactement le total voulu.
    return total * poids / poids.sum()
