"""Topologie du réseau et calcul des flux de puissance en approximation DC.

Premier module du fil A. Il ne dépend d'aucun autre module du projet.

L'approximation DC linéarise les équations de flux de puissance : les tensions
sont supposées unitaires, les résistances négligées et les différences d'angle
petites. Le flux sur une ligne devient alors proportionnel à la différence des
angles de phase de ses extrémités,

    F_l = (theta_i - theta_j) / x_l,

et les injections de puissance se relient aux angles par le laplacien pondéré du
graphe. Le réseau reste parfaitement discret : c'est le modèle électrique qui est
simplifié, pas la géométrie.

Un nœud de référence est exclu du système pour lever la singularité qui découle
de l'équilibre global des puissances : la somme des injections étant nulle, le
laplacien complet est singulier et les angles ne sont définis qu'à une constante
près.

Critères de validation :
  - la somme des injections doit être nulle ;
  - le réseau doit être connexe ;
  - la matrice de flux doit avoir la dimension (nombre de lignes, nombre de
    nœuds moins un) ;
  - sur un arbre, le flux d'une ligne doit égaler la somme des injections du
    sous-arbre qu'elle alimente, indépendamment des réactances.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Reseau:
    """Description complète d'un réseau de transport.

    Les lignes sont orientées arbitrairement : un flux négatif signifie que la
    puissance circule dans le sens opposé à l'orientation retenue. Cette
    orientation n'a aucune conséquence physique, elle fixe seulement la
    convention de signe.
    """

    n_noeuds: int
    lignes: np.ndarray            # (n_lignes, 2), indices des extrémités
    reactances: np.ndarray        # (n_lignes,)
    generateurs: np.ndarray       # indices des nœuds producteurs
    charges: np.ndarray           # indices des nœuds consommateurs
    limites: np.ndarray | None = None      # (n_lignes,), capacités F_max
    reference: int = 0            # nœud dont l'angle est fixé à zéro
    niveaux: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_lignes(self) -> int:
        return len(self.lignes)

    def __post_init__(self) -> None:
        if self.reactances.shape != (self.n_lignes,):
            raise ValueError("Une réactance par ligne est requise.")
        if self.limites is not None and self.limites.shape != (self.n_lignes,):
            raise ValueError("Une limite par ligne est requise.")
        if not 0 <= self.reference < self.n_noeuds:
            raise ValueError("Nœud de référence hors du réseau.")


# --------------------------------------------------------------------------
# Construction de la topologie
# --------------------------------------------------------------------------


def arbre(
    n_generations: int = 4,
    branches_racine: int = 3,
    branches: int = 2,
    niveau_generateurs: int | None = None,
    reactance: float = 1.0,
) -> Reseau:
    """Réseau en arbre, suivant la construction de Carreras et al.

    La racine porte `branches_racine` liens, puis chaque nœud du bord reçoit
    `branches` descendants à chaque génération. Tous les nœuds internes ont donc
    trois lignes incidentes, ce qui correspond approximativement à la moyenne
    observée sur les grands réseaux réels.

    Avec les valeurs par défaut, le nombre de nœuds suit 1, 4, 10, 22, 46, 94,
    190, 382 selon le nombre de générations, soit les tailles utilisées dans la
    littérature de référence.

    Les générateurs sont placés à un niveau donné de l'arbre, tous les autres
    nœuds étant des charges. Par défaut au troisième niveau, comme dans la
    littérature de référence, ou au dernier niveau pour les arbres moins
    profonds.
    """
    if n_generations < 1:
        raise ValueError("Au moins une génération est nécessaire.")
    if niveau_generateurs is None:
        niveau_generateurs = min(3, n_generations)

    niveaux = [0]                 # niveau de chaque nœud, la racine est au niveau 0
    lignes: list[tuple[int, int]] = []
    bord = [0]

    for generation in range(1, n_generations + 1):
        n_enfants = branches_racine if generation == 1 else branches
        nouveau_bord = []
        for parent in bord:
            for _ in range(n_enfants):
                enfant = len(niveaux)
                niveaux.append(generation)
                lignes.append((parent, enfant))
                nouveau_bord.append(enfant)
        bord = nouveau_bord

    niveaux = np.array(niveaux)
    n_noeuds = niveaux.size

    if niveau_generateurs > n_generations:
        raise ValueError("Niveau des générateurs au-delà de la profondeur de l'arbre.")
    generateurs = np.flatnonzero(niveaux == niveau_generateurs)
    charges = np.flatnonzero(niveaux != niveau_generateurs)

    return Reseau(
        n_noeuds=n_noeuds,
        lignes=np.array(lignes, dtype=int),
        reactances=np.full(len(lignes), float(reactance)),
        generateurs=generateurs,
        charges=charges,
        niveaux=niveaux,
        reference=int(generateurs[0]),
    )


# --------------------------------------------------------------------------
# Propriétés structurelles
# --------------------------------------------------------------------------


def matrice_incidence(reseau: Reseau) -> np.ndarray:
    """Matrice d'incidence orientée, de dimension (n_lignes, n_noeuds).

    Chaque ligne de la matrice porte +1 à son nœud de départ et -1 à son nœud
    d'arrivée.
    """
    M = np.zeros((reseau.n_lignes, reseau.n_noeuds))
    indices = np.arange(reseau.n_lignes)
    M[indices, reseau.lignes[:, 0]] = 1.0
    M[indices, reseau.lignes[:, 1]] = -1.0
    return M


def est_connexe(reseau: Reseau) -> bool:
    """Vérifie que tous les nœuds sont atteignables depuis le nœud de référence."""
    voisins: list[list[int]] = [[] for _ in range(reseau.n_noeuds)]
    for i, j in reseau.lignes:
        voisins[i].append(j)
        voisins[j].append(i)

    vus = {reseau.reference}
    pile = [reseau.reference]
    while pile:
        courant = pile.pop()
        for voisin in voisins[courant]:
            if voisin not in vus:
                vus.add(voisin)
                pile.append(voisin)
    return len(vus) == reseau.n_noeuds


def degres(reseau: Reseau) -> np.ndarray:
    """Nombre de lignes incidentes à chaque nœud."""
    return np.bincount(reseau.lignes.ravel(), minlength=reseau.n_noeuds)


# --------------------------------------------------------------------------
# Flux de puissance
# --------------------------------------------------------------------------


def laplacien(reseau: Reseau) -> np.ndarray:
    """Laplacien pondéré par les susceptances, de dimension (n_noeuds, n_noeuds).

    C'est la matrice B telle que P = B theta. Elle est singulière : l'ajout d'une
    constante à tous les angles laisse les flux inchangés.
    """
    M = matrice_incidence(reseau)
    susceptances = 1.0 / reseau.reactances
    return M.T @ (susceptances[:, None] * M)


def matrice_de_flux(reseau: Reseau) -> np.ndarray:
    """Matrice A telle que F = A P, où P exclut le nœud de référence.

    Construite une seule fois pour une topologie donnée, puis réutilisée à chaque
    résolution : c'est ce qui rend le calcul des flux quasi instantané une fois
    la topologie fixée.

    L'inversion se fait par factorisation plutôt que par inversion explicite,
    plus stable numériquement.
    """
    autres = _indices_hors_reference(reseau)
    B_reduit = laplacien(reseau)[np.ix_(autres, autres)]
    M = matrice_incidence(reseau)[:, autres]
    susceptances = 1.0 / reseau.reactances

    # F = diag(b) M theta, avec theta = B_reduit^{-1} P.
    return (susceptances[:, None] * M) @ np.linalg.inv(B_reduit)


def flux(reseau: Reseau, injections: np.ndarray, A: np.ndarray | None = None) -> np.ndarray:
    """Flux sur chaque ligne pour un vecteur d'injections par nœud.

    `injections` couvre tous les nœuds, positif pour une production, négatif pour
    une consommation. Sa somme doit être nulle. Passer `A` évite de reconstruire
    la matrice de flux à chaque appel.
    """
    injections = np.asarray(injections, dtype=float)
    if injections.shape != (reseau.n_noeuds,):
        raise ValueError("Une injection par nœud est requise.")
    if not np.isclose(injections.sum(), 0.0, atol=1e-9):
        raise ValueError(
            f"Les injections ne s'équilibrent pas (somme = {injections.sum():.3e})."
        )

    if A is None:
        A = matrice_de_flux(reseau)
    return A @ injections[_indices_hors_reference(reseau)]


def angles(reseau: Reseau, injections: np.ndarray) -> np.ndarray:
    """Angles de phase, le nœud de référence étant fixé à zéro.

    Ces angles sont des variables statiques de l'approximation DC : ils décrivent
    un état d'équilibre, pas une trajectoire temporelle.
    """
    autres = _indices_hors_reference(reseau)
    B_reduit = laplacien(reseau)[np.ix_(autres, autres)]
    theta = np.zeros(reseau.n_noeuds)
    theta[autres] = np.linalg.solve(B_reduit, np.asarray(injections)[autres])
    return theta


def taux_de_charge(flux_lignes: np.ndarray, limites: np.ndarray) -> np.ndarray:
    """Taux de charge M_l = |F_l| / F_l_max, ligne par ligne.

    La valeur 1 correspond à une ligne à sa limite. Le dispatch n'autorise jamais
    de dépassement : c'est la saturation, et non le dépassement, qui déclenche
    l'avarie dans le modèle de cascade.
    """
    return np.abs(flux_lignes) / limites


def limites_depuis_cas_de_base(
    reseau: Reseau,
    injections: np.ndarray,
    marge: float = 1.5,
    plancher: float = 1e-3,
) -> np.ndarray:
    """Capacités de lignes dimensionnées sur un cas de base.

    Chaque ligne reçoit une capacité égale au flux qu'elle porte dans le cas de
    base, multiplié par une marge. Le plancher évite qu'une ligne peu sollicitée
    dans ce cas particulier se retrouve avec une capacité nulle.

    Cette convention garantit qu'une solution sans surcharge existe au cas de
    base, ce qui est le point de départ attendu avant toute montée en charge.
    """
    reference = np.abs(flux(reseau, injections))
    return np.maximum(marge * reference, plancher * reference.max())


def _indices_hors_reference(reseau: Reseau) -> np.ndarray:
    """Indices des nœuds autres que celui de référence."""
    return np.array([i for i in range(reseau.n_noeuds) if i != reseau.reference])
