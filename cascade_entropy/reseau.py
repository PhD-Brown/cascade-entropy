"""
Topologie du réseau et calcul des flux de puissance en approximation DC.

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
    """
    Description complète d'un réseau de transport.

    Une instance de cette classe contient toute la géométrie du réseau :
    - le nombre de nœuds,
    - la liste des lignes connectées,
    - leurs réactances,
    - les nœuds producteurs et consommateurs,
    - les limites de transport éventuelles.

    Les lignes sont orientées arbitrairement : un flux négatif signifie que la
    puissance circule dans le sens opposé à l'orientation retenue. Cette
    orientation n'a aucune conséquence physique, elle fixe seulement la
    convention de signe.
    """

    n_noeuds: int
    lignes: np.ndarray            # (n_lignes, 2), indices des extrémités
    reactances: np.ndarray        # (n_lignes,), résistance électrique linéarisée
    generateurs: np.ndarray       # indices des nœuds producteurs
    charges: np.ndarray           # indices des nœuds consommateurs
    limites: np.ndarray | None = None      # (n_lignes,), capacités F_max
    reference: int = 0            # nœud dont l'angle est fixé à zéro
    niveaux: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_lignes(self) -> int:
        """Nombre total de lignes du réseau."""
        return len(self.lignes)

    def __post_init__(self) -> None:
        """Vérifie rapidement que le réseau est cohérent avant utilisation."""
        # Une réactance par ligne est nécessaire pour définir la susceptance de la branche.
        if self.reactances.shape != (self.n_lignes,):
            raise ValueError("Une réactance par ligne est requise.")
        # Si des limites sont fournies, il faut en avoir une par ligne.
        if self.limites is not None and self.limites.shape != (self.n_lignes,):
            raise ValueError("Une limite par ligne est requise.")
        # Le nœud de référence doit appartenir au graphe ; sinon, le système de résolution est mal posé.
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
    """
    Construit un arbre hiérarchique avec un niveau de générateurs fixé.

    La logique est simple : on part de la racine, puis on crée des descendants
    successifs. Chaque génération ajoute de nouveaux nœuds, de sorte qu'un arbre
    de profondeur n contient un nombre de nœuds bien défini et une topologie
    reproductible.

    Cette construction est importante car elle produit les graphes utilisés dans
    les simulations et les tests : ce sont des arbres dont les propriétés de
    calcul sont analytiques, notamment pour vérifier les flux.
    """
    if n_generations < 1:
        raise ValueError("Au moins une génération est nécessaire.")
    if niveau_generateurs is None:
        niveau_generateurs = min(3, n_generations)

    # Chaque nœud est repéré par son niveau dans l'arbre: 0 pour la racine.
    niveaux = [0]
    lignes: list[tuple[int, int]] = []
    bord = [0]

    # On étend l'arbre génération par génération.
    # À chaque étape, chaque nœud du bord crée plusieurs enfants.
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

    # Les générateurs sont les nœuds d'un niveau choisi ; les autres sont des charges.
    generateurs = np.flatnonzero(niveaux == niveau_generateurs)
    charges = np.flatnonzero(niveaux != niveau_generateurs)

    return Reseau(
        n_noeuds=n_noeuds,
        lignes=np.array(lignes, dtype=int),
        reactances=np.full(len(lignes), float(reactance)),
        generateurs=generateurs,
        charges=charges,
        niveaux=niveaux,
        # fixe le premier générateur comme nœud de référence
        reference=int(generateurs[0]),
    )


# --------------------------------------------------------------------------
# Propriétés structurelles
# --------------------------------------------------------------------------


def matrice_incidence(reseau: Reseau) -> np.ndarray:
    """
    Construit la matrice d'incidence orientée, de dimension (n_lignes, n_noeuds).

    Chaque ligne correspond à une branche. Sur cette ligne, on marque :
    - +1 au nœud de départ,
    - -1 au nœud d'arrivée.

    Cette matrice est l'outil central pour écrire le laplacien du réseau et ainsi
    relier injections et angles de phase.
    """
    M = np.zeros((reseau.n_lignes, reseau.n_noeuds))
    indices = np.arange(reseau.n_lignes)
    M[indices, reseau.lignes[:, 0]] = 1.0   # le flux part du premier nœud
    M[indices, reseau.lignes[:, 1]] = -1.0  # et arrive au second nœud
    return M


def est_connexe(reseau: Reseau) -> bool:
    """
    Vérifie qu'un graphe est entièrement connecté.

    On part du nœud de référence, puis on explose le graphe en profondeur via une pile
    de parcours. Si on visite tous les nœuds, le réseau est connexe ; sinon, il
    comporte au moins une composante isolée.
    """
    voisins: list[list[int]] = [[] for _ in range(reseau.n_noeuds)]
    for i, j in reseau.lignes:
        voisins[int(i)].append(int(j))
        voisins[int(j)].append(int(i))

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
    """
    Compte le nombre de lignes incidentes à chaque nœud.
    Carreras vise une moyenne de 3 lignes par nœud, 
    (proche de la réalité des grands réseaux).
    """
    return np.bincount(reseau.lignes.ravel(), minlength=reseau.n_noeuds)


# --------------------------------------------------------------------------
# Flux de puissance
# --------------------------------------------------------------------------


def laplacien(reseau: Reseau) -> np.ndarray:
    """
    Construit le laplacien pondéré du réseau.

    On multiplie la matrice d'incidence par ses susceptances, afin d'obtenir la
    matrice B qui relie injections et angles : P = B * theta. Dans ce modèle,
    la puissance injectée à un nœud dépend de la différence d'angle avec ses
    voisins et de la susceptance de la ligne.
    """
    M = matrice_incidence(reseau)
    susceptances = 1.0 / reseau.reactances
    return M.T @ (susceptances[:, None] * M)


def matrice_de_flux(reseau: Reseau) -> np.ndarray:
    """
    Calcule la matrice de transfert des flux.

    Le but est de passer directement des injections de puissance aux flux de ligne
    par une relation linéaire F = A * P, sans refaire chaque fois toute la
    résolution du système. C'est un gain de performance majeur lorsqu'on simule
    beaucoup de configurations.
    """
    autres = _indices_hors_reference(reseau)
    B_reduit = laplacien(reseau)[np.ix_(autres, autres)]
    M = matrice_incidence(reseau)[:, autres]
    susceptances = 1.0 / reseau.reactances

    # F = diag(b) * M * theta, et theta = B_reduit^{-1} * P.
    # On construit la matrice A directement pour les nœuds hors référence.
    return (susceptances[:, None] * M) @ np.linalg.inv(B_reduit)


def flux(reseau: Reseau, injections: np.ndarray, A: np.ndarray | None = None) -> np.ndarray:
    """
    Calcule les flux sur les lignes à partir des injections.

    Les injections positives correspondent à la production, négatives à la
    consommation. La somme des injections doit être nulle : l'énergie produite
    doit être égale à l'énergie consommée, sinon le système ne représente pas un
    point d'équilibre physique.
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
    # La matrice A donne le flux sur chaque ligne à partir des injections hors référence.
    return A @ injections[_indices_hors_reference(reseau)]


def angles(reseau: Reseau, injections: np.ndarray) -> np.ndarray:
    """
    Résout les angles de phase du réseau dans l'approximation DC.

    Les angles sont définis à une constante près, donc on fixe le nœud de
    référence à zéro. Cela rend le système solvable sans ambiguïté.
    """
    autres = _indices_hors_reference(reseau)
    B_reduit = laplacien(reseau)[np.ix_(autres, autres)]
    theta = np.zeros(reseau.n_noeuds)
    theta[autres] = np.linalg.solve(B_reduit, np.asarray(injections)[autres])
    return theta


def taux_de_charge(flux_lignes: np.ndarray, limites: np.ndarray) -> np.ndarray:
    """
    Calcule le rapport entre le flux et la capacité supportée par chaque ligne.

    Une valeur de 1 signifie que la ligne est exactement à sa limite. Les modèles
    de cascade traitent la saturation comme un seuil critique : la ligne peut
    saturer, mais on ne veut pas qu'elle dépasse sa capacité dans le modèle.
    """
    return np.abs(flux_lignes) / limites


def limites_par_niveau(reseau: Reseau, capacites: np.ndarray | None = None) -> np.ndarray:
    """
    Retourne une capacité par ligne selon la profondeur du nœud aval.

    L'idée est qu'une ligne proche de la racine doit transporter la puissance de
    tout le sous-arbre qu'elle alimente. Les capacités décroissent donc en
    fonction de la hiérarchie. C'est une convention de modélisation, pas un
    théorème physique universel.
    """
    if capacites is None:
        # Valeurs de référence pour les arbres de profondeur croissante.
        capacites = np.array([15620.0, 7748.7, 3812.9, 1844.9, 860.97, 368.99, 123.00])

    # Le niveau d'une ligne est celui du nœud enfant, donc le sous-arbre qu'elle contient.
    niveau_ligne = reseau.niveaux[reseau.lignes[:, 1]] - 1
    if niveau_ligne.max() >= capacites.size:
        raise ValueError("Pas assez de capacités fournies pour la profondeur de l'arbre.")
    return capacites[niveau_ligne]


def limites_depuis_cas_de_base(
    reseau: Reseau,
    injections: np.ndarray,
    marge: float = 1.5,
    plancher: float = 1e-3,
) -> np.ndarray:
    """
    Dimensionne les limites à partir d'un cas de base.

    On prend le flux observé dans le cas de base, on le multiplie par une marge de
    sécurité, puis on impose un plancher minimal. Cela garantit que le cas de base
    est sans surcharge et qu'une ligne peu chargée ne se retrouve pas avec une
    capacité nulle.
    """
    reference = np.abs(flux(reseau, injections))
    return np.maximum(marge * reference, plancher * reference.max())


def _indices_hors_reference(reseau: Reseau) -> np.ndarray:
    """Renvoie les indices de tous les nœuds sauf le nœud de référence."""
    return np.array([i for i in range(reseau.n_noeuds) if i != reseau.reference])
