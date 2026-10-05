"""
Paramètres et fluctuations régionales pour reproduire Carreras et al. (2002).

Ce module est volontairement additif : il ne change pas la convention générale
de `cascade.journee()`. Pour l'expérience de réplication, on construit ici la
demande régionale corrélée, puis on appelle `journee(..., g=0)` afin de ne pas
appliquer une deuxième fluctuation indépendante par nœud.

Ce qui est explicitement publié dans Carreras et al. (2002)
-----------------------------------------------------------
- arbres idéalisés de 46, 94, 190 et 382 nœuds ;
- 12 générateurs, placés au troisième niveau ;
- P_L = -74 et P_G = 2623.9 dans la Table I ;
- limites de lignes par niveau :
  15620, 7748.7, 3812.9, 1844.9, 860.97, 368.99, 123.00 ;
- toutes les réactances/impédances des arbres valent 1 ;
- P_C = somme_j P_j^max ;
- facteur de charge r borné par 2-gamma <= r <= gamma ;
- gamma = 1.9 dans l'expérience statistique de la section V ;
- p0 = 1e-4 ;
- 60 000 réalisations par jeu de paramètres ;
- les charges peuvent être regroupées en régions et les nœuds d'une même
  région varient ensemble.

Interprétation de P_G
---------------------
La Table I donne P_G comme paramètre nodal des générateurs. Le texte définit
P_C comme la somme des P_j^max. Comme les 12 générateurs des arbres sont
identiques dans cette construction, nous prenons donc

    P_j^max = P_G = 2623.9
    P_C = 12 * 2623.9 = 31486.8

et NON P_C = 2623.9.

Points NON spécifiés dans l'article
-----------------------------------
Le papier ne donne pas, dans le texte disponible, la valeur numérique de N_F
(nombre de régions indépendantes), ni une loi de probabilité explicite au-delà
des bornes de r. Pour ne pas inventer silencieusement ces éléments :

- le choix de N_F est un paramètre explicite ;
- la reproduction contrôlée par défaut utilise N_F=3, correspondant aux trois
  branches principales de l'arbre, ce qui fournit des régions topologiquement
  contiguës et le même N_F pour toutes les tailles ;
- les facteurs régionaux sont tirés uniformément dans [2-gamma, gamma], qui est
  l'hypothèse déjà utilisée dans le projet pour les fluctuations, mais cette loi
  uniforme est documentée comme une hypothèse de réplication, pas comme une
  valeur publiée.

La Table I imprime "384" pour la dernière taille, tandis que le texte, la
formule N_N = 3*2^n - 2 et les figures parlent de 382 nœuds. Le code utilise
382, cohérent avec la topologie décrite dans le texte.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dispatch import SEUIL_SATURATION
from .reseau import Reseau, arbre, limites_par_niveau


P_L_TABLE = -74.0
P_G_TABLE = 2623.9

GAMMA_TABLE = 1.9
P0_TABLE = 1e-4

# p1 n'est pas redonné explicitement dans le paragraphe de la Fig. 12.
# La valeur 1.0 est donc un choix de reproduction contrôlée, cohérent avec
# le cas déterministe p1=1 discuté juste avant dans l'article.
P1_REFERENCE = 1.0

CAPACITES_TABLE = np.array(
    [15620.0, 7748.7, 3812.9, 1844.9, 860.97, 368.99, 123.00],
    dtype=float,
)

TAILLE_VERS_GENERATIONS = {
    46: 4,
    94: 5,
    190: 6,
    382: 7,
}


@dataclass(frozen=True)
class ConfigurationCarreras:
    """Réseau et paramètres fixes d'un arbre de la Table I."""

    n_noeuds: int
    reseau: Reseau
    limites: np.ndarray
    puissance_max: np.ndarray
    p_c: float

    @property
    def n_generateurs(self) -> int:
        return int(self.reseau.generateurs.size)

    @property
    def n_charges(self) -> int:
        return int(self.reseau.charges.size)

    @property
    def ratio_table(self) -> float:
        """
        P_D/P_C si chaque charge vaut |P_L|=74.

        Ce rapport est un audit de la Table I. Lors d'un balayage, P_D/P_C reste
        le paramètre de contrôle et les charges sont redimensionnées en conséquence.
        """
        return abs(P_L_TABLE) * self.n_charges / self.p_c


def configuration_arbre(n_noeuds: int) -> ConfigurationCarreras:
    """Construit exactement l'arbre et les paramètres de la Table I."""
    if n_noeuds not in TAILLE_VERS_GENERATIONS:
        raise ValueError(
            f"Taille non supportée : {n_noeuds}. "
            f"Choisir {tuple(TAILLE_VERS_GENERATIONS)}."
        )

    reseau = arbre(
        TAILLE_VERS_GENERATIONS[n_noeuds],
        branches_racine=3,
        branches=2,
        niveau_generateurs=3,
        reactance=1.0,
    )

    if reseau.n_noeuds != n_noeuds:
        raise RuntimeError(
            f"La topologie demandée ({n_noeuds}) a produit "
            f"{reseau.n_noeuds} nœuds."
        )
    if reseau.generateurs.size != 12:
        raise RuntimeError(
            f"La topologie Carreras doit avoir 12 générateurs, "
            f"pas {reseau.generateurs.size}."
        )

    limites = limites_par_niveau(reseau, CAPACITES_TABLE)
    puissance_max = np.full(reseau.generateurs.size, P_G_TABLE, dtype=float)
    p_c = float(puissance_max.sum())

    return ConfigurationCarreras(
        n_noeuds=n_noeuds,
        reseau=reseau,
        limites=limites,
        puissance_max=puissance_max,
        p_c=p_c,
    )


def groupes_trois_regions(reseau: Reseau) -> np.ndarray:
    """
    Assigne chaque charge à l'une des trois branches principales de l'arbre.

    Pour tout nœud autre que la racine, la région est définie par son ancêtre
    au niveau 1. La racine est une charge centrale qui n'appartient à aucun
    sous-arbre ; on l'affecte déterministement à la région 0. Cela donne des
    groupes presque exactement équilibrés :
      - 46 nœuds : 12, 11, 11 charges ;
      - 94 nœuds : 28, 27, 27 charges ;
      - 190 nœuds : 60, 59, 59 charges ;
      - 382 nœuds : 124, 123, 123 charges.

    Ce N_F=3 est un CHOIX DE RÉPLICATION explicite, pas une valeur numérique
    publiée par Carreras et al.
    """
    if reseau.niveaux is None:
        raise ValueError("Le réseau doit contenir ses niveaux hiérarchiques.")

    niveau1 = np.flatnonzero(reseau.niveaux == 1)
    if niveau1.size != 3:
        raise ValueError(
            "La construction régionale attend exactement trois branches de niveau 1."
        )

    parent = np.full(reseau.n_noeuds, -1, dtype=int)
    for a, b in reseau.lignes:
        # `arbre()` oriente chaque ligne du parent vers l'enfant.
        parent[int(b)] = int(a)

    region_par_noeud = np.full(reseau.n_noeuds, -1, dtype=int)
    region_par_noeud[0] = 0

    region_niveau1 = {int(noeud): i for i, noeud in enumerate(niveau1)}

    for noeud in range(1, reseau.n_noeuds):
        courant = noeud
        while parent[courant] != 0:
            courant = parent[courant]
            if courant < 0:
                raise RuntimeError("Parent introuvable dans l'arbre.")
        region_par_noeud[noeud] = region_niveau1[courant]

    groupes = region_par_noeud[reseau.charges]
    if np.any(groupes < 0):
        raise RuntimeError("Certaines charges n'ont pas reçu de région.")
    return groupes


def groupes_regions(reseau: Reseau, n_regions: int = 3) -> np.ndarray:
    """
    Retourne les groupes de fluctuation.

    N_F=3 est le choix topologique principal. N_F=1 est disponible comme contrôle
    de sensibilité (toutes les charges varient ensemble). D'autres découpages ne
    sont pas inventés ici : ils nécessiteraient une règle supplémentaire absente
    de l'article.
    """
    if n_regions == 1:
        return np.zeros(reseau.charges.size, dtype=int)
    if n_regions == 3:
        return groupes_trois_regions(reseau)
    raise ValueError(
        "Cette reproduction contrôlée supporte N_F=1 ou N_F=3. "
        "Le papier ne spécifie pas assez la partition pour inventer d'autres N_F."
    )


def facteurs_regionaux(
    n_regions: int,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Tire un facteur commun par région dans [2-gamma, gamma].

    La loi uniforme est une hypothèse explicite de réplication. L'article publie
    les bornes, mais pas une densité de probabilité détaillée dans le passage
    disponible.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng doit être un numpy.random.Generator.")
    if not 1.0 <= gamma <= 2.0:
        raise ValueError("gamma doit être dans [1, 2] pour cette reproduction.")
    return rng.uniform(2.0 - gamma, gamma, size=n_regions)


def demande_regionale(
    configuration: ConfigurationCarreras,
    ratio_pd_pc: float,
    rng: np.random.Generator,
    *,
    gamma: float = GAMMA_TABLE,
    n_regions: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit une demande corrélée par régions.

    Retourne
    --------
    demande : ndarray
        Demande réellement appliquée aux charges.
    facteurs : ndarray
        Un facteur aléatoire par région.
    groupes : ndarray
        Région de chaque charge.

    Le paramètre de contrôle reste la DEMANDE MOYENNE
        P_D = ratio_pd_pc * P_C.
    Chaque charge reçoit P_D/N_L avant fluctuation, puis toutes les charges
    d'une même région sont multipliées par le même facteur.
    """
    ratio_pd_pc = float(ratio_pd_pc)
    if not np.isfinite(ratio_pd_pc) or ratio_pd_pc <= 0:
        raise ValueError("ratio_pd_pc doit être un réel strictement positif.")

    groupes = groupes_regions(configuration.reseau, n_regions=n_regions)
    facteurs = facteurs_regionaux(n_regions, gamma, rng)

    demande_moyenne_totale = ratio_pd_pc * configuration.p_c
    base = np.full(
        configuration.n_charges,
        demande_moyenne_totale / configuration.n_charges,
        dtype=float,
    )
    demande = base * facteurs[groupes]
    return demande, facteurs, groupes


def sigma_relatif_uniforme(
    groupes: np.ndarray,
    gamma: float = GAMMA_TABLE,
) -> float:
    """
    Écart-type relatif théorique de la demande totale sous l'hypothèse uniforme.

    Pour r_f ~ U[2-gamma, gamma], Var(r_f)=(gamma-1)^2/3.
    Si w_f est la fraction de charges dans la région f,

        sigma(P_D reel / P_D moyen)
        = (gamma-1)/sqrt(3) * sqrt(sum_f w_f^2).

    Cette formule est celle de NOTRE hypothèse uniforme, pas une équation
    attribuée au papier.
    """
    groupes = np.asarray(groupes, dtype=int)
    if groupes.ndim != 1 or groupes.size == 0:
        raise ValueError("groupes doit être un vecteur non vide.")
    comptes = np.bincount(groupes)
    poids = comptes / comptes.sum()
    return float((gamma - 1.0) / np.sqrt(3.0) * np.sqrt(np.sum(poids**2)))


def sigma_relatif_papier(
    n_regions: int,
    gamma: float = GAMMA_TABLE,
) -> float:
    """
    Valeur relative correspondant à la formule publiée :
        sigma = (gamma - 1)/(2 sqrt(N_F)) * P_D.
    """
    if n_regions < 1:
        raise ValueError("n_regions doit être >= 1.")
    return float((gamma - 1.0) / (2.0 * np.sqrt(n_regions)))


def bandes_ordonnees(configuration: ConfigurationCarreras) -> list[dict]:
    """
    Frontières analytiques des bandes ordonnées de la Fig. 10 (Carreras 2002).

    Hypothèse : le délestage vide les couronnes extérieures en premier
    (`departage="exterieur_dabord"`), comme le décrit Carreras : « it reaches a
    point at which all the loads on the outermost ring of the system are blacked
    out. At this point the system behaves as a tree network with 192 nodes ».

    Pour chaque profondeur de troncature D (niveau_max, niveau_max − 1, …, 4),
    le réseau se comporte comme l'arbre restreint aux niveaux <= D :

    - début de bande : la génération étant saturée (r > 1), seules N_L / r
      charges sont servies ; toutes les charges plus profondes que D sont
      coupées dès que N_L / r <= N_L(D), soit
          r_debut(D) = N_L / N_L(D)        (= 1 pour l'arbre complet) ;
    - fin de bande : seuil de transport de l'arbre restreint, même formule
      que r_T(N) mais en ne comptant que les charges de niveau <= D en aval :
          r_fin(D) = min_l  F_l^max N_L / (n_l(D) P_C),
      sur les lignes extérieures (niveau > 3) de l'arbre restreint.

    Une bande n'existe que si r_debut < r_fin. Ces valeurs ne dépendent que de
    la Table I : aucun paramètre n'est ajusté.

    Retourne une liste de dicts : profondeur, n_charges_servies, r_debut,
    r_fin, r_fin_critere (= SEUIL_SATURATION × r_fin, seuil effectivement vu
    par la cascade) et equivalent_n_noeuds (taille de l'arbre restreint).
    """
    reseau = configuration.reseau
    if reseau.niveaux is None:
        raise ValueError("Le réseau doit contenir ses niveaux hiérarchiques.")

    niveaux = np.asarray(reseau.niveaux, dtype=int)
    enfants = reseau.lignes[:, 1].astype(int)
    niveau_ligne = niveaux[enfants]

    parent = np.full(reseau.n_noeuds, -1, dtype=int)
    for a, b in reseau.lignes:
        parent[int(b)] = int(a)

    est_charge = np.zeros(reseau.n_noeuds, dtype=bool)
    est_charge[reseau.charges] = True
    n_l = int(est_charge.sum())

    bandes = []
    for profondeur in range(int(niveaux.max()), 3, -1):
        # Charges conservées et, pour chaque nœud, nombre de ces charges en aval.
        garde = est_charge & (niveaux <= profondeur)
        en_aval = garde.astype(int)
        for noeud in np.argsort(-niveaux, kind="stable"):
            if parent[noeud] >= 0:
                en_aval[parent[noeud]] += en_aval[noeud]

        n_servies = int(garde.sum())
        r_debut = n_l / n_servies

        masque = (niveau_ligne > 3) & (niveau_ligne <= profondeur)
        charges_ligne = en_aval[enfants[masque]]
        r_lignes = (
            configuration.limites[masque] * n_l
            / (charges_ligne * configuration.p_c)
        )
        r_fin = float(r_lignes.min())

        if r_debut < r_fin:
            bandes.append({
                "profondeur": profondeur,
                "equivalent_n_noeuds": int(3 * 2 ** profondeur - 2),
                "n_charges_servies": n_servies,
                "r_debut": float(r_debut),
                "r_fin": r_fin,
                "r_fin_critere": SEUIL_SATURATION * r_fin,
            })
    return bandes
