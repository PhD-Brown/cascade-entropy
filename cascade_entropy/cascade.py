"""
Cascade de pannes : avaries aléatoires, redispatch et convergence.

Troisième module du fil A. Il orchestre `reseau` et `dispatch` en boucle pour
simuler une « journée » : une réalisation possible de la demande et des avaries
de ligne, jusqu'à ce que plus aucune ligne ne soit en surcharge.

Trois sources d'aléatoire interviennent, à des moments distincts et avec des
rôles distincts. Toutes proviennent d'un seul générateur (`numpy.random.Generator`)
transmis en paramètre à `journee()` — jamais d'un état aléatoire global — pour
garantir la reproductibilité à graine fixée. L'ordre des tirages est fixe :
demande, puis avaries accidentelles, puis avaries par surcharge.

1. Fluctuation de la demande (avant tout dispatch)
   La demande du jour n'est pas fixe : chaque charge est multipliée par un
   facteur tiré indépendamment, uniformément dans [1 - g, 1 + g], autour de la
   demande moyenne. C'est le paramètre g de Carreras et al. (2002, section V),
   et g = 0 supprime toute fluctuation. C'est ce qui permet à une ligne de
   saturer certains jours et pas d'autres, même en l'absence de toute avarie.

   Correspondance de convention avec Carreras et al. (2002, section V) :
   g = 1,9 chez eux correspond à g = 0,9 ici.

   Hypothèse de modélisation : le facteur est indépendant d'un nœud de charge à
   l'autre (et non commun à tout le réseau) et de loi uniforme. La demande du
   jour est renvoyée dans `Journee.demande`.

2. Avarie accidentelle (avant le premier dispatch)
   Indépendamment pour chaque ligne, un tirage de Bernoulli(p0) détermine si
   elle subit une panne accidentelle, avant même de résoudre le dispatch.
   Carreras et al. (2004) : « we also assign a probability p0 for a random
   outage of a line. This value represents possible failures caused by
   phenomena such as accidents and weather related events. » La plupart des
   jours, ce tirage ne touche aucune ligne.

3. Avarie par surcharge (dans la boucle de cascade)
   Après chaque résolution du dispatch, chaque ligne dont le taux de charge
   atteint le seuil de saturation (`SEUIL_SATURATION = 0,99`, défini dans
   `dispatch.py`) subit, indépendamment, un tirage de Bernoulli(p1) qui
   détermine si elle tombe à son tour. Carreras et al. (2002) : « we assume
   that there is a probability, p1, that an overloaded line will suffer an
   outage. » Si au moins une ligne tombe, on redispatch et on recommence ;
   sinon, la journée est terminée et le délestage total de la dernière
   solution est la taille du blackout du jour.

Une ligne tombée (par p0 ou par p1) n'est jamais retirée du réseau : sa
réactance est multipliée par 1e6 et sa limite par 1e-6, comme validé dans le
notebook 02 — et elle est explicitement exclue de tout calcul de taux de
charge pour le reste de la journée, pour éviter qu'une ligne morte (flux et
capacité quasi nuls) ne soit prise pour une ligne saturée. Le taux de charge
d'une ligne morte, tel que le renvoie `Solution`, ne veut donc rien dire :
on lit `Journee.hors_service` et `Journee.taux_maximal`, jamais
`Journee.solution.taux_maximal`.

Dégénérescence du dispatch. Plusieurs répartitions peuvent avoir le même coût
optimal avec des flux différents. Les lignes saturées, donc les avaries par
surcharge, dépendent du choix fait parmi elles. `journee(..., departage=...)`
rend ce choix explicite et le transmet à chaque dispatch de la journée :
  - ``"highs"`` (défaut) : choix du solveur, sans règle — comportement
    historique, conservé à l'identique ;
  - ``"exterieur_dabord"`` : à puissance servie égale, les charges les plus
    extérieures de l'arbre sont délestées en premier (voir `dispatch`).
Le délestage total à chaque dispatch est identique pour les deux règles ; ce
sont la localisation du délestage, les lignes saturées et donc la suite de la
cascade qui changent. Toute conclusion doit préciser la règle utilisée.

Critères de validation :
  - p0 = 0 et p1 = 0 : `journee()` produit exactement le même résultat qu'un
    simple `dispatch.resoudre()` sur la demande tirée ce jour-là — aucune
    avarie ne doit apparaître ;
  - graine fixée : la séquence d'avaries, le nombre d'itérations et le
    délestage final sont reproductibles ;
  - une ligne hors service n'est jamais retenue comme candidate à une
    nouvelle avarie par surcharge, quel que soit son taux de charge apparent ;
  - la boucle termine en au plus `n_lignes + 1` dispatchs (un dispatch initial,
    puis un par vague d'avaries par surcharge), puisqu'une ligne ne peut tomber
    qu'une seule fois et que chaque vague en fait tomber au moins une.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real

import numpy as np

from .dispatch import (
    DEPARTAGE_HIGHS,
    SEUIL_SATURATION,
    Solution,
    resoudre,
    valider_departage,
)
from .reseau import Reseau, matrice_de_flux

FACTEUR_REACTANCE_AVARIE = 1e6
FACTEUR_LIMITE_AVARIE = 1e-6


@dataclass
class Journee:
    """
    Résultat d'une journée simulée.

    On garde la demande tirée, la séquence des avaries (accidentelles puis
    vague par vague) et deux solutions du dispatch. La dernière décrit l'état du
    réseau à la fin de la cascade : son délestage total est la taille du blackout
    du jour. La première est celle du tout premier dispatch, avant toute avarie
    par surcharge : c'est l'état que le réseau présentait avant que la cascade
    ne le déforme, et donc la bonne base pour observer la répartition des flux
    sans la confondre avec le blackout lui-même.
    """

    demande: np.ndarray
    solution: Solution
    solution_initiale: Solution
    hors_service: np.ndarray
    avaries_accidentelles: np.ndarray
    avaries_par_surcharge: list[np.ndarray]
    n_iterations: int
    # Nombre de dispatchs de la journée résolus par une stratégie de secours
    # du solveur (voir `dispatch.STRATEGIES_SOLVEUR`) ; 0 presque toujours.
    n_secours: int = 0

    @property
    def delestage_total(self) -> float:
        """Taille du blackout du jour : délestage de la dernière solution."""
        return self.solution.delestage_total

    @property
    def lignes_tombees(self) -> np.ndarray:
        """Indices de toutes les lignes tombées, dans l'ordre chronologique."""
        return np.concatenate([self.avaries_accidentelles, *self.avaries_par_surcharge])

    @property
    def hors_service_initial(self) -> np.ndarray:
        """Masque des lignes hors service lors du premier dispatch (avaries p0 seules)."""
        masque = np.zeros(self.hors_service.size, dtype=bool)
        masque[self.avaries_accidentelles] = True
        return masque

    @property
    def taux_maximal(self) -> float:
        """
        M_max sur les seules lignes en service, à la fin de la cascade.

        `Solution.taux_maximal` inclurait les lignes mortes, dont le taux de
        charge (rapport de deux quantités quasi nulles) n'a aucun sens.
        """
        taux = self.solution.taux_de_charge[~self.hors_service]
        return float(taux.max()) if taux.size else 0.0

    @property
    def taux_maximal_initial(self) -> float:
        """M_max du premier dispatch, sur les lignes encore en service à ce moment."""
        taux = self.solution_initiale.taux_de_charge[~self.hors_service_initial]
        return float(taux.max()) if taux.size else 0.0


def _probabilite(valeur, nom: str) -> float:
    """Exige un nombre réel dans [0, 1] ; refuse les booléens et les NaN."""
    if isinstance(valeur, (bool, np.bool_)) or not isinstance(valeur, Real):
        raise ValueError(f"{nom} doit être un nombre dans [0, 1].")
    if not 0.0 <= valeur <= 1.0:
        raise ValueError(f"{nom} doit être un nombre dans [0, 1].")
    return float(valeur)


def _redispatch(reseau: Reseau, demande: np.ndarray, limites: np.ndarray,
                puissance_max: np.ndarray, hors_service: np.ndarray,
                departage: str = DEPARTAGE_HIGHS) -> Solution:
    """
    Résout le dispatch avec les lignes hors service dégradées.

    On repart toujours du réseau et des limites d'origine, sans jamais les
    modifier : la réactance de chaque ligne morte est multipliée par 1e6 et sa
    limite par 1e-6. La matrice de flux dépend des réactances : elle doit être
    recalculée après chaque avarie, on ne peut pas réutiliser celle du jour 0.
    """
    reactances = np.where(hors_service,
                          reseau.reactances * FACTEUR_REACTANCE_AVARIE,
                          reseau.reactances)
    limites_courantes = np.where(hors_service,
                                 limites * FACTEUR_LIMITE_AVARIE, limites)
    reseau_courant = replace(reseau, reactances=reactances)
    return resoudre(reseau_courant, demande, limites=limites_courantes,
                    puissance_max=puissance_max,
                    A=matrice_de_flux(reseau_courant),
                    departage=departage)


def journee(
    reseau: Reseau,
    demande_moyenne: np.ndarray,
    puissance_max: np.ndarray,
    p0: float,
    p1: float,
    g: float,
    rng: np.random.Generator,
    limites: np.ndarray | None = None,
    departage: str = DEPARTAGE_HIGHS,
) -> Journee:
    """
    Simule une journée : demande aléatoire, avaries, redispatch jusqu'à convergence.

    Déroulement :
    1. la demande du jour est tirée autour de `demande_moyenne` (fluctuation g) ;
    2. chaque ligne tombe accidentellement avec la probabilité p0 ;
    3. on résout le dispatch ;
    4. chaque ligne en service et saturée tombe avec la probabilité p1 ; si au
       moins une tombe, on redispatch et on revient à l'étape 4, sinon on s'arrête.

    Les tirages utilisent `rng.random(n) < p` : avec p = 0, aucune ligne ne peut
    tomber ; avec p = 1, toutes les candidates tombent. Le `Reseau` et les
    limites de l'appelant ne sont jamais modifiés. `limites` vaut par défaut
    `reseau.limites`.

    `departage` (``"highs"`` par défaut, ou ``"exterieur_dabord"``) est appliqué
    à tous les dispatchs de la journée, initial comme après chaque vague. Il ne
    consomme aucun tirage aléatoire : à graine égale, les deux règles voient la
    même demande et les mêmes avaries p0.
    """
    departage = valider_departage(departage)
    demande_moyenne = np.asarray(demande_moyenne, dtype=float)
    if demande_moyenne.shape != reseau.charges.shape:
        raise ValueError("Une valeur de demande par nœud de charge est requise.")
    if np.any(demande_moyenne < 0):
        raise ValueError("La demande doit être positive.")
    p0 = _probabilite(p0, "p0")
    p1 = _probabilite(p1, "p1")
    g = _probabilite(g, "g")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng doit être un numpy.random.Generator.")

    limites = reseau.limites if limites is None else limites
    if limites is None:
        raise ValueError("Aucune limite de ligne définie.")
    limites = np.asarray(limites, dtype=float)
    if limites.shape != (reseau.n_lignes,):
        raise ValueError("Une limite par ligne est requise.")

    # 1. Fluctuation de la demande : un facteur uniforme par nœud de charge.
    demande = demande_moyenne * rng.uniform(1.0 - g, 1.0 + g,
                                            size=demande_moyenne.shape)

    # 2. Avaries accidentelles, avant tout dispatch.
    hors_service = rng.random(reseau.n_lignes) < p0
    avaries_accidentelles = np.flatnonzero(hors_service)

    # 3. Premier dispatch, conservé tel quel pour l'observation avant cascade.
    solution = _redispatch(reseau, demande, limites, puissance_max, hors_service,
                           departage)
    solution_initiale = solution
    n_iterations = 1
    n_secours = int(solution.secours)

    # 4. Cascade. Les candidates sont les lignes saturées ET en service : une
    # ligne morte a un taux de charge apparent d'environ 1 qu'il faut ignorer.
    avaries_par_surcharge: list[np.ndarray] = []
    while True:
        candidates = np.flatnonzero(
            (solution.taux_de_charge >= SEUIL_SATURATION) & ~hors_service
        )
        tombent = candidates[rng.random(candidates.size) < p1]
        if tombent.size == 0:
            break
        avaries_par_surcharge.append(tombent)
        hors_service[tombent] = True
        solution = _redispatch(reseau, demande, limites, puissance_max, hors_service,
                               departage)
        n_iterations += 1
        n_secours += int(solution.secours)

    return Journee(
        demande=demande,
        solution=solution,
        solution_initiale=solution_initiale,
        hors_service=hors_service,
        avaries_accidentelles=avaries_accidentelles,
        avaries_par_surcharge=avaries_par_surcharge,
        n_iterations=n_iterations,
        n_secours=n_secours,
    )
