"""Propagation N−1 avec suppression exacte des lignes et redispatch par îlot.

Projet I — Validation A6, uniquement sur des graphes synthétiques pour le moment.

L'algorithme *ne modifie pas* ``cascade.journee`` ni ``n1.redispatch_exact``.
Il réutilise ce dernier pour chaque état après une vague de défaillances.

Convention de validation (pas encore décision du protocole scientifique) :
* chaque scénario part d'un réseau et d'une demande inchangés ;
* une ligne est retirée de façon déterministe à t=0 ;
* après chaque redispatch, les lignes actives ayant un taux de charge >= 0.99
  sont candidates à une panne secondaire ; chacune tombe avec probabilité p1 ;
* si aucune candidate ne tombe lors d'une vague, la cascade s'arrête ;
* p1 doit être fourni explicitement. Les essais p1=1 sont déterministes.

Le programme linéaire de dispatch limite déjà les flux à la capacité :
« surchargée » signifie donc ici *saturée* au seuil 0.99, et non |F| > C.
Cette convention historique devra être explicitement validée avec Patrick.

Tous les indices de ligne correspondent au réseau original. Aucune donnée à
accès restreint n'est requise. Aucune écriture sur disque n'est effectuée.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .dispatch import SEUIL_SATURATION
from .n1 import ResultatRedispatchN1, redispatch_exact
from .reseau import Reseau


@dataclass
class ResultatCascadeN1:
    """Historique complet d'une seule cascade N−1.

    ``etat_reference`` décrit le réseau intact AVANT la perte.
    ``etats`` contient l'état immédiatement APRÈS le retrait initial, puis
    un état après chaque vague de défaillances secondaires. Il contient
    donc toujours au moins une entrée.
    """

    ligne_initiale: int
    p1: float
    demande: np.ndarray
    etat_reference: ResultatRedispatchN1
    etats: list[ResultatRedispatchN1]
    vagues_secondaires: list[np.ndarray]

    @property
    def etat_final(self) -> ResultatRedispatchN1:
        return self.etats[-1]

    @property
    def hors_service(self) -> np.ndarray:
        return self.etat_final.hors_service.copy()

    @property
    def lignes_tombees(self) -> np.ndarray:
        """Déclencheur suivi des lignes tombées dans chaque vague secondaire."""
        return np.concatenate(
            [np.array([self.ligne_initiale], dtype=int), *self.vagues_secondaires]
        )

    @property
    def n_vagues_secondaires(self) -> int:
        return len(self.vagues_secondaires)

    @property
    def n_redispatchs(self) -> int:
        """Nombre de redispatchs post-défaillance (sans l'état intact)."""
        return len(self.etats)

    @property
    def n_secours(self) -> int:
        """Secours du solveur utilisés après retrait (sans l'état intact)."""
        return sum(etat.n_secours for etat in self.etats)

    @property
    def delestage_total(self) -> float:
        return self.etat_final.delestage_total

    @property
    def fraction_delestee(self) -> float:
        return self.etat_final.fraction_delestee

    @property
    def delestage_initial(self) -> float:
        """Délestage du réseau intact ; permet de distinguer le niveau de base."""
        return self.etat_reference.delestage_total

    @property
    def delestage_incremental(self) -> float:
        """Délestage final moins le délestage du réseau intact."""
        return self.delestage_total - self.delestage_initial


def _valider_ligne(reseau: Reseau, ligne_initiale: int) -> int:
    if isinstance(ligne_initiale, (bool, np.bool_)) or not isinstance(
        ligne_initiale, Integral
    ):
        raise ValueError("ligne_initiale doit être un indice entier (pas un booléen).")
    ligne = int(ligne_initiale)
    if not 0 <= ligne < reseau.n_lignes:
        raise ValueError("ligne_initiale hors du réseau.")
    return ligne


def _valider_p1(
    p1: float, rng: np.random.Generator | None, *, exiger_rng: bool = True
) -> float:
    if isinstance(p1, (bool, np.bool_)) or not isinstance(p1, Real):
        raise ValueError("p1 doit être un réel dans [0, 1].")
    p = float(p1)
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p1 doit être fini et compris dans [0, 1].")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng doit être un numpy.random.Generator ou None.")
    if exiger_rng and 0 < p < 1 and rng is None:
        raise ValueError("Un rng explicite est requis lorsque 0 < p1 < 1.")
    return p


def simuler_cascade_n1(
    reseau: Reseau,
    demande: np.ndarray,
    puissance_max: np.ndarray,
    ligne_initiale: int,
    *,
    p1: float,
    rng: np.random.Generator | None = None,
    limites: np.ndarray | None = None,
    departage: str = "highs",
) -> ResultatCascadeN1:
    """Simule une perte N−1 imposée, puis la propagation secondaire.

    ``p1`` est obligatoire : cela évite de choisir une convention de cascade
    par défaut sans l'accord scientifique. Pour p1=0 ou p1=1, aucun RNG n'est
    nécessaire ; pour 0<p1<1, fournir un ``np.random.default_rng(graine)``.

    Aucune mutation du réseau, de la demande, des capacités ou du générateur
    aléatoire fourni (sauf la consommation normale de tirages RNG).
    Les erreurs de redispatch remontent explicitement et ne se transforment
    jamais en délestage nul silencieux.
    """
    ligne = _valider_ligne(reseau, ligne_initiale)
    p = _valider_p1(p1, rng)

    # Les données d'entrée sont figées localement pour préserver le cas de base.
    demande_fixe = np.array(demande, dtype=float, copy=True)
    puissance_fixe = np.array(puissance_max, dtype=float, copy=True)
    limites_fixes = None if limites is None else np.array(limites, dtype=float, copy=True)
    masque = np.zeros(reseau.n_lignes, dtype=bool)

    # Base intacte : nécessaire pour rendre observable tout délestage préexistant.
    etat_reference = redispatch_exact(
        reseau, demande_fixe, puissance_fixe, masque,
        limites=limites_fixes, departage=departage,
    )

    masque[ligne] = True
    etat = redispatch_exact(
        reseau, demande_fixe, puissance_fixe, masque,
        limites=limites_fixes, departage=departage,
    )
    etats = [etat]
    vagues: list[np.ndarray] = []

    # Terminaison garantie : chaque itération productive retire >= 1 ligne
    # supplémentaire, et une ligne retirée n'est jamais candidate de nouveau.
    while p > 0:
        actifs = ~masque
        if np.any(~np.isfinite(etat.taux_de_charge[actifs])):
            raise RuntimeError("Taux de charge non finis sur des lignes en service.")
        candidates = np.flatnonzero(
            actifs & (etat.taux_de_charge >= SEUIL_SATURATION)
        )
        if candidates.size == 0:
            break
        if p == 1:
            tombent = candidates
        else:
            assert rng is not None  # validé ci-dessus
            tombent = candidates[rng.random(candidates.size) < p]
        if tombent.size == 0:
            break

        masque[tombent] = True
        vagues.append(tombent.copy())
        etat = redispatch_exact(
            reseau, demande_fixe, puissance_fixe, masque,
            limites=limites_fixes, departage=departage,
        )
        etats.append(etat)

    return ResultatCascadeN1(
        ligne_initiale=ligne,
        p1=p,
        demande=demande_fixe,
        etat_reference=etat_reference,
        etats=etats,
        vagues_secondaires=vagues,
    )


def simuler_tous_retraits_n1(
    reseau: Reseau,
    demande: np.ndarray,
    puissance_max: np.ndarray,
    *,
    p1: float,
    graine: int | None = None,
    limites: np.ndarray | None = None,
    departage: str = "highs",
) -> list[ResultatCascadeN1]:
    """Évalue chaque retrait N−1 à partir du même réseau intact.

    Pour 0<p1<1, chaque ligne utilise un RNG distinct construit à partir du
    couple (graine, indice de ligne). Les résultats d'une ligne ne dépendent
    ainsi ni des tirages ni de l'ordre de simulation des autres lignes.

    Ceci est une commodité de validation A6, pas encore une campagne de
    production avec stockage et analyses statistiques.
    """
    if graine is not None:
        if isinstance(graine, (bool, np.bool_)) or not isinstance(graine, Integral) or graine < 0:
            raise ValueError("graine doit être un entier non négatif.")
    p = _valider_p1(p1, None, exiger_rng=False)
    if 0 < p < 1 and graine is None:
        raise ValueError("graine obligatoire pour une campagne avec 0 < p1 < 1.")
    resultats = []
    for ligne in range(reseau.n_lignes):
        rng = None if p in (0, 1) else np.random.default_rng(
            np.random.SeedSequence([int(graine), ligne])
        )
        resultats.append(
            simuler_cascade_n1(
                reseau, demande, puissance_max, ligne,
                p1=p, rng=rng, limites=limites, departage=departage,
            )
        )
    return resultats
