"""
Cascade de pannes : avaries aléatoires, redispatch et convergence.

Troisième module du fil A. Il orchestre `reseau` et `dispatch` en boucle pour
simuler une « journée » : une réalisation possible de la demande et des avaries
de ligne, jusqu'à ce que plus aucune ligne ne soit en surcharge.

Trois sources d'aléatoire interviennent, à des moments distincts et avec des
rôles distincts. Toutes proviennent d'un seul générateur (`numpy.random.Generator`)
transmis en paramètre à `journee()` — jamais d'un état aléatoire global — pour
garantir la reproductibilité à graine fixée.

1. Fluctuation de la demande (avant tout dispatch)
   La demande du jour n'est pas fixe : elle est tirée autour d'une moyenne avec
   une fluctuation aléatoire, comme le paramètre g de Carreras et al. (2002,
   section V). C'est ce qui permet à une ligne de saturer certains jours et pas
   d'autres, même en l'absence de toute avarie.

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
capacité quasi nuls) ne soit prise pour une ligne saturée.

Critères de validation :
  - p0 = 0 et p1 = 0 : `journee()` produit exactement le même résultat qu'un
    simple `dispatch.resoudre()` sur la demande tirée ce jour-là — aucune
    avarie ne doit apparaître ;
  - graine fixée : la séquence d'avaries, le nombre d'itérations et le
    délestage final sont reproductibles ;
  - une ligne hors service n'est jamais retenue comme candidate à une
    nouvelle avarie par surcharge, quel que soit son taux de charge apparent ;
  - la boucle termine en au plus `n_lignes` itérations, puisqu'une ligne ne
    peut tomber qu'une seule fois.
"""

from __future__ import annotations

import numpy as np

SEUIL_SURCHARGE = 0.99


def journee(reseau, demande: np.ndarray, p1: float, rng: np.random.Generator):
    """Simule une journée et retourne le délestage et les lignes tombées."""
    raise NotImplementedError
