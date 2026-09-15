"""Analyse entropique des pannes en cascade dans les réseaux de transport.

Le paquet est organisé en deux fils indépendants.

Fil A, modèle du réseau :
    reseau      topologie, matrice de flux, limites de lignes
    dispatch    une demande donne les flux sur chaque ligne
    cascade     un état du réseau donne un blackout
    evolution   N jours donnent des séries temporelles

Fil B, outils d'analyse :
    synthetiques    séries de comportement connu, pour la validation
    entropie        une série donne une mesure informationnelle
    indicateurs     une série donne un indicateur de référence

Les deux fils ne communiquent que par les séries temporelles écrites sur disque.
Aucun module du fil B n'importe un module du fil A.
"""

__version__ = "0.1.0"
