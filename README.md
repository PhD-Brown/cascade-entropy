# cascade-entropy

Analyse entropique des séries temporelles de pannes en cascade dans les réseaux
électriques de transport.

Projet PHY-3202 (Projet I), Université Laval, automne 2026.
Superviseur : Patrick Desrosiers.

## Question

Les séries temporelles produites par l'évolution d'un réseau électrique
présentent-elles des signatures informationnelles permettant de distinguer des
régimes présentant différentes vulnérabilités aux pannes en cascade ? Et si oui,
ces signatures apportent-elles une information complémentaire aux indicateurs
électriques et statistiques déjà utilisés ?

Le point de comparaison est l'analyse par l'exposant de Hurst menée par Carreras
et al. sur les mêmes séries. Si les mesures entropiques n'apportent rien de plus
que la charge moyenne, le taux de charge maximal des lignes ou le Hurst, c'est un
résultat en soi.

## Architecture

Le développement est séparé en deux fils indépendants qui ne communiquent que par
les séries temporelles produites par le modèle.

**Fil A — modèle du réseau**

| Module | Entrée → sortie |
|---|---|
| `reseau.py` | topologie, matrice de flux, limites de lignes |
| `dispatch.py` | demande → flux sur chaque ligne |
| `cascade.py` | un état → délestage, lignes en panne |
| `evolution.py` | N jours → séries temporelles |

**Fil B — outils d'analyse**

| Module | Entrée → sortie |
|---|---|
| `entropie.py` | une série → entropie de permutation, SampEn, MSE |
| `indicateurs.py` | une série → exposant de Hurst, charge, taux de charge maximal |

Le fil B ne dépend d'aucune donnée électrique et se valide sur des séries
synthétiques de comportement connu.

## Critères de validation

Chaque module est associé à un critère explicite, vérifié avant intégration.

- `dispatch` : réseau réduit résolu à la main ; aucune solution avec délestage ni
  ligne saturée à faible charge.
- `cascade` : probabilité d'avarie nulle → aucune panne ; graine aléatoire fixée →
  cascade reproductible.
- `evolution` : convergence du taux de charge moyen vers un plateau ; queue en loi
  de puissance dans la distribution des tailles de blackout.
- `entropie` : entropie de permutation basse sur un signal périodique, proche du
  maximum sur un bruit non corrélé ; croisement multiéchelle entre bruit non
  corrélé et bruit corrélé à longue portée.
- `indicateurs` : exposant de Hurst d'un bruit non corrélé voisin de 0,5.

## Données

Aucune donnée réelle n'est versionnée dans ce dépôt. Les séries analysées sont
générées par les simulations. Si des données externes deviennent disponibles,
elles resteront hors du dépôt.

## Références principales

- Carreras, Lynch, Dobson, Newman, *Critical Points and Transitions in an Electric
  Power Transmission Model for Cascading Failure Blackouts*, Chaos 12 (2002) 985.
- Carreras, Lynch, Dobson, Newman, *Complex Dynamics of Blackouts in Power
  Transmission Systems*, Chaos 14 (2004) 643.
- Bandt, Pompe, *Permutation Entropy*, PRL 88 (2002) 174102.
- Richman, Moorman, *Physiological Time-Series Analysis Using Approximate Entropy
  and Sample Entropy*, Am. J. Physiol. 278 (2000) H2039.
- Costa, Goldberger, Peng, *Multiscale Entropy Analysis of Complex Physiologic
  Time Series*, PRL 89 (2002) 068102.

## Licence

MIT.
