# CLAUDE.md — cascade-entropy

Projet PHY-3202 (Projet I), Université Laval, automne 2026. Auteur : Alex Baker.
Superviseur : Patrick Desrosiers.

Question : les séries temporelles d'un réseau électrique de transport portent-elles
des signatures entropiques qui distinguent des régimes de vulnérabilité aux pannes
en cascade, au-delà de la charge moyenne, du taux de charge maximal et de l'exposant
de Hurst (référence : Carreras et al.) ? Un résultat négatif est un résultat valable.

## Commandes

```bash
python -m pip install -e ".[dev]"   # installer avant toute chose
python -m pytest                    # lancer avant de déclarer une tâche terminée
python scripts/02_melange_temporel.py --n 2048 --melanges 20 --echelles 5 --graine 20260915 --sortie data/essai
```

Python >= 3.10. Dépendances : NumPy, SciPy, Matplotlib (+ pytest en dev).
N'ajoute aucune dépendance sans me le demander.

## Architecture

- `cascade_entropy/` : fonctions réutilisables. `scripts/` : expériences.
  `tests/` : pytest. `notebooks/` : carnets pédagogiques. `docs/` : journal et protocoles.
- **Fil A — modèle du réseau** : `reseau.py` (topologie, flux DC), `dispatch.py`
  (programmation linéaire), `cascade.py` et `evolution.py` (non implémentés,
  lèvent `NotImplementedError`).
- **Fil B — analyse** : `synthetiques.py`, `entropie.py`, `indicateurs.py`,
  `controles.py`, `figures.py`, `_validation.py`.
- **Règle stricte : aucun module du fil B n'importe un module du fil A.**

Conventions physiques : injections positives en production, négatives en
consommation. Taux de charge `M = |F| / F_max`, saturation si `M >= 0.99`.
Coût du dispatch `C = somme(production) - W × somme(charge_servie)`, `W = 100`.

## Conventions de code

- Noms, docstrings et commentaires **en français**, dans le style pédagogique
  existant (la docstring explique ce que mesure la fonction et pourquoi).
- Valider les entrées via `_validation.py` plutôt que de dupliquer les contrôles.
- Toute aléa passe par une graine explicite ; les résultats doivent être reproductibles.
- Les sorties vont dans `data/` et `figures/` (non versionnés). Chaque script
  enregistre ses paramètres (`parametres.json`) avec ses résultats.
- Changements petits et ciblés. Ne modifie jamais un test pour le faire passer :
  signale plutôt l'écart.

## Rigueur scientifique

- Un test qui passe ne démontre pas la validité physique du modèle.
- Ajuster un exposant ne démontre pas une loi de puissance.
- Une différence d'entropie ne démontre pas une différence de vulnérabilité.
- Plusieurs dispatchs peuvent avoir le même coût optimal avec des flux différents :
  ne tire aucune conclusion physique d'une saturation sans traiter cette dégénérescence.
- Signale explicitement toute hypothèse de modélisation que tu introduis.

## Données

**Aucune donnée réelle dans le dépôt, qui est public.** Ne lis pas, ne crée pas
et ne commite rien dans `donnees_externes/`. Les données d'Hydro-Québec, si elles
arrivent, restent hors de Git et hors de tes sessions sans mon accord explicite.

## Façon de travailler

1. Pour toute tâche non triviale : lis le code concerné, propose un plan, attends
   mon accord avant de modifier.
2. Pour une revue : audit d'abord (bugs, incohérences, tests manquants), sans
   rien modifier.
3. Termine par `python -m pytest` et un résumé court de ce qui a changé.
4. Pour une décision technique notable, propose une entrée pour `docs/journal.md`
   sans l'écrire toi-même.

## Priorités actuelles

- Implémenter `cascade.journee()` (avaries et redispatchs sur une journée) :
  aucune avarie si la probabilité est nulle, reproductible à graine fixée.
- Implémenter `evolution.py` (évolution sur plusieurs jours, export des séries).
- Définir une règle de sélection parmi les dispatchs de même coût optimal.
