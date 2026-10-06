# CLAUDE.md — cascade-entropy

Projet PHY-3202 (Projet I), Université Laval, automne 2026. Auteur : Alex Baker.
Superviseur : Patrick Desrosiers.

Question : les séries temporelles d'un réseau électrique de transport portent-elles
des signatures entropiques qui distinguent des régimes de vulnérabilité aux pannes
en cascade, au-delà de la charge moyenne, du taux de charge maximal et de l'exposant
de Hurst (référence : Carreras et al.) ? Un résultat négatif est un résultat valable.

## État du projet (5 octobre 2026)

- **SO1** (Carreras et al., 2002) : fermé. Fig. 3 à 11 reproduites sans coefficient ajusté.
- **SO2** (Carreras et al., 2004) : produit et analysé (balayage `so2-G-scan`, 21 cas).
  Reproduit en partie ; les écarts (Fig. 7, Fig. 9, Table I) sont documentés.
- **SO3** (entropies sur les séries de SO2) : prochaine étape.
- Les chiffres de validation vivent dans `docs/so1_carreras_2002.md` et
  `docs/so2_carreras_2004.md`. **Ne les recopie pas ailleurs** et ne les cite pas de mémoire.

## Commandes

```bash
python -m pip install -e ".[dev]"   # installer avant toute chose
python -m pytest -q                 # lancer avant de déclarer une tâche terminée
```

Scripts (lancés depuis la racine ; options complètes dans `scripts/README.md`) :

```bash
python scripts/05_diagnostics_carreras.py --departage exterieur_dabord   # Fig. 3-11 de SO1
python scripts/06_validation_so1.py                                      # audit chiffré SO1
python scripts/07_dynamique_lente.py run --tailles 46 94 190 --G 0.1 0.25 0.5 0.75 1.0 1.5 2.0 \
    --jours 120000 --workers 8 --run-id so2-G-scan                       # production SO2
python scripts/08_analyse_so2.py --help                                  # analyse SO2
```

Python >= 3.10. La liste des dépendances est celle de `pyproject.toml`.
N'ajoute aucune dépendance sans me le demander.

## Architecture

- `cascade_entropy/` : fonctions réutilisables. `scripts/` : expériences et campagnes.
  `tests/` : pytest. `notebooks/` : carnets pédagogiques. `docs/` : résultats détaillés
  (SO1, SO2), journal et protocoles. Chaque dossier de code a son README.
- **Fil A — modèle du réseau** : `reseau.py` (topologie, flux DC), `dispatch.py`
  (programme linéaire, option `departage`), `cascade.py` (une journée), `evolution.py`
  (dynamique multi-jours, reprise exacte), `carreras.py` (protocole de la Table I).
- **Fil B — analyse** : `synthetiques.py`, `entropie.py`, `indicateurs.py`,
  `analyse_series.py`, `controles.py`.
- **Transversaux** : `figures.py`, `chemins.py`, `_validation.py`.
- **Règle stricte : aucun module du fil B n'importe un module du fil A.** Les fils
  communiquent par des fichiers sur disque. Seule exception, dans l'autre sens et déjà
  documentée dans `__init__.py` : `evolution` (A) importe `entropie.nombre_effectif` (B).
  N'en ajoute aucune autre.
- `chemins.py` trouve la racine en remontant jusqu'à un dossier nommé `cascade_entropy/`
  et crée `data/` et `figures/` à l'import. **Ne renomme et ne déplace jamais
  `cascade_entropy/`** (ni `data/` pendant un run).

Conventions physiques : injections positives en production, négatives en
consommation. Taux de charge `M = |F| / F_max`, saturation si `M >= 0.99`.
Coût du dispatch `C = somme(production) - W × somme(charge_servie)`, `W = 100`.
Notations : `P_G = 2623.9` par générateur, `P_C = 12 × P_G = 31 486.8`,
`g = γ − 1` (facteurs uniformes sur `[1−g, 1+g]`, `g = 0.9` pour `γ = 1.9`),
`ρ = r / r_T(N)` (distance relative au seuil de transport).

## Conventions de code

- Noms, docstrings et commentaires **en français**, dans le style pédagogique
  existant (la docstring explique ce que mesure la fonction et pourquoi).
- Valider les entrées via `_validation.py` plutôt que de dupliquer les contrôles.
- **Toute nouveauté est une option.** La valeur par défaut garde le comportement
  historique (ex. `--departage` ne change jamais un résultat existant sans qu'on le demande).
- Toute aléa passe par une graine explicite (`numpy.random.Generator`, jamais d'état
  global). Les résultats doivent être reproductibles.
- Runs lourds : `--run-id`, paramètres et graines dans `metadata.json`, reprise avec
  `--resume`. Un run est non destructif : il n'écrase jamais un autre run.
- Chaque ajout vient avec ses docstrings, ses validations d'entrée et ses tests pytest.
  **Les nouveaux tests ne dépendent pas des tests existants et ne les modifient pas.**
- Changements petits et ciblés. Ne modifie jamais un test pour le faire passer :
  signale plutôt l'écart.
- Sorties : `data/` n'est pas versionné. Dans `figures/`, seules les figures PNG de
  référence sont versionnées ; les PDF et `figures/**/runs/` ne le sont pas. Chaque
  script enregistre ses paramètres avec ses résultats.
- Windows / PowerShell : ferme les figures ouvertes avant de relancer un script
  (fichier verrouillé).

## Rigueur scientifique

- **Aucun coefficient ajouté pour faire coïncider avec la littérature** : on mesure
  l'écart, puis on l'explique.
- Distingue toujours ce qui est **mesuré** (simulation), **dérivé analytiquement**
  (formule fermée vérifiée numériquement) et **hypothèse de réplication** (convention
  que le papier ne fixe pas). Signale explicitement toute hypothèse que tu introduis.
- Si un article est incohérent avec lui-même, signale-le au lieu de le contourner.
- Un test qui passe ne démontre pas la validité physique du modèle.
- Ajuster un exposant ne démontre pas une loi de puissance (la lognormale est favorisée
  dans nos résultats). Une différence d'entropie ne démontre pas une différence de
  vulnérabilité.
- Plusieurs dispatchs peuvent avoir le même coût optimal avec des flux différents.
  Règle de référence : `departage = exterieur_dabord` ; `highs` (choix du solveur) reste
  un contrôle de sensibilité, dépendant de la plateforme. **Le défaut de `--departage`
  n'est pas le même dans tous les scripts** (voir `scripts/README.md`) : ne le change pas
  sans me le demander.
- Entre Linux et Windows, seules les **statistiques** des longues trajectoires de SO2
  sont comparables, pas les trajectoires (divergence après ≈ 5 000 jours).

## Données

**Aucune donnée réelle dans le dépôt, qui est public.** Ne lis pas, ne crée pas
et ne commite rien dans `donnees_externes/`. Les données d'Hydro-Québec, si elles
arrivent, restent hors de Git et hors de tes sessions sans mon accord explicite.

## Git

- Ne commite pas et ne pousse pas sans que je le demande.
- **Jamais `git add .`** : ajoute les fichiers un par un. Du code peut être en cours.
- Ne réécris pas l'historique.

## Façon de travailler

Je prépare avec Claude (conversation du projet) un **ordre de travail** que je te donne.

1. Lis d'abord le code concerné (audit), puis fais ce que l'ordre demande, dans son
   périmètre. Rien de plus : une idée hors périmètre se signale, elle ne s'implémente pas.
2. Si l'ordre contredit le code ou ce fichier, arrête-toi et dis-le avant de modifier.
3. Pour une tâche non triviale sans ordre écrit : propose un plan et attends mon accord.
4. Pour une revue : audit d'abord (bugs, incohérences, tests manquants), sans rien modifier.
5. Termine par `python -m pytest -q`. Rends compte en français : fichiers modifiés,
   tests ajoutés, résultat de pytest, hypothèses introduites, points à vérifier.
   Ce compte rendu me sert à l'analyse avec Claude : sois précis et honnête sur ce
   qui n'est pas vérifié.
6. Pour une décision technique notable, propose une entrée pour `docs/journal.md`
   sans l'écrire toi-même.
7. Ne modifie pas les README ni `docs/so*_carreras_*.md` sans que l'ordre le demande :
   leurs chiffres viennent des tableaux de validation.

## Priorités actuelles

- Poser les tags `so1-v1` et `so2-v1` (je m'en occupe) et mettre `docs/journal.md` à jour.
- SO2 : refaire la Table I de Carreras 2004 sur la fréquence cumulée relative
  (`08_analyse_so2.py`) ; instruire l'écart de la Fig. 7 (plateau pour `G >= 1`).
- SO3 : appliquer entropie de permutation, SampEn et MSE aux séries de `so2-G-scan`,
  avec les mêmes références par mélange temporel que pour Hurst.
