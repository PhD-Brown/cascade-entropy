# Expérience 02 : mêmes valeurs, ordre différent

## Objectif et portée

Comparer la courbe MSE d'une réalisation de bruit 1/f à celles de vingt
permutations de ses valeurs. Les permutations conservent exactement le
multiensemble des valeurs (donc l'histogramme, les extrema, les moments et les
éventuels zéros), tout en modifiant leur ordre temporel. De minuscules écarts
d'arrondi peuvent affecter les moments calculés dans un autre ordre.

Cette expérience est un contrôle descriptif de l'effet de l'ordre temporel.
Elle ne teste pas la vulnérabilité électrique, ne prouve pas une supériorité sur
Hurst et ne constitue pas une validation universelle de MSE. La variabilité
affichée est conditionnelle à UNE série originale, pas à un ensemble de réseaux
ou de réalisations originales indépendantes.

## Protocole par défaut

- N = 8192, une série 1/f générée par le générateur spectral du projet.
- 20 mélanges, échelles 1 à 10, m = 2, r = 0,15.
- Chaque courbe garde sa tolérance fixée sur l'écart-type de la série avant
  granularisation. L'écart-type est mathématiquement le même après permutation.
- Graine racine = 20260915 ; deux flux SeedSequence distincts pour la génération
  et les mélanges. Ajouter des mélanges conserve la série et les premiers mélanges.
- Bande = moyenne ± écart-type des permutations (ddof = 1), pas un intervalle de
  confiance ni une enveloppe de significativité.
- Aucun mélange avec SampEn non finie n'est retiré silencieusement : arrêt explicite.

## Exécution

À la racine du dépôt, avec Python >= 3.10 :

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/02_melange_temporel.py
```

Pour vérifier rapidement l'installation avant l'expérience complète :

```bash
python scripts/02_melange_temporel.py --n 2048 --melanges 3 --echelles 5 --sortie data/02_essai_rapide
```

Le calcul SampEn existant est quadratique en N. Les blocs réduisent l'occupation
mémoire, pas le nombre de comparaisons. Le script affiche sa progression.

## Sorties

Par défaut dans `data/02_melange_temporel/` :

- `02_melange_temporel.png` : figure complète avec légende.
- `courbes.csv` : échelle, originale, moyenne des mélanges, écart-type.
- `resultats.npz` : série originale, chaque courbe de mélange, résumés.
- `parametres.json` : paramètres, versions et empreintes SHA-256 des sources.

Relancer avec le même dossier remplace ces quatre sorties. Utiliser `--sortie`
pour conserver une autre expérience. Les paramètres et la version NumPy sont
consignés pour pouvoir régénérer les permutations.

Lecture des données :

```python
import numpy as np
with np.load("data/02_melange_temporel/resultats.npz") as donnees:
    print(donnees["melanges"].shape)  # (20, 10) avec les paramètres par défaut
```

## Corrections incluses avec cette expérience

- Rejet des séries non réelles, non finies ou non unidimensionnelles par les mesures.
- Vérification des paramètres de dimension, délai, tolérance et granularisation.
- Documentation du tri stable pour les égalités en entropie de permutation ;
  comptage direct des motifs, sans codage entier susceptible de déborder.
- Nom corrigé du test sur le bruit brun spectral ; documentation du générateur.
- Figure 01 : légende de la dispersion et écart-type avec ddof = 1.
- Tests indépendants de comptage SampEn, de tolérance MSE fixe et de reproductibilité.

Le comportement historique de SampEn sur une série constante ou sans suffisamment
d'appariements reste `NaN`. Ces cas ne doivent jamais être interprétés comme une
entropie nulle. La gestion des zéros des futures séries OPA demandera une analyse
spécifique. Le journal ancien reste une trace historique : le bruit en 1/f² n'y
doit pas être interprété comme une mesure de « forte persistance » d'une marche
brownienne.

## Vérification effectuée avec ce lot

- 41 tests réussis
- Exécution complète : 8192 points, 20 mélanges, 10 échelles.
- Échelle 1 : originale 1,941 ; mélanges 2,475 ± 0,010.
- Échelle 10 : originale 1,899 ; mélanges 1,341 ± 0,024.
- Multiensembles exactement conservés pour tous les mélanges.
- Figure inspectée visuellement. L'expérience montre ici une sensibilité à
  l'ordre temporel ; elle ne démontre aucune capacité de prédiction électrique.
