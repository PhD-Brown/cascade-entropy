# cascade-entropy

Analyse entropique des séries temporelles de pannes en cascade dans les réseaux
électriques de transport.

Projet PHY-3202 (Projet I), Université Laval, automne 2026.
Superviseur : Patrick Desrosiers.

## Question scientifique

Les séries temporelles produites par l'évolution d'un réseau électrique
présentent-elles des signatures informationnelles permettant de distinguer des
régimes présentant différentes vulnérabilités aux pannes en cascade ? Et si oui,
ces signatures apportent-elles une information complémentaire aux indicateurs
électriques et statistiques déjà utilisés ?

Le point de comparaison est l'analyse par l'exposant de Hurst menée par Carreras
et al. sur les séries de blackouts de leur modèle. Si les mesures entropiques
n'apportent rien de plus que la charge moyenne, le taux de charge maximal des
lignes ou le Hurst, c'est un résultat en soi.

## État actuel et limites

Le dépôt contient des outils d'analyse de séries temporelles, des expériences
sur signaux synthétiques, un calcul de flux DC et un dispatch sous contraintes.
**Le simulateur complet de pannes en cascade n'est pas encore implémenté.**

| Composante | État actuel |
|---|---|
| Signaux synthétiques et mesures informationnelles | Implémentés, avec tests et expériences de référence |
| Contrôle par mélange de l'ordre temporel | Implémenté, avec export des résultats |
| Topologie et flux électriques DC | Implémentés |
| Dispatch de production et de charge servie | Implémenté par programmation linéaire |
| Cascade sur une journée | À implémenter : `cascade.journee()` lève `NotImplementedError` |
| Évolution sur plusieurs jours et export des séries électriques | À implémenter : fonctions de `evolution.py` levant `NotImplementedError` |

Le dispatch minimise le coût
`C = somme(production) - W × somme(charge_servie)`, avec `W = 100` par défaut.
Dans le modèle sans pertes, production totale et charge servie totale sont
égales : pour `W > 1`, cet objectif maximise donc la charge totale servie.

**Plusieurs répartitions peuvent avoir le même coût optimal tout en produisant
des flux et des saturations différents.** Le code ne définit pas d'objectif
secondaire pour les départager. Une montée de la saturation obtenue lors d'un
balayage de demande ne suffit donc pas à établir une « seconde transition »
physique. La règle de sélection des solutions et sa fidélité au modèle de
référence restent à examiner avant de construire la cascade.

Les calculs DC ne décrivent pas la dynamique complète des tensions et des
fréquences. La gestion des îlots après déconnexion reste également à définir
pour la cascade.

## Architecture

Le développement suit deux fils : le modèle électrique produit les séries que
les outils d'analyse doivent ensuite étudier. Pour l'instant, les expériences
temporelles utilisent des signaux synthétiques.

**Fil A — modèle du réseau**

| Module dans `cascade_entropy/` | Rôle |
|---|---|
| `reseau.py` | Construire la topologie, les matrices électriques et les limites ; calculer les angles et flux DC pour des injections imposées |
| `dispatch.py` | À partir de la demande et des capacités, déterminer production, charge servie, délestage, injections, flux et taux de charge |
| `cascade.py` | Prévu : enchaîner avaries et nouveaux dispatchs sur une journée |
| `evolution.py` | Prévu : faire évoluer le réseau sur plusieurs jours et enregistrer les séries |

Les injections sont positives en production et négatives en consommation.
Le dispatch manipule une demande et une charge servie positives, puis les
convertit en injections signées. Le taux de charge d'une ligne est
`M = |F| / F_max`. Le dispatch contraint les flux aux limites de transport ;
le seuil actuel de détection de saturation est `M >= 0.99`.

**Fil B — outils d'analyse et de contrôle**

| Module dans `cascade_entropy/` | Rôle |
|---|---|
| `synthetiques.py` | Générer signaux périodiques, bruit blanc et bruits à spectre de puissance |
| `entropie.py` | Calculer Shannon, entropie de répartition, nombre effectif, entropie de permutation, SampEn et entropie multiéchelle (MSE) |
| `indicateurs.py` | Estimer Hurst par R/S et sa distribution nulle simulée ; calculer charge moyenne/maximale, survie empirique et exposant de loi de puissance |
| `controles.py` | Comparer la MSE d'une série à celle de permutations de ses valeurs |
| `figures.py` | Fournir un style graphique et des tracés réutilisables |
| `_validation.py` | Centraliser les contrôles d'entrée des outils d'analyse |

Les outils temporels ne dépendent pas du simulateur électrique. L'entropie de
répartition peut aussi décrire un état spatial du réseau : préciser alors si
les poids proviennent des amplitudes de flux ou des taux de charge. Ces deux
choix ne mesurent pas la même répartition, et leur lien avec la vulnérabilité
reste à tester.

## Démarrage rapide

Prérequis : Python 3.10 ou plus récent et Git. Depuis un terminal :

```bash
git clone https://github.com/PhD-Brown/cascade-entropy.git
cd cascade-entropy
python -m venv .venv
```

Activer l'environnement :

- Windows PowerShell : `.venv\Scripts\Activate.ps1`
- macOS / Linux : `source .venv/bin/activate`

Installer le paquet et les dépendances de développement, puis lancer les tests :

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Les dépendances principales sont NumPy, SciPy et Matplotlib ; l'option
`dev` ajoute pytest. Le nombre de tests évolue : consulter le résultat de
l'exécution pour connaître l'état de la version utilisée.

### Première expérience : mêmes valeurs, ordre différent

Depuis la racine du dépôt :

```bash
python scripts/02_melange_temporel.py --n 2048 --melanges 20 --echelles 5 --graine 20260915 --sortie data/mon_essai
```

L'expérience compare une série synthétique à spectre `1/f` à des permutations
qui conservent ses valeurs et modifient leur ordre. Elle enregistre :

| Fichier dans le dossier de sortie | Contenu |
|---|---|
| `02_melange_temporel.png` | Courbes MSE de l'originale et des mélanges |
| `courbes.csv` | Valeurs résumées par échelle |
| `resultats.npz` | Série originale, courbes individuelles et résumés numériques |
| `parametres.json` | Paramètres, versions et empreintes des sources actuellement suivies par le script |

La bande représente **± un écart-type entre mélanges**, pas un intervalle de
confiance. Cette expérience porte sur une seule réalisation originale :
augmenter le nombre de mélanges ne réduit pas l'incertitude liée au choix de
cette réalisation. Dans le NPZ, `melanges` contient les courbes MSE, pas les
signaux permutés.

Le script `scripts/01_croisement_multiechelle.py` propose aussi une comparaison
de bruits blanc et rose ; ses paramètres sont définis dans le fichier.

## Organisation et carnets

| Dossier | Contenu |
|---|---|
| `cascade_entropy/` | Fonctions de calcul réutilisables |
| `scripts/` | Expériences exécutables |
| `tests/` | Vérifications automatisées |
| `notebooks/` | Carnets pédagogiques ; voir [les instructions](notebooks/README.md) |
| `docs/` | Documentation et protocoles |
| `data/`, `figures/` | Sorties locales de calcul, exclues du suivi Git |

Le carnet versionné `notebooks/01_laboratoire.py` utilise le format percent de
Jupytext. Pour l'ouvrir en notebook dans VS Code ou Jupyter :

```bash
python -m pip install jupytext
jupytext --to ipynb notebooks/01_laboratoire.py
```

Sélectionner l'environnement Python dans lequel le paquet a été installé.
Certaines mentions d'avancement du premier carnet sont historiques :
le tableau « État actuel et limites » ci-dessus reflète le dispatch désormais
implémenté.

## Vérification numérique et validation scientifique

Les tests vérifient des propriétés précises : cas électriques calculables à la
main, conservation de puissance, contraintes de production et de transport,
comptages entropiques de référence, contrôles d'entrée et reproductibilité.
Leur réussite ne démontre pas à elle seule la validité physique du modèle.

Les contrôles et objectifs scientifiques sont distincts :

- **Analyse temporelle :** comparer des signaux de référence, étudier la
  sensibilité aux paramètres et à la longueur des séries, et examiner le
  croisement multiéchelle entre bruit blanc et bruit corrélé.
- **Hurst :** comparer l'estimation R/S à un témoin simulé de même longueur.
  La distribution nulle actuelle utilise un bruit gaussien non corrélé ; elle
  ne constitue pas un témoin universel pour toutes les séries.
- **Cascade, à implémenter :** vérifier notamment l'absence d'avaries lorsque
  leur probabilité est nulle et la reproductibilité à graine fixée.
- **Évolution, à implémenter :** rechercher, dans les conditions du modèle de
  référence, un plateau du taux de charge moyen et une éventuelle queue en loi
  de puissance des tailles de blackout. Ce sont des objectifs de reproduction,
  pas des comportements garantis ni des résultats déjà obtenus.

Ajuster un exposant ne suffit pas à démontrer une loi de puissance. De même, une
différence d'entropie ou de nombre effectif ne démontre pas, à elle seule, une
différence de vulnérabilité électrique.

## Données et reproductibilité

Aucune donnée réelle n'est versionnée. Les expériences actuelles produisent des
séries synthétiques ; les séries de blackouts seront produites lorsque le fil A
sera complété. Les données externes éventuelles resteront hors du dépôt.

Conserver ensemble paramètres, résultats et version du code pour chaque
expérience. Un commit des scripts ne sauvegarde pas les fichiers ignorés dans
`data/` et `figures/`. Les empreintes du script de mélanges ne couvrent pas encore
`controles.py`, où se trouve désormais le calcul : conserver aussi le commit
utilisé et toute modification locale.

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
