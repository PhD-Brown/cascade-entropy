# B4 — Attribution électrique configurable sur topologies synthétiques

Compte rendu du 17 novembre 2026. Référence de départ :
`28eeb9bdfd7fe99bd5137780134e03bf1f68d42b`.

Ce travail réalise le mécanisme technique B4 de la [feuille de route #11](https://github.com/PhD-Brown/cascade-entropy/issues/11),
en appliquant ses commentaires [A3](https://github.com/PhD-Brown/cascade-entropy/issues/11#issuecomment-6861560057),
[A6](https://github.com/PhD-Brown/cascade-entropy/issues/11#issuecomment-6861363593)
et [B](https://github.com/PhD-Brown/cascade-entropy/issues/11#issuecomment-6861346701).
Le contexte Carreras de `CLAUDE.md` demeure historique ; aucun paramètre de cette
reproduction ne sert de défaut à la nouvelle attribution.

## Architecture et décisions techniques

Seul `cascade_entropy/attribution_n1.py` est ajouté au code du moteur de réseau
(fil A au sens de `CLAUDE.md`). La « phase B » de la feuille de route ne désigne
pas le fil B d'analyse des entropies. Aucun import entre ces deux fils n'est
introduit. Aucun module historique, README, dépendance ou défaut existant n'est
modifié.

- `Topologie` conserve les lignes canoniques A3 : `u<v`, ordre lexicographique,
  indices d'origine stables. Une topologie modifiée après construction est
  revalidée ; un réordonnancement qui changerait les identifiants est refusé.
- `vers_reseau()` valide et copie les indices, rôles disjoints, réactances et
  limites. L'ordre des producteurs et des charges explicites est conservé :
  il détermine l'association aux vecteurs de puissances.
- `redispatch_exact()` calcule l'intact avec un masque tout faux et
  `departage="highs"`, seule option admise. Aucun équilibrage par un slack à
  puissance illimitée n'est ajouté. L'objectif et les bornes du dispatch
  existant sont conservés.
- `angles()` et `matrice_incidence()` contrôlent après résolution les lois DC,
  la conservation nodale et les bilans par composante.
- `EtatElectrique.reseau`, `.demande` et `.puissance_max` se passent directement
  aux interfaces N−1 existantes. Aucun calcul de scores ou de campagne n'est
  exécuté par le module.

Les erreurs du solveur remontent avec leur cause ; une sortie non finie, de
mauvaise dimension, hors bornes ou incohérente avec la loi DC provoque une
`RuntimeError`. Les statuts d'îlots et le nombre de secours restent accessibles.
Le contrôle emploie une tolérance relative de `1e-7` et une tolérance absolue
de `1e-8 pu`, multipliée par la plus grande demande individuelle si celle-ci
est inférieure à 1 pu. Un résultat peut donc être refusé par B4 même si HiGHS
l'accepte avec sa propre tolérance : il faut alors examiner les échelles et le
problème, jamais interpréter cet échec comme un délestage nul. Aucun arrondi ou
écrêtage ne corrige secrètement les sorties.

## Options explicites

`Valeurs(mode, parametres, graine=None)` décrit chaque vecteur :

| Mode | Paramètres | Graine |
| --- | --- | --- |
| `explicites` | Une valeur par charge, producteur ou ligne concernée | Absente |
| `constante` | Un nombre à répéter | Absente |
| `uniforme` | Deux bornes non négatives croissantes | Obligatoire, entier ≥ 0 |

Chaque champ aléatoire utilise son propre `numpy.random.default_rng(graine)`.
L'état aléatoire global n'est pas consommé et le tirage d'un champ ne dépend
pas de l'ordre d'appel des autres. Des graines identiques entre champs peuvent
induire des dépendances : leur choix appartient à l'appelant. Ces options ne
constituent pas des distributions physiques recommandées.

`Placement(generateurs, charges, reference)` permet des listes explicites,
avec au moins un producteur et un consommateur, disjoints et sans doublons.
Des nœuds neutres sont autorisés. `placement_aleatoire()` propose un tirage
uniforme sans remise de deux groupes disjoints, de tailles et graine fournies.
Chaque groupe tiré est trié ; la référence est son premier producteur.
Les entrées et la provenance du tirage sont enregistrées.

Les demandes, Pmax et réserves minimales sont fournis individuellement :

- `0 <= G <= Pmax` et `0 <= L_servie <= demande` sont imposés par le moteur ;
- la réserve disponible vaut `Pmax-G`, par producteur ;
- `reserves_minimales` décrit une réserve souhaitée à contrôler **après**
  dispatch. Elle ne retranche rien de Pmax et n'ajoute pas de contrainte à
  l'optimisation. Si elle n'est pas atteinte, le résultat porte l'avertissement
  `reserves_minimales_non_satisfaites`. Une réserve demandée supérieure à Pmax
  est une configuration invalide ;
- cette marge de génération ne garantit pas sa disponibilité après un retrait
  ou son acheminement dans le réseau. Une politique de réserve maintenue dans
  le dispatch nécessiterait une décision scientifique séparée.

Toutes les entrées de puissance utilisent la même unité explicitement donnée :
`pu` ou `MW`. Dans le second cas, demandes, Pmax, réserves, limites de référence
et plancher de capacité sont divisés par `base_mva`, finie et strictement
positive. Les sorties sont **toujours en pu**, les angles en radians. Les
réactances doivent déjà être en pu sur cette même base ; `b=1/x>0` est calculé
dans l'ordre des lignes. Les ohms et changements de base de réactance sont
refusés plutôt que devinés. La base numérique n'a aucun défaut approuvé.

## Capacités et dépendance au dispatch

Deux voies sont offertes, sans préférence scientifique :

1. `Capacites("independantes", valeurs)` affecte les capacités directement.
   Le vecteur peut être explicite, constant ou tiré avec une graine.
2. `Capacites("flux_reference", valeurs, marge=..., plancher=...)` effectue
   exactement deux étapes. `valeurs` décrit les **limites provisoires
   explicites** utilisées pour résoudre un intact de référence. Ensuite :
   `C_l = max(marge * abs(F_reference_l), plancher)`, où le plancher est absolu
   dans l'unité d'entrée et strictement positif. Ces capacités sont figées
   avant de résoudre de nouveau l'intact final.

La seconde voie ne suppose ni un dispatch sans contraintes de transport, ni
l'égalité `F_reference = F_initial`. Elle n'itère pas jusqu'à un point fixe.
Les deux résultats sont accessibles séparément (`resultat_reference` et
`resultat`) et la configuration contient les limites provisoires. La dépendance
au dispatch est toujours signalée. Un changement des flux, un délestage ou un
secours du solveur dans le calcul de référence sont signalés séparément.

Si le plancher ne domine pas et si le flux final reste identique au flux de
référence, le taux de charge vaut `1/marge` sur chaque ligne à flux non nul.
Avec des capacités uniformes, le taux de charge est proportionnel au flux
absolu. Ces redondances sont des conséquences mathématiques à déclarer, pas des
critères pour choisir des limites qui favoriseraient une centralité. Le cas
de flux partout nul conserve des capacités positives grâce au plancher absolu.

## Exemple minimal : nombres de validation uniquement

```python
import numpy as np
from cascade_entropy.topologies_n1 import Topologie
from cascade_entropy.attribution_n1 import (
    Placement, Valeurs, Capacites, attribuer,
    sauver_configuration, charger_configuration,
)

t = Topologie("chaine_validation", 3, np.array([[0, 1], [1, 2]]))
e = attribuer(
    t,
    placement=Placement([0], [1, 2], reference=0),
    demande=Valeurs("explicites", [1.0, 2.0]),
    puissance_max=Valeurs("explicites", [4.0]),
    reserves_minimales=Valeurs("explicites", [0.5]),
    reactances=Valeurs("explicites", [0.2, 0.3]),
    capacites=Capacites("independantes", Valeurs("constante", [10.0])),
    base_mva=100.0, unite_puissance="pu", unite_reactance="pu",
)
assert np.allclose(e.resultat.flux, [3.0, 2.0])
assert np.allclose(e.angles, [0.0, -0.6, -1.2])
# Écriture facultative : un chemin existant n'est jamais écrasé.
# sauver_configuration(e, "configuration_validation.json")
# rejeu = charger_configuration("configuration_validation.json")
```

Le manifeste JSON contient la topologie complète, les groupes ordonnés, tous
les paramètres, graines, unités, règle de capacités et versions Python/NumPy/
SciPy. Le chargement revalide les entrées et recalcule l'intact ; il ne charge
pas une solution de solveur sans contrôle. Il s'agit d'un manifeste B4,
pas d'un stockage des scénarios d'une campagne. Des tableaux fournis à la main
restent une voie pour des stratégies externes ; leur provenance peut être
documentée par l'appelant.

La reproductibilité exacte testée vaut dans le même environnement et pour le
même code. Le choix d'un optimum HiGHS dégénéré peut varier avec les versions
et plateformes ; aucune unicité scientifique ni reproductibilité binaire
interplateforme n'est affirmée.

## États dégradés et décisions scientifiques ouvertes

Tout délestage strictement positif retourné est signalé, même inférieur à la
tolérance de contrôle. Il reste chiffré dans `resultat.delestage`, sans rejet
automatique ni assimilation à zéro. Les topologies déconnectées avec au moins
une ligne sont résolues par îlot et signalées ; les nœuds isolés sont conservés.
Une demande totale nulle est autorisée comme cas technique mais signalée : sa
fraction délestée est indéfinie. Une topologie sans aucune ligne est refusée.

Restent ouverts avec Patrick : placement, distribution des demandes, Pmax et
réserves, distribution des réactances, base numérique, capacités et référence
éventuelle de leur calcul, admission des états avec délestage ou réserve
insuffisante, densités, nombre de réalisations et départage définitif. La règle
de propagation finale ne relève pas de B4. Les exemples et tests ne choisissent
aucune de ces hypothèses pour la campagne.

## Validation locale

- Installation : `python -m pip install -e ".[dev]"`, réussie sans nouvelle
  dépendance déclarée.
- Référence avant B4 : `python -m pytest -q` → **509 passed in 138.89s**.
- Sélection : `python -m pytest tests/test_attribution_n1.py tests/test_topologies_n1.py tests/test_n1.py tests/test_cascade_n1.py -q`
  → **168 passed in 1.02s**, dont **97 tests B4**.
- Analyse syntaxique AST des deux nouveaux fichiers Python : réussie.
- Suite complète finale : résultat consigné dans la PR après exécution.

Environnement : Windows, Python 3.11.9, NumPy 2.3.4, SciPy 1.16.3,
pytest 9.1.1. Tests uniquement sur réseaux synthétiques : oracles analytiques
chaîne/triangle, bilans nodaux indépendants sur les quatre familles à 50 nœuds
et l'arbre témoin, unités, indices, dimensions, valeurs invalides,
reproductibilité, non-mutation, réserves, génération insuffisante, transport
limité, demande nulle, îlots, plancher, deux dispatchs et erreurs numériques.
Une injection de courant cyclique conserve le bilan nodal mais doit être
refusée par le contrôle de la loi DC. D'autres sorties corrompues et erreurs
de solveur sont injectées pour vérifier qu'elles ne deviennent pas des succès.

Ces résultats valident les mécanismes testés, pas les hypothèses physiques
d'une campagne ni les performances prédictives des centralités. Aucun matériel
restreint, campagne finale ou comparaison des centralités n'est utilisé.

## Proposition pour le journal (sans modifier `docs/journal.md`)

> B4 : ajout d'une attribution électrique configurable au-dessus de
> `vers_reseau` et du redispatch exact. Unités explicites, réserves contrôlées
> après dispatch, capacités indépendantes ou dérivées d'un dispatch de référence
> explicitement contraint ; recalcul final sans point fixe implicite. Les états
> avec délestage restent signalés. Les paramètres numériques de campagne et
> les règles d'admission demeurent à valider avec Patrick.
