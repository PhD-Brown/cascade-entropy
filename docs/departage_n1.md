# Départage des dispatchs N−1 : diagnostic technique

Diagnostic du 17 novembre 2026, révisé le 18 novembre sur la branche
`codex/departage-diagnostic`. Le run synthétique de référence est
`departage-20261118-v3`. `highs` reste le défaut. Le choix scientifique reste
à valider avec Patrick Desrosiers.

## Mécanisme et portée

[lu] Le moteur historique minimise `somme(G) − W * somme(S)` avec bilan
`somme(G) = somme(S)`. Ses coûts de production identiques laissent généralement
plusieurs dispatchs primaires optimaux. HiGHS choisit un sommet ; ce sommet
peut saturer des lignes qu'un autre optimum déchargerait. La cascade N−1
retire les lignes actives à `|F| / C >= 0.99` avec probabilité `p1`. Elle
n'interdit pas les saturations après une panne : certaines sont inévitables.

[dérivé] Une ligne saturée est **évitable individuellement** si le minimum
de son taux sur la face primaire optimale est strictement inférieur à 0.99.
Elle est **forcée** sinon. Les extrema du flux signé donnent ce minimum :
zéro si l'intervalle contient zéro, sinon `min(|F_min|, |F_max|) / C`.
Les minima de lignes différentes peuvent correspondre à des solutions
différentes. « Toutes évitables individuellement » ne signifie donc pas
« toutes évitables ensemble ».

[dérivé] Le diagnostic d'une vague utilise son état immédiatement précédent.
Le retrait initial imposé est exclu des nombres de pannes évitables/forcées.
Le marqueur « déjà saturée intacte » compare avec la référence propre à la
cascade. La part des premières vagues entièrement évitables a pour
dénominateur les retraits **avec au moins une vague**, pas tous les retraits.
Pour une cascade isolée, cet indicateur vaut 0 ou 1, et est indéfini sans vague.
Ces attributions conditionnelles ne constituent pas une preuve contrefactuelle
que l'élimination d'une saturation sauverait toute la cascade.

## Options et hypothèses

- [lu] `redispatch_exact(..., departage="highs")` conserve le chemin historique,
  sans second objectif ni contrainte supplémentaire.
- [dérivé] `min_charge_max` maximise d'abord le service par l'appel historique,
  puis minimise `t` avec `|F_l| <= t C_l`, les contraintes physiques originales
  et un service au moins égal à `S* − tol`. C'est une hypothèse de répartition
  recherchant une marge de transport, sans coût, rampe ni délai de réglage.
  Le scalaire optimal `t*` est déterminé ; tous les flux ne le sont pas.
- [dérivé] `min_redispatch` minimise ensuite `somme(|G_i − G_i_precedent|)`.
  L'ancienne production peut dépasser une nouvelle capacité disponible, mais
  doit être finie, non négative et alignée sur les générateurs. Elle est
  obligatoire pour cette règle et refusée pour les autres. Cet objectif ne
  pénalise ni la variation des flux ni la répartition du service entre charges.
- [dérivé] La cascade avec `min_redispatch` exige de choisir explicitement
  `departage_intact="highs"` ou `"min_charge_max"`. Elle utilise ensuite la
  production de l'état immédiatement précédent, y compris après îlotage.
  Le diagnostic comparatif choisit `min_charge_max`. Le paramètre est réservé
  à `min_redispatch`.
- [lu] `attribuer` accepte ces règles. Pour `min_redispatch`, sa production
  antérieure explicite utilise l'unité d'entrée (pu ou MW). En mode
  `flux_reference`, la même règle et la même production antérieure sont
  utilisées dans les deux dispatchs. La configuration sauvegarde les options ;
  les conditions et les textes des avertissements existants sont conservés.
- [dérivé] `etat_intact_prorata` sert la demande entière de chaque îlot et
  impose `G_i = demande_ilot * Pmax_i / somme(Pmax_ilot)`, puis calcule les
  flux DC sans PL. Il signale les violations de Pmax, de transport ou de bilan,
  sans écrêter les productions ni délester pour rendre le témoin faisable.
  Il ne peut pas être choisi comme règle de cascade.

[lu] `variabilite_flux` décrit par défaut la face **primaire**, même si on
la compare à un résultat départagé. Son option diagnostique
`departage="min_charge_max"` mesure aussi la non-unicité résiduelle sur la
face minimax, avec une tolérance supplémentaire de `1e-7` sur les taux.
Elle ne mesure pas la face secondaire de `min_redispatch`.

[dérivé] Un objectif strictement convexe dans les flux, tel que la somme des
carrés des taux, pourrait rendre ces flux uniques sur l'ensemble admissible.
L'intégration d'un solveur quadratique et sa validation sortent de cette
mission. Cela n'établirait d'ailleurs pas l'unicité de toutes les productions.

### Persistance de l'intact B4 pour C

[dérivé] `integration_n1` accepte `highs` et `min_charge_max`. Pour une
expérience `min_redispatch`, on calcule d'abord le B4 avec la règle de
`departage_intact`. La cascade recalcule cet intact, puis en utilise la
production pour son premier redispatch. Elle ne reçoit aucune production
antérieure externe qui permettrait de réutiliser un B4 `min_redispatch`.

[dérivé] L'intégration écrit `highs` dans le format historique
`cascade-entropy-integration-b4-c-v1`. Pour `min_charge_max`, le contrat
`cascade-entropy-integration-b4-c-v2` ajoute `sources_dispatch_sha256` avec
les empreintes de `dispatch.py`, `n1.py`, `_optimisation_dispatch_n1.py` et
`attribution_n1.py`. Le lecteur accepte les deux formats et vérifie la
présence et la forme des quatre empreintes en v2. Le schéma du manifeste
B5/B6 est commun aux deux formats.

[mesuré] Les tests comparent les octets des fichiers `highs` à ceux produits
par `a2f2d90` avec une provenance figée. Ils vérifient aussi l'aller-retour
disque et l'égalité des flux relus avec l'intact de la cascade à `1e-12`
près pour les deux règles, en capacités indépendantes et `flux_reference`.

## Tolérances et contrôles

- [lu] Service : `TOLERANCE_SERVIE * max(1, demande_ilot)`, soit `1e-9` fois
  cette échelle. La même contrainte est utilisée pour les diagnostics et les
  objectifs secondaires. La face explorée est donc un voisinage numérique de
  la face optimale exacte. Un très petit délestage positif peut apparaître ;
  les avertissements historiques ne sont pas supprimés pour le cacher.
- [lu] Non-unicité : plage strictement supérieure à
  `1e-6 * max(1, C_l)` en unités de puissance. Ce seuil de diagnostic dépasse
  la tolérance usuelle du solveur aux échelles du banc. Il ne borne pas
  l'amplification possible de la tolérance de service dans les cas sensibles ;
  ce n'est pas une tolérance physique universelle ni un test symbolique d'unicité.
- [lu] HiGHS : le solveur primaire historique reste non modifié (faisabilité
  usuelle de l'ordre de `1e-7` absolu). Les nouveaux PL emploient des tolérances
  primale et duale de `1e-10` ; un échec ou un résultat non admissible déclenche
  des essais précis sans présolve (simplexe dual puis point intérieur, avec
  mise à l'échelle), puis un point intérieur dont le coût est multiplié par
  0.01, puis les secours historiques sur le même nouvel objectif. Le facteur
  positif 0.01 conserve mathématiquement les minimiseurs ; il change le
  conditionnement numérique du coût, pas le critère physique.
  Les erreurs de bornes au plus
  égales à `1e-9 * max(1, plus_grande_borne_finie)` sont ramenées exactement
  dans les bornes, puis toutes les contraintes sont contrôlées avec cette
  tolérance. Une violation supérieure lève une erreur. Cette normalisation
  numérique évite de transmettre des productions négatives à l'état suivant ;
  elle ne s'applique jamais au défaut historique. Le test comparant le minimax
  réalisé à son optimum utilise `1e-7` en taux. Les inégalités du minimax sont
  exprimées en taux pour améliorer l'échelle.
- [lu] Classification : comparaison stricte au seuil, sans bande inventée.
  Une valeur très proche de 0.99 reste numériquement fragile.
- [lu] Prorata : violations au-delà de `1e-8 * max(1, grandeur)` ; un îlot
  sans production disponible et à demande positive est explicitement
  infaisable, ses flux non calculables restent à NaN.
- [lu] Fraction incrémentale considérée nulle dans le CSV : valeur absolue
  `<= 1e-9`. Quantile 90 % : interpolation linéaire de NumPy.

[lu] Les erreurs des objectifs secondaires remontent avec leur cause ; aucun
repli vers le résultat primaire n'est effectué. Les secours du solveur gardent
le même objectif. Leur utilisation est comptée par îlot dans `n_secours`.
Les diagnostics réutilisent en mémoire les PL d'îlots identiques : cache borné
à 1024 entrées, clé comprenant toute la topologie locale, les paramètres
électriques, le service cible, les lignes mesurées et l'option de face. Les valeurs cachées sont
immuables ; les résultats retournés possèdent leurs propres tableaux.

[dérivé] A4 utilise les mêmes noyaux de variabilité et de classement qu'A2/A3,
en calculant seulement les deux extrema des lignes effectivement tombées.
**Toutes** les lignes actives restent dans les contraintes physiques : il
s'agit de supprimer des objectifs inutilisés, pas de réduire le réseau.
L'API publique A2 continue à calculer les plages de toutes les lignes actives.
Les résultats partiels ne sont pas exposés publiquement.

## Protocole reproductible

[dérivé] `scripts/diagnostic_departage_n1.py` énumère les retraits de huit
réseaux synthétiques de 50 nœuds selon les défauts de `topologies_n1`.
L'anneau à cordes, Gilbert et le géométrique utilisent les graines 20261113
et 20261114. La grille et l'arbre témoin sont fixes. Le script de comparaison
apparie les mêmes retraits entre règles de départage.

[lu] Réglages électriques provisoires repris de l'audit : 8 producteurs et
20 charges placés uniformément sans remise, graine 11 ; demandes uniformes
sur `[0.5, 1.5]` pu, graine 12 ; Pmax de 5 pu par producteur ; réactances
uniformes sur `[0.2, 0.8]` pu, graine 13. Référence de transport calculée avec
limites de 100 pu ; capacités finales `max(1.5 * |F_reference|, 0.01 pu)`.
`p1=1` exclusivement, sans tirage aléatoire de panne.

[dérivé] Le banc fige les capacités issues d'une référence `highs` pour
toutes les règles. Les cascades `highs` et `min_charge_max` utilisent leur
propre règle pour l'intact et les vagues. La cascade `min_redispatch` part d'un
intact minimax. Le banc compare donc des politiques complètes. Une expérience
qui isole le redispatch après le premier retrait demanderait un état intact
commun et reste une extension future.

[lu] Le script refuse un répertoire de run existant sans `--resume` ; une
reprise exige des paramètres, versions et empreintes de sources identiques.
Les JSON complets par règle sont publiés sans écrasement. Les cas déjà
terminés sont relus. Métadonnées : graines, paramètres, Python, NumPy, SciPy,
HiGHS embarqué, système, réglages de threads, HEAD et SHA-256 des sources
effectivement exécutées. Les temps CSV excluent le diagnostic intact commun,
dont la durée est conservée dans le JSON de configuration du réseau.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python scripts/diagnostic_departage_n1.py --run-id departage-20261118-v3 --workers 8
python scripts/diagnostic_departage_n1.py --run-id departage-20261118-v3 --workers 8 --resume
python scripts/comparer_regles_departage_n1.py --run-id departage-20261118-v3
python scripts/diagnostic_departage_n1.py --run-id departage-20261117-audit-verifie --audit-reference --workers 3
```

### Capacités de référence et non-unicité résiduelle

[mesuré] Avec les capacités taillées sur les flux `highs`, l'écart maximal
entre les flux intacts `min_charge_max` et la référence est d'environ
`4,4e-9` pu sur les sept maillages. Sur l'arbre témoin, il atteint environ
5 pu sur la ligne d'indice 2, de `(0, 3)`. Le taux maximal est proche de
`2/3` sur les huit réseaux. L'arbre conserve huit flux non uniques sur la
face minimax.

[dérivé] Sur les maillages, le minimax retrouve à la tolérance numérique le
dispatch `highs` qui a servi à dimensionner les limites. Le choix du sommet
`highs` intervient donc dans la comparaison par les capacités, ce qui
favorise ici la règle qui retrouve cet état de dimensionnement. Sur l'arbre,
le même taux maximal laisse plusieurs flux possibles. Le départage du
dispatch de référence constitue une décision scientifique à part entière.
Un troisième objectif demanderait une justification scientifique.

[dérivé] En mode `flux_reference`, B4 applique la règle demandée au dispatch
provisoire puis au dispatch final. Un B4 minimax façonne donc ses propres
capacités `max(1,5·|F_min_charge_max|, 0,01 pu)`.
[mesuré] Sur ces huit réseaux, le recalcul B4 minimax donne un taux final
de `2/3` à environ `2,12e-12` près sur les lignes au-dessus du plancher.
Les statistiques du banc concernent les capacités communes calibrées par
`highs`. Leur application à un B4 minimax demanderait une nouvelle mesure.

## Validation et résultats

[mesuré] Validation initiale du 17 novembre sous Windows.

| Commande | Environnement Windows | Résultat | Durée |
|---|---|---:|---:|
| `python -m pytest -q` avant modification | Python 3.11.9, NumPy 2.3.4, SciPy 1.16.3 | 858 réussis | 138.59 s |
| `python -m pytest -q` après le premier diagnostic | mêmes versions | 935 réussis | 136.83 s |
| `py -3.13 -m pytest -q` après le premier diagnostic | Python 3.13.7, NumPy 2.3.3, SciPy 1.16.2 | 935 réussis | 140.64 s |
| `python -m pytest tests/test_departage_diagnostic_n1.py tests/test_script_diagnostic_departage_n1.py -q --tb=short` | Python 3.11.9 | 77 réussis | 3.53 s |

[mesuré] Validation de la révision du 18 novembre sous Windows, avec
`OPENBLAS_NUM_THREADS=1` et `OMP_NUM_THREADS=1`. La suite complète inclut les
935 tests précédents et les 28 nouveaux, sans modification de test existant.

| Commande | Environnement Windows | Suite complète | 28 nouveaux tests seuls |
|---|---|---:|---:|
| `python -m pytest -q` | Python 3.11.9, NumPy 2.3.4, SciPy 1.16.3 | 963 réussis en 202,29 s | 28 réussis en 2,48 s |
| `py -3.13 -m pytest -q` | Python 3.13.7, NumPy 2.3.3, SciPy 1.16.2 | 963 réussis en 201,05 s | 28 réussis en 2,49 s |

[mesuré] Les 77 tests du premier diagnostic couvrent le radial symétrique (1 pu par
producteur et taux maximal 0.5), la conservation du service primaire et les
objectifs secondaires, bornes et bilans nodaux/par îlot, loi DC indépendante
par résolution des angles, flux retirés exactement nuls, plages analytiques
dont changement de signe, saturations évitables et forcées dans un même état,
historique par vague et marqueur intact, îlots triviaux, non-mutation,
déterminisme, non-unicité résiduelle minimax, configuration/rejeu et validations.
Les erreurs de solveur, les secours à objectif conservé et les solutions hors
bornes sont injectés dans les tests. Le cas numérique de production légèrement
négative rencontré pendant le diagnostic possède son test de régression.
Le diagnostic ciblé est comparé au diagnostic complet sur tous les retraits
du banc fixé de 8 nœuds.

[mesuré] Lors des contrôles du 17 novembre, les 24 résultats par réseau/règle déjà terminés avec
le calcul complet (deux anneaux, arbre, grille et deux cas de référence)
ont été comparés à leurs nouvelles sorties ciblées : mêmes champs de résumé,
mêmes fractions et mêmes diagnostics par vague, temps exclus.
Un second problème synthétique archivé reproduit le résidu de bilan de
`6.12e-9` pu annoncé réussi par HiGHS ; le réessai précis sans présolve donne
`1.11e-15` pu et conserve l'objectif.

[mesuré] Un troisième problème archivé vient du graphe géométrique de graine
20261114, retrait initial 52, lors d'une vague en `min_redispatch`. HiGHS
renvoie le statut `Unknown` pour les premiers essais précis ; un PL primaire
précis confirme pourtant que les contraintes de service sont faisables.
Le point intérieur avec coût multiplié par 0.01 trouve un coût L1 original
de `2.2484223109` pu, avec résidus de contraintes de l'ordre de `4e-15` pu.
Les 175 retraits de ce réseau ont ensuite été rejoués en `min_redispatch`,
tous réussis, avec un seul îlot post-retrait ayant utilisé un secours.

[mesuré] La non-régression bit à bit compare les trois fonctions demandées
aux sources de `bebe96d`, figées dans une nouvelle fixture, et exécutées avec
les bibliothèques courantes. Le banc fixé comporte 8 nœuds, deux générateurs,
quatre charges, des capacités flux_reference et chacun de ses 12 retraits.
Tous les tableaux, avertissements, configurations et historiques sont
comparés. Cette méthode vérifie la conservation du défaut sur chaque pile
testée sans figer artificiellement un sommet d'une autre version de HiGHS.
Elle n'est pas une preuve pour tous les réseaux possibles.

[mesuré] Les quatre résultats de l'arbre témoin, y compris les diagnostics
par retrait et par vague, sont identiques en exécution série et en processus
indépendants, hors temps. La comparaison complète n'a pas été rejouée en série.

[mesuré] Comparaison directe avec `references_audit/degenerescence.py` fourni
dans l'archive, exécuté sans intégrer son code au moteur (Python 3.13.7,
NumPy 2.3.3, SciPy 1.16.2). Les graines sont 7 pour les graphes aléatoires
et 11/12/13 pour l'électricité. Les résultats et les empreintes de l'archive
sont dans [validation_references.json](departage_n1/validation_references.json).

| Famille | Capacités | Saturées | Évitables / forcées | t minimal |
|---|---|---:|---:|---:|
| Gilbert | uniformes | 5 | 5 / 0 | 0.731355 |
| Gilbert | flux_reference | 8 | 8 / 0 | 0.666667 |
| Anneau | uniformes | 13 | 0 / 13 | 1.000000 |
| Anneau | flux_reference | 3 | 3 / 0 | 0.666667 |
| Grille | uniformes | 11 | 0 / 11 | 1.000000 |
| Grille | flux_reference | 5 | 5 / 0 | 0.666667 |

[mesuré] L'écart maximal des bornes de flux avec le script de référence est
`4.13e-8` pu en mode flux_reference, et l'écart du t minimal reste inférieur
à `2.1e-9` sur les six cas. Les faces ne sont pas strictement identiques : le
script de référence impose S* en égalité, le nouveau diagnostic autorise
S* − tol. Pour l'anneau à capacités uniformes, cela donne 9 plages dépassant
`1e-6` pu au lieu de 0 dans le script exact, avec un écart maximal de
`2.95e-6` pu. Ces 9 plages ne démontrent pas une non-unicité de la face exacte.
Les 13 saturations restent classées forcées. Ce cas illustre pourquoi il faut
conserver la tolérance et l'échelle avec tout nombre de flux « non uniques ».

[mesuré] Le script original `cascade_artefact.py`, exécuté sous le même
environnement 3.13, retrouve pour Gilbert 127/127 retraits avec vague,
97 premières vagues contenant une ligne saturée intacte, une fraction
incrémentale médiane de 0.647 et un maximum de 0.827 (arrondis de son affichage).
Pour la grille : 85/85, 79 premières vagues contenant une ligne saturée intacte,
médiane 0.558, maximum 0.646. Ce contrôle est distinct du comptage évitable/
forcé de chaque vague, absent du script original.

## Comparatif final sur huit réseaux

[mesuré] Le run v3 utilise Windows, Python 3.11.9, NumPy 2.3.4, SciPy 1.16.3
et HiGHS 1.8.0. Il couvre 918 retraits par règle, soit 2 754 cascades. Les
huit témoins prorata sont évalués à l'intact. Le tableau donne le nombre de
retraits avec vague et la médiane de fraction délestée incrémentale. Ces
effectifs et médianes arrondies retrouvent ceux de l'archive du 17 novembre.

| Réseau | Lignes | `highs` avec vague · médiane | `min_charge_max` avec vague · médiane | `min_redispatch` avec vague · médiane |
|---|---:|---:|---:|---:|
| Anneau 20261113 | 75 | 75/75 · 35.3 % | 50/75 · 9.8 % | 67/75 · 14.5 % |
| Anneau 20261114 | 75 | 75/75 · 57.0 % | 55/75 · 4.1 % | 69/75 · 7.6 % |
| Gilbert 20261113 | 132 | 132/132 · 60.9 % | 76/132 · 0.0 % | 95/132 · 0.0 % |
| Gilbert 20261114 | 137 | 132/137 · 74.0 % | 71/137 · 0.0 % | 92/137 · 0.0 % |
| Géométrique 20261113 | 190 | 190/190 · 63.5 % | 83/190 · 0.0 % | 130/190 · 0.0 % |
| Géométrique 20261114 | 175 | 175/175 · 67.3 % | 106/175 · 0.0 % | 123/175 · 0.0 % |
| Grille | 85 | 85/85 · 55.8 % | 42/85 · 0.0 % | 71/85 · 24.0 % |
| Arbre témoin | 49 | 45/49 · 10.7 % | 6/49 · 3.4 % | 15/49 · 3.4 % |

[mesuré] Les 35 lignes saturées intactes sous highs sont toutes évitables
individuellement. Les deux options ont zéro ligne saturée intacte. La référence
intacte de min_redispatch est ici le minimax. Tous les optima minimax intacts
atteignent environ 2/3. Le témoin prorata conserve ses violations de transport.

| Réseau | Saturées highs (évitables) | Flux non uniques primaire / minimax | t minimal | Taux maximal prorata | Prorata faisable |
|---|---:|---:|---:|---:|---|
| Anneau 20261113 | 3 (3) | 75 / 0 | 0.666667 | 63.33 | non |
| Anneau 20261114 | 5 (5) | 75 / 0 | 0.666667 | 33.74 | non |
| Gilbert 20261113 | 4 (4) | 132 / 0 | 0.666667 | 228.86 | non |
| Gilbert 20261114 | 4 (4) | 137 / 0 | 0.666667 | 73.27 | non |
| Géométrique 20261113 | 7 (7) | 190 / 0 | 0.666667 | 16.35 | non |
| Géométrique 20261114 | 6 (6) | 172 / 0 | 0.666667 | 26.04 | non |
| Grille | 5 (5) | 85 / 0 | 0.666667 | 29.91 | non |
| Arbre témoin | 1 (1) | 15 / 8 | 0.666667 | 228.86 | non |

[mesuré] Les huit proratas ont une production disponible suffisante et
violent les limites de transport. [dérivé] Le test dépend des capacités
dimensionnées par les flux `highs`. D'autres capacités pourraient rendre
le prorata admissible. La classification évitable/forcée est non applicable
à ces huit témoins. Le CSV laisse les effectifs vides, le JSON les représente
par `null`, et `classification_intacte_applicable` vaut `false`.

[mesuré] Statistiques regroupant tous les retraits de chaque règle, sans
moyenner les médianes par réseau. Les valeurs numériquement nulles sont
affichées comme zéro ; les valeurs brutes, y compris les minuscules fractions
négatives liées aux tolérances de service, restent dans les CSV.

| Règle | Avec vague | Délestage incrémental nul | Médiane | Q90 | Maximum | Pannes évitables / forcées |
|---|---:|---:|---:|---:|---:|---:|
| `highs` | 909/918 | 8/918 | 63.5 % | 74.0 % | 100.0 % | 20484 / 58157 |
| `min_charge_max` | 489/918 | 579/918 | 0.0 % | 63.5 % | 89.3 % | 2179 / 18544 |
| `min_redispatch` | 662/918 | 478/918 | 0.0 % | 67.3 % | 93.1 % | 3885 / 24787 |

[dérivé] **Sous `highs`, 20 484 / 78 641 = 26,047 % des pannes secondaires
sont classées évitables**, contre `20 482 / 78 638 = 26,046 %` dans l'archive.
Les deux valeurs s'arrondissent à 26,05 %. En incluant les 918 retraits
imposés au dénominateur, la part v3 est de 25,75 %. Les premières vagues sont
entièrement évitables dans 422/909 cascades avec vague, soit 46,42 %.
Les parts évitables des deux autres règles passent de 10,52 % à 10,51 %
pour le minimax et de 13,51 % à 13,55 % pour `min_redispatch`.

[mesuré] Les fractions brutes v3 sont négatives dans 0 retrait sous `highs`,
162 sous `min_charge_max` et 130 sous `min_redispatch`. Leurs minima par
règle sont respectivement `0`, `−3,6123e-12` et `−9,9999993e-10`.
[dérivé] Ce sont des résidus numériques. Le second objectif permet un service
`≥ S* − 1e-9·max(1, Σd)` par îlot. Le dernier état peut servir légèrement plus
de charge que l'intact et donner un incrément négatif. Le compte et le
minimum par réseau et par règle sont dans les sorties, sans écrêtage.
[Comparatif CSV](departage_n1/comparatif.csv) · [Synthèse](departage_n1/synthese.json)

[dérivé] Passer de highs au minimax réduit ici les retraits avec vague de 909
à 489 ; min_redispatch les réduit à 662. Les médianes regroupées deviennent
numériquement nulles, mais les queues restent élevées : Q90 de 63.5 % et
67.3 %, maxima de 89.3 % et 93.1 %. Il n'y a pas d'amélioration uniforme de
chaque retrait : le maximum du second anneau passe par exemple de 70.6 %
sous highs à 71.7 % sous minimax. Ces différences ne donnent pas une part
causale du délestage imputable aux seules saturations évitables.

[mesuré] Référence Gilbert de l'audit, graine 7 :

| Règle | Avec vague | Délestage nul | Médiane | Q90 | Maximum | Évitables / forcées |
|---|---:|---:|---:|---:|---:|---:|
| highs | 127/127 | 0/127 | 64.7 % | 74.4 % | 82.7 % | 2181 / 9891 |
| min_charge_max | 66/127 | 75/127 | 0.0 % | 66.5 % | 88.4 % | 479 / 4158 |
| min_redispatch | 94/127 | 59/127 | 5.2 % | 70.4 % | 79.8 % | 660 / 5119 |

[dérivé] Sur ce seul Gilbert : 2 181/12 072 = **18.07 %** de pannes
secondaires évitables sous highs. La médiane passe de 64.7 % à 0 % ou 5.2 %.
Cette sensibilité forte au départage coexiste avec une minorité de pannes
individuellement évitables : les pannes suivantes sont classées dans un réseau
déjà dégradé. Le maximum minimax (88.4 %) dépasse celui de highs (82.7 %) ;
aucune domination scénario par scénario n'est établie.

[mesuré] Le temps mural est de 809,89 s pour v3 (8 processus) et de 170,03 s
pour le run de référence de l'audit (3 processus). Les diagnostics intacts
communs représentent respectivement 46,45 s et 5,60 s cumulés.

Les temps CSV excluent cette étape commune et l'attribution initiale. Ils
incluent cache et concurrence, dans l'ordre highs, minimax, min_redispatch,
prorata : ce n'est pas un benchmark isolé du solveur. La reprise d'un run
complet a été exécutée, sans recalcul ni écrasement.

[dérivé] Les métadonnées v3 indiquent le HEAD de départ `a2f2d90` et les
empreintes des sources exécutées avec les modifications locales. La
synthèse ajoute le SHA-256 de la copie LF versionnée de `retraits.csv`, en
complément du SHA-256 du CSV brut enregistré lors de la comparaison.
[Métadonnées du run](departage_n1/metadata.json) · [Métadonnées de comparaison](departage_n1/comparaison_metadata.json)

### Classement et gravité selon la règle

[dérivé] Le Spearman apparie les mêmes retraits. Pour calculer les rangs, il
ramène `|f| ≤ 1e-9` à zéro, arrondit les autres fractions à neuf décimales,
puis applique les rangs moyens de `scipy.stats.spearmanr`. Le CSV indique
aussi le coefficient brut, le nombre de zéros et le nombre de valeurs dans
des groupes ex æquo avant et après arrondi. Les nombreux ex æquo limitent
l'interprétation d'un classement fin, surtout près de zéro.
[Comparaison complète](departage_n1/spearman_regles.csv)

[mesuré] Corrélations v3 avec les rangs traités.

| Réseau | `highs` / minimax | `highs` / redispatch | Minimax / redispatch |
|---|---:|---:|---:|
| Anneau 20261113 | 0,328 | 0,333 | 0,802 |
| Anneau 20261114 | 0,222 | 0,208 | 0,758 |
| Gilbert 20261113 | 0,124 | 0,101 | 0,734 |
| Gilbert 20261114 | 0,393 | 0,353 | 0,832 |
| Géométrique 20261113 | 0,251 | 0,246 | 0,842 |
| Géométrique 20261114 | 0,111 | 0,114 | 0,602 |
| Grille | 0,096 | 0,023 | 0,515 |
| Arbre témoin | 0,776 | 0,815 | 0,988 |

[mesuré] Dans le CSV archivé du 17 novembre, les 28 valeurs `highs` de
l'arbre affichées à `0,106817` occupent trois représentations binaires, sur
une amplitude de `2,78e-17`. Pour `highs`/minimax, le Spearman de cette archive
passe de `0,565` sur le brut à `0,738` en rétablissant seulement cet ex æquo,
puis à `0,776` avec tout le traitement. Le Spearman brut v3 vaut `0,561`.
[dérivé] Le départage doit être fixé avant toute comparaison avec les
centralités.

[mesuré] Gravité v3 sur les fractions brutes. Chaque ligne agrège 918
retraits. Les cinq effectifs totalisent 918 par règle.

| Règle | Nulle | ]0, 5 %[ | [5 %, 30 %[ | ≥ 30 % | Résidu négatif hors tolérance | Médiane hors classe nulle |
|---|---:|---:|---:|---:|---:|---:|
| `highs` | 0,87 % | 0,00 % | 7,95 % | 91,18 % | 0,00 % | 63,54 % |
| `min_charge_max` | 63,07 % | 4,03 % | 10,02 % | 22,88 % | 0,00 % | 44,12 % |
| `min_redispatch` | 52,07 % | 3,59 % | 10,46 % | 33,88 % | 0,00 % | 51,81 % |

[dérivé] La classe nulle couvre `|f| ≤ 1e-9`. La cinquième classe couvre
`f < −1e-9` et entre dans la médiane des cas non nuls. Les répartitions
par réseau utilisent les mêmes bornes. [Répartition détaillée](departage_n1/repartition_gravite.csv)

[hypothèse] La concentration des retraits à gravité nulle ou élevée peut
venir de la propagation déterministe. Avec `p1=1`, chaque ligne active
à `0,99` ou plus tombe. Si un retrait rend le service dépendant de lignes à
leur limite, leur perte simultanée peut fragmenter le réseau et accentuer
le délestage. Cette interprétation reste à tester. Le seuil, `p1` et la
mesure de gravité sont donc aussi des choix du protocole.

## Reproductibilité

[mesuré] V3 a les mêmes paramètres, graines et versions déclarées que le run
principal du 17 novembre. Les 2 754 clés `(réseau, règle, retrait)`
coïncident. Parmi elles, 25 lignes sur huit couples `(réseau, règle)`
diffèrent sur un compteur ou sur la fraction de plus de `1e-12`. Le détail
donne les valeurs d'archive et de v3 pour chaque champ modifié. Cinq fractions
dépassent la tolérance, avec un écart maximal de 21,79 points de pourcentage
pour le retrait 153 du géométrique 20261114 sous `min_redispatch`. Le rejeu
ciblé des 25 cas avec `a2f2d90` et avec le code révisé retrouve les lignes v3
et les mêmes octets de flux entre les deux versions.
[Écarts par retrait](departage_n1/ecarts_archive_20261117.csv)

[mesuré] Selon le rejeu indépendant communiqué par Alex, Linux avec NumPy
2.5.3 et SciPy 1.18.1 retrouve les cinq fractions v3 à `1e-10` près. Ce
rejeu signale également des différences avec l'archive sur le retrait 39
de l'anneau 20261114 et sur les retraits 58, 61, 80 et 109 du Gilbert
20261113 sous `min_redispatch`. Ces cas témoignent d'une sensibilité à la
plateforme et à la pile numérique. Le run Windows v3 sert de référence.

[mesuré] Pour le retrait 25 du géométrique 20261113 sous `min_redispatch`,
la fraction passe de `−1,0000000037990338e-9` à `−9,99999682408678e-10`.
Cette variation sous `1e-12` déplace le retrait de la cinquième classe à
la classe nulle, qui passe de 477 à 478 cas. Les médianes regroupées, Q90
et maxima restent identiques à l'arrondi du tableau. Les compteurs au
voisinage de zéro et les attributions par vague varient. Ainsi, les
effectifs bruts de fractions négatives passent de 405 à 162 sous le minimax
et de 333 à 130 sous `min_redispatch`.

[hypothèse] Le seuil `0,99` avec `p1=1` et la dégénérescence des programmes
linéaires secondaires peuvent amplifier des écarts numériques de l'ordre
de `1e-15` sur environ 1 % des retraits. Les statistiques globales de
gravité restent comparables à la précision d'affichage du banc. Ce
mécanisme constitue une explication plausible, sans attribution causale
retrait par retrait.

[dérivé] Le critère de reproductibilité retient cette comparabilité des
statistiques et documente les écarts par retrait. La fixture `bebe96d`
continue à vérifier séparément la non-régression bit à bit des défauts.

[mesuré] L'ancien dossier local `departage-20261117-verifie` et un éventuel
run v2 sont absents de ce workspace. Le début enregistré est
`2026-11-18T03:06:07.389331Z`. La comparaison des dates de ses JSON et
l'hypothèse d'une copie depuis v2 restent non vérifiables ici.
[dérivé] Le script de `a2f2d90` produisait les JSON par règle et
`comparatif.csv`. Son `retraits.csv` archivé a été ajouté dans ce même
commit. Sa structure correspond à une consolidation des détails JSON par
retrait. La commande d'origine et l'identité de ses fichiers d'entrée
restent inconnues. V3 exporte directement `retraits.csv` et l'archive du
17 novembre reste consultable dans `a2f2d90`.

## Fichiers livrés

[dérivé] Le tableau regroupe les fichiers du diagnostic initial et de sa
révision. Les sorties proviennent des réseaux synthétiques du banc.

| Fichier | Rôle |
|---|---|
| `cascade_entropy/n1.py` | Options de départage et production antérieure par îlot. |
| `cascade_entropy/cascade_n1.py` | Transmission de la règle et production d'une vague à la suivante. |
| `cascade_entropy/attribution_n1.py` | Options intactes cohérentes, configuration et rejeu. |
| `cascade_entropy/integration_n1.py` | Persistance B4/C en v1 et v2 selon la règle intacte. |
| `cascade_entropy/_optimisation_dispatch_n1.py` | Formulations secondaires et résolutions contrôlées. |
| `cascade_entropy/diagnostics_dispatch_n1.py` | Plages, classement, attribution des vagues et prorata. |
| `scripts/diagnostic_departage_n1.py` | Banc technique, métadonnées, sorties exclusives, reprise et processus. |
| `scripts/comparer_regles_departage_n1.py` | Spearman apparié, ex æquo et classes de gravité. |
| `tests/test_departage_diagnostic_n1.py` | Oracles analytiques, physiques, non-régression et cas numériques. |
| `tests/test_script_diagnostic_departage_n1.py` | Sorties, reprise et refus d'écrasement. |
| `tests/test_integration_departage_n1.py` | Octets v1, contrat v2 et concordance entre B4 et cascade. |
| `tests/test_cascade_departage_intact_n1.py` | Validation du choix explicite de l'intact. |
| `tests/test_comparer_regles_departage_n1.py` | Rangs, bornes des classes et prorata infaisable. |
| `tests/fixtures/departage_bebe96d.json` | Sources témoins au commit de départ. |
| `tests/fixtures/departage_precision.json` | PL reproduisant le défaut de précision du bilan. |
| `tests/fixtures/departage_cout_echelle.json` | PL reproduisant le statut Unknown de min_redispatch. |
| `docs/departage_n1.md` | Définitions, mesures, limites et décisions à soumettre. |
| `docs/departage_n1/comparatif.csv` | Les 32 lignes réseau/règle avec tous les indicateurs demandés. |
| `docs/departage_n1/retraits.csv` | Les 2 754 scénarios permettant de recalculer les statistiques. |
| `docs/departage_n1/metadata.json` | Paramètres, graines, versions et empreintes du run principal. |
| `docs/departage_n1/spearman_regles.csv` | Corrélations brutes et arrondies par réseau et paire de règles. |
| `docs/departage_n1/repartition_gravite.csv` | Effectifs, parts et médianes par règle et réseau. |
| `docs/departage_n1/comparaison_metadata.json` | Versions, paramètres et empreinte du CSV brut de v3. |
| `docs/departage_n1/ecarts_archive_20261117.csv` | Champs modifiés dans les 25 lignes en écart. |
| `docs/departage_n1/audit_comparatif.csv` | Comparatif des trois réseaux de référence de l'audit. |
| `docs/departage_n1/audit_retraits.csv` | Les 861 scénarios de contrôle des références. |
| `docs/departage_n1/audit_metadata.json` | Provenance du run de contrôle. |
| `docs/departage_n1/validation_references.json` | Confrontation des six cas aux PL de référence. |
| `docs/departage_n1/synthese.json` | Consolidation, dénominateurs et temps explicites. |

[dérivé] La consolidation vérifie l'exhaustivité des retraits, leur unicité
et l'égalité entre les sommes par retrait et les totaux CSV. Les JSON
détaillés par vague restent dans les runs locaux sous
`data/diagnostic_departage_n1/`, régénérables par les commandes ci-dessus.

## Limites et décisions scientifiques

[lu] Ce banc contient deux réalisations par famille aléatoire, un seul
placement électrique commun et `p1=1`. Il ne mesure pas une distribution de
vulnérabilité, une réponse temporelle réelle, ni une validation électrique
d'un réseau réel. Aucun matériel restreint n'est utilisé.

[dérivé] Une réduction de cascade sous un autre départage démontre une
sensibilité à la règle dans ce modèle et ces paramètres. Elle n'attribue pas
automatiquement la même part du délestage final aux pannes évitables :
réactions successives, îlotage et redispatch se répondent.

[dérivé] Les erreurs de solveur sont testées par injection sur des cas
ciblés. La couverture ne s'étend pas à tous les modes d'échec, aux échelles
extrêmes ou aux performances sur de grands réseaux. HiGHS peut choisir
d'autres sommets entre versions et plateformes, y compris après les
objectifs secondaires. La section Reproductibilité donne le périmètre du
rejeu Linux communiqué.

Décisions à discuter avec Patrick

- La saturation au seuil 0.99 doit-elle provoquer des pannes secondaires dans
  ce modèle où le dispatch respecte déjà les capacités ?
- Quel objectif secondaire et quelle règle d'état intact sont physiquement
  défendables ? Quels délais, rampes et coûts faudrait-il ensuite représenter ?
- Les capacités doivent-elles rester communes entre règles ou être recalées
  selon un dispatch de référence choisi scientifiquement ?
- Quel départage appliquer au dispatch de référence qui dimensionne les
  capacités ?
- Quelle mesure de gravité retenir en présence de nombreux ex æquo à zéro ?
  [hypothèse] Une piste à tester est de choisir `p1 < 1` et de moyenner la
  gravité sur plusieurs tirages.
- Quel placement, quelles puissances, marges et probabilités de panne utiliser
  dans un futur protocole scientifique ?
- Faut-il explorer ultérieurement un objectif strictement convexe ?

Proposition pour `docs/journal.md` (non écrite dans ce fichier) :

> 2026-11-17 — Instrumentation du départage N−1 sur graphes synthétiques.
> Ajout d'options minimax des taux et minimisation du redispatch, sans changer
> le défaut HiGHS ni les conventions Carreras. Les saturations sont classées
> individuellement dans l'état précédant chaque vague. Comparaison technique
> à capacités figées ; choix scientifique du départage et du seuil à valider
> avec Patrick. Voir `docs/departage_n1.md` pour les mesures et leurs limites.
