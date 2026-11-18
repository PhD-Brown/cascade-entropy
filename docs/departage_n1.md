# Départage des dispatchs N−1 : diagnostic technique

Travail du 17 novembre 2026, base `bebe96d`, branche
`codex/departage-diagnostic`. Graphes synthétiques uniquement. Aucun choix
scientifique définitif n'est introduit ; `highs` reste le défaut.

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
- [lu] La cascade avec `min_redispatch` prend sa référence selon
  `departage_intact="highs"` (défaut) ou `"min_charge_max"`, puis utilise
  toujours la production de l'état immédiatement précédent, y compris après
  îlotage. Le diagnostic comparatif choisit explicitement `min_charge_max`.
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

[lu] `scripts/diagnostic_departage_n1.py` n'effectue aucune analyse de
centralité, aucun classement ni corrélation. Il énumère seulement les retraits
des huit réseaux techniques suivants : anneau à cordes, Gilbert et géométrique
avec graines 20261113 et 20261114 ; grille et arbre témoin fixes. Tailles et
autres paramètres topologiques : défauts de `topologies_n1`, soit 50 nœuds.

[lu] Réglages électriques provisoires repris de l'audit : 8 producteurs et
20 charges placés uniformément sans remise, graine 11 ; demandes uniformes
sur `[0.5, 1.5]` pu, graine 12 ; Pmax de 5 pu par producteur ; réactances
uniformes sur `[0.2, 0.8]` pu, graine 13. Référence de transport calculée avec
limites de 100 pu ; capacités finales `max(1.5 * |F_reference|, 0.01 pu)`.
`p1=1` exclusivement, sans tirage aléatoire de panne.

[lu] **Capacités communes entre les règles** : elles sont calculées une seule
fois avec `attribuer(..., departage="highs")`, puis figées. Le deuxième
dispatch intact et tous les redispatchs emploient la règle comparée. C'est une
hypothèse technique explicite pour isoler son effet. Recalculer les capacités
depuis la référence propre à chaque règle, bien que permis par `attribuer`,
changerait aussi le réseau physique et répondrait à une autre question.

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
python scripts/diagnostic_departage_n1.py --run-id departage-20261117-verifie --workers 8
python scripts/diagnostic_departage_n1.py --run-id departage-20261117-verifie --workers 8 --resume
python scripts/diagnostic_departage_n1.py --run-id departage-20261117-audit-verifie --audit-reference --workers 3
```

## Validation et résultats

[mesuré] Validation finale locale, sans modification d'un test existant :

| Commande | Environnement Windows | Résultat | Durée |
|---|---|---:|---:|
| `python -m pytest -q` avant modification | Python 3.11.9, NumPy 2.3.4, SciPy 1.16.3 | 858 réussis | 138.59 s |
| `python -m pytest -q` final | mêmes versions | 935 réussis | 136.83 s |
| `py -3.13 -m pytest -q` final | Python 3.13.7, NumPy 2.3.3, SciPy 1.16.2 | 935 réussis | 140.64 s |
| `python -m pytest tests/test_departage_diagnostic_n1.py tests/test_script_diagnostic_departage_n1.py -q --tb=short` | Python 3.11.9 | 77 réussis | 3.53 s |

[mesuré] Les 77 nouveaux tests couvrent : le radial symétrique (1 pu par
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

[mesuré] En complément, les 24 résultats par réseau/règle déjà terminés avec
le calcul complet (deux anneaux, arbre, grille et deux cas de référence)
ont été comparés à leurs nouvelles sorties ciblées : mêmes champs de résumé,
mêmes fractions et mêmes diagnostics par vague, temps exclus.
Un second problème synthétique archivé reproduit le résidu de bilan de
`6.12e-9` pu annoncé réussi par HiGHS ; le réessai précis sans présolve donne
`1.11e-15` pu et conserve l'objectif. Les runs ayant échoué avant ces corrections
restent locaux et ne sont pas mélangés aux mesures finales.

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

## Comparatif final : huit réseaux

[mesuré] Run principal : Windows, Python 3.11.9, NumPy 2.3.4, SciPy 1.16.3,
HiGHS 1.8.0. 918 retraits par règle, soit 2 754 scénarios de cascade ; les
huit témoins prorata ne sont pas propagés. Le tableau donne le nombre de
retraits avec vague et la médiane de fraction délestée incrémentale.

| Réseau | Lignes | highs : vagues ; médiane | min_charge_max : vagues ; médiane | min_redispatch : vagues ; médiane |
|---|---:|---:|---:|---:|
| Anneau 20261113 | 75 | 75/75 ; 35.3 % | 50/75 ; 9.8 % | 67/75 ; 14.5 % |
| Anneau 20261114 | 75 | 75/75 ; 57.0 % | 55/75 ; 4.1 % | 69/75 ; 7.6 % |
| Gilbert 20261113 | 132 | 132/132 ; 60.9 % | 76/132 ; 0.0 % | 95/132 ; 0.0 % |
| Gilbert 20261114 | 137 | 132/137 ; 74.0 % | 71/137 ; 0.0 % | 92/137 ; 0.0 % |
| Géométrique 20261113 | 190 | 190/190 ; 63.5 % | 83/190 ; 0.0 % | 130/190 ; 0.0 % |
| Géométrique 20261114 | 175 | 175/175 ; 67.3 % | 106/175 ; 0.0 % | 123/175 ; 0.0 % |
| Grille | 85 | 85/85 ; 55.8 % | 42/85 ; 0.0 % | 71/85 ; 24.0 % |
| Arbre témoin | 49 | 45/49 ; 10.7 % | 6/49 ; 3.4 % | 15/49 ; 3.4 % |

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

[mesuré] L'arbre conserve huit flux non uniques sur la face minimax : cette
limite n'est donc pas seulement un exemple analytique. Aucun des huit prorata
n'est faisable avec ces capacités, malgré une production disponible suffisante.

[mesuré] Statistiques regroupant tous les retraits de chaque règle, sans
moyenner les médianes par réseau. Les valeurs numériquement nulles sont
affichées comme zéro ; les valeurs brutes, y compris les minuscules fractions
négatives liées aux tolérances de service, restent dans les CSV.

| Règle | Avec vague | Délestage incrémental nul | Médiane | Q90 | Maximum | Pannes évitables / forcées |
|---|---:|---:|---:|---:|---:|---:|
| highs | 909/918 | 8/918 | 63.5 % | 74.0 % | 100.0 % | 20482 / 58156 |
| min_charge_max | 489/918 | 579/918 | 0.0 % | 63.5 % | 89.3 % | 2180 / 18543 |
| min_redispatch | 662/918 | 477/918 | 0.0 % | 67.3 % | 93.1 % | 3872 / 24794 |

[dérivé] **Sous highs, 20 482 / 78 638 = 26.05 % des pannes secondaires sont
classées évitables**, retraits imposés exclus. En incluant les 918 retraits
imposés au dénominateur, la part est 25.75 %. Les premières vagues sont
entièrement évitables dans 422/909 cascades avec vague, soit 46.42 %.
Les parts évitables parmi les pannes secondaires des deux autres règles sont
10.52 % (minimax) et 13.51 % (min_redispatch).

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

[mesuré] Temps muraux finaux : 338.73 s pour le run principal
(8 processus), 170.03 s pour les références (3 processus).
Les diagnostics intacts communs représentent 21.20 s et 5.60 s cumulés.

Les temps CSV excluent cette étape commune et l'attribution initiale. Ils
incluent cache et concurrence, dans l'ordre highs, minimax, min_redispatch,
prorata : ce n'est pas un benchmark isolé du solveur. La reprise d'un run
complet a été exécutée, sans recalcul ni écrasement.

[lu] Les métadonnées conservent le HEAD initial bebe96d et les empreintes des
sources nouvelles exécutées avant leur commit : ces empreintes identifient
les octets mesurés. Le HEAD initial ne contient pas l'implémentation nouvelle.
Les tableaux de cascade ont été mesurés sur la pile 3.11 seulement ; la suite
verte sous 3.13 ne prouve pas des trajectoires identiques entre ces piles.
Les capacités de référence peuvent elles-mêmes changer si le sommet primaire
change entre plateformes.

## Fichiers livrés

Les trois modules N−1 existants sont modifiés ; les autres fichiers ci-dessous
sont nouveaux. Aucun fichier historique interdit, test existant, README ou
document existant n'a été modifié. Les sorties sont entièrement synthétiques.

| Fichier | Rôle |
|---|---|
| `cascade_entropy/n1.py` | Options de départage et production antérieure par îlot. |
| `cascade_entropy/cascade_n1.py` | Transmission de la règle et production d'une vague à la suivante. |
| `cascade_entropy/attribution_n1.py` | Options intactes cohérentes, configuration et rejeu. |
| `cascade_entropy/_optimisation_dispatch_n1.py` | Formulations secondaires et résolutions contrôlées. |
| `cascade_entropy/diagnostics_dispatch_n1.py` | Plages, classement, attribution des vagues et prorata. |
| `scripts/diagnostic_departage_n1.py` | Banc technique, métadonnées, sorties exclusives, reprise et processus. |
| `tests/test_departage_diagnostic_n1.py` | Oracles analytiques, physiques, non-régression et cas numériques. |
| `tests/test_script_diagnostic_departage_n1.py` | Sorties, reprise et refus d'écrasement. |
| `tests/fixtures/departage_bebe96d.json` | Sources témoins au commit de départ. |
| `tests/fixtures/departage_precision.json` | PL reproduisant le défaut de précision du bilan. |
| `tests/fixtures/departage_cout_echelle.json` | PL reproduisant le statut Unknown de min_redispatch. |
| `docs/departage_n1.md` | Définitions, mesures, limites et décisions à soumettre. |
| `docs/departage_n1/comparatif.csv` | Les 32 lignes réseau/règle avec tous les indicateurs demandés. |
| `docs/departage_n1/retraits.csv` | Les 2 754 scénarios permettant de recalculer les statistiques. |
| `docs/departage_n1/metadata.json` | Paramètres, graines, versions et empreintes du run principal. |
| `docs/departage_n1/audit_comparatif.csv` | Comparatif des trois réseaux de référence de l'audit. |
| `docs/departage_n1/audit_retraits.csv` | Les 861 scénarios de contrôle des références. |
| `docs/departage_n1/audit_metadata.json` | Provenance du run de contrôle. |
| `docs/departage_n1/validation_references.json` | Confrontation des six cas aux PL de référence. |
| `docs/departage_n1/synthese.json` | Consolidation, dénominateurs et temps explicites. |

[dérivé] La consolidation vérifie l'exhaustivité des retraits, les partitions
évitable/forcée par vague, l'absence de retrait répété et l'égalité entre
sommes par retrait et totaux CSV. Les métadonnées copiées sont inchangées.
Les JSON détaillés par vague restent dans les runs locaux sous
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

[lu] Les erreurs de solveur sont testées par injection. Tous les modes
d'échec numérique, les échelles extrêmes et les performances sur de grands
réseaux ne sont pas couverts. HiGHS peut choisir d'autres sommets entre
versions et plateformes, y compris après les objectifs secondaires. Aucun
résultat Linux de cette branche n'est présumé à partir des essais Windows.

Décisions à discuter avec Patrick :

- La saturation au seuil 0.99 doit-elle provoquer des pannes secondaires dans
  ce modèle où le dispatch respecte déjà les capacités ?
- Quel objectif secondaire et quelle règle d'état intact sont physiquement
  défendables ? Quels délais, rampes et coûts faudrait-il ensuite représenter ?
- Les capacités doivent-elles rester communes entre règles ou être recalées
  selon un dispatch de référence choisi scientifiquement ?
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
