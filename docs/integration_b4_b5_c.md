# Intégration B4 → B5/B6 → C

Correction post-fusion des PR #13, #12 et #14, sur `origin/main` à
`3fc7289d65a5b07b929221c85006fab7bcd3116b`. Cadre : protocole discuté avec
Patrick Desrosiers le 28 octobre 2026, [issue #11](https://github.com/PhD-Brown/cascade-entropy/issues/11)
et ses [conventions A3](https://github.com/PhD-Brown/cascade-entropy/issues/11#issuecomment-6861560057).
Les nombres ci-dessous sont exclusivement synthétiques et provisoires.

Le raccordement permanent est `cascade_entropy.integration_n1`, un module
transversal extérieur aux fils A/B. Il importe les producteurs B4/B5 et les
types d'entrée de C ; aucun de ces modules ne l'importe. En particulier, C,
les entropies et les indicateurs historiques ne reçoivent aucun import du modèle
ou de la persistance. Les valeurs passent réellement par les fichiers B5/B6.
Aucun moteur, score scientifique, comportement par défaut ou dépendance n'est modifié.

**Défaut numérique corrigé.** Les générateurs acceptent `p=1` et `rayon=2`,
mais conservent respectivement `1.0` et `2.0`. La comparaison des JSON canoniques
les refusait à la sauvegarde ; la régénération pouvait aussi refuser des
paramètres topologiques équivalents de types différents.

L'égalité numérique est désormais admise uniquement pour `gilbert.p` et
`geometrique.rayon`, à valeur réelle finie et exactement représentable en float.
Les booléens, chaînes, valeurs hors domaine, NaN/infinis et entiers nécessitant
un arrondi sont refusés. Exemple : `2**53` peut être représenté exactement comme
rayon, `2**53+1` est refusé par la persistance plutôt qu'arrondi implicitement.
La probabilité immédiatement inférieure à 1 reste distincte de 1.

Cette règle ne réécrit aucun argument, champ ni ancien fichier ; le format v1
et sa canonicalisation restent inchangés. L'identité dépend toujours des octets
JSON canoniques de la définition topologique. Si cette définition contient
`p:1` au lieu de `p:1.0`, les IDs restent différents même si les contrôles de
concordance reconnaissent la même valeur. Une représentation changée dans un
manifeste ne passe jamais avec son ancien checksum. Dimensions, graines,
identifiants, dtypes, statistiques et positions gardent leurs exigences strictes.

**API de l'adaptateur.**

```python
from cascade_entropy.integration_n1 import sauver_etat_initial, charger_etat_initial
from cascade_entropy.centralites_lignes import calculer_scores

# etat est un EtatElectrique produit par attribuer(), avec ses entrées explicites.
# Les arguments de génération décrivent exactement sa topologie B4.
id_reseau = sauver_etat_initial(
    etat, nouveau_dossier, statut="validation",
    arguments_generation=arguments_generation,
    revision_code=sha_code_execute, arbre_modifie=arbre_effectivement_modifie,
    parametres_experience={"provisoire": True},
)
relu = charger_etat_initial(nouveau_dossier)
scores = calculer_scores(relu.etat_initial, **variantes_c_explicitement_choisies)
# Pour un pilote N−1 : relu.etat_electrique.reseau, .demande et .puissance_max.
# Les groupes ordonnés restent dans relu.etat_electrique.reseau.generateurs/charges.
# Pour les poids G–C, conserver ces ordres comme axes de la matrice de poids.
```

Tous les mots-clés B5/B6 restent obligatoires. L'écriture renvoie l'ID
**topologique**, comme `sauver_realisation()`. La lecture renvoie un
`EtatInitialIntegre` comportant `realisation` (topologie/manifeste),
`etat_electrique` (snapshot B4 reconstitué sans nouvelle optimisation) et
`etat_initial` (véritable `EtatInitialIntact`, copié/protégé pour C).

L'identifiant de C est `id_reseau + ":" + sha256_metadata` : deux variantes
électriques d'une topologie ne sont plus confondues. Cette identité dépend
aussi des métadonnées du manifeste ; ce n'est pas un identifiant minimal de
classe d'équivalence physique. Le contrat est vérifié à la lecture, mais aucun
checksum ne certifie l'authenticité d'une provenance arbitrairement réécrite.

**Contenu persistant.** Le dossier reste `topologie.json` + `metadata.json`,
écrits exclusivement par B5/B6, avec création exclusive et contrôles existants.
Dans `parametres_experience`, la clé `integration_b4_c` est réservée. Son format
est `cascade-entropy-integration-b4-c-v1`. Elle conserve :

- la configuration B4 complète, dont placement ordonné, référence nodale,
  demandes/Pmax/réserves, réactances, unités d'entrée, règles de capacités,
  paramètres/graines de chaque tirage et définition de topologie ;
- l'environnement B4 d'origine (Python, NumPy, SciPy), sans le remplacer par
  celui de la lecture ; les versions B5/B6 d'écriture restent dans sa provenance ;
- les avertissements B4 et, distinctement pour `final` et `reference`, les
  statuts d'îlots et le nombre de secours du solveur.

L'argument d'expérience libre permet de conserver les variantes C et les
éventuelles autres graines. L'adaptateur n'en choisit aucune et ne calcule pas
les scores à l'écriture. Une révision déclarée avec arbre modifié ne suffit pas
à rejouer du code local perdu : conserver le code exécuté avant tout usage scientifique.

| Champs électriques | Axes et unités | Association |
|---|---|---|
| `rang_generateur`, `rang_charge` | nœud, `1`, entiers | `-1` hors rôle ; sinon rang réel dans la liste B4, sans tri |
| `demande` | nœud, pu | Demande au nœud consommateur, zéro ailleurs |
| `puissance_max`, `reserves_minimales`, `reserves_disponibles` | nœud, pu | Valeur au producteur, zéro ailleurs |
| `reserves_satisfaites` | nœud, `1`, booléen | Diagnostic B4 au producteur, faux ailleurs |
| `final_production`, `reference_production` | nœud, pu | Production au producteur, zéro ailleurs |
| `final_charge_servie`, `final_delestage`, variantes `reference_` | nœud, pu | Valeur au consommateur, zéro ailleurs |
| `final_flux`, `reference_flux` | ligne, pu | Flux signé dans l'orientation A3 d'origine |
| `final_taux_de_charge`, `reference_taux_de_charge` | ligne, `1` | Taux de l'intact concerné |
| `final_hors_service`, `reference_hors_service` | ligne, `1`, booléen | Masque intact entièrement faux |
| `reactances`, `limites` | ligne, pu | Réactances et limites **finales figées** |
| `injections`, `angles` | nœud, pu / rad | État intact final, référence B4 conservée |

Les champs `reference_` n'existent que pour le mode `flux_reference`. Les
valeurs de référence ne remplacent jamais `final_flux` ou les limites finales
dans l'entrée C. MW → pu est réalisé exclusivement par B4 avant le passage à
l'adaptateur ; tous les champs de puissance persistés sont en pu sur la base
MVA déclarée. `champ_numerique()` et `tableau_champ()` conservent les dtypes et
valeurs, sans nouveau sérialiseur. Les rangs et la configuration empêchent de
perdre l'ordre des groupes lorsqu'ils sont non triés.

**Validation et refus.** L'écriture travaille sur une copie B4. Elle compare
les entrées matérialisées par `Valeurs.realiser()` à la configuration, revalide
les indices/rôles avec `vers_reseau()`, contrôle les limites finales et la
séparation des deux dispatchs. Elle réutilise `_controler()` de B4 pour les
bornes, bilans nodaux/par îlot et loi DC, sans nouveau dispatch ni nouveau
calcul scientifique. Ces helpers privés restent internes au même paquet et
leur raccordement est couvert par les tests transversaux.

La lecture réutilise les contrôles d'intégrité B5/B6 puis vérifie le contrat
transversal : format, topologie, base, unités, axes, dtypes, rangs, champs attendus,
valeurs hors rôle et diagnostics. Elle reconstitue les objets et applique les
mêmes contrôles B4 ; elle ne relance pas le solveur d'optimisation. Les erreurs
de contrat produisent `ValueError`/`TypeError`, les incohérences physiques
détectées par B4 `RuntimeError`, les erreurs E/S gardent leur type. Un dossier
existant reste refusé ; rien n'est effacé ni écrasé.

Les résultats post-cascade, masques actifs non intacts, configurations modifiées
sans correspondance avec les sorties, diagnostics invalides et unités inconnues
sont refusés. L'adaptateur ne peut pas prouver l'origine de nombres reconstruits
frauduleusement comme un objet B4 cohérent. Il ne traite pas les résultats
post-retrait et leurs NaN : ils demanderont le contrat de résultats de phase D.

Le domaine de C reste un graphe simple connexe. Les intacts B4 déconnectés
sont refusés explicitement par cet adaptateur sans changer B4. La persistance
reste limitée aux cinq générateurs reproductibles B5/B6. Un intact présentant
du délestage est conservé avec ses avertissements ; son admission scientifique
n'est pas décidée. Une demande totale nulle reste un cas technique signalé,
avec fraction délestée indéfinie. Les instantanés de C restent indépendants des
mutations ultérieures de l'état électrique ou des cascades.

**Tests et résultats locaux.**

Windows `10.0.26300`, Python 3.11.9, NumPy 2.3.4, SciPy 1.16.3, pytest 9.1.1.
Installation éditable dans le venv dédié, avec les dépendances déjà installées
(`--system-site-packages`), sans ajout de dépendance. Aucune exécution CI
distante n'est revendiquée ici.

| Commande (`python` du venv dédié) | Résultat |
|---|---|
| Avant correction, `python -m pytest tests/test_persistance_parametres_n1.py -q -k regression` | 2 failed, 24 deselected in 0.49s : défaut reproduit pour `p` et `rayon` |
| `python -m pytest tests/test_persistance_parametres_n1.py tests/test_persistance_n1.py tests/test_topologies_n1.py -q` | 129 passed in 1.34s |
| `python -m pytest tests/test_integration_n1.py -q` | 54 passed in 2.33s |
| `python -m pytest -q` | 858 passed in 140.84s (0:02:20), sans avertissement pytest |

Les 34 nouveaux tests numériques gardent des contrôles stricts sur les IDs,
graines, types et empreintes. Les 54 tests d'intégration sont indépendants des
tests existants. Les vingt combinaisons famille × capacités × unités emploient
six nœuds avec rôles non triés `[3,0]` et `[5,2,1]`, graines séparées 101/102/103,
et conservent toutes les associations après un vrai aller-retour disque.
Le rejeu B4 et la régénération topologique sont explicitement vérifiés.

La suite complète comporte les 770 tests de référence et 88 nouveaux tests.
Les moteurs N−1, les modèles Carreras, les définitions des sept scores et les
dépendances sont identiques à la base. Parsing AST des quatre fichiers Python
modifiés/ajoutés et contrôles Git des espaces réussis ; les tests existants
restent inchangés. Les validations ci-dessus sont locales.

L'oracle analytique principal est la chaîne `0–1–2`, demandes 2 pu au nœud 2
et 1 pu au nœud 1, producteur de Pmax 4 pu, `x=[0.2,0.3]`, limites 10 pu,
base 100 MVA. Les sommes de sous-arbres donnent `F=[3,2]`, les chutes `xF`
`theta=[0,-0.6,-1.2]`. Les sept scores sont vérifiés par des formules de chaîne,
dont son vecteur propre pondéré. Les retraits indépendants donnent 3 et 2 pu
délestés, fractions 1 et 2/3 ; les bilans sont accumulés par extrémités sans
réutiliser la matrice d'incidence du modèle.

Un triangle avec redispatch vérifie la propagation. Un cas dérivé de flux
impose `F_reference=[1,1]`, puis `F_final=[2,2]`, avec délestage initial 1 pu :
C reçoit `[2,2]` et les limites finales ; après retrait, la gravité garde le
dénominateur demandé `D0=3` pu. Les tests de corruption recalculent le checksum
pour atteindre les contrôles sémantiques, plutôt que tester seulement le hash.
Un processus frais bloque les imports des modèles dans les modules d'analyse.

Un nouveau test a initialement utilisé le nom de dossier Windows réservé `nul` :
53 passed / 1 failed sur ce problème de fixture E/S. Il a été renommé
`demande_zero` sans changer les assertions. La suite complète lancée trop tôt
a été interrompue, sans résultat utilisable ; elle a été relancée après les
54 succès. Les journaux restent locaux sous `data/validation_integration/`.
Les tests existants n'ont pas été édités.

**Limites scientifiques et décision.** La chaîne est préparée pour un pilote
technique synthétique, sans campagne finale, Spearman, classement de centralités,
sélection a posteriori ou changement Carreras/N−1. HiGHS peut choisir un autre
optimum selon plateforme/version ; le rejeu exact testé concerne le même
environnement. La lecture garde donc les sorties intactes enregistrées.
La concordance des tests ne démontre pas la validité physique du modèle ni le
pouvoir prédictif des indicateurs.

Restent à valider avec Patrick : paramètres/distributions électriques et base
numérique, réserves, règle de capacités, admission des intacts délestés,
départage final, seuil/probabilité de propagation, variantes et normalisations
des scores, densités/effectifs et protocole statistique. Aucun choix de campagne
n'est adopté par cet adaptateur.

**Proposition de progression pour #11, non publiée.**

> Après les fusions #13 → #12 → #14, une PR de correction propose l'équivalence
> numérique ciblée de `p`/`rayon` et le raccordement transversal permanent
> B4 → B5/B6 → C. Les rôles ordonnés, unités, graines, configurations et deux
> dispatchs distincts sont persistés ; C utilise l'intact final relu sur disque.
> Les tests permanents couvrent les cinq familles, les deux capacités, MW/pu,
> conservation, délestage initial, intégrité et non-fuite post-cascade.
> Validation locale : 129 tests numériques/persistance/topologies, 54 tests
> d'intégration et 858 tests complets réussis, sans avertissement pytest.
> Les paramètres restent synthétiques et provisoires ; aucun modèle historique,
> campagne finale ou comparaison des performances n'est modifié/lancé.
> Résultats et limites : voir le tableau de validation de cette PR.

Proposition d'entrée au journal, non inscrite : « Intégration transversale
B4/B5–B6/C par snapshot intact persistant ; équivalence numérique limitée aux
paramètres réels de génération, identité et types stricts conservés. Validation
synthétique permanente, paramètres de campagne et politique d'admission ouverts. »
