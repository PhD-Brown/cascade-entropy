# Journal de développement

Décisions techniques, résultats de validation et points à vérifier. Complète le
journal de bord du cours, qui porte sur le raisonnement scientifique plutôt que
sur l'implémentation.

---

## 2026-09-15

### Structure du dépôt

Paquet installable `cascade_entropy`, installé en mode éditable avec
`pip install -e ".[dev]"`. Ce choix évite toute manipulation de `sys.path` dans
les tests et les scripts : les imports fonctionnent depuis n'importe quel
répertoire.

Séparation en deux fils, conformément au plan de travail. Les quatre modules du
fil A (`reseau`, `dispatch`, `cascade`, `evolution`) sont pour l'instant des
squelettes portant leur docstring et leur critère de validation. Aucun module du
fil B n'importe un module du fil A.

`data/` et `figures/` sont hors du versionnement : les séries et les figures se
régénèrent par les scripts. `donnees_externes/` est exclu par précaution, pour le
cas où des données d'Hydro-Québec deviendraient disponibles.

### `indicateurs.hurst` — exposant de Hurst par méthode R/S

Découpage en segments disjoints, statistique R/S moyennée par échelle, pente de
log(R/S) contre log(n). Échelles espacées logarithmiquement entre 8 points et un
quart de la longueur de la série.

Validation sur 30 réalisations de 8192 points :

| série | H |
|---|---|
| bruit non corrélé | 0,544 ± 0,018 |
| bruit en 1/f | 0,899 ± 0,023 |
| marche aléatoire | 0,993 ± 0,010 |

**Point à vérifier avant toute interprétation.** L'écart à 0,5 sur le bruit non
corrélé n'est pas une erreur d'implémentation : c'est le biais connu de la méthode
R/S sur des séries finies. Or Carreras rapporte H = 0,55 ± 0,02 aux échelles
courtes sur ses séries de blackouts, qu'il interprète comme une faible
persistance. Les deux valeurs se recouvrent.

Cela ne signifie pas que son résultat est faux — ses séries sont beaucoup plus
longues et son biais n'est pas nécessairement le mien. Mais avant de conclure à
une persistance sur mes propres séries, il faudra soit appliquer la correction
d'Anis-Lloyd, soit construire une distribution nulle par simulation : générer un
millier de bruits non corrélés de même longueur, observer la distribution des H
obtenus, et ne déclarer une persistance que si la valeur mesurée en sort.

C'est la logique du témoin appliquée au témoin lui-même. À développer pour le
rapport d'étape.

### `entropie` — trois mesures informationnelles

`permutation` : motifs ordinaux encodés en entiers, comptage vectorisé sans
dictionnaire. Normalisée par log(m!).

`echantillon` : SampEn, distance de Tchebychev. Les comptes pour m et m+1 sont
calculés ensemble, puisque la distance en m+1 est le maximum de la distance en m
et de l'écart sur le point supplémentaire. Calcul par blocs de 512 lignes : une
matrice de distances complète pour 16384 points dépasserait 2 Go.

`multiechelle` : SampEn de la série granularisée à chaque échelle.

**Détail d'implémentation qui compte.** La tolérance r est calculée une seule
fois, sur l'écart-type de la série d'origine, et reste fixe pour toutes les
échelles. La granularisation réduit la variance ; recalculer la tolérance à chaque
échelle la ferait suivre et effacerait exactement l'effet que la mesure cherche à
révéler. C'est l'erreur classique des implémentations maison de MSE.

Validation, entropie de permutation normalisée :

| série | H |
|---|---|
| sinusoïde | 0,49 |
| bruit non corrélé | 0,9998 |
| bruit en 1/f | 0,993 |
| marche aléatoire | 0,943 |

### Résultat 1 — croisement multiéchelle

Script `scripts/01_croisement_multiechelle.py`, figure
`figures/01_croisement_multiechelle.png`. Cinq réalisations de 16384 points,
vingt échelles, m = 2, r = 0,15.

| échelle | bruit non corrélé | bruit en 1/f |
|---|---|---|
| 1 | 2,472 | 1,883 |
| 20 | 1,017 | 1,803 |

Croisement à l'échelle 4.

Le bruit non corrélé part de la valeur la plus élevée puis s'effondre ; le bruit
corrélé à longue portée conserve son entropie sur toutes les échelles. C'est le
résultat de Costa, reproduit indépendamment.

**Ce que ça justifie dans le plan.** Le signal le plus irrégulier de tous est
aussi celui dont la complexité chute le plus vite avec l'échelle. Irrégularité et
complexité sont donc deux propriétés distinctes, et c'est la raison pour laquelle
SO3 prévoit du multiéchelle plutôt que de l'entropie brute. Ce point n'a plus
besoin d'être justifié par citation seule.

Reproductibilité vérifiée : valeurs identiques à trois décimales entre une
machine Linux et une machine Windows, graines fixées.

### État de la suite de tests

13 tests, environ 19 secondes. Les deux modules du fil B sont couverts. Les
critères de validation du fil A sont écrits dans les docstrings mais pas encore
testés, faute d'implémentation.

### Prochaines étapes

`reseau.py` est le prochain module à écrire : il est sans risque quelle que soit
la décision sur la portée du projet, puisque toute modélisation de réseau en aura
besoin. `dispatch.py` attend cette décision, étant spécifique au modèle quasi
statique.

## 2026-09-15 (suite) — contrôles d'entrée et expérience 02

### Validation des entrées

Nouveau module `_validation.py`, appliqué à toutes les mesures du fil B. Les
séries non réelles, non finies, vides ou multidimensionnelles sont rejetées, de
même que les paramètres de dimension, délai, tolérance et granularisation
invalides. Avant, une série contenant un NaN produisait silencieusement un
résultat. Les séries issues de données réelles en contiendront.

Comportement conservé : SampEn retourne NaN lorsqu'aucun appariement n'est
trouvé. Ce NaN ne doit jamais être lu comme une entropie nulle.

Note sur le comptage des motifs ordinaux : le passage d'un codage entier à
`np.unique(axis=0)` est neutre. La justification inscrite dans le code — un
risque de dépassement d'entier — est fausse : les poids valent d^k, donc le code
maximal vaut 26 pour d = 3, et il faudrait d de l'ordre de 20 pour approcher la
limite d'un entier 64 bits, ce qui représenterait 20! motifs. `np.unique(axis=0)`
trie lexicographiquement et est donc un peu plus lent. Sans conséquence aux
tailles de série actuelles.

### Résultat 2 — sensibilité à l'ordre temporel

Script `scripts/02_melange_temporel.py`. Une réalisation de bruit 1/f est
comparée à vingt permutations de ses propres valeurs. La permutation conserve
exactement le multiensemble des valeurs — histogramme, extrema, moments,
écart-type — et ne modifie que l'ordre temporel.

| échelle | série originale | mélanges |
|---|---|---|
| 1 | 1,941 | 2,475 ± 0,010 |
| 10 | 1,899 | 1,341 ± 0,024 |

MSE répond donc bien à l'ordre temporel et non à la distribution des valeurs.
C'est le contrôle nul approprié pour la mesure, et c'est la même logique que le
témoin M_max appliquée cette fois à l'outil plutôt qu'au système.

Portée à ne pas dépasser : contrôle descriptif sur une seule réalisation
originale, bande à ± un écart-type entre permutations et non intervalle de
confiance. N'établit aucune supériorité sur le Hurst, ni aucune capacité de
prédiction électrique.

### Point de vigilance — égalités exactes et séries riches en zéros

Les mesures du fil B départagent les égalités par ordre temporel, via un tri
stable. Sur une série comportant beaucoup de valeurs identiques, cette convention
domine le résultat. Mesures sur des séries de 8192 points à proportion de zéros
croissante, valeurs non nulles tirées d'une loi log-normale :

| zéros | PE | PE après ajout d'un bruit de 1e-9 | SampEn |
|---|---|---|---|
| 0 % | 0,9998 | 0,9998 | 1,466 |
| 80 % | 0,578 | 0,9999 | 0,405 |
| 90 % | 0,378 | 0,9999 | 0,214 |
| 99 % | 0,068 | 0,9999 | 0,022 |

Même série, même information : un bruit de 1e-9, sans aucune signification
physique, fait passer PE de 0,068 à 0,9999. Le contrôle sur les valeurs non
nulles seules donne 0,999, confirmant que les zéros portent tout l'effet.

**Conséquence directe pour le projet.** La série de délestage quotidien produite
par la dynamique lente sera exactement de cette forme : la plupart des jours ne
comportent aucun blackout, donc un zéro exact. Une entropie de permutation
calculée sur cette série mesurerait la fréquence des zéros plutôt que la
dynamique du réseau. Elle produirait une courbe nette en fonction de G,
entièrement explicable par un indicateur trivial — soit exactement le mode
d'échec que SO3 cherche à détecter, mais provenant de l'outil et non du système.

L'exposant de Hurst n'est pas affecté de la même façon : R/S repose sur des
sommes cumulées et un écart-type, que les égalités ne perturbent pas. C'est
pourquoi Carreras a pu analyser ces séries sans rencontrer ce problème.

**Décision.** Aucun bruit ajouté aux données : ce serait inventer de
l'information. L'analyse par PE et SampEn portera sur l'entropie de la
distribution des taux de charge des lignes, observable disponible à chaque pas de
temps en sortie du dispatch et dépourvue de zéros exacts — celle qui a été
ajoutée à SO2 du plan de travail. Le délestage et le nombre de lignes en panne
restent les observables du Hurst et des statistiques de taille de blackout.

Vérification systématique à faire avant toute interprétation d'une courbe
d'entropie : mesurer d'abord la fraction de valeurs identiques dans la série.

### Dette technique

`scripts/README.md` pose qu'un script ne contient pas de logique réutilisable.
`02_melange_temporel.py` contient `analyser()`, que `test_melange_temporel.py`
doit charger par `runpy.run_path`. À déplacer dans le paquet, en
`cascade_entropy/controles.py`, le script redevenant une simple interface en
ligne de commande.

### État de la suite de tests

60 tests, environ 21 secondes.