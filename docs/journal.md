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