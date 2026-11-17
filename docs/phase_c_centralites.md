# Phase C — cinq indicateurs et deux références initiales

Statut au 17 novembre 2026 : implémentation et validation mathématique sur
graphes synthétiques publics. **Les variantes offertes ne sont pas des décisions
de protocole prises avec Patrick Desrosiers.** La phase E et les paramètres
quantitatifs B4–B6 restent hors de ce travail.

Cadre consulté avant modification : [issue #11](https://github.com/PhD-Brown/cascade-entropy/issues/11),
[conventions A3](https://github.com/PhD-Brown/cascade-entropy/issues/11#issuecomment-6861560057),
`CLAUDE.md`, `reseau.py`, `topologies_n1.py`, `n1.py`,
[archive #6](https://github.com/PhD-Brown/cascade-entropy/issues/6) et
[vérifications #10](https://github.com/PhD-Brown/cascade-entropy/issues/10).
Les résultats numériques et affirmations de performance archivés par Claude
n'ont pas été validés par ce travail.

## Interface et domaine

`cascade_entropy.centralites_lignes` reçoit des tableaux et n'importe ni
`reseau`, ni `dispatch`, ni les moteurs de cascade. Il n'intervient pas dans
les deux fils historiques d'analyse entropique et de simulation Carreras.
Il utilise seulement NumPy, la bibliothèque standard et `_validation.py`.
Aucune dépendance supplémentaire.

`GrapheLignes(n_noeuds, lignes, reactances)` copie un graphe simple, non orienté,
connexe, avec au moins deux nœuds. Boucles, parallèles, graphe vide et composantes
disjointes sont refusés explicitement. Ce domaine correspond aux générateurs A3.
Les nœuds sont `0…n−1`, les lignes `0…m−1`. L'orientation et l'ordre reçus sont
conservés exactement. Le constructeur ne trie et ne fusionne aucune ligne.
Pour une `Topologie` déjà triée ou un `Reseau`, transmettre directement les
tableaux dans leur ordre d'origine ; ne pas les reconstruire depuis un ensemble.

`EtatInitialIntact(graphe, flux, limites, hors_service, identifiant)` exige :

- un flux signé fini par ligne, dans l'orientation fournie ;
- une capacité finie strictement positive par ligne, dans la même unité ;
- un masque booléen explicite entièrement faux ;
- un identifiant d'état non vide, choisi par l'appelant.

Les tableaux d'entrée sont copiés et protégés contre les écritures ordinaires.
Le masque refuse un état déclaré post-retrait. **Il ne certifie pas la provenance
du flux** : le producteur de l'état doit garantir son origine intacte, la
convergence du dispatch et l'alignement avec ce graphe, et conserver injections,
paramètres et diagnostics. Ce module ne calcule pas le dispatch, n'impose aucun
critère d'admission de délestage initial et ne fixe pas de capacités.

Les fonctions de référence et d'agrégation exigent réellement le type
`EtatInitialIntact` à l'exécution : un objet de simulation portant un attribut
`flux` est refusé, même si ses dimensions sont compatibles.

`calculer_scores` renvoie les sept colonnes, `identifiants_lignes`, les extrémités,
l'identifiant d'état et un dictionnaire de paramètres sérialisable en JSON
(y compris rôles et poids G-C copiés). **Tous les choix scientifiques sont des
mots-clés obligatoires** ; seules les tolérances numériques ont un défaut.
Les fonctions unitaires sont également disponibles. Aucun rang n'est calculé ;
aucun résultat de cascade n'est accepté comme objet d'entrée ou consulté.

## Définitions et sens des valeurs

Notations : pour la ligne orientée arbitrairement e=(i,j),
\(\nu_e=e_i-e_j\), \(x_e>0\), \(b_e=1/x_e\).
L'incidence M a pour ligne \(\nu_e^T\). On utilise
\(L=M^T\operatorname{diag}(b)M\), et son pseudo-inverse de Moore–Penrose \(L^+\).
La « résistance effective » est celle du circuit auxiliaire de conductances b,
**pas** une résistance ohmique des lignes de transport. Avec les conventions A3,
x est en pu et flux/capacités en pu sur une base commune encore à fixer en B4.

### 1. Degré adapté aux lignes

\[
k_i=\sum_e\mathbb{1}_{i\in e},\qquad
D_e^{\mathrm{somme}}=k_i+k_j,\qquad
D_e^{\mathrm{produit}}=k_i k_j.
\]

`degre_lignes(variante='somme'|'produit')` : degrés non pondérés, scores bruts sans
unité ni normalisation. Grande valeur = extrémités davantage connectées.
La somme et le produit peuvent donner des classements différents ; aucune des
deux variantes n'est adoptée automatiquement. Aucun degré électrique, coreness
ou autre centralité n'est ajouté.

### 2. Intermédiarité géodésique

\[
B_e=\sum_{s<t}\frac{\sigma_{st}(e)}{\sigma_{st}},\qquad
\bar B_e=\frac{B_e}{\binom n2}.
\]

`intermediarite_geodesique(longueurs='unitaires'|'reactances',
normalisation='somme'|'moyenne')`. Chaque paire non ordonnée est comptée une fois.
Les longueurs sont 1 ou x, jamais b. Les chemins de même longueur sont partagés
uniformément **par chemin complet**, sans sélection aléatoire et sans partage
uniforme naïf entre les seuls prédécesseurs. Score sans unité ; grande valeur =
participation plus élevée aux chemins géodésiques.

Implémentation : Dijkstra, puis comptage et accumulation des dépendances sur le
graphe acyclique des distances strictement croissantes. Pour les réactances,
les longueurs sont divisées par max(x). L'égalité des distances utilise
`egalite_rtol=1e-12`, `atol=0` ; `egalite_rtol=0` demande l'égalité flottante
stricte. Avec une tolérance non nulle, des chemins numériquement presque égaux
peuvent donc être traités comme ex æquo. La valeur est enregistrée dans les
paramètres. Un incrément de distance perdu en flottants produit une erreur.

La définition fractionnaire, les poids interprétés comme distances et leur
positivité sont vérifiés dans la
[documentation officielle NetworkX, « edge_betweenness_centrality »](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.edge_betweenness_centrality.html),
rubriques définition et poids, consultées le 17 novembre 2026. NetworkX n'est
ni une dépendance ni un oracle des tests.

### 3. Vecteur propre pondéré

\[
A_{ij}=b_{ij}\ (i\ne j),\quad A_{ii}=0,\quad
Ac=\lambda_{\max}c,\quad c_i>0,\qquad V_e=b_e c_i c_j.
\]

`centralite_vecteur_propre(normalisation='l2'|'l1'|'maximum')` :
\(\sum c_i^2=1\), \(\sum c_i=1\) ou \(\max c_i=1\), respectivement.
Score en unités de b, sans normalisation supplémentaire. À graphe fixé,
ces trois choix multiplient tous les scores par une même constante positive.
La valeur absolue entre réseaux dépend de ce choix.

La diagonalisation symétrique sélectionne la plus grande valeur propre
algébrique, même sur un graphe biparti. Si les sommes des poids incidents sont
constantes, c est uniforme. Ainsi, avec l2, \(V_e=b_e/n\) ; avec l1,
\(V_e=b_e/n^2\) ; avec maximum, \(V_e=b_e\). La régularité du **degré seul** ne
suffit pas si les susceptances sont hétérogènes.

Pour une diminution relative infinitésimale \(b_e\mapsto(1-\epsilon)b_e\),
\[
\left.\frac{d\lambda_{\max}}{d\epsilon}\right|_0
=-\frac{2b_e c_i c_j}{c^T c}.
\]

C'est une dérivée locale, pas l'effet exact d'un retrait fini. Elle mesure une
contribution à l'adjacence, pas une conséquence du laplacien DC. Grande valeur =
contribution spectrale élevée ; son lien à la gravité reste à tester.
Source vérifiée : [Restrepo, Ott et Hunt, arXiv:cond-mat/0606122, PDF v1](https://arxiv.org/pdf/cond-mat/0606122),
page 2, équations (3)–(4). Le facteur 2 ci-dessus est la spécialisation dérivée
ici au retrait des **deux** entrées symétriques d'une arête non orientée.
La sensibilité est contrôlée par différences finies centrées dans les tests.

### 4. Intermédiarité de courant

\[
f_e^{st}=b_e\nu_e^TL^+(e_s-e_t),\qquad
I_e=\sum_{(s,t)\in\mathcal P}q_{st}|f_e^{st}|,\qquad
\bar I_e=\frac{I_e}{\sum q_{st}}.
\]

Chaque paire représente un **transfert unitaire séparé**. La valeur absolue est
prise avant la somme. Il ne s'agit pas du flux net sous transferts simultanés.
`normalisation='somme'|'moyenne'` sélectionne ces deux agrégations.

- `paires='uniformes'` : toutes les paires s<t, q=1. Rôles et poids supplémentaires
  sont refusés pour éviter une option ignorée silencieusement.
- `paires='generateur_charge'` : produit cartésien des listes explicites G et C,
  non vides, uniques et disjointes. `poids_paires` est une matrice obligatoire
  (|G|, |C|), finie, positive ou nulle, de total strictement positif. Une matrice
  de uns donne la version G-C uniforme. Des poids différents sont exploratoires
  et doivent être annoncés avec leur origine et leurs unités.

La moyenne est une fraction sans unité d'un transfert unitaire ; la somme
conserve l'unité éventuelle des poids. Grande valeur = participation électrique
élevée. Pas de pondération automatique par capacité ou demande, ni de limites
de transit dans ce calcul linéaire. Les variantes G-C ne sont pas attribuées aux
formules publiées sous « extended/electrical betweenness », dont les règles
distinctes de pondération/agrégation restent à contrôler dans #10.

La relation courant–potentiel–laplacien est vérifiée dans
[Brandes et Fleischer, STACS 2005, version auteurs](https://kim246.wwwdns.kim.uni-konstanz.de/publications/bf-cmbcf-05.pdf),
§2, pages imprimées 534–535, loi d'Ohm et équation (1). La moyenne par
\(\binom n2\) utilisée ici est annoncée explicitement, sans prétendre reproduire
toutes les normalisations publiées. Sur un arbre, le transfert vaut 1 sur
l'unique chemin et 0 ailleurs : I et B coïncident avec les mêmes paires et
normalisations. Cette égalité est testée, sans conclure sur le maillage général.

### 5. Redondance électrique locale

\[
R_e=\nu_e^TL^+\nu_e,
\qquad K_e=b_eR_e,\qquad E_e=1-K_e.
\]

`redondance_electrique(convention='criticite'|'redondance')` renvoie K ou E,
deux variantes du même indicateur local. `resistance_effective` expose R comme
diagnostic mathématique en unités de x, sans ajouter une sixième centralité.
K et E sont sans unité et dans [0,1] à l'arrondi près. Grand K signifie moins
de chemins alternatifs ; grand E signifie davantage de redondance. Leur lien
avec la gravité n'est pas supposé validé.

Un pont a R=x, K=1, E=0. Les valeurs restent finies ; aucun LODF ni division par
1−K n'est calculé. On ne classe pas les lignes presque ponts comme des ponts
exacts par une tolérance arbitraire. L'identification topologique exacte reste
disponible séparément dans `topologies_n1.ponts`.

Identités contrôlées : \(\sum_e K_e=n-1\), K=1 sur tout arbre, K=2/3 sur un
triangle uniforme, et K=(n−1)/n sur un cycle uniforme. On a aussi
\(\sum_eK_e=\operatorname{tr}(L^+L)=n-1\), puisque le graphe est connexe.
Les tests utilisent en plus une énumération indépendante des arbres couvrants
pondérés : K égale leur probabilité d'inclure l'arête.

Sources vérifiées le 17 novembre 2026 :

- [Spielman, cours 13, 10 octobre 2018](https://www.cs.yale.edu/homes/spielman/561/lect13-18.pdf),
  p. 13-1 : définition par transfert unitaire et pseudo-inverse ; §13.3 : séries
  et parallèles.
- [Spielman, cours 14, 15 octobre 2018](https://www.cs.yale.edu/homes/spielman/561/lect14-18.pdf),
  §14.6, p. 14-5 : poids bR, ponts et théorème 14.6.1 des arbres couvrants pondérés.
- [Spielman, cours 21, 12 novembre 2018](https://www.cs.yale.edu/homes/spielman/561/lect21-18.pdf),
  pp. 21-4–21-5 : somme des poids et preuve par la trace.

L'implémentation ne reprend pas les autres identités de #6 (LODF, « report
total », variation de résistance globale), et ne les marque pas vérifiées.

### 6 et 7. Références électriques

\[
Q_e=|F_e^{(0)}|,\qquad T_e=|F_e^{(0)}|/C_e.
\]

`flux_initial_absolu` et `taux_charge_initial` prennent l'objet intact explicite.
Q est dans l'unité du flux, T sans unité ; grandes valeurs = flux/charge relative
élevés, sans validation prédictive. Un flux nul donne exactement zéro, sans
plancher ajouté. Les capacités nulles ou non finies sont refusées.
Si C est uniforme, Q et T sont proportionnels ; si C=alpha*Q et Q>0, T=1/alpha.
Ces dégénérescences sont testées, pas contournées par une règle artificielle.

## Calcul numérique et invariances

Les opérateurs électriques et l'adjacence sont calculés après division des
susceptances par max(b) ; l'échelle est rétablie pour R et V. Un seul L+ est
utilisé dans `calculer_scores`. La diagonalisation du laplacien retire le seul
mode constant, puis inverse **tous** les autres modes. Si
\(\lambda_2\le10^{-12}\lambda_{\max}\), le calcul est refusé comme mal conditionné,
sans transformer numériquement le réseau en composantes disjointes. Une valeur
propre principale d'adjacence séparée d'au plus \(10^{-12}\lambda_{\max}\), ou
un vecteur de Perron numériquement non positif, produit aussi une erreur.

Seuls les arrondis hors des bornes [0,1] du score K sont rabattus sur ces bornes
(écart maximal admis 1e-9) ; les valeurs intérieures ne sont pas arrondies.
Les ex æquo analytiques peuvent donc différer de quelques ulp. Les tests de
symétrie utilisent des tolérances, sans fabriquer de rangs ni de départage.

Les scores sont invariants aux orientations et aux permutations des nœuds ; une
permutation des lignes permute les sorties de la même façon. Si x est multiplié
par a>0 : degré, géodésique, courant et K/E restent identiques, R est multiplié
par a et V est divisé par a. La géodésique pondérée conserve les chemins sous
ce changement d'échelle ; sa tolérance numérique reste relative.

La méthode est dense et exacte à l'arrondi près, destinée aux petits graphes
de la mission (~50 nœuds). Coût dominant de diagonalisation O(n³), mémoire du
calcul des courants toutes paires O(m n²). Aucun passage à l'échelle industrielle
ou validation sur données réelles n'est revendiqué.

## Exemple minimal reproductible

Ces nombres sont un triangle synthétique, avec un transfert intact de 1 pu de
0 vers 1, sans calcul de cascade ni choix de paramètres de campagne :

```python
import numpy as np
from cascade_entropy.centralites_lignes import (
    GrapheLignes, EtatInitialIntact, calculer_scores,
)

g = GrapheLignes(3, np.array([[0, 1], [1, 2], [0, 2]]), np.ones(3))
initial = EtatInitialIntact(
    g, flux=np.array([2/3, -1/3, 1/3]), limites=np.full(3, 2.0),
    hors_service=np.zeros(3, dtype=bool), identifiant="triangle-unitaire-intact",
)
resultat = calculer_scores(
    initial, degre="somme", longueurs="unitaires",
    normalisation_geodesique="moyenne", normalisation_vecteur="l2",
    paires_courant="uniformes", normalisation_courant="moyenne",
    convention_resistance="redondance",
)
# ID 0, 1, 2 ; colonnes respectivement :
# degré = 4 ; géodésique = 1/3 ; vecteur propre = 1/3 ;
# courant = 4/9 ; redondance = 1/3 ;
# |F0| = [2/3, 1/3, 1/3] ; |F0|/C = [1/3, 1/6, 1/6].
# Conserver resultat.parametres et les paramètres complets de l'état producteur.
```

## Validation locale

Les 97 tests `tests/test_centralites_lignes.py` utilisent chaîne, arbre branché,
triangle, cycle et maillage asymétrique pondéré avec feuille-pont. Les oracles
comprennent : coupes a(n−a) d'arbres ; résistances série/parallèle ; vecteur
trigonométrique de la chaîne ; chemins simples exhaustifs ; élimination de
Gauss en fractions exactes avec plusieurs références ; arbres couvrants
pondérés exhaustifs ; sensibilité spectrale par différences finies.
Ils vérifient aussi paires G-C, poids nuls, valeur absolue avant somme,
invariances, références électriques constantes/nulles, identifiants stables,
non-mutation, refus d'objets de simulation et d'entrées hors domaine. Un processus
frais bloque les imports des moteurs pour contrôler l'indépendance du module.
Aucun accès réseau dans les tests.

Validation finale locale, dans le worktree isolé
`C:\Users\alexb\Documents\Projets\cascade-entropy-phase-c`, branche
`codex/phase-c-centralites-isole`, fondé sur `origin/main` au SHA
`28eeb9bdfd7fe99bd5137780134e03bf1f68d42b` :

```text
python -m pytest tests/test_centralites_lignes.py -q
97 passed in 0.48s

python -m pytest -q
606 passed in 134.21s (0:02:14)
```

Python 3.11.9, Windows 10.0.26300, NumPy 2.3.4, SciPy 1.16.3, pytest 9.1.1.
Aucun avertissement dans ces deux sorties pytest. L'exemple documenté a aussi
été exécuté et ses sept colonnes vérifiées ; parsing AST et contrôle des espaces
Git réussis. Pas de validation CI distante annoncée.

Empreintes SHA-256 des sources finales testées (avant normalisation éventuelle
des fins de ligne par Git) :

```text
cascade_entropy/centralites_lignes.py
8556d6109e3eb0db082664c862170dc019a5621feff141a0a4319faac213cf85
tests/test_centralites_lignes.py
86c2c7108ca88722dd0437efb83e1df884f9142076ef639e95f134b2182f74c3
```

L'isolation a été nécessaire parce qu'un commit B4 et des fichiers B5/B6 sont
apparus dans le répertoire partagé pendant la mission. La PR ne reprend aucun
de ces changements et ne modifie aucun fichier existant. Ces tests ne valident
ni le code concurrent, ni une campagne réelle, ni le pouvoir prédictif des scores.

## Décisions scientifiques restant ouvertes

- Choisir somme ou produit du degré ; géodésiques unitaires ou par réactance
  (ou contrôle de sensibilité annoncé), et normalisation à publier.
- Fixer la normalisation du vecteur propre entre réseaux ; ne pas interpréter
  sa contribution spectrale comme une preuve de vulnérabilité DC.
- Fixer les paires de courant, la place de la variante G-C et l'origine de ses
  poids éventuels ; aucune formule bibliographique G-C n'est adoptée ici.
- Choisir K (criticité) ou E (redondance) comme représentation du cinquième
  indicateur et annoncer le sens du classement en phase E. Pas de nouveau score
  global, ni de mélange flux–résistance ajouté.
- Définir avant E le traitement statistique des ex æquo, constantes et signes
  attendus. Aucun Spearman, seuil top-k ou classement des cascades n'est calculé.
- B4–B6 conserve tous ses choix : injections/placement, capacités, réactances,
  graines, admission des états déjà délestés et paramètres de campagne.

## Entrée proposée pour le journal (non écrite dans `docs/journal.md`)

> 17 novembre 2026 — Phase C : module indépendant de scores de lignes, cinq
> indicateurs et deux références intactes, variantes mathématiques obligatoires
> et paramètres traçables. Validation analytique sur graphes synthétiques ;
> aucune modification Carreras/N−1/A3, aucune campagne ni comparaison phase E.
> Les choix détaillés de protocole restent à discuter avec Patrick. Voir la PR
> et `docs/phase_c_centralites.md` pour la preuve d'exécution des tests.
