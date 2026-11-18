# Décisions scientifiques à discuter avec Patrick — Projet I (PHY-3202)

> **Document de préparation à la discussion — non approuvé.**  
> État au **18 novembre 2026 (Québec)**, après fusion des PR #13, #12, #14, #15 et **#16**.  
> **Objet :** réunir les questions dont les réponses modifieraient le protocole scientifique de la seconde partie du Projet I. **Aucune option listée ici n'est considérée comme adoptée du seul fait de son implémentation ou de sa présence dans les tests.**

**Question de recherche maintenue :** les cinq indicateurs de lignes permettent-ils d'identifier les déclencheurs des cascades les plus graves **au-delà** du flux initial absolu et du taux de charge initial ?

**Référence décisionnelle :** protocole discuté avec Patrick Desrosiers le **28 octobre 2026**. Ce document ne prétend pas retranscrire mot à mot cette réunion et ne modifie aucune décision déjà prise. Toute révision de son cadre doit être discutée avec Patrick.

## Comment utiliser cette page

- **À valider avec Patrick :** hypothèses physiques, définitions finales des indicateurs, règles d'inclusion et de comparaison statistique qui influencent l'interprétation des résultats.
- **À préparer techniquement sans attendre :** essais synthétiques, oracles numériques, traçabilité, mesures de temps d'exécution, diagnostics d'îlotage et d'échec, sans sélection de réglages selon les performances prédictives.
- **À décider/documenter par l'équipe dans le cadre approuvé :** organisation du code, identifiants de fichiers, choix de graines *après* définition du plan d'échantillonnage, présentation des journaux et autres détails d'ingénierie ne changeant pas les hypothèses.
- **État des décisions :** toutes les questions ci-dessous restent *ouvertes* sauf indication explicite d'une convention déjà fixée. Les valeurs numériques employées dans les tests ne sont pas des valeurs de campagne.

Les priorités **P0** sont à trancher **avant la campagne scientifique finale** ; les **P1** concernent les modalités, analyses secondaires ou extensions à confirmer selon le temps disponible. Le pilote **technique** peut être exécuté avec des paramètres explicitement provisoires, sans utiliser ses résultats pour optimiser les centralités ou les paramètres physiques.

## 1. Ce qui est déjà fixé — ne pas rouvrir implicitement

| Élément | Cadre existant |
| --- | --- |
| Objet d'étude | Quatre centralités de lignes + une redondance électrique locale, comparées à deux références électriques |
| Références | `abs(F_initial)` et `abs(F_initial) / C_ligne` calculés sur **le même intact final** que les cinq indicateurs |
| Réseaux principaux | Anneaux avec cordes, grilles, Gilbert et géométriques aléatoires ; arbre témoin ; IEEE 118 seulement comme extension éventuelle |
| Modèle | Flux **DC quasi statique**, retrait exact et indépendant de chaque ligne, redispatch par îlot, propagation secondaire |
| Convention A3 | Indices de lignes stables, orientation et unités explicites, puissance/capacité en pu sur base commune, réactances en pu, `F=0` exactement sur une ligne retirée |
| Gravité principale | Fraction **finale** de demande délestée, définie pour une demande totale strictement positive |
| Principe prédictif | Calculer les indicateurs **avant les retraits** ; ne jamais employer les conséquences d'une cascade comme entrée d'un score |
| Historique | Ne pas modifier les reproductions Carreras (SO1–SO3), ni les conventions A3, sans décision explicite |

**État logiciel au 18 novembre :** phases A et B intégrées ; les **sept scores de C sont implémentés et testés** ; la PR #15 a fourni l'adaptateur permanent B4 → B5/B6 → C. La **PR #16 est maintenant fusionnée** : diagnostics de non-unicité et de saturations, alternatives de départage optionnelles, contrat de persistance v2 pour l'intact minimax, choix explicite de l'intact pour `min_redispatch` et résultats de validation synthétiques. **963 tests réussis** dans l'exécution locale d'Alex après fusion ; Codex et Claude rapportent également 963 réussites sur leurs environnements Windows et Linux. Il s'agit d'une validation **technique**, non d'une approbation des variantes, des hypothèses physiques ou de la campagne. Les deux références électriques et les tests de petits graphes sont déjà cochés dans l'issue #11 ; les cinq cases de définition scientifique de C restent ouvertes.

## 2. Priorité P0 — clore scientifiquement la phase C

### C1 — Degré adapté aux lignes

Implémenté : pour une ligne `(i,j)` et les degrés nodaux `k_i`, `k_j` :

- `somme` : `k_i + k_j` ;
- `produit` : `k_i * k_j`.

**Questions à arbitrer :** quelle variante constitue **l'indicateur principal** ? L'autre doit-elle être exclue ou préspécifiée comme analyse de sensibilité ? Quelle justification physique/topologique donner à ce choix, sachant que les classements peuvent différer ?

**Décision :** _à consigner_.

### C2 — Intermédiarité géodésique des arêtes

Implémenté : fraction de chemins les plus courts passant par chaque arête, avec partage entre chemins de même longueur.

- Longueurs `unitaires` (nombre d'arêtes) **ou** `reactances` (somme des réactances `x`, et non des susceptances `1/x`).
- Agrégation `somme` **ou** `moyenne` sur toutes les paires non ordonnées.
- Tolérance numérique relative d'égalité des distances configurable (`egalite_rtol`).

**Questions :** quelle notion de « chemin court » répond à notre hypothèse ? Quelle normalisation publier, notamment pour comparer des réseaux de tailles différentes ? La seconde définition est-elle une sensibilité annoncée ou hors protocole ? Faut-il fixer une politique explicite pour les chemins presque ex æquo ?

**Décision :** _à consigner_.

### C3 — Vecteur propre pondéré

Cadre déjà retenu dans #11 : matrice d'adjacence pondérée par `b_ij=1/x_ij`, vecteur propre principal `c`, puis score de ligne `V_ij = b_ij * c_i * c_j`.

Implémenté : normalisation du vecteur `l2`, `l1` ou `maximum`.

**Questions :** quelle normalisation devient la convention de publication et de comparaison entre réalisations ? Comment présenter correctement ce score comme **contribution spectrale locale**, sans présumer qu'il mesure directement la gravité d'une cascade DC ? Faut-il annoncer une sensibilité à la normalisation si l'analyse porte aussi sur des amplitudes entre réseaux ?

**Remarque :** sur un même réseau, ces normalisations changent l'échelle des scores par un facteur positif commun, mais pas leur ordre ; l'interprétation entre réseaux exige néanmoins une convention.

**Décision :** _à consigner_.

### C4 — Intermédiarité de courant

Implémenté : courant absolu transporté par une ligne lorsqu'on injecte **séparément** un transfert unitaire entre une paire source–puits, puis agrégation pondérée des paires.

- Paires `uniformes` : toutes les paires de nœuds non ordonnées.
- Paires `generateur_charge` : couples entre producteurs et consommateurs, avec poids explicites `q_GC` suivant **l'ordre conservé des rôles**.
- Normalisation `somme` **ou** `moyenne` pondérée.

**Questions :** veut-on mesurer la centralité électrique **globale** (toutes paires) ou une centralité liée aux rôles **production–consommation** ? Si G–C, d'où proviennent les poids (uniformes, demandes, autre hypothèse justifiée) et comment évitera-t-on un choix orienté par les résultats ? Quelle agrégation et quelle normalisation utiliser ? Les variantes non retenues deviennent-elles des sensibilités préannoncées ?

**Attention :** cette construction ne doit pas être présentée comme reproduisant automatiquement une formule publiée d'« electrical/extended betweenness » si ses pondérations et conventions sont différentes.

**Décision :** _à consigner_.

### C5 — Résistance effective et redondance locale

Implémenté sur le laplacien pondéré par les susceptances :

- `R_e` : résistance effective entre les extrémités (diagnostic mathématique, **pas** une résistance ohmique de la ligne de transport) ;
- `K_e = b_e * R_e` : convention `criticite`, proche de **1** pour un pont ;
- `E_e = 1 - K_e` : convention `redondance`, proche de **0** pour un pont.

**Questions :** faut-il publier `K_e` (importance/criticité structurelle) ou `E_e` (redondance) comme **cinquième indicateur** ? Dans quel sens doit-on ordonner les lignes pour anticiper les cascades graves ? Comment distinguer « beaucoup de chemins alternatifs » et « conséquence électrique grave » sans les déclarer équivalents ?

**Remarque :** `K` et `E` sont des transformations inverses de **la même** quantité ; ce ne sont pas deux prédicteurs indépendants.

**Décision :** _à consigner_.

### C6 — Convention commune d'interprétation des scores

**Questions transversales :** fixe-t-on à l'avance, pour chaque score, si une **valeur élevée** est censée désigner une ligne plus dommageable ? Comment traiter les scores inversés, les ex æquo et les cas constants, sans choisir après coup le sens qui maximise Spearman ? Quelles variantes constituent les **cinq indicateurs principaux**, et lesquelles, le cas échéant, sont uniquement des sensibilités ?

**Livrable attendu pour clore C scientifiquement :** une **configuration de sept scores** explicitement identifiée, documentée, reproductible et approuvée, sans modifier leurs définitions en fonction des cascades observées.

## 3. Priorité P0 — hypothèses physiques et échantillonnage (phase B)

### B1 — Taille, connectivité et échantillons topologiques

**Questions :** confirmer l'ordre de grandeur d'**environ 50 nœuds**, les paramètres des quatre familles (nombre de cordes, dimensions des grilles, probabilité Gilbert, rayon géométrique), leurs densités et la comparabilité recherchée. Comment traiter le conditionnement à la connexité, les ponts et la variabilité entre réalisations ? Quel nombre de réalisations par famille est compatible avec l'échéance et une interprétation raisonnable ? **« Environ cinq » a été envisagé, pas justifié comme puissance statistique.** Quel rôle donner à l'arbre témoin ?

**Décision :** _à consigner_.

### B2 — Producteurs, consommateurs, demandes et génération

**Questions :** combien et où placer les générateurs et charges (explicite ou aléatoire) ? Quelles distributions et échelles pour les demandes et `Pmax` ? Quelle proportion ou marge de capacité de production viser ? Faut-il des réserves minimales et, si oui, sont-elles **diagnostiques** comme dans B4 ou doivent-elles devenir de vraies contraintes d'optimisation (ce qui nécessiterait une décision/modification séparée) ?

**Décision :** _à consigner_.

### B3 — Réactances et base électrique

**Questions :** quelle valeur de `base_mva` et quelle provenance des réactances en pu ? Réactances constantes, explicites ou distribuées, avec quelles bornes/justifications ? L'homogénéité des réactances doit-elle être un témoin ou un scénario principal ? Les paramètres sont-ils comparables d'une famille à l'autre ?

**Attention :** les exemples en **100 MVA** ne fixent **pas** la base scientifique. B4 convertit MW → pu mais ne devine pas une conversion des ohms ni un changement de base des réactances.

**Décision :** _à consigner_.

### B4 — Capacités des lignes : décision particulièrement structurante

Deux voies sont implémentées, **sans choix approuvé** :

1. `independantes` : capacités explicites, constantes ou tirées indépendamment ;
2. `flux_reference` : limites provisoires → dispatch de référence → `C_e = max(marge * abs(F_reference,e), plancher)` → limites figées → **nouveau dispatch intact final**.

**Questions :** quelle voie utiliser ? Quelles marges, planchers, valeurs ou distributions et quelle justification physique ? Le dispatch de référence est-il adapté au phénomène étudié ? Quelle répartition des taux de charge initiaux accepte-t-on **sans sélectionner des réglages d'après les performances prédictives** ?

**Point critique de comparabilité :** avec capacités uniformes, flux et taux de charge conduisent au même classement ; si `C_e` suit proportionnellement `abs(F_e)`, le taux de charge peut devenir quasi constant. Il faut **documenter** ces dégénérescences éventuelles et les implications pour la question de recherche, pas ajuster les capacités afin de favoriser artificiellement un indicateur.

**Découverte du diagnostic de départage (PR #16) :** sur le banc synthétique à capacités `flux_reference` communes, ces limites ont été taillées depuis un flux de référence `highs` ; sur les sept réseaux maillés, l'intact `min_charge_max` retrouve ce flux à quelques `10^-9` pu près. **La référence de dimensionnement conserve ainsi l'empreinte d'un optimum potentiellement arbitraire.** Sur l'arbre témoin, l'écart atteint environ **5 pu** pour une ligne alors que le taux de charge maximal reste proche de **2/3** : minimiser un maximum n'impose pas nécessairement tous les flux. Avec B4 en mode `flux_reference`, la règle sélectionnée est utilisée **tant pour le dispatch de référence que pour le dispatch final** ; un B4 minimax construit donc ses capacités depuis son propre flux minimax. Ce mécanisme doit être discuté indépendamment de toute « amélioration » apparente des cascades.

**Questions supplémentaires à Patrick (P0) :** faut-il construire les capacités **indépendamment** d'un état de flux optimisé ou à partir de `flux_reference` ? Dans le second cas, quelle règle de dispatch pour la **référence de dimensionnement**, et comment éviter une circularité entre construction des limites, dispatch et taux de charge observés ? Quelle vérification de sensibilité serait définie **avant** de regarder les performances des indicateurs ?

**Décision :** _à consigner_.

### B5 — Admission des états initiaux dégradés

B4 conserve et signale le délestage intact, les réserves insuffisantes et la demande totale nulle ; l'adaptateur ne prend **aucune décision d'exclusion scientifique**.

**Questions :** accepter ou exclure les états déjà délestés ? À partir de quels critères **prédéfinis** ? Rapporter séparément délestage initial, final et incrémental ? Que faire des cas `D0=0`, pour lesquels la fraction de demande délestée est indéfinie, et des réalisations techniquement valides mais peu pertinentes physiquement ?

**Décision :** _à consigner_.

## 4. Priorité P0 — cascade N−1, départage et résultats (phase D)

### D1 — Propagation secondaire, seuil et probabilité de panne

Le moteur technique impose la ligne initiale et examine les candidates avec `rho >= 0.99`, chacune tombant avec la probabilité explicite `p1`. Dans le diagnostic synthétique **PR #16**, `p1=1` ; il s'agit d'un **réglage de test**, non d'un choix scientifique approuvé.

**Observations sur huit réseaux synthétiques (918 retraits par règle, capacités communes fixées depuis `highs`) :** nombre de retraits entraînant une vague secondaire : **909** avec `highs`, **489** avec `min_charge_max`, **662** avec `min_redispatch` initialisé sur l'intact minimax. Il ne faut pas en déduire que le minimax est physiquement préférable ni attribuer automatiquement une part de délestage final aux saturations évitables. Les secondes défaillances dépendent de toute l'histoire des redispatchs et de l'îlotage.

**Questions à arbitrer :** que doit représenter une ligne à 99 % ou 100 % de sa capacité alors que le dispatch respecte déjà des contraintes de transport ? Le seuil `0.99` et `p1` sont-ils justifiables physiquement ? Si `0 < p1 < 1`, quel nombre de réalisations stochastiques par retrait et quelles conventions d'agrégation, de graines et d'incertitude utiliser ? Ne pas choisir `p1` dans le seul but de rendre la distribution de gravité plus continue.

**Décision :** _à consigner_.

### D2 — Départage de l'optimum DC et état intact utilisé

Le service maximal constitue l'objectif primaire. Lorsque plusieurs dispatchs satisfont ce même optimum, le solveur peut sélectionner différentes productions et saturations. La PR #16 propose, **sans les adopter**, trois comportements : `highs` (défaut historique préservé), `min_charge_max` (minimiser le taux de charge maximal à service maximal, à la tolérance déclarée près) et `min_redispatch` (minimiser l'écart de production avec l'état précédent).

**Interface maintenant disponible :** la persistance B4/B5/C accepte les intacts `highs` (contrat v1 inchangé) et `min_charge_max` (contrat v2, avec empreintes de sources du dispatch). `min_redispatch` est une politique de **cascade**, dont le départ est obligatoirement défini par `departage_intact="highs"` ou `"min_charge_max"`. Les empreintes du contrat v2 documentent les sources et leur forme ; elles ne garantissent pas à elles seules que l'environnement de relecture exécute exactement le même code.

**Observations :** au seuil numérique testé, les sept réseaux maillés n'ont pas de flux déclarés non uniques sur la face minimax ; l'arbre témoin conserve **huit flux** déclarés non uniques. Cela ne démontre **pas** une unicité mathématique universelle, ni une identité bit à bit des flux entre plateformes.

**Questions à arbitrer :** quel départage de l'intact et quelle politique de redispatch après retrait représentent les hypothèses physiques de l'étude ? Que signifie « minimiser le redispatch » sans rampes, coûts et délais d'exploitation ? Comment traiter un optimum secondaire lui-même non unique ? Quelle règle appliquer lors de la construction du dispatch de référence des **capacités** (voir B4) ? Faut-il imposer un départage tertiaire et, si oui, à partir de quelle justification, **pas** en fonction des classements obtenus ?

**Décision :** _à consigner_.

### D3 — Gravité, délestage préexistant, îlots et échecs

**Déjà fixé dans l'issue #11 :** gravité principale `S_e = (D0 - D_servie_finale,e) / D0` pour `D0 > 0`, avec observables complémentaires de pertes de lignes et d'îlotage. **Le diagnostic PR #16 utilise pour sa comparaison technique une fraction de délestage *incrémental*.** Ce n'est **pas** une modification automatique de la définition principale. Il faut expliciter le traitement des états initialement délestés pour relier les deux mesures.

Dans le banc de diagnostic, les retraits avec fraction incrémentale considérée nulle (tolérance `|f| <= 1e-9`) sont **8/918** sous `highs`, **579/918** sous minimax et **478/918** sous `min_redispatch`. Quelques fractions brutes sont très légèrement négatives à cause des tolérances du solveur : elles sont **conservées** et comptées, non transformées silencieusement en zéro. Les égalités nombreuses réduisent l'information d'un classement de Spearman ; cela n'autorise pas à modifier la gravité pour rendre l'analyse plus commode.

**Questions :** quelles règles préspécifiées pour accepter/exclure les intacts délestés, pour `D0=0` et pour les cas non résolus ? Quels diagnostics d'îlotage, de saturation, de délai et de solveur doivent figurer dans les tables de résultat ? Comment présenter séparément la gravité principale finale et les métriques incrémentales complémentaires, sans convertir un échec numérique en gravité nulle ?

**Décision :** _à consigner_.

### D4 — Reproductibilité numérique et seuils discontinus

Le run synthétique `departage-20261118-v3` a comparé **2 754 retraits** à l'archive antérieure : **25** lignes diffèrent sur au moins un compteur ou une fraction, dont **cinq fractions** de plus de `10^-12`. Le plus grand écart de fraction est d'environ **0,2179**. Les 25 écarts sont archivés ; les rejeux ciblés du code initial reproduisent les nouvelles valeurs et le rejeu Linux indépendant communiqué à Codex confirme les cinq fractions v3 considérées à environ `10^-10` près. L'origine des valeurs de l'ancienne archive n'a **pas** été reconstituée. Certaines trajectoires individuelles restent sensibles aux conditions numériques ; les agrégats publiés demeurent proches à leur précision d'affichage.

**Hypothèse non démontrée :** le seuil `0.99`, `p1=1` et la dégénérescence des programmes linéaires peuvent amplifier de faibles différences numériques en changements discontinus des vagues. Ne pas présenter cette explication comme une cause prouvée pour chaque retrait.

**Questions :** quel niveau de reproductibilité exiger pour les trajectoires individuelles, pour les agrégats par réseau et pour l'interprétation physique ? Quelles tolérances et conventions de classement/ex æquo doivent être enregistrées à l'avance ? Quels écarts justifient une investigation numérique, la qualification d'un cas comme indéterminé ou une analyse de sensibilité préspécifiée ?

**Décision :** _à consigner_.

## 5. Priorité P0 — comparaison statistique (phase E, avant l'examen des performances)

**Cadre déjà prévu :** Spearman et identification des retraits les plus dommageables ; comparaison par réseau puis entre réalisations/familles, **sans traiter les lignes d'un même réseau comme des observations indépendantes**.

**Questions à trancher avant la campagne :**

1. **Sens et unité d'analyse :** comment calculer et résumer les corrélations par réalisation, famille et au total ? Quels scores sont orientés vers la gravité croissante ?
2. **Événements les plus graves :** seuil de gravité, fraction `top-k` ou autre règle de sélection, **fixée avant** de voir les succès des indicateurs.
3. **Ex æquo et constantes :** que rapporter lorsque tous les délestages sont nuls/identiques, que les scores sont constants ou que des lignes ont le même rang ? Une corrélation non définie ne doit pas être remplacée silencieusement par zéro. Le diagnostic PR #16 calcule les Spearman **entre politiques de départage** après passage de `|f| <= 10^-9` à zéro puis arrondi à neuf décimales, en conservant aussi la version brute : ce traitement n'est **pas** automatiquement la règle statistique finale. Sur les sept réseaux maillés du banc, les Spearman traités `highs`–minimax vont d'environ **0,096 à 0,393** ; ce ne sont **pas** des corrélations centralité–gravité.
4. **Répétitions stochastiques et incertitude :** niveau d'agrégation, mesure de variabilité entre réseaux et nombre minimal d'échantillons interprétables.
5. **Comparaisons multiples :** quels résultats seront principaux, quels résultats exploratoires, et comment éviter la sélection a posteriori du meilleur indicateur ?
6. **Diagnostics complémentaires :** dans quelle mesure le nombre de lignes perdues, la présence de ponts et le type de cascade seront-ils discutés sans remplacer le critère principal ?

**Décision :** _à consigner_.

## 6. Priorité P1 — arbitrages de portée et restitution

- Faut-il retenir **quelques sensibilités explicitement préspécifiées** (capacité, densité, variante de score, `p1`) si le cœur du protocole et l'échéance le permettent ?
- Quels exemples de cascades et quelles figures sont les plus convaincants pour expliquer le mécanisme physique, sans sélectionner uniquement les cas qui favorisent une hypothèse ?
- L'**IEEE 118** reste une extension éventuelle, **pas** une exigence ni une raison de retarder le cœur du travail.
- Jusqu'où discuter la portée du modèle DC quasi statique, qui ne représente pas les transitoires électromécaniques, et quelles limites de généralisation mettre en avant ?

**Décision :** _à consigner si utile_.

## 7. Ce que nous pouvons faire sans ces décisions

- Exécuter un **pilote technique synthétique** avec des paramètres et variantes explicitement étiquetés « validation technique — provisoire ».
- Contrôler bilans nodaux/par îlot, conversions MW/pu, intégrité des sauvegardes, graines, ID stables, absence de fuite post-cascade, non-mutation et conformité aux conventions A3.
- Mesurer temps de génération, attribution, calcul des sept scores, simulations N−1, écriture/lecture, et documenter les échecs numériques.
- Vérifier séparément délestage initial, final et incrémental ; vérifier les cas de ponts, îlots, demande nulle et optima non uniques.
- Préparer le **format de résultats de la phase D**, ses statuts d'échec et sa traçabilité, sans lancer la campagne finale ni choisir le protocole statistique à partir des observations.
- Préparer pour Patrick des **options argumentées** et une estimation de coût informatique de chaque option, sans transformer ces propositions en décisions approuvées.

**Ne pas faire avant accord :** lancer la campagne scientifique finale ; choisir les variantes ou paramètres selon Spearman, les pires retraits ou toute performance de prédiction déjà observée ; retoucher les reproductions Carreras ; présenter les résultats synthétiques du pilote comme des conclusions physiques.

## 8. Registre des décisions — à compléter **après** discussion

| ID | Décision retenue / justification | Date et accord de Patrick | Conséquence sur code/protocole |
| --- | --- | --- | --- |
| C1–C6 | _En attente_ | _En attente_ | Configuration des cinq indicateurs et conventions de classement |
| B1–B5 | _En attente_ | _En attente_ | Topologies, état physique initial et règles d'inclusion |
| D1–D4 | _En attente_ | _En attente_ | Propagation, départage, observables, reproductibilité |
| E | _En attente_ | _En attente_ | Analyse statistique préspécifiée |
| P1 | _En attente, facultatif_ | _En attente_ | Sensibilités et extensions éventuelles |

**Après validation :** consigner la décision datée et sa justification, figer une configuration de campagne versionnée, vérifier les tests correspondants, puis noter l'avancement en **commentaire** de l'[issue #11](https://github.com/PhD-Brown/cascade-entropy/issues/11). Ne pas modifier silencieusement le corps de cette feuille de route.

## Références internes

- [Feuille de route officielle #11](https://github.com/PhD-Brown/cascade-entropy/issues/11) et [conventions A3](https://github.com/PhD-Brown/cascade-entropy/issues/11#issuecomment-6861560057)
- [Définitions, équations et tests de la phase C](phase_c_centralites.md)
- [Attribution électrique configurable B4](attribution_electrique_b4.md)
- [Persistance des réalisations B5/B6](persistance_n1.md)
- [Adaptateur transversal B4 → B5/B6 → C](integration_b4_b5_c.md)
- [Historique d'intégration de la PR #15](https://github.com/PhD-Brown/cascade-entropy/pull/15)
- [PR #16 — diagnostic du départage N−1, fusionnée](https://github.com/PhD-Brown/cascade-entropy/pull/16)
- [Rapport du départage : méthodes, tests, audit v3 et écarts numériques](departage_n1.md)
- [Archive des 25 écarts de reproductibilité](departage_n1/ecarts_archive_20261117.csv)
- [Tableau de vérification des sources #10](https://github.com/PhD-Brown/cascade-entropy/issues/10)

**Note de confidentialité :** ce document public ne contient aucun jeu de données, banc d'essai ou matériel à diffusion restreinte.
