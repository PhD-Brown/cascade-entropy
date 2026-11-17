# Commentaire proposé pour l'issue #11

Texte préparé pour relecture ; **non publié**. Le corps de la feuille de route
reste inchangé.

---

## Suivi d'avancement — Phase C (17 novembre 2026)

**Livrable proposé :** `cascade_entropy/centralites_lignes.py`, avec les cinq
indicateurs prévus et les deux références électriques initiales. Les définitions,
équations, unités, sens des valeurs, sources mathématiques vérifiées et limites
numériques sont dans `docs/phase_c_centralites.md`.

Les variantes restent explicites et configurables :

- degré : somme ou produit des degrés des extrémités ;
- géodésique : longueurs unitaires ou réactances, partage des chemins multiples,
  somme ou moyenne sur les paires non ordonnées ;
- vecteur propre : adjacence pondérée par b=1/x, score b*c_i*c_j, normalisation
  l2, l1 ou maximum ;
- courant : toutes paires uniformes ou générateur–charge à poids fournis,
  valeur absolue avant somme, somme ou moyenne pondérée ;
- résistance effective locale : bR pour la criticité ou 1−bR pour la redondance,
  sans divergence sur les ponts ;
- références : |F0| et |F0|/C à partir d'un objet d'état initial intact fourni
  explicitement, sans règle de capacité ajoutée.

La sortie conserve les identifiants des lignes d'origine et les paramètres des
variantes. Aucun résultat de cascade n'est utilisé dans le calcul des scores.
Le module n'importe aucun moteur de simulation.

**Validation locale :** 97 nouveaux tests sur chaîne, arbre, triangle, cycle et
maillage asymétrique pondéré. Oracles indépendants : chemins exhaustifs, calculs
de courants en fractions exactes, arbres couvrants pondérés, formules analytiques
et sensibilité spectrale. Contrôles des symétries, ex æquo, flux nuls, ponts,
invariances d'échelle/orientation, non-mutation, provenance déclarée et erreurs
sur les entrées hors domaine.

`python -m pytest tests/test_centralites_lignes.py -q` → **97 passed in 0.48s**.

`python -m pytest -q` → **606 passed in 134.21s (0:02:14)**,
sans avertissement, dans le worktree isolé `cascade-entropy-phase-c` fondé sur
`main` à `28eeb9b`, avec les seuls ajouts de phase C. Environnement : Python
3.11.9, NumPy 2.3.4, SciPy 1.16.3, pytest 9.1.1, Windows 10.0.26300.
Ce résultat local ne comprend pas les développements concurrents B4/B5/B6 ;
aucune validation CI distante n'est annoncée.

**Décisions encore ouvertes avec Patrick :** adaptation du degré, variantes et
normalisations à retenir pour la campagne, paires/poids du courant et sens du
cinquième indicateur. Les variantes de l'archive #6 ne sont pas présentées comme
adoptées ; les autres vérifications bibliographiques restent suivies dans #10.
Les paramètres électriques B4–B6, le traitement des états initiaux déjà délestés
et les conventions de comparaison de phase E ne sont pas fixés ici.

**Périmètre préservé :** aucun changement Carreras, moteurs N−1 ou conventions
A3 ; aucun fichier B4/B5/B6 modifié. Aucune campagne, corrélation de Spearman ou
recherche des pires cascades. La validation porte sur les définitions et leur
implémentation, pas sur leur pouvoir prédictif ni sur des données réelles.
