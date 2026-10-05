# SO2 : dynamique lente de Carreras et al. (2004)

*Séries de délestage et de lignes tombées, balayage en `G`, exposant de Hurst `H(G)`, cycles déterministes.*

Ce document est le détail scientifique de SO2. Le [README racine](../README.md) n'en garde que les résultats principaux.

**État au 5 octobre 2026 (soir)** : le balayage `so2-G-scan` est terminé et analysé en entier avec `scripts/08_analyse_so2.py` (21 cas × 100 000 jours analysés, 50 mélanges par série), plus deux contrôles sur `μ`. SO2 est **partiellement reproduit**, avec des écarts documentés.

- **Reproduits** : les deux régimes selon `G`, `H(G)` du délestage (indépendant de la taille), `H` des lignes à faible `G`, le second pic de la Fig. 8.
- **Non reproduits** : Fig. 7 (plateau pour `G ≥ 1`), Fig. 9 (10 à 35 fois trop de grands blackouts), `H` long des lignes à `G` élevé (il s'atténue quand la taille augmente), Table I (à refaire sur la fréquence cumulée).
- **Interprétation** : le `H` court publié (0.55) est surtout le biais de R/S. Deux cycles déterministes sont prédits analytiquement et vérifiés sur les 21 cas.

Aucun de ces résultats n'établit une criticalité auto-organisée. Le fil B (entropies, SO3) n'a pas encore été appliqué à ces séries.

**Navigation** : [Cibles de validation](#cibles-de-validation) · [Constats du balayage](#constats-du-balayage) · [Hypothèses de réplication](#hypothèses-de-réplication) · [Avancement](#avancement-et-points-ouverts) · [Provenance](#provenance)

## Comment lire les statuts

- **✓** : accord avec la valeur publiée (ou tendance publiée reproduite).
- **~** : accord partiel, ou valeur reproduite mais expliquée par un artefact de méthode.
- **✗** : comportement publié non reproduit.
- **⏳** : comparaison à refaire.

Distinction maintenue partout : **mesuré** (simulation), **dérivé analytiquement** (formule fermée vérifiée sur les 21 cas) et **hypothèse de réplication** (convention que le papier ne fixe pas).

## Cibles de validation

| Observable (Carreras 2004) | Référence publiée | Modèle (`so2-G-scan`) | Statut |
|---|---|---|---|
| Transitoire (Fig. 2) | ≈ 20 000 jours, légère hausse des blackouts | MSER 0 à 6 900 j sur les 21 cas, aucune alerte ; aucun renforcement de ligne avant ≈ 3 000–8 000 j | ✓ cohérent |
| Deux régimes selon `G` (Sec. IV) | faible `G` : blackouts fréquents, peu de lignes ; `G` élevé : rares, en cascade | 94 nœuds : 122 → 3.7 blackouts / 300 j et 0.25 → 8.1 lignes par blackout de `G` = 0.1 à 2 ; même tendance à 46 et 190 | ✓ reproduit |
| `H` aux échelles courtes (Fig. 3) | 0.55 ± 0.02, indépendant de `G` | 21 cas : délestage 0.52–0.56, lignes 0.54–0.58. Mélanges : 0.52–0.56. Délestage à `G ≤ 0.5` : identique au mélange (z < 2) sur les 3 tailles. Ailleurs : +0.01 à +0.02 (z = 2 à 9) | ~ valeur reproduite ; ≈ 95 % de la valeur est le biais de R/S |
| `H` long du délestage, 600 < t < 10⁵ (Fig. 4a) | ≈ 0.5 si `G < 1`, diminue si `G > 1` | 21 cas : 0.48–0.52 pour `G ≤ 0.25` (= mélange) ; 0.37–0.43 à `G = 0.5` ; 0.24–0.29 pour `G ≥ 0.75` (z ≈ −9 à −12). Même courbe pour 46, 94 et 190 nœuds | ✓ qualitatif, indépendant de la taille (baisse dès `G ≈ 0.5`) |
| `H` long des lignes tombées (Fig. 4b) | 0.2–0.4, antipersistant, selon le réseau | Antipersistant (0.18–0.28) jusqu'à `G = 0.25` (46), 0.5 (94), 0.75 (190). Au-delà, montée vers 0.9 (46), 0.82–0.90 (94), 0.36 / 0.51 / 0.60 (190, `G` = 1 / 1.5 / 2). Le seuil de la montée recule et son amplitude baisse quand la taille augmente | ✓ à faible `G` ; ✗ à `G` élevé, écart qui s'atténue avec la taille |
| Puissance servie selon `G` (Fig. 7, 94 nœuds) | faible aux deux extrémités de `G` | 0.938 → 0.998, puis plateau dès `G ≈ 0.75` (aucune baisse à `G` élevé) ; idem à 46 et 190 | ✗ non reproduit |
| Lignes par blackout selon `G` (Fig. 7) | croissant avec `G` | 0.25 / 0.40 / 1.1 / 3.4 / 6.6 / 7.9 / 8.1 | ✓ |
| Distribution des lignes par blackout (Fig. 8, 94 nœuds) | pic à 4 à faible `G` ; second pic ≈ 17 à `G` élevé | 94 nœuds : pics à 8, 16–17 et 24–28 lignes ; le pic 16–17 croît avec `G` (0.4 % → 6 %) ✓. 190 nœuds : pics à 8, 16, 32, 40, 48. Faible `G` : 91–95 % des blackouts sans ligne, puis décroissance monotone dès 1 ligne (pas de pic à 4) | ~ second pic ✓ ; premier pic ✗ |
| Blackouts > 15 lignes / blackouts (Fig. 9, 94 nœuds) | ≈ 0.001 à faible `G`, 0.007 si `G > 1` | 94 : 0.007 / 0.012 / 0.033 / 0.10 / 0.20 / 0.24 / 0.25 ; 190 : 0.013 → 0.23 ; 46 : 0 → 0.06 | ✗ ≈ 30 fois plus haut à `G` élevé (voir l'incohérence possible plus bas) |
| Queue du délestage normalisé (Table I) | indice −0.56 / −0.51 / −0.55 / −0.58 ; étendue 4 / 8 / 13 / 31 | Densité presque plate jusqu'à ≈ 0.15 à faible `G`, bimodale à `G ≥ 1` (petits délestages et cascades à 0.2–0.7). Plages trouvées : 46 et 94 nœuds, étendue ≤ 9 ; 190 nœuds, `G ≥ 1` : pentes −1.9 / −1.6 / −2.6, étendue 8 à 11 | ⏳ à refaire : le texte de Carreras 2004 (Sec. III) définit la Table I sur la fréquence cumulée relative (fonction de rang), pas sur la densité malgré la légende. Pente cumulée −0.55 ↔ pente de densité ≈ −1.55 ; or 190 nœuds, `G` = 1.5 donne −1.56 en densité. Ajustement sur la fréquence cumulée à ajouter à `08` |

`H` : R/S, plages 10–365 j et 600–25 000 j (le quart de la série). Référence : 50 copies mélangées par série, intervalle à 95 % (`08_analyse_so2.py`). Les séries font 120 000 jours, dont 20 000 de transitoire écartés.

## Constats du balayage

### `H` court ≈ biais de R/S

Les pentes R/S sur 10–365 jours valent 0.53–0.56. Sur 50 copies mélangées de chaque série (mélange = mémoire détruite, distribution conservée), elles valent 0.52–0.56. C'est le biais connu de R/S aux petites échelles (Anis et Lloyd, 1976). La référence par mélange est très étroite (± 0.005), ce qui rend visible un petit excès réel : nul pour le délestage à `G ≤ 0.5`, +0.01 pour les lignes, +0.01 à +0.02 à `G ≥ 0.75`.

La « faible persistance » de 0.55 de Carreras est donc à ≈ 95 % le biais de la méthode ; la mémoire réelle aux échelles courtes est de l'ordre de 0.01–0.02. C'est un résultat **mesuré** : il justifie que toute mesure de mémoire de SO3 soit comparée à une référence par mélange.

### Cycle déterministe de 1000 jours (lignes)

Le spectre des lignes renforcées et des lignes tombées a un pic à 1000 jours. La prédiction **dérivée analytiquement** : une ligne renforcée d'un facteur `μ` redevient saturée quand la demande a crû du même facteur.

$$
T_{\text{lignes}} = \frac{\ln \mu}{\ln \lambda_{\text{jour}}}, \qquad \lambda_{\text{jour}} = 1.018^{1/365}
\quad\Rightarrow\quad T = \frac{\ln 1.05}{\ln(1.018)/365} = 998\ \text{jours}
$$

Dans un arbre symétrique, les lignes qui tombent ensemble sont renforcées ensemble et restent en phase. Signature dans R/S (lignes, 46 nœuds) : pente 0.73–0.79 entre 365 et 1000 jours, puis 0.22–0.52 au-delà. L'ajustement publié 600 < t < 10⁵ chevauche cette cassure.

**Hypothèse** (non démontrée) : l'antipersistance des lignes publiée par Carreras (0.2–0.4) serait la même signature, puisque `μ ∈ [1.01, 1.1]` donne une période de 200 à 1950 jours, dans leur fenêtre.

### Contrôle `μ` : prédiction confirmée ✓

Runs `so2-controle-mu102` et `so2-controle-mu110` (94 nœuds, `G` = 0.5 et 1).

| `μ` | Période prédite `ln μ / ln λ_jour` | Période mesurée |
|---|---|---|
| 1.02 | 405.2 j | 405 j et 405 j |
| 1.05 | 998.2 j | 1000 j |
| 1.10 | 1950.0 j | 1961 j à `G` = 1 |

À `μ = 1.10` et `G = 0.5`, le pic le plus haut est l'harmonique `P/2` (971 j, 143 fois la médiane du spectre) ; le fondamental est présent (62 fois). Le cycle suit `μ` sans aucun paramètre ajusté : son mécanisme est établi.

Effet secondaire : `μ` règle la taille des cascades. À `G` = 1, lignes par blackout 11.9 / 6.6 / 4.9 et part > 15 lignes 0.37 / 0.20 / 0.14 pour `μ` = 1.02 / 1.05 / 1.10. Même à `μ` = 1.10, on reste 20 fois au-dessus des 0.007 de la Fig. 9. `H` long des lignes : ≈ 0.2 à `G` = 0.5 et ≈ 0.75 à `G` = 1 pour les trois `μ` ; la modulation lente à `G ≥ 1` ne dépend donc pas de `μ`.

### Cycle de génération

La marge de génération reste collée au seuil `G · g`, en dents de scie d'amplitude `k / N_G` = 0.17 %. L'intervalle entre deux mises à niveau est **dérivé analytiquement** :

$$
\Delta t_{\text{gén}} = \frac{k / N_G}{(1 + G\,g)\, \ln \lambda_{\text{jour}}}
$$

Prédit 31.3 → 12.2 jours de `G` = 0.1 à 2, mesuré 31 → 12 jours, écart ≤ 0.5 jour sur les 21 cas.

### Modulation lente des lignes à `G ≥ 1`

À 94 nœuds, `G` = 1.5, le nombre de lignes tombées par bloc de 5 000 jours varie de 3 250 à 6 170 (± 25 %), contre ± 3 % à `G` = 0.5. Cette modulation de 7 000 à 25 000 jours rend R/S persistant au-delà du cycle (`H` ≈ 0.7–0.9). À 46 nœuds, elle domine même le spectre : le pic mesuré passe de 1 000 j à 20 000 j (`G` = 1) et 8 333 j (`G` = 2).

Elle s'atténue avec la taille (`H` long des lignes à `G` = 2 : 0.92 / 0.82 / 0.60 pour 46 / 94 / 190), ce qui suggère une moyenne sur davantage de groupes de lignes indépendants. Elle ne dépend pas de `μ` (contrôle). **Origine à trouver** ; avec 100 000 jours, on n'en voit que 4 à 14 périodes.

### Plateau pour `G ≥ 1`

Dès que la marge `G · g` dépasse la fluctuation maximale `g`, la génération ne contraint plus jamais le dispatch. Augmenter `G` n'ajoute alors que de la capacité inutilisée, et tout devient indépendant de `G` (fréquence, puissance servie). C'est **dérivé analytiquement** de l'Éq. (5) : la frontière à `G ≈ 1` n'est pas un résultat émergent.

Carreras montre au contraire une baisse de la puissance servie à `G` élevé (Fig. 7). Leur modèle contient donc un couplage entre marge de génération et lignes que le nôtre n'a pas, ou qui n'est pas décrit. **À instruire avant toute conclusion.**

### Incohérence possible dans Carreras 2004

Le texte de la Fig. 8 dit qu'à `G` maximal le pic à ≈ 17 lignes est « comparable » au pic à faible nombre de lignes. La Fig. 9 donne pourtant un rapport (> 15 lignes) / (tous les blackouts) de 0.007 seulement. Les deux sont difficiles à concilier, sauf si la Fig. 8 est en échelle logarithmique. Le titre de la Table I parle de PDF, mais la Fig. 5 trace une fréquence cumulée : quantité à préciser avant de comparer les pentes.

### Plateforme

Les trajectoires Linux et Windows d'un même cas divergent après ≈ 5 000 jours : une ligne à 0.98999… d'un côté et 0.99000… de l'autre franchit ou non le seuil de saturation, puis les histoires se séparent. Les statistiques coïncident (délestage moyen 0.06056 contre 0.06061 ; lignes par jour 0.0502 contre 0.0498). **Seuls les résultats statistiques sont donc comparables d'une machine à l'autre, pas les trajectoires.**

### Solveur

4 jours sur 2.52 millions de jours simulés ont utilisé une stratégie de secours de HiGHS (46 nœuds, `G` = 1 : 1 jour ; `G` = 1.5 : 3 jours). Effet négligeable.

## Hypothèses de réplication

| Hypothèse | Valeur retenue | Source | Statut |
|---|---|---|---|
| Croissance de la demande | 1.8 %/an (`λ ≈ 1.00005` / jour) | publié | ✓ |
| Seuil de marge `(ΔP/P)_c` | `G · g` (Éq. 5, `γ̃ = g`) | publié ; `γ̃` = écart relatif maximal de la demande = `g`, déduit | ✓ |
| Amplitude des fluctuations `g` | 0.9 (`γ = 1.9`, comme SO1) | non publié dans Carreras 2004 | ~ option `--g` |
| Type de fluctuation | régionale, `N_F = 3` | annexe : « per load or regional groups » | ~ option `--fluctuation` |
| Renforcement `μ` | 1.05 | publié dans [1.01, 1.1] ; fixe la période du cycle des lignes (vérifié à 1.02, 1.05, 1.10) et la taille des cascades | ~ option `--mu` |
| Ajout de génération `k` | 0.02 | publié : « a few percent » | ~ |
| `p1` | 1 | publié dans [0.1, 1] | ~ |
| Lignes renforcées | lignes tombées par surcharge un jour de blackout | publié : lignes surchargées ; identique si `p1 = 1` | ✓ |
| Départage du délestage | `exterieur_dabord` | validé en SO1 (Fig. 9–10) ; diagnostics déterministes identiques sur toute plateforme (les longues trajectoires, elles, divergent : voir Plateforme) | ✓ |
| Demande initiale | 0.7 × `P_C` (Table I) | sans effet sur le régime stationnaire | ✓ |
| Cycle de la génération | `Δt = (k / N_G) / ((1 + G·g) ln λ_jour)` | dérivé ; mesuré 31 / 28 / 24 / 20 / 18 / 15 / 12 j contre 31.3 / 27.8 / 23.5 / 20.4 / 17.9 / 14.5 / 12.2 prédits (94 nœuds, `G` = 0.1 à 2) | ✓ |
| Robustesse du solveur | stratégies de secours (mise à l'échelle, simplexe dual, point intérieur) | technique ; appel historique inchangé bit pour bit | ✓ |

Relation avec la notation de Carreras 2002 : `g = γ − 1` (`g = 0.9` correspond à `γ = 1.9`). Paramètres du balayage `so2-G-scan` : fluctuations régionales, `N_F = 3`, `g = 0.9`, `μ = 1.05`, `k = 0.02`, `p1 = 1`, transitoire de 20 000 jours, 120 000 jours par cas, `G` ∈ {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}, tailles 46 / 94 / 190.

## Avancement et points ouverts

- [x] Observables, fluctuations régionales, `marge_pour_G`, reprise exacte (`EtatEvolution`)
- [x] `07_dynamique_lente.py` : checkpoints, reprise bit pour bit, MSER
- [x] Correctif solveur (07 v1.1.0) : secours HiGHS, cas en échec isolé, programme fautif sauvegardé
- [x] Balayage `so2-G-scan` : 21 cas × 120 000 jours (terminé le 5 octobre)
- [x] Contrôle du cycle : `μ` = 1.02 et 1.10, périodes mesurées 405 et 1961 j contre 405 et 1950 j prédites
- [x] `cascade_entropy/analyse_series.py` et `scripts/08_analyse_so2.py`
- [x] Correctif `pdf_logarithmique` : la plus grande valeur pouvait sortir des classes (`10**log10(x)` < `x` d'un ulp), détecté par le test Pareto sous Windows
- [x] `08_analyse_so2.py` sur les 21 cas : `H(G)` avec intervalles pour les 3 tailles
- [ ] Tag Git `so2-v1`
- [ ] Instruire l'écart de la Fig. 7 (plateau pour `G ≥ 1`) et la modulation lente (question ouverte pour le superviseur)
- [ ] Refaire la Table I sur la fréquence cumulée relative
- [ ] Appliquer entropie de permutation, SampEn et MSE aux séries de `so2-G-scan` avec les mêmes références par mélange (SO3)

## Provenance

| Résultat | Script | Run |
|---|---|---|
| Essai 46 nœuds, `G` = 1, 32 000 j | [`scripts/07_dynamique_lente.py`](../scripts/README.md#07_dynamique_lentepy) | `runs/essai-so2` (Windows) |
| Essai 46 nœuds, `G` = 0.5, 60 000 j ; test de reprise | `scripts/07_dynamique_lente.py` | `t-cont`, `t-int` (Linux, identiques) |
| Balayage en `G` (21 cas) | `scripts/07_dynamique_lente.py` 1.0.0, repris en 1.1.0 | `runs/so2-G-scan` (Windows) |
| Cycle de 1000 j, `H` court du mélange, distribution des lignes | analyse ad hoc des séries 46 nœuds (`G` = 0.1, 0.25, 0.5) | `runs/so2-G-scan` |
| `H` avec mélanges, cycles, Fig. 4, 7, 8, 9, Table I (21 cas) | [`scripts/08_analyse_so2.py`](../scripts/README.md#08_analyse_so2py) 1.0.0 | `data/08_analyse_so2/so2-G-scan` (Windows) |
| Contrôle `μ` = 1.02 et 1.10 | `scripts/07_dynamique_lente.py` puis `scripts/08_analyse_so2.py` | `runs/so2-controle-mu102`, `runs/so2-controle-mu110` (Windows) |
| Divergence Linux / Windows | comparaison jour par jour, 46 nœuds `G` = 0.1 et 0.25 | `stress-46` (Linux) contre `so2-G-scan` (Windows) |

Tests : `tests/test_evolution_so2.py`, `tests/test_regime_stationnaire.py`, `tests/test_solveur_secours.py`, `tests/test_analyse_series.py`, `tests/test_analyse_so2_script.py`.

Précédent : [SO1, réplication de Carreras et al. (2002)](so1_carreras_2002.md). Retour au [README racine](../README.md).
