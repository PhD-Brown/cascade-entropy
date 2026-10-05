# scripts

Chaque script produit une figure ou un résultat chiffré destiné au rapport. Un script ne contient pas de logique réutilisable : il orchestre des fonctions du paquet [`cascade_entropy`](../cascade_entropy/README.md), trace le résultat et l'enregistre.

## Conventions

- **Nommage** : `NN_sujet.py`, où `NN` est l'ordre chronologique de production.
- **Lancement** : depuis la racine du dépôt, après `python -m pip install -e ".[dev]"`. Exemple : `python scripts/06_validation_so1.py`.
- **Sorties** : les séries et les runs vont dans `data/` (non versionné). Les figures vont dans `figures/`, où seules les figures de référence en PNG sont versionnées (voir le [README racine](../README.md#ce-qui-est-versionné-et-ce-qui-ne-lest-pas)). Tout est régénérable en relançant le script.
- **Runs reproductibles** : les campagnes lourdes (`05_reproduction_carreras.py`, `07_dynamique_lente.py`) ont un `--run-id`, enregistrent leurs paramètres et leurs graines dans `metadata.json` et se reprennent avec `--resume`.
- **Windows** : fermer les figures ouvertes avant de relancer un script (un fichier ouvert est verrouillé).
- **Dépendance optionnelle** : `powerlaw` (`python -m pip install powerlaw`) pour les analyses de queue de `04_loi_puissance_taille_finie.py` et de `05_reproduction_carreras.py`.

## Vue d'ensemble

| Script | Fil | Rôle | Sorties principales | Statut |
|---|---|---|---|---|
| `01_croisement_multiechelle.py` | B | Entropie multiéchelle d'un bruit blanc et d'un bruit rose (croisement de Costa et al.) | `figures/01_croisement_multiechelle.png` | validation de l'outil sur signaux synthétiques |
| `02_melange_temporel.py` | B | Contrôle de l'ordre temporel : série 1/f et ses permutations | `resultats.npz`, `courbes.csv`, `parametres.json`, figure | contrôle descriptif, une seule réalisation |
| `03_balayage_charge.py` | A | Balayage du niveau de charge : les deux transitions du modèle | `figures/03_balayage_charge.png` | **historique, qualitatif** (voir plus bas) |
| `04_loi_puissance_taille_finie.py` | A | Queue des tailles de blackout, 46 contre 94 nœuds (scan puis production) | `data/04_loi_puissance_taille_finie/`, figures | **historique**, remplacé par `05_reproduction_carreras.py` |
| `05_diagnostics_carreras.py` | A | Figures déterministes 3 à 11 de Carreras et al. (2002) | `data/` et `figures/05_reproduction_carreras/deterministe*/` | SO1 validé |
| `05_reproduction_carreras.py` | A | Campagnes stochastiques de SO1 (`audit`, `scan`, `production`, `runs`) | `data/` et `figures/05_reproduction_carreras/runs/<RUN_ID>/` | SO1 |
| `06_validation_so1.py` | A | Audit chiffré de SO1 contre les valeurs publiées | `data/05_reproduction_carreras/validation_so1/` | SO1 validé |
| `07_dynamique_lente.py` | A | Dynamique lente de Carreras et al. (2004), cas taille × `G` | `data/` et `figures/07_dynamique_lente/runs/<run-id>/` | SO2 en cours |

### Attention : le défaut de `--departage` n'est pas le même partout

Le départage choisit, parmi les dispatchs de même coût, lequel est retenu quand il faut délester. La règle de référence du projet est `exterieur_dabord` ; `highs` est le choix du solveur (comportement historique).

| Script | Valeur par défaut de `--departage` |
|---|---|
| `05_diagnostics_carreras.py` | `highs` |
| `05_reproduction_carreras.py` | `highs` |
| `07_dynamique_lente.py` | `exterieur_dabord` (valeur affichée par le script quand l'option est omise) |

Pour retrouver les résultats validés de SO1 avec les scripts `05_*`, passer `--departage exterieur_dabord`.

---

## `01_croisement_multiechelle.py`

Compare l'entropie multiéchelle d'un bruit non corrélé et d'un bruit corrélé à longue portée. À l'échelle 1, le bruit non corrélé a l'entropie la plus élevée ; après granularisation, il la perd rapidement, tandis que le bruit corrélé la conserve. La figure justifie le recours au multiéchelle plutôt qu'à l'entropie brute.

**Options** : aucune. Les paramètres sont fixés en tête de fichier (`N = 16384`, 20 échelles, 5 réalisations, graine `20260915`).

```powershell
python scripts/01_croisement_multiechelle.py
```

## `02_melange_temporel.py`

Compare la courbe d'entropie multiéchelle d'une série 1/f à celles de ses permutations (mêmes valeurs, ordre détruit). Les bandes sont ± un écart-type entre mélanges, pas un intervalle de confiance. L'expérience porte sur une seule réalisation et ne conclut ni à une significativité ni à une vulnérabilité électrique. Les sources exécutées sont identifiées par leur empreinte SHA-256 dans `parametres.json`.

**Options** : voir `python scripts/02_melange_temporel.py --help`. Détails de l'expérience : [`docs/02_melange_temporel.md`](../docs/02_melange_temporel.md).

```powershell
python scripts/02_melange_temporel.py
```

## `05_diagnostics_carreras.py`

Banc de validation **déterministe** des Fig. 3 à 11 de Carreras et al. (2002). Le balayage principal n'a aucune fluctuation de demande, avec `p0 = 0` et `p1 = 1` pour les Fig. 3 à 10. Il est séparé de `05_reproduction_carreras.py` pour ne jamais toucher aux fichiers d'une production en cours. Son contenu pourra être fusionné dans ce dernier sous une sous-commande `deterministe`.

| Option | Rôle |
|---|---|
| `--taille {46,94,190,382}` | Arbre étudié (382 pour les figures du papier ; les autres tailles servent à l'audit) |
| `--ratio-min`, `--ratio-max`, `--pas` | Balayage principal de `P_D/P_C` |
| `--ratio-min-etendu`, `--ratio-max-etendu`, `--pas-etendu` | Balayage étendu (carte de la Fig. 10) |
| `--graine` | Graine aléatoire |
| `--departage {highs,exterieur_dabord}` | Règle de départage (défaut `highs`) |
| `--p1 P1 [P1 ...]` | Valeurs de `p1` pour la sensibilité analogue à la Fig. 11 |
| `--repetitions-p1` | Répétitions par ratio pour les `p1` intermédiaires |
| `--sans-p1` | Ne calcule pas la sensibilité à `p1` (Fig. 11) |

```powershell
# Règle de référence : sorties dans deterministe_exterieur_dabord/
python scripts/05_diagnostics_carreras.py --departage exterieur_dabord

# Règle historique (défaut) : sorties dans deterministe/
python scripts/05_diagnostics_carreras.py

# Balayage plus fin, ou autre taille (audit uniquement)
python scripts/05_diagnostics_carreras.py --pas 0.0025
python scripts/05_diagnostics_carreras.py --taille 190
```

Avec `--departage exterieur_dabord`, les sorties vont dans des dossiers suffixés `deterministe_exterieur_dabord/` sans jamais écraser celles de `highs` : on peut comparer les deux règles figure par figure. Le script vérifie alors aussi les frontières analytiques des bandes ordonnées de la Fig. 10.

**Sorties** (dossiers `data/05_reproduction_carreras/deterministe[_exterieur_dabord]/` et `figures/05_reproduction_carreras/deterministe[_exterieur_dabord]/`) :

- `scan_382.csv/.npz`, `scan_etendu_382.csv/.npz`, `sensibilite_p1_382.csv`, `metadata_deterministe_382.json` ;
- `05_carreras_fig03_...` à `05_carreras_fig11_...`, en `.png` et `.pdf`.

Pour les lignes tombées, `M_ij` n'est pas physiquement défini dans la représentation numérique. Le script garde donc deux matrices : `M_final_service` (`NaN` pour les lignes hors service) et `M_final_carreras` (`0`, uniquement pour une visualisation analogue aux figures du papier).

## `05_reproduction_carreras.py`

Campagnes stochastiques de SO1. Chaque exécution crée son propre dossier et ne peut écraser aucun autre run :

```text
data/05_reproduction_carreras/runs/<RUN_ID>/
figures/05_reproduction_carreras/runs/<RUN_ID>/
```

Sans `--run-id`, le `RUN_ID` est construit avec le type de run, les tailles, le nombre de réalisations, les ratios et un horodatage. Si le nom existe déjà, le script refuse d'écraser.

**Sous-commandes** : `audit`, `scan`, `production`, `runs` (liste les runs existants).

**Options communes** à `audit`, `scan` et `production` :

| Option | Rôle |
|---|---|
| `--tailles T [T ...]` | Tailles d'arbres |
| `--gamma` | Paramètre `γ` des fluctuations régionales (facteurs uniformes sur `[2-γ, γ]`) |
| `--p0` | Probabilité d'avarie accidentelle |
| `--p1` | Probabilité d'avarie conditionnelle à la surcharge |
| `--n-regions {1,3}` | Nombre de régions `N_F` |
| `--departage {highs,exterieur_dabord}` | Règle de départage (défaut `highs`). Ignoré avec `--resume` : la valeur du run d'origine est reprise |

**Options de `scan`** :

| Option | Rôle |
|---|---|
| `--run-id`, `--resume` | Nom explicite d'un nouveau run ; reprise d'un run (paramètres relus depuis `metadata.json`) |
| `--n` | Nombre de réalisations par cas |
| `--chunk-size` | Réalisations par bloc (défaut 250). Petit : progression plus fine et meilleure reprise |
| `--ratios R [R ...]` | Liste explicite de ratios ; prioritaire sur `--ratio-min`, `--ratio-max` et `--pas` |
| `--ratio-min`, `--ratio-max`, `--pas` | Intervalle régulier de ratios |
| `--min-tail` | Nombre minimal de points dans la queue pour classer un ratio comme candidat commun |
| `--workers` | Nombre de processus |

**Options de `production`** :

| Option | Rôle |
|---|---|
| `--run-id`, `--resume` | Comme pour `scan` |
| `--ratio` | Ratio commun (requis pour un nouveau run ; relu des métadonnées avec `--resume`) |
| `--n` | Nombre de réalisations par taille |
| `--chunk-size` | Réalisations par bloc (défaut 1000) |
| `--workers` | Nombre de processus |

```powershell
# Scan à ratios explicites
python scripts/05_reproduction_carreras.py scan `
    --run-id scan-fin-190-382 `
    --tailles 190 382 `
    --ratios 0.78 0.80 0.81 `
    --n 1000 --chunk-size 100 --min-tail 50 --workers 16

# Production longue
python scripts/05_reproduction_carreras.py production `
    --run-id production-4-tailles-r080 `
    --tailles 46 94 190 382 `
    --ratio 0.80 --n 60000 --chunk-size 500 --workers 16

# Lister les runs, puis reprendre un run interrompu
python scripts/05_reproduction_carreras.py runs
python scripts/05_reproduction_carreras.py scan --resume <RUN_ID> --workers 16
```

**Interruption.** Les workers ignorent `Ctrl+C` ; le processus principal les arrête, conserve les blocs déjà terminés, met `status.json` à l'état `interrupted`, réécrit le CSV avec tous les cas complets et affiche la commande exacte de reprise.

**Fichiers d'un run de scan** :

| Fichier | Contenu |
|---|---|
| `metadata.json` | Paramètres immuables du run |
| `status.json` | État (`running`, `completed`, `interrupted`, `failed`) et progression |
| `run.log` | Journal horodaté |
| `scan.csv` | Cas complètement terminés |
| `candidats.json` | Classement final des ratios communs |
| `checkpoints/*.npz` | Blocs de simulation réutilisables à la reprise |

Le CSV contient aussi les statistiques de cascades de lignes (fréquence de cascade multi-ligne, nombre moyen, médian, `q95` et maximum de lignes tombées sachant au moins une), et le run produit la figure `scan_cascades_lignes.png`.

## `06_validation_so1.py`

Audit quantitatif de SO1. Il **ne modifie aucun paramètre** du modèle et n'effectue aucun ajustement : il demande au modèle tel qu'il est codé de produire des nombres, puis les compare aux références publiées.

1. **Fig. 3 et 4 (382 nœuds)** : comparaison de deux lectures du papier, l'étiquette publiée `P_D/P_C = 0.30` et la Table I lue littéralement (`|P_L| = 74` par charge).
2. **Seuil de transport** `r_T(N)` pour `N = 46, 94, 190, 382` : mesure numérique (balayage grossier puis bissection sur le dispatch), prédiction analytique indépendante, écart entre les deux.
3. **Ratios du mini-scan à distance relative commune** `ρ = r / r_T(N)`, avec la probabilité que la demande fluctuée dépasse `P_C` (délestage imposé par la génération et non par le transport).
4. **Bandes ordonnées de la Fig. 10** (382 nœuds) : frontières analytiques et vérification au milieu de chaque bande avec les deux règles de départage. Attendu : bande présente avec `exterieur_dabord`, absente avec `highs`.

**Options** : aucune.

```powershell
python scripts/06_validation_so1.py
```

**Sorties** (`data/05_reproduction_carreras/validation_so1/`) : `audit_figures_3_4.json`, `seuils_transport.csv`, `cibles_rho.csv`, `commandes_mini_scan.txt` (les commandes du mini-scan à `ρ` constant), `bandes_fig10.csv`.

## `07_dynamique_lente.py`

Production de SO2 : dynamique lente de Carreras et al. (2004) pour des cas `taille × G`. Sous-commande : `run`. Chaque cas est enregistré sous un nom du type `N94_G1p5000` (taille 94, `G = 1.5`, la virgule décimale devient `p`). Des checkpoints sont écrits tous les `--bloc` jours : un run interrompu se reprend à l'identique avec `--resume`.

| Option | Rôle |
|---|---|
| `--tailles T [T ...]` | Tailles d'arbres |
| `--G G [G ...]` | Valeurs de `G` (marge de génération, Éq. 5 de Carreras 2004) |
| `--jours` | Durée de chaque cas, en jours |
| `--bloc` | Jours entre deux checkpoints (défaut 5000) |
| `--transitoire` | Jours écartés avant l'analyse (Carreras 2004 : 20 000) |
| `--g` | Amplitude des fluctuations : facteur dans `[1-g, 1+g]` |
| `--fluctuation {noeud,regionale}` | Fluctuation par nœud ou par région |
| `--n-regions {1,3}` | Nombre de régions `N_F` |
| `--p0`, `--p1` | Probabilités d'avarie accidentelle et conditionnelle à la surcharge |
| `--mu` | Renforcement des lignes tombées par surcharge |
| `--k` | Ajout de génération |
| `--lambda-annuel` | Croissance annuelle de la demande |
| `--ratio-initial` | Demande initiale divisée par `P_C` de la Table I |
| `--departage {highs,exterieur_dabord}` | Règle de départage |
| `--graine` | Graine aléatoire |
| `--workers` | Nombre de processus |
| `--run-id`, `--resume` | Nom explicite d'un nouveau run ; reprise (paramètres relus depuis `metadata.json`) |

Paramètres du balayage `so2-G-scan` : fluctuations régionales, `N_F = 3`, `g = 0.9`, `μ = 1.05`, `k = 0.02`, `p1 = 1`, transitoire de 20 000 jours, 120 000 jours par cas. La relation avec la notation de Carreras est `g = γ − 1` (`g = 0.9` correspond à `γ = 1.9`).

```powershell
# Essai court (un cas)
python scripts/07_dynamique_lente.py run `
    --tailles 46 --G 1.0 --jours 32000 --transitoire 20000 --run-id essai-so2

# Balayage en G (3 tailles x 7 valeurs de G)
python scripts/07_dynamique_lente.py run `
    --tailles 46 94 190 `
    --G 0.1 0.25 0.5 0.75 1.0 1.5 2.0 `
    --jours 120000 --workers 8 --run-id so2-G-scan

# Reprise après interruption
python scripts/07_dynamique_lente.py run --resume so2-G-scan --workers 8
```

**Sorties** : `data/07_dynamique_lente/runs/<run-id>/` (checkpoints, `metadata.json`, `resume.csv`) et `figures/07_dynamique_lente/runs/<run-id>/`.

---

## Scripts historiques (`03` et `04`)

Ces deux scripts sont conservés pour la traçabilité. **Ils ne servent pas de validation.**

### `03_balayage_charge.py` : qualitatif, avec une échelle calibrée

Reproduit qualitativement la figure de Carreras et al. : en augmentant la demande à profil fixe, deux changements de régime apparaissent (la demande atteint la capacité de production, puis les flux atteignent les limites de transport).

Attention aux deux différences avec la réplication validée :

- l'**échelle des capacités de lignes est calibrée** pour que la seconde transition tombe à 1.45 (`CIBLE_SECONDE`). C'est le seul paramètre libre du script, donc ses résultats ne testent pas le modèle contre la littérature ;
- le script prend 2623.9 comme **capacité totale** de production (`P_C` dans le fichier), alors que la réplication de `05` et `06` utilise `P_G = 2623.9` par générateur et `P_C = 12 × P_G = 31 486.8`.

La validation de SO1 repose sur `05_diagnostics_carreras.py` et `06_validation_so1.py`, qui lisent la Table I littéralement, sans calibration.

**Options** : aucune.

### `04_loi_puissance_taille_finie.py` : expérience historique 46 contre 94

Teste, sur deux arbres (46 et 94 nœuds), si une région compatible avec une loi de puissance apparaît près du régime critique et si elle s'élargit avec la taille. Deux étapes :

1. **Scan** exploratoire : balayage fin de `P_D/P_C`, estimation de `α`, `x_min`, de la distance KS et du nombre de points de queue, comparaison à la lognormale et à l'exponentielle. Aucune conclusion confirmatoire à cette étape.
2. **Production** confirmatoire : 60 000 réalisations i.i.d. par taille, graine différente de celle du scan, ratio choisi avant le lancement.

Selon le docstring du script, les fluctuations de cette expérience sont indépendantes par charge, sans les régions de Carreras, et `g = 0.9` y correspond à `γ = 1.9`. La réplication de référence avec fluctuations régionales est `05_reproduction_carreras.py`. L'usage détaillé (étapes 1 et 2) est donné dans l'en-tête du script.
