# scripts

Chaque script produit une figure ou un résultat chiffré destiné au rapport.

Convention de nommage : `NN_sujet.py`, où NN est l'ordre chronologique de
production. Les figures sont écrites dans `figures/`, les séries dans `data/`,
deux dossiers non versionnés.

Un script ne contient pas de logique réutilisable : il orchestre des fonctions du
paquet `cascade_entropy` et trace le résultat.

# Guide — `05_reproduction_carreras.py` v2

## Installation

Remplacer :

```text
scripts/05_reproduction_carreras.py
```

par le fichier `05_reproduction_carreras_v2.py` fourni, puis le renommer :

```text
05_reproduction_carreras.py
```

Aucune modification de `cascade_entropy/carreras.py` n'est nécessaire.

---

## Ce qui change

### 1. Aucun run ne peut écraser un autre

Chaque exécution crée automatiquement :

```text
data/05_reproduction_carreras/runs/<RUN_ID>/
figures/05_reproduction_carreras/runs/<RUN_ID>/
```

Le `RUN_ID` contient le type de run, les tailles, le nombre de réalisations,
les ratios et un horodatage.

On peut imposer un nom lisible :

```powershell
python scripts/05_reproduction_carreras.py scan `
    --run-id scan-fin-190-382 `
    --tailles 190 382 `
    --ratios 0.78 0.80 0.81 `
    --n 1000 `
    --chunk-size 100 `
    --workers 16
```

Si ce nom existe déjà, le script refuse d'écraser les données.

---

### 2. Checkpoints bloc par bloc

Un scan de 1000 réalisations avec `--chunk-size 100` est découpé en 10 blocs.

Chaque bloc terminé est sauvegardé immédiatement dans :

```text
checkpoints/
```

Une interruption ne détruit donc que les blocs qui étaient *en cours* au
moment du Ctrl+C; tous les blocs déjà terminés sont conservés.

---

### 3. Reprise d'un run interrompu

Lister les runs :

```powershell
python scripts/05_reproduction_carreras.py runs
```

Puis reprendre :

```powershell
python scripts/05_reproduction_carreras.py scan `
    --resume <RUN_ID> `
    --workers 16
```

Les paramètres scientifiques, ratios, tailles, `n` et `chunk-size` sont relus
depuis `metadata.json`. Il n'est pas nécessaire de les retaper.

Pour une production :

```powershell
python scripts/05_reproduction_carreras.py production `
    --resume <RUN_ID> `
    --workers 16
```

---

### 4. Ctrl+C propre

Les workers ignorent `Ctrl+C`; le processus principal gère l'interruption.

Le script :
1. arrête les workers;
2. conserve les checkpoints terminés;
3. met `status.json` à l'état `interrupted`;
4. réécrit le CSV avec tous les cas complets;
5. affiche la commande exacte permettant de reprendre.

---

### 5. Ratios explicites

On peut maintenant écrire :

```powershell
--ratios 0.78 0.80 0.81
```

au lieu de construire artificiellement un intervalle régulier.

Si `--ratios` est présent, il est prioritaire sur `--ratio-min`,
`--ratio-max` et `--pas`.

---

### 6. Nouvelle statistique des cascades de lignes

En plus de la fréquence des blackouts et de la queue du délestage, le CSV contient :

```text
freq_lignes_tombees
freq_cascade_multiligne
fraction_multiligne_conditionnelle
lignes_tombees_moyenne_conditionnelle
lignes_tombees_mediane_conditionnelle
lignes_tombees_q95_conditionnelle
lignes_tombees_max
```

Une nouvelle figure :

```text
scan_cascades_lignes.png
```

montre notamment

\[
E[N_{\rm outages}\mid N_{\rm outages}>0].
\]

---

## Prochaine expérience recommandée

Pour départager `0.78`, `0.80` et `0.81` sur 190/382 :

```powershell
python scripts/05_reproduction_carreras.py scan `
    --run-id scan-fin-190-382 `
    --tailles 190 382 `
    --ratios 0.78 0.80 0.81 `
    --n 1000 `
    --chunk-size 100 `
    --min-tail 50 `
    --workers 16
```

`--min-tail 50` est volontairement exploratoire ici. Une fois la région
retenue, refaire une confirmation avec `n=2500` et `--min-tail 100`.

---

## Production longue

Exemple :

```powershell
python scripts/05_reproduction_carreras.py production `
    --run-id production-4-tailles-r080 `
    --tailles 46 94 190 382 `
    --ratio 0.80 `
    --n 60000 `
    --chunk-size 500 `
    --workers 16
```

Pour 190/382, commencer avec des blocs plus petits (`250` à `500`) est
préférable : meilleure répartition du travail et reprise plus fine.

---

## Fichiers d'un run

```text
metadata.json
```
Paramètres immuables du run.

```text
status.json
```
État courant (`running`, `completed`, `interrupted`, `failed`) et progression.

```text
run.log
```
Journal horodaté.

```text
scan.csv
```
Cas complètement terminés du scan.

```text
candidats.json
```
Classement final des ratios communs.

```text
checkpoints/*.npz
```
Blocs de simulation réutilisables en reprise.

Aucune donnée d'un run n'est écrasée par un autre run.
