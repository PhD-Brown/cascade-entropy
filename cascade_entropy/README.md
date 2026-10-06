# cascade_entropy

Le paquet Python du projet : le modèle de réseau (fil A) et les outils d'analyse informationnelle (fil B). Les notebooks racontent la démarche ; les [scripts](../scripts/README.md) orchestrent des campagnes ; **tout calcul réutilisable et testable vit ici**.

## Deux fils indépendants

```text
Fil A : modèle du réseau                         Fil B : outils d'analyse
  reseau      topologie, flux, limites             synthetiques   séries de comportement connu
  dispatch    demande -> flux sur les lignes       entropie       série -> mesure informationnelle
  cascade     état du réseau -> blackout           indicateurs    série -> indicateur de référence
  evolution   N jours -> séries temporelles          analyse_series séries de SO2 -> H, cycles, queues
                                                   substituts     série -> mélanges, IAAFT
                                                   analyse_entropie séries de SO3 -> PE, MSE, références

                  séries temporelles écrites sur disque
        (le fil A les produit, le fil B les lit : le fil B n'importe jamais le fil A)
```

### La règle : le fil B n'importe jamais le fil A

Aucun module du fil B n'importe un module du fil A. Les deux fils communiquent par les séries temporelles écrites sur disque. Cela permet de valider les mesures entropiques sur des signaux de comportement connu, indépendamment de tout modèle électrique.

**Une seule exception, dans l'autre sens** : `evolution` (fil A) importe `entropie.nombre_effectif` (fil B) pour calculer l'une de ses observables. La dépendance va du fil A vers le fil B uniquement. Elle est documentée dans la docstring de `__init__.py`.

## Les modules

| Module | Fil | Rôle | Éléments clés |
|---|---|---|---|
| `reseau.py` | A | Topologie en arbre, matrice de flux DC, limites de lignes | `arbre`, `matrice_de_flux`, `limites_par_niveau` |
| `dispatch.py` | A | Dispatch par programme linéaire : charge servie, flux, délestage | `resoudre`, `demande_uniforme`, `DEPARTAGES`, `SEUIL_SATURATION` |
| `cascade.py` | A | Une journée : avaries `p0`, cascade `p1`, redispatch après chaque panne | `journee` |
| `evolution.py` | A | Dynamique multi-jours (modes `independant` et `auto_organise`), fluctuations régionales, reprise exacte | `simuler`, `EtatEvolution`, `marge_pour_G` |
| `carreras.py` | A | Protocole contrôlé de Carreras et al. (2002) : constantes de la Table I, régions, bandes ordonnées | `configuration_arbre`, `groupes_regions`, `bandes_ordonnees`, `P_G_TABLE`, `P_L_TABLE`, `GAMMA_TABLE` |
| `synthetiques.py` | B | Séries de comportement connu (sinusoïde, bruits en `f^-β`) pour valider les analyses | `periodique`, `bruit_puissance`, `bruit_blanc`, `bruit_rose` |
| `entropie.py` | B | Shannon, entropie de répartition, nombre effectif, entropie de permutation, SampEn, MSE ; option `methode="arbre"` (arbre k-d, mêmes comptes, praticable sur 10⁵ points) et liste d'échelles | `nombre_effectif`, `multiechelle`, `METHODES_SAMPEN` |
| `indicateurs.py` | B | Indicateurs de référence : exposant de Hurst `R/S`, détection du transitoire (MSER), diagnostics de queue | |
| `analyse_series.py` | B | Analyse des séries de SO2 : `R/S` vectorisé par plage, référence par mélange, période dominante, queue de distribution | |
| `substituts.py` | B | Séries de substitution : IAAFT (même distribution et même spectre, phases aléatoires, Schreiber et Schmitz 1996) | `substitut_iaaft`, `ecart_spectral` |
| `analyse_entropie.py` | B | Batterie de SO3 : PE multiéchelle, MSE, références par mélange et IAAFT, diagnostic d'égalités | `analyser_avec_references`, `resume_substituts`, `diagnostic_egalites` |
| `controles.py` | B | Témoins : mélanges temporels (mêmes valeurs, ordre détruit) | `analyser` |
| `figures.py` | transversal | Visualisations réutilisables | |
| `chemins.py` | transversal | Chemins canoniques `data/` et `figures/` à la racine du dépôt | `racine_projet`, `DOSSIER_DATA`, `DOSSIER_FIGURES` |
| `_validation.py` | transversal | Validation commune des entrées | |

La colonne « Éléments clés » ne liste que des noms vérifiés dans les scripts du dépôt ; chaque module contient d'autres fonctions, documentées par leurs docstrings. Le classement par fil de `carreras`, `controles`, `figures`, `chemins` et `_validation` suit leur rôle : ces modules ne figurent pas dans la docstring de `__init__.py`.

## Conventions

- **Docstrings en français**, explicatives : elles disent ce que la fonction fait et pourquoi.
- **Validation des entrées** : les fonctions publiques vérifient leurs arguments (validation commune dans `_validation.py`) et échouent avec un message clair.
- **Toute nouveauté est une option.** La valeur par défaut garde le comportement historique. Exemple : l'option `departage` du dispatch vaut `highs` par défaut ; `exterieur_dabord` est la règle de référence du projet, choisie explicitement.
- **Reproductibilité** : les fonctions stochastiques reçoivent un générateur `numpy.random.Generator` explicite (`rng`), jamais un état global. Les runs enregistrent graines et paramètres dans `metadata.json`.
- **Chaque ajout vient avec ses tests** (`tests/`, `python -m pytest -q`).

### Notations à ne pas confondre

| Grandeur | Convention du code | Lien avec Carreras |
|---|---|---|
| Capacités de production | `P_G = 2623.9` par générateur, `P_C = 12 × P_G = 31 486.8` | Table I de Carreras et al. (2002) |
| Amplitude des fluctuations | `gamma` (`05_*`) : facteurs uniformes sur `[2-γ, γ]` ; `g` (`07_*`) : facteurs dans `[1-g, 1+g]` | `g = γ − 1` (`g = 0.9` pour `γ = 1.9`) |
| Départage du délestage | `departage` ∈ `DEPARTAGES` = {`highs`, `exterieur_dabord`} | hypothèse de réplication : le papier utilise un simplexe |

## À savoir sur `chemins.py`

Importer `cascade_entropy.chemins` **crée** `data/` et `figures/` à la racine du dépôt s'ils n'existent pas. La racine est trouvée en remontant depuis le dossier courant jusqu'à un dossier nommé `cascade_entropy/` : un notebook ou un script doit donc être lancé quelque part à l'intérieur du dépôt. À importer en premier, avant toute écriture de figure ou de donnée :

```python
from cascade_entropy.chemins import DOSSIER_DATA, DOSSIER_FIGURES
```

## Exemple minimal (fil B seul)

Entropie multiéchelle d'un bruit rose, sans aucun module du fil A :

```python
import numpy as np

from cascade_entropy.synthetiques import bruit_puissance
from cascade_entropy.entropie import multiechelle

rng = np.random.default_rng(20260915)
serie = bruit_puissance(16384, beta=1.0, rng=rng)       # bruit rose
echelles, valeurs = multiechelle(serie, echelles=20)    # entropie multiéchelle
```

Pour des appels du fil A (réseau, dispatch, cascade, évolution), les scripts sont les exemples de référence : [`03_balayage_charge.py`](../scripts/03_balayage_charge.py) pour `reseau` et `dispatch`, [`06_validation_so1.py`](../scripts/06_validation_so1.py) pour `carreras` et `cascade`, [`07_dynamique_lente.py`](../scripts/07_dynamique_lente.py) pour `evolution`. Pour `analyse_series` (fil B), l'exemple de référence est [`08_analyse_so2.py`](../scripts/08_analyse_so2.py).
