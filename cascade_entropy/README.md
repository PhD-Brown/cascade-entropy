# Carnets

Les carnets racontent la démarche scientifique ; le paquet [`cascade_entropy`](../cascade_entropy/README.md) effectue les calculs.

**Règle** : un carnet ne contient aucune logique de calcul. Toute fonction appelée vient du paquet `cascade_entropy` et est couverte par la suite de tests.

## Les quatre carnets principaux

Ils suivent l'ordre logique du projet :

| Carnet | Rôle |
|---|---|
| `00_fondations_probabilistes.ipynb` | Probabilités, Shannon, ordres relatifs et bases des mesures informationnelles |
| `01_laboratoire.ipynb` | Validation des méthodes sur des signaux synthétiques connus |
| `02_modele_evolution.ipynb` | Fil A complet : réseau, dispatch, cascade, évolution, réplication de Carreras |
| `03_analyse_entropique.ipynb` | Interface avec le fil B et premières analyses informationnelles des séries électriques |

## Archives : `autres/`

`autres/` contient des versions antérieures et des brouillons de carnets (`02_dispatch.ipynb`, `02_modele_evolution.ipynb`, `03_evolution.ipynb`). Ils ne sont pas maintenus : les quatre carnets ci-dessus font foi.

## Politique de versionnement

- **Les `.ipynb` sont versionnés, avec leurs sorties.** GitHub les affiche avec leurs figures : on peut relire le déroulé sans rien installer.
- **Pas de jupytext.** Les carnets ne sont pas appariés à des fichiers `.py` : le `.ipynb` est la source.
- **Les diffs Git sont peu lisibles**, car les sorties contiennent des images encodées. Pour relire un changement, utiliser l'aperçu GitHub ou `git diff --stat`.
- **Avant de versionner un carnet**, redémarrer le noyau et tout exécuter (*Restart & Run All*), pour que les sorties enregistrées correspondent au code actuel.

## Lancer un carnet

Depuis la racine du dépôt :

```powershell
python -m pip install -e ".[dev]"
```

Puis ouvrir le carnet dans VS Code ou Jupyter, avec l'interpréteur de l'environnement du projet. Le lancement peut se faire depuis n'importe quel sous-dossier du dépôt : `cascade_entropy.chemins` retrouve la racine, de sorte qu'un carnet lancé depuis `notebooks/` écrit ses données dans `data/` et ses figures dans `figures/` à la racine, comme les scripts.

Les figures sont régénérables ; seules celles de référence sont versionnées (voir le [README racine](../README.md#ce-qui-est-versionné-et-ce-qui-ne-lest-pas)).
