# Réalisations synthétiques N−1 — format B5/B6 v1

Cette interface prépare la traçabilité de la phase B de l'[issue #11](https://github.com/PhD-Brown/cascade-entropy/issues/11).
Elle respecte les [conventions A3](https://github.com/PhD-Brown/cascade-entropy/issues/11#issuecomment-6861560057)
et le [suivi du prototype B](https://github.com/PhD-Brown/cascade-entropy/issues/11#issuecomment-6861346701).
Elle ne fixe ni les choix électriques quantitatifs B4, ni les densités, ni le
nombre de réalisations, ni les paramètres de propagation. Elle ne lance aucune
campagne et n'apporte aucune validation physique.

## Réutilisation et périmètre

`cascade_entropy.persistance_n1` enveloppe `topologies_n1.sauver_json()` et
`charger_json()` ; il ne remplace pas leur format ni les générateurs. Le fichier
`topologie.json` reste directement lisible par le lecteur historique. Il contient
déjà famille, nombre de nœuds, arêtes, graine, paramètres, nombre d'essais,
positions et statistiques structurelles. Ces champs ne sont pas dupliqués dans
le manifeste.

Les extensions nécessaires sont un identifiant réseau, une provenance, les
arguments effectifs de génération, les autres paramètres d'expérience, un statut,
des propriétés électriques optionnelles et des contrôles d'intégrité. Le
sérialiseur historique peut écraser un fichier : la nouvelle interface réserve
un **nouveau dossier** pour rendre son utilisation non destructive, sans changer
les appels historiques ni le script de démonstration existant.

Les cinq familles sont prises en charge : `anneau_cordes`, `grille`, `gilbert`,
`geometrique`, `arbre_temoin`. La grille et l'arbre ont une graine `null`, car leurs
générateurs sont déterministes. Pour les deux familles conditionnées à la
connexité, `essais` reste dans la topologie et `max_essais` figure dans les
arguments effectifs du manifeste. Aucun ajout caché de ligne n'est réalisé.

## Disposition et identité

Utiliser un chemin sous `DOSSIER_DATA`, par exemple
`data/n1/realisations/<identifiant-local>/`. `data/` est ignoré par Git. La fonction
accepte un chemin explicite pour permettre des expériences temporaires ; elle
ne choisit pas automatiquement une destination et ne parcourt aucune donnée.

```text
<identifiant-local>/
    topologie.json    # cascade-entropy-topologie-v1, sérialiseur existant
    metadata.json     # cascade-entropy-realisation-v1, publié en dernier
```

`id_reseau = "n1-" + SHA256(definition_topologique_canonique)`. La définition est
le contenu JSON complet de `topologie.json`, statistiques et positions incluses,
avec clés triées, séparateurs `,`/`:`, UTF-8 et sans nombres non finis. Cet ID est
indépendant du chemin, du statut, de la provenance et des propriétés électriques.
Il identifie une définition générée, pas une classe d'isomorphisme : deux graines
ou paramètres différents ne sont pas assimilés même s'ils donnent le même graphe.
Deux variantes électriques d'une même définition gardent cet ID ; les distinguer
par leurs dossiers et l'empreinte de leur manifeste. Ne jamais remplacer l'ancienne
variante pour lui ajouter des paramètres B4.

Les nœuds sont `0…n−1`, les lignes `0…m−1`. Le manifeste enregistre ces listes
dans `noeuds` et `lignes`. Les arêtes sont simples, uniques, orientées `u→v` avec
`u<v` et rangées lexicographiquement. Une permutation de lignes ou une inversion
d'arête dans un fichier est **refusée**, même si le lecteur historique pourrait la
normaliser : les valeurs électriques ne doivent jamais changer d'association.
Les positions suivent l'ordre des nœuds ; elles ne constituent pas des distances
physiques. Celles du graphe géométrique sont dans le carré unité du générateur.

## Manifeste

Tous les champs suivants sont obligatoires ; les champs supplémentaires sont
refusés dans cette version. Un changement de contrat exige une autre version,
sans migration implicite.

| Champ | Contenu |
| --- | --- |
| `format` | `cascade-entropy-realisation-v1` |
| `id_reseau` | Identité de la définition topologique décrite ci-dessus |
| `statut` | `demonstration`, `validation` ou `campagne_scientifique` |
| `noeuds`, `lignes` | Identifiants d'origine dans l'ordre des tableaux |
| `reproduction` | Nom qualifié du générateur, dictionnaire `arguments` complet (défauts inclus), convention d'aléa `alea` |
| `provenance` | `revision_code`, booléen `arbre_modifie`, `versions` (Python, NumPy, SciPy, paquet), `plateforme`, `sources_sha256` |
| `parametres_experience` | Dictionnaire explicite ; `{}` admis si rien d'autre n'est défini |
| `electrique` | `null` avant B4, sinon le conteneur décrit plus bas |
| `sha256_topologie` | SHA-256 des octets exacts de `topologie.json` |
| `sha256_metadata` | SHA-256 du manifeste canonique en excluant ce seul champ |

Les métadonnées libres doivent être du JSON strict : chaînes, booléens, entiers,
flottants finis, listes, dictionnaires à clés texte et `null`. Convertir
explicitement les scalaires NumPy et les tuples avant de les y placer. Les
tableaux électriques passent par `champ_numerique()`. Les clés JSON répétées,
NaN, infinis et types implicitement convertibles sont refusés.

Les graines propres à d'autres étapes et leurs algorithmes, les options de
solveur, règles de départage, seuils et paramètres de propagation devront être
déposés dans `parametres_experience` dès qu'ils seront décidés. La graine de la
topologie seule ne reproduit pas une cascade stochastique.

`revision_code` et `arbre_modifie` sont des déclarations de l'appelant, pas des
résultats devinés depuis le dossier courant. Utiliser le SHA complet du code
effectivement exécuté (`git rev-parse HEAD`) et vérifier l'arbre de travail
(`git status --short`). Une révision avec `arbre_modifie=True` ne suffit pas à
retrouver des modifications locales perdues : conserver le code correspondant
dans une révision durable avant une utilisation scientifique. Les versions
installées sont collectées automatiquement ; elles ne sont pas un verrou complet
de l'environnement. `sources_sha256` enregistre les empreintes des octets de
`persistance_n1.py`, `topologies_n1.py`, `reseau.py` et `_validation.py` exécutés.
Ces empreintes permettent de vérifier une archive de code ; elles ne remplacent
pas cette archive. Elles ne sont pas comparées automatiquement au code courant
à la lecture. Les différences de plateforme/solveur restent à contrôler.

Le statut est une déclaration d'usage, **pas une certification scientifique**.
Un statut de campagne exige un conteneur électrique présent ; les contrôles de
faisabilité, de conservation et l'approbation du protocole restent extérieurs à
cette interface. Une démonstration sans paramètres électriques reste possible.

## Interface proposée pour B4

Aucune classe, fonction ou structure interne B4 n'est supposée. L'appelant adapte
ses valeurs au conteneur suivant, passé par l'argument optionnel `electrique` :

```python
electrique = {
    "base_puissance": {"valeur": base_mva, "unite": "MVA"},
    "conventions": {
        "orientation": "u vers v ; incidence +1 à u, -1 à v",
        "injections": "production positive, consommation servie négative",
        "attribution": "référence de la méthode B4 et de ses hypothèses",
    },
    "champs": {
        "reactances": champ_numerique(
            reactances, axes=["ligne"], unite="pu",
            convention="x positif ; b=1/x ; base commune déclarée"),
        "demande": champ_numerique(
            demande, axes=["noeud"], unite="pu",
            convention="demande positive par nœud initial"),
    },
}
```

Les noms de champs sont libres ; ceux de l'exemple ne sont pas une API B4 adoptée.
Chaque champ possède exactement `valeurs`, `dtype`, `axes`, `unite`, `convention`.
La base numérique n'a aucun défaut : elle doit être finie, positive et exprimée
en MVA. Les unités et les conventions de chaque champ doivent être explicites.
Conformément à A3, puissances et capacités seront en pu sur cette base, réactances
en pu, angles en radians ; les quantités sans dimension utilisent `unite="1"`.
L'interface vérifie la présence de ces descriptions, pas leur vérité physique.

Les axes autorisés sont `noeud` et `ligne`. Une liste vide représente un scalaire,
un axe représente un vecteur et deux axes peuvent représenter une matrice. Chaque
dimension doit correspondre au nombre initial de nœuds/lignes. Des masques
booléens nodaux permettent de conserver les rôles producteurs/consommateurs
indépendamment de la représentation B4 ; les indices clairsemés de longueur
variable nécessitent un adaptateur, sans réindexation du réseau.

Les types booléens, entiers signés/non signés et flottants NumPy jusqu'à 64 bits
sont pris en charge. Le `dtype` conserve taille et boutisme. Le rechargement
refuse les conversions avec perte ; `tableau_champ(champ)` restitue une copie
NumPy. Le JSON conserve exactement les flottants binaires pris en charge, y
compris les zéros signés. Aucun pickle n'est utilisé. Les NaN de taux de charge
post-retrait prévus par A3 ne sont pas acceptés : ce format porte sur les
réalisations initiales, pas sur les sorties de cascades. Un format de résultats
devra décrire les masques et valeurs manquantes séparément.

Dépendances à trancher avec B4 avant intégration : noms et liste des grandeurs
effectivement livrées, représentation des rôles, base numérique, unités et
conventions, paramètres/graines de l'attribution, définition de l'état initial
et référence du protocole validé. Aucune modification des modules B4 ou C n'est
nécessaire pour examiner cette proposition. Un éventuel adaptateur sera une
modification distincte après accord sur cette interface.

## Exemple technique sans paramètres électriques

Après installation du paquet (`python -m pip install -e ".[dev]"`), depuis la
racine, exemple à adapter avec la révision réellement exécutée :

```python
from cascade_entropy.chemins import DOSSIER_DATA
from cascade_entropy.topologies_n1 import anneau_cordes
from cascade_entropy.persistance_n1 import (
    sauver_realisation, charger_realisation, verifier_reproduction,
)

arguments = {"n_noeuds": 12, "n_cordes": 4, "graine": 31}
t = anneau_cordes(**arguments)
cible = DOSSIER_DATA / "n1" / "realisations" / "demo-anneau-31"
identifiant = sauver_realisation(
    t, cible, statut="demonstration", arguments_generation=arguments,
    revision_code=revision_git_effective, arbre_modifie=arbre_effectivement_modifie,
    parametres_experience={},
)
r = charger_realisation(cible)
assert r.id_reseau == identifiant
verifier_reproduction(r)
```

Les paramètres de cet exemple sont une fixture technique, sans statut de choix
scientifique. L'écriture régénère une seule topologie et vérifie identité des
arêtes, paramètres, graines, essais et positions **avant** de créer la cible.
La lecture ne relance aucun générateur ; `verifier_reproduction()` est explicite
pour tester une régénération avec l'environnement courant. Les topologies sauvées
restent la référence si un changement de version modifie les générateurs.

## Erreurs, intégrité et non-écrasement

- Une cible existante, même vide ou incomplète, produit `FileExistsError`.
  Deux écritures concurrentes vers la même cible ont un seul gagnant.
- Une entrée invalide produit `ValueError` avant la création de la cible.
- Un format inconnu, un JSON invalide, une empreinte incorrecte, des indices ou
  formes incohérents produisent `ValueError` à la lecture.
- Les erreurs E/S gardent leur type et leur chemin : `FileNotFoundError`,
  `PermissionError`, etc. Elles ne deviennent pas des réalisations vides.
- `metadata.json` est publié par renommage atomique après fermeture des fichiers.
  Une interruption peut laisser un dossier incomplet et `metadata.json.tmp` :
  le lecteur le refuse, et l'écriture ne le reprend ni ne le nettoie. Inspecter
  ce dossier puis choisir une nouvelle cible ; aucune suppression automatique.
- Les empreintes détectent l'altération accidentelle, pas une modification
  malveillante avec recalcul des empreintes. La mise en page du manifeste n'a
  pas d'incidence ; celle de `topologie.json` est couverte octet par octet.
  Il n'y a pas de garantie transactionnelle contre une panne matérielle.

Les tests dans `tests/test_persistance_n1.py` utilisent uniquement `tmp_path` et
des réseaux synthétiques. Commandes de validation :

```powershell
python -m pytest tests/test_persistance_n1.py tests/test_topologies_n1.py -q
python -m pytest -q
```

## Proposition d'entrée pour le journal (non inscrite)

« B5/B6 : réutilisation du JSON topologique v1 dans un dossier exclusif avec
manifeste versionné, identité de définition topologique et empreintes SHA-256.
Interface électrique optionnelle par champs numériques à unités et axes
explicites, à discuter avec B4. Contrôles techniques synthétiques ; choix
électriques, densités, effectifs et validation de campagne toujours ouverts. »
