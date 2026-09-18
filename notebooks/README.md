# Carnets

Les carnets sont appariés à un fichier `.py` au format percent par jupytext. Le
`.py` est versionné, le `.ipynb` ne l'est pas : il contient les sorties, y
compris les images encodées, et produit des diffs illisibles.

```
pip install jupytext
jupytext --to ipynb notebooks/01_laboratoire.py
```

Règle : un carnet ne contient aucune logique de calcul. Toute fonction appelée
vient du paquet `cascade_entropy` et est couverte par la suite de tests. Le
carnet raconte, le paquet calcule.
