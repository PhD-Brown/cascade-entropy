# scripts

Chaque script produit une figure ou un résultat chiffré destiné au rapport.

Convention de nommage : `NN_sujet.py`, où NN est l'ordre chronologique de
production. Les figures sont écrites dans `figures/`, les séries dans `data/`,
deux dossiers non versionnés.

Un script ne contient pas de logique réutilisable : il orchestre des fonctions du
paquet `cascade_entropy` et trace le résultat.
