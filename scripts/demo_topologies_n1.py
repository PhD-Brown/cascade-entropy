"""Examine les cinq familles synthétiques ; aucun paramètre électrique n'est fixé.

Exécuter depuis la racine du dépôt : python -m scripts.demo_topologies_n1
Option : --sortie MON_DOSSIER pour enregistrer les topologies en JSON.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cascade_entropy.topologies_n1 import (
    anneau_cordes, arbre_temoin, geometrique, gilbert, grille,
    sauver_json, statistiques,
)


def main() -> None:
    parser=argparse.ArgumentParser(description="Banc de topologies N-1 purement synthétiques")
    parser.add_argument("--graine",type=int,default=20261113)
    parser.add_argument("--realisations",type=int,default=1)
    parser.add_argument("--sortie",type=Path,default=None,
                        help="Si défini, écrit des JSON de topologies (aucune donnée électrique).")
    args=parser.parse_args()
    if args.realisations<1:
        parser.error("--realisations doit être >= 1")
    if args.sortie is not None:
        args.sortie.mkdir(parents=True,exist_ok=True)
    topologies=[grille(5,10),arbre_temoin(50,3)]
    for j in range(args.realisations):
        seed=args.graine+j
        topologies.extend([
            anneau_cordes(50,25,seed),
            gilbert(50,.10,seed,max_essais=500),
            geometrique(50,.25,seed,max_essais=500),
        ])
    print("PHASE B — banc de topologies synthétiques, pas de paramètres électriques")
    print(f"{'famille':17s} {'n':>3s} {'m':>4s} {'deg moy':>9s} {'ponts':>6s} {'cycles':>7s} {'essais':>7s} {'graine':>10s}")
    for k,t in enumerate(topologies):
        s=statistiques(t)
        print(f"{t.famille:17s} {t.n_noeuds:3d} {t.n_lignes:4d} {s['degre_moyen']:9.3f} {s['n_ponts']:6d} {s['rang_cyclomatique']:7d} {t.essais:7d} {str(t.graine):>10s}")
        if args.sortie is not None:
            sauver_json(t,args.sortie/f"{k:02d}_{t.famille}_seed-{t.graine if t.graine is not None else 'fixe'}.json")
    if args.sortie is not None:
        print("Définitions enregistrées dans",args.sortie)
    print("NOTE : p=0.10, rayon=0.25 et 25 cordes sont des paramètres PILOTES, non validés par Patrick.")


if __name__=="__main__":
    main()
