"""Compare les règles sur les mêmes retraits d'un run du diagnostic N−1."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.stats import spearmanr

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from scripts.diagnostic_departage_n1 import environnement, sauver_csv, _sauver


TOLERANCE_ZERO = 1e-9
DECIMALES_RANGS = 9
SEUILS_GRAVITE = (0.05, 0.30)


def lire_retraits(chemin):
    reseaux = defaultdict(lambda: defaultdict(dict))
    with chemin.open(encoding="utf-8", newline="") as fichier:
        lecteur = csv.DictReader(fichier)
        colonnes = {"reseau", "regle", "retrait", "fraction_incrementale"}
        if not colonnes <= set(lecteur.fieldnames or ()):
            raise ValueError("Colonnes de retrait manquantes.")
        for ligne in lecteur:
            nom, regle = ligne["reseau"], ligne["regle"]
            retrait = int(ligne["retrait"])
            fraction = float(ligne["fraction_incrementale"])
            valeurs = reseaux[nom][regle]
            if retrait in valeurs or not np.isfinite(fraction):
                raise ValueError(f"Retrait répété ou fraction non finie : {nom}, {regle}, {retrait}.")
            valeurs[retrait] = fraction
    regles = sorted({regle for retraits in reseaux.values() for regle in retraits})
    if len(regles) < 2:
        raise ValueError("Au moins deux règles ayant des retraits sont requises.")
    tableaux = {}
    for nom, retraits in sorted(reseaux.items()):
        indices = sorted(retraits[regles[0]])
        if not indices or any(sorted(retraits[regle]) != indices for regle in regles):
            raise ValueError(f"Retraits différents entre règles pour {nom}.")
        tableaux[nom] = {regle: np.array([retraits[regle][i] for i in indices]) for regle in regles}
    return tableaux, regles


def n_valeurs_ex_aequo(fractions):
    _, effectifs = np.unique(fractions, return_counts=True)
    return int(effectifs[effectifs > 1].sum())


def correlation(a, b):
    if np.all(a == a[0]) or np.all(b == b[0]):
        return None
    return float(spearmanr(a, b).statistic)


def comparer(nom, retraits):
    arrondies = {r: np.round(np.where(np.abs(f) <= TOLERANCE_ZERO, 0.0, f), DECIMALES_RANGS)
                 for r, f in retraits.items()}
    for a, b in combinations(retraits, 2):
        x, y = retraits[a], retraits[b]
        xa, ya = arrondies[a], arrondies[b]
        yield {"reseau": nom, "regle_a": a, "regle_b": b, "n_retraits": len(x),
               "spearman": correlation(xa, ya), "spearman_brut": correlation(x, y),
               "n_nuls_a": int(np.count_nonzero(np.abs(x) <= TOLERANCE_ZERO)),
               "n_nuls_b": int(np.count_nonzero(np.abs(y) <= TOLERANCE_ZERO)),
               "n_ex_aequo_a": n_valeurs_ex_aequo(xa), "n_ex_aequo_b": n_valeurs_ex_aequo(ya),
               "n_ex_aequo_brut_a": n_valeurs_ex_aequo(x), "n_ex_aequo_brut_b": n_valeurs_ex_aequo(y)}


def repartition(nom, regle, fractions):
    s5, s30 = SEUILS_GRAVITE
    nuls = np.abs(fractions) <= TOLERANCE_ZERO
    classes = {"nulle": nuls, "moins_5pct": (fractions > TOLERANCE_ZERO) & (fractions < s5),
               "de_5_a_30pct": (fractions >= s5) & (fractions < s30), "au_moins_30pct": fractions >= s30,
               "residu_negatif_hors_tolerance": fractions < -TOLERANCE_ZERO}
    n = len(fractions)
    non_nulles = fractions[~nuls]
    ligne = {"reseau": nom, "regle": regle, "n_retraits": n}
    for classe, masque in classes.items():
        effectif = int(np.count_nonzero(masque))
        ligne[f"n_{classe}"] = effectif
        ligne[f"part_{classe}"] = effectif / n
    ligne["mediane_non_nulle"] = float(np.median(non_nulles)) if len(non_nulles) else None
    return ligne


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sortie", type=Path, default=RACINE / "data/diagnostic_departage_n1")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("run-id doit être un nom simple, sans chemin.")
    dossier = args.sortie / args.run_id
    source = dossier / "retraits.csv"
    reseaux, regles = lire_retraits(source)
    spearman = [ligne for nom, retraits in reseaux.items() for ligne in comparer(nom, retraits)]
    gravite = [repartition("", regle, np.concatenate([retraits[regle] for retraits in reseaux.values()]))
               for regle in regles]
    gravite.extend(repartition(nom, regle, retraits[regle])
                   for nom, retraits in reseaux.items() for regle in regles)
    metadata = {"run_id": args.run_id, "source": source.name,
                "sha256_source": hashlib.sha256(source.read_bytes()).hexdigest(),
                "environnement": environnement(), "parametres": {
                    "regles": regles, "tolerance_zero_fraction": TOLERANCE_ZERO,
                    "decimales_rangs": DECIMALES_RANGS, "seuils_gravite": list(SEUILS_GRAVITE),
                    "spearman": "scipy.stats.spearmanr, rangs moyens pour les ex aequo",
                    "ex_aequo": "nombre de valeurs appartenant à un groupe d'au moins deux valeurs égales",
                    "gravite_et_mediane": "fractions brutes, médiane de toutes les valeurs hors de la classe nulle",
                    "reseau_vide": "agrégation de tous les retraits, sans pondération par réseau"}}
    chemin_meta = dossier / "comparaison_metadata.json"
    if chemin_meta.exists() and json.loads(chemin_meta.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Métadonnées de comparaison existantes différentes.")
    sauver_csv(dossier / "spearman_regles.csv", spearman)
    sauver_csv(dossier / "repartition_gravite.csv", gravite)
    if not chemin_meta.exists():
        _sauver(chemin_meta, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
