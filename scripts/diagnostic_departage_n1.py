"""Diagnostic technique synthétique du départage, sans analyse de centralité.

Exécution depuis la racine : python scripts/diagnostic_departage_n1.py --run-id NOM
Les capacités de la comparaison sont figées depuis la référence highs : changer
simultanément les capacités et la règle confondrait deux effets distincts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
import scipy

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from cascade_entropy import attribution_n1 as B, topologies_n1 as T
from cascade_entropy import diagnostics_dispatch_n1 as D
from cascade_entropy.cascade_n1 import simuler_tous_retraits_n1
from cascade_entropy.n1 import redispatch_exact


def reseaux(audit=False):
    """Huit réseaux : deux tirages par famille aléatoire, deux témoins fixes."""
    if audit:
        return [("audit_gilbert_7", T.gilbert(graine=7)),
                ("audit_anneau_7", T.anneau_cordes(graine=7)), ("audit_grille", T.grille())]
    cas = []
    for famille in (T.anneau_cordes, T.gilbert, T.geometrique):
        for graine in (20261113, 20261114):
            cas.append((f"{famille.__name__}_{graine}", famille(graine=graine)))
    return cas + [("grille", T.grille()), ("arbre_temoin", T.arbre_temoin())]


def attribuer(topologie):
    """Réglages provisoires de l'audit : graines électriques 11, 12 et 13."""
    return B.attribuer(topologie,
        placement=B.placement_aleatoire(topologie, n_generateurs=8, n_charges=20, graine=11),
        demande=B.Valeurs("uniforme", [.5, 1.5], graine=12),
        puissance_max=B.Valeurs("constante", [5.]),
        reserves_minimales=B.Valeurs("constante", [0.]),
        reactances=B.Valeurs("uniforme", [.2, .8], graine=13),
        capacites=B.Capacites("flux_reference", B.Valeurs("constante", [100.]), marge=1.5, plancher=.01),
        base_mva=100., unite_puissance="pu", unite_reactance="pu")


def _json(objet):
    return json.dumps(objet, ensure_ascii=False, indent=2, allow_nan=False)


def _sauver(chemin, objet):
    # Le lien exclusif publie seulement un fichier complet et refuse l'écrasement.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=chemin.parent,
                                     suffix=".partial", delete=False) as f:
        temporaire = Path(f.name)
        f.write(_json(objet) + "\n")
    try:
        os.link(temporaire, chemin)
    finally:
        temporaire.unlink()


def environnement():
    """Version du HiGHS embarqué ; aucune dépendance highspy ajoutée."""
    try:
        from scipy.optimize._highspy import _core
        highs = ".".join(str(getattr(_core, f"HIGHS_VERSION_{v}")) for v in ("MAJOR", "MINOR", "PATCH"))
    except (ImportError, AttributeError):
        highs = "indisponible via cette version de scipy"
    return {"python": platform.python_version(), "plateforme": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "highs": highs,
            "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


def empreintes():
    """Empreintes des octets exécutés, distinctes du SHA de base Git."""
    fichiers = [Path(__file__), *sorted((RACINE / "cascade_entropy").glob("*.py"))]
    return {p.relative_to(RACINE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in fichiers}


def mesurer(nom, etat, regle, variabilite, variabilite_minmax):
    """Une ligne CSV ; le témoin prorata ne subit jamais de cascade."""
    debut = perf_counter()
    r, d, p = etat.reseau, etat.demande, etat.puissance_max
    prorata = None
    if regle == "prorata":
        prorata = D.etat_intact_prorata(r, d, p)
        intact = prorata.resultat
    else:
        intact = redispatch_exact(r, d, p, np.zeros(r.n_lignes, bool),
                                 departage="min_charge_max" if regle == "min_redispatch" else regle)
    applicable = prorata is None or prorata.faisable
    n_saturees = n_evitables = None
    if applicable:
        sat, _ = D.lignes_saturees(intact)
        ev, _ = D.classer_saturations(intact, variabilite)
        n_saturees, n_evitables = len(sat), len(ev)
    ligne = {"reseau": nom, "regle": regle, "n_lignes": r.n_lignes,
        "n_saturees_intact": n_saturees, "n_evitables_intact": n_evitables,
        "classification_intacte_applicable": applicable,
        "taux_max_intact": float(np.nanmax(intact.taux_de_charge)),
        "charge_max_minimale": float(max(variabilite.charge_max_minimale)),
        "n_flux_non_uniques": variabilite.n_flux_non_uniques,
        "n_flux_non_uniques_face_minmax": variabilite_minmax.n_flux_non_uniques,
        "prorata_faisable": None if prorata is None else prorata.faisable,
        "prorata_violations": None if prorata is None else ";".join(prorata.violations)}
    details = []
    if regle != "prorata":
        options = {"departage_intact": "min_charge_max"} if regle == "min_redispatch" else {}
        cascades = simuler_tous_retraits_n1(r, d, p, p1=1., departage=regle, **options)
        for cascade in cascades:
            diagnostic = D.diagnostiquer_cascade(cascade, r, p)
            details.append({"retrait": cascade.ligne_initiale,
                "fraction_incrementale": cascade.delestage_incremental / d.sum(),
                "diagnostic": asdict(diagnostic)})
        fractions = np.array([a["fraction_incrementale"] for a in details])
        avec = sum(bool(a["diagnostic"]["vagues"]) for a in details)
        uniquement = sum(a["diagnostic"]["part_premiere_vague_uniquement_evitable"] == 1 for a in details)
        ligne.update({"n_retraits": len(cascades), "n_retraits_avec_vague": avec,
            "n_retraits_delestage_incremental_nul": int(np.count_nonzero(np.abs(fractions) <= 1e-9)),
            "n_fractions_incrementales_negatives": int(np.count_nonzero(fractions < 0)),
            "fraction_incrementale_min": float(fractions.min()),
            "fraction_incrementale_mediane": float(np.median(fractions)),
            "fraction_incrementale_q90": float(np.quantile(fractions, .9)),
            "fraction_incrementale_max": float(max(fractions)),
            "n_tombees_evitables": sum(a["diagnostic"]["n_evitables"] for a in details),
            "n_tombees_forcees": sum(a["diagnostic"]["n_forcees"] for a in details),
            "n_premieres_vagues_uniquement_evitables": uniquement,
            "part_premieres_vagues_uniquement_evitables": uniquement / avec if avec else None})
    else:
        for cle in ("n_retraits", "n_retraits_avec_vague", "n_retraits_delestage_incremental_nul",
                    "n_fractions_incrementales_negatives", "fraction_incrementale_min",
                    "fraction_incrementale_mediane", "fraction_incrementale_q90", "fraction_incrementale_max",
                    "n_tombees_evitables", "n_tombees_forcees", "n_premieres_vagues_uniquement_evitables",
                    "part_premieres_vagues_uniquement_evitables"):
            ligne[cle] = None
    ligne["secondes"] = perf_counter() - debut
    return {"resume": ligne, "retraits": details}


def traiter_reseau(travail):
    """Calcule un réseau indépendant, avec reprise au niveau de chaque règle."""
    nom, topo, sortie, regles = travail
    e = attribuer(topo)
    v = v_minmax = None
    resultats = []
    for regle in regles:
        chemin = sortie / f"{nom}__{regle}.json"
        if chemin.exists():
            contenu = json.loads(chemin.read_text(encoding="utf-8"))
        else:
            if v is None:
                debut = perf_counter()
                v = D.variabilite_flux(e.reseau, e.demande, e.puissance_max, np.zeros(e.reseau.n_lignes, bool))
                v_minmax = D.variabilite_flux(e.reseau, e.demande, e.puissance_max,
                                            np.zeros(e.reseau.n_lignes, bool), departage="min_charge_max")
                config = sortie / f"{nom}__configuration.json"
                if not config.exists():
                    _sauver(config, {"configuration": e.configuration,
                        "secondes_variabilite_intact": perf_counter() - debut})
            print(f"Calcul {nom} / {regle}", flush=True)
            contenu = mesurer(nom, e, regle, v, v_minmax)
            _sauver(chemin, contenu)
            print(f"Terminé {nom} / {regle} : {contenu['resume']['secondes']:.1f} s", flush=True)
        resultats.append(contenu)
    return resultats


def ligne_retrait(resume, detail):
    diagnostic = detail["diagnostic"]
    vagues = diagnostic["vagues"]
    return {"reseau": resume["reseau"], "regle": resume["regle"],
            "retrait": detail["retrait"], "fraction_incrementale": detail["fraction_incrementale"],
            "n_vagues": len(vagues), "n_evitables": diagnostic["n_evitables"],
            "n_forcees": diagnostic["n_forcees"],
            "premiere_vague_uniquement_evitable": diagnostic["part_premiere_vague_uniquement_evitable"],
            "premiere_vague_contient_saturation_intacte": bool(vagues and vagues[0]["deja_saturees_intact"]),
            "n_deja_saturees_intact": sum(len(vague["deja_saturees_intact"]) for vague in vagues)}


def sauver_csv(chemin, lignes):
    f = io.StringIO(newline="")
    writer = csv.DictWriter(f, fieldnames=list(lignes[0]))
    writer.writeheader()
    writer.writerows(lignes)
    texte = f.getvalue()
    if chemin.exists():
        if chemin.read_bytes() != texte.encode("utf-8"):
            raise ValueError(f"CSV existant différent : reprise refusée ({chemin.name}).")
    else:
        with chemin.open("x", encoding="utf-8", newline="") as fichier:
            fichier.write(texte)


def main(argv=None):
    """Crée un run exclusif ou reprend seulement ses calculs manquants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Processus indépendants par réseau (1 à 8).")
    parser.add_argument("--audit-reference", action="store_true", help="Trois cas exacts de l'audit, graines 7 et 11.")
    parser.add_argument("--sortie", type=Path, default=RACINE / "data/diagnostic_departage_n1")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("workers doit être compris entre 1 et 8.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("run-id doit être un nom simple, sans chemin.")
    sortie = args.sortie / args.run_id
    params = {"audit_reference": args.audit_reference, "workers": args.workers,
        "graines_topologies": [7] if args.audit_reference else [20261113, 20261114],
        "graine_placement": 11, "graine_demande": 12, "graine_reactances": 13,
        "n_generateurs": 8, "n_charges": 20, "demande": [.5, 1.5], "Pmax": 5., "x": [.2, .8],
        "capacites": {"mode": "flux_reference", "regle_reference": "highs", "limite_reference": 100.,
                      "marge": 1.5, "plancher_absolu": .01, "figees_entre_regles": True},
        "p1": 1., "departage_intact_minredispatch": "min_charge_max",
        "regles": ["highs", "min_charge_max", "min_redispatch", "prorata"],
        "tolerance_zero_fraction": 1e-9, "tolerance_non_unicite": "1e-6 * max(1, C_l)",
        "tolerance_service": "1e-9 * max(1, demande_ilot)", "quantile": "numpy linear"}
    version = environnement()
    hashes = empreintes()
    if args.resume:
        metadata = json.loads((sortie / "metadata.json").read_text(encoding="utf-8"))
        if metadata["parametres"] != params or metadata["environnement"] != version or metadata["sha256_sources"] != hashes:
            raise ValueError("Reprise refusée : paramètres, environnement ou sources différents.")
        if (sortie / "complete.json").exists():
            print(f"Run déjà complet : {sortie}")
            return 0
    else:
        sortie.mkdir(parents=True, exist_ok=False)
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RACINE, text=True).strip()
        _sauver(sortie / "metadata.json", {"format": "diagnostic-departage-n1-v1", "run_id": args.run_id,
            "debut_utc": datetime.now(timezone.utc).isoformat(), "git_head": sha,
            "parametres": params, "environnement": version, "sha256_sources": hashes})
    travaux = [(nom, topo, sortie, params["regles"]) for nom, topo in reseaux(args.audit_reference)]
    if args.workers == 1:
        groupes = list(map(traiter_reseau, travaux))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            groupes = list(pool.map(traiter_reseau, travaux))
    resultats = [resultat for groupe in groupes for resultat in groupe]
    resumes = [resultat["resume"] for resultat in resultats]
    retraits = [ligne_retrait(resultat["resume"], detail)
                for resultat in resultats for detail in resultat["retraits"]]
    sauver_csv(sortie / "comparatif.csv", resumes)
    sauver_csv(sortie / "retraits.csv", retraits)
    _sauver(sortie / "complete.json", {"fin_utc": datetime.now(timezone.utc).isoformat(), "n_lignes_csv": len(resumes)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
