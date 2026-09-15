"""Contrôle de l'ordre temporel : une série 1/f et ses permutations.

Depuis la racine, après `python -m pip install -e ".[dev]"` :
    python scripts/02_melange_temporel.py

Les bandes représentent ± un écart-type entre mélanges, pas un intervalle de
confiance. L'expérience porte sur UNE réalisation originale, sans test de
significativité ni affirmation de vulnérabilité électrique.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cascade_entropy.entropie import multiechelle
from cascade_entropy.synthetiques import bruit_rose
from cascade_entropy._validation import entier_positif

RACINE = Path(__file__).resolve().parents[1]


def analyser(n=8192, n_melanges=20, echelles=10, m=2, r=0.15,
             graine=20260915, progression=True):
    """Retourne la série originale et toutes les courbes, sans écrire de fichier."""
    n = entier_positif(n, "n")
    n_melanges = entier_positif(n_melanges, "n_melanges", minimum=2)
    echelles = entier_positif(echelles, "echelles")
    m = entier_positif(m, "m")
    if not np.isfinite(r) or r <= 0:
        raise ValueError("r doit être fini et strictement positif.")
    if n // echelles < 10 * (m + 1):
        raise ValueError("Trop peu de points à la dernière échelle : augmenter n "
                         "ou réduire echelles.")
    # Deux flux indépendants : changer le nombre de mélanges ne change pas x.
    semences = np.random.SeedSequence(graine).spawn(2)
    rng_signal = np.random.default_rng(semences[0])
    rng_melanges = np.random.default_rng(semences[1])
    originale = bruit_rose(n, rng_signal)
    triee = np.sort(originale)
    if progression:
        print(f"Originale : N={n}, {echelles} échelles", flush=True)
    tau, mse_originale = multiechelle(originale, echelles=echelles, m=m, r=r)
    courbes = []
    for i in range(n_melanges):
        melangee = rng_melanges.permutation(originale)
        # Vérification exacte du multiensemble, plus forte qu'un histogramme.
        if not np.array_equal(np.sort(melangee), triee):
            raise RuntimeError("Un mélange a modifié les valeurs de la série.")
        tau_m, mse = multiechelle(melangee, echelles=echelles, m=m, r=r)
        if not np.array_equal(tau_m, tau):
            raise RuntimeError("Les échelles diffèrent entre les courbes.")
        courbes.append(mse)
        if progression:
            print(f"Mélange {i + 1}/{n_melanges} terminé", flush=True)
    courbes = np.asarray(courbes)
    if not np.isfinite(mse_originale).all() or not np.isfinite(courbes).all():
        raise ValueError("SampEn indéfinie à au moins une échelle : augmenter N "
                         "ou revoir r. Aucune courbe n'est omise de la moyenne.")
    return dict(serie=originale, echelles=tau, originale=mse_originale,
                melanges=courbes, moyenne=courbes.mean(axis=0),
                ecart_type=courbes.std(axis=0, ddof=1))


def enregistrer(resultats, dossier, parametres):
    """Enregistre données, paramètres, CSV et figure dans un même dossier."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dossier / "resultats.npz", **resultats)
    table = np.column_stack([resultats[k] for k in
                             ("echelles", "originale", "moyenne", "ecart_type")])
    np.savetxt(dossier / "courbes.csv", table, delimiter=",", comments="",
               header="echelle,mse_originale,mse_melanges_moyenne,mse_melanges_ecart_type",
               fmt="%.12g")
    # Les sources effectivement exécutées sont identifiées, même hors de Git.
    sources = [Path(__file__), RACINE / "cascade_entropy" / "entropie.py",
               RACINE / "cascade_entropy" / "synthetiques.py",
               RACINE / "cascade_entropy" / "_validation.py"]
    metadata = dict(parametres=parametres, python=platform.python_version(),
                    numpy=np.__version__, matplotlib=matplotlib.__version__,
                    ecart_type_reference=float(resultats["serie"].std(ddof=0)),
                    verification_valeurs="multiensembles identiques, comparaison exacte",
                    barres="ecart-type entre permutations, ddof=1; pas un IC",
                    portee="une realisation originale; controle descriptif de l'ordre temporel",
                    sources_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sources})
    (dossier / "parametres.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tau, moyenne, ecart = (resultats[k] for k in
                           ("echelles", "moyenne", "ecart_type"))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(tau, resultats["originale"], "o-", label="Série 1/f originale")
    ax.plot(tau, moyenne, "s-", label="Mélanges : moyenne")
    ax.fill_between(tau, moyenne - ecart, moyenne + ecart,
                    alpha=0.22, color="C1", label="Mélanges : ± un écart-type")
    ax.set(xlabel="Échelle de granularisation", ylabel="Entropie d'échantillon (nats)",
           title=f"Mêmes valeurs, ordre différent — N = {parametres['n']}")
    ax.set_xticks(tau[::max(1, len(tau) // 10)])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.text(0.5, 0.015,
             f"Une série originale ; {parametres['n_melanges']} mélanges ; "
             f"m = {parametres['m']} ; r = {parametres['r']} × σ originale\n"
             "Bande descriptive entre mélanges, pas un intervalle de confiance.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(dossier / "02_melange_temporel.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--melanges", type=int, default=20)
    parser.add_argument("--echelles", type=int, default=10)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--r", type=float, default=0.15)
    parser.add_argument("--graine", type=int, default=20260915)
    parser.add_argument("--sortie", type=Path,
                        default=RACINE / "data" / "02_melange_temporel")
    args = parser.parse_args()
    parametres = dict(n=args.n, n_melanges=args.melanges, echelles=args.echelles,
                      m=args.m, r=args.r, graine=args.graine)
    try:
        resultats = analyser(**parametres)
    except ValueError as erreur:
        parser.error(str(erreur))
    enregistrer(resultats, args.sortie, parametres)
    print("\nÉchelle | originale | mélanges (moyenne ± écart-type)")
    for i in dict.fromkeys([0, len(resultats["echelles"]) - 1]):
        print(f"{resultats['echelles'][i]:7.0f} | {resultats['originale'][i]:9.3f} | "
              f"{resultats['moyenne'][i]:.3f} ± {resultats['ecart_type'][i]:.3f}")
    print(f"Résultats : {args.sortie.resolve()}")


if __name__ == "__main__":
    main()
