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

from cascade_entropy.controles import analyser

RACINE = Path(__file__).resolve().parents[1]


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
