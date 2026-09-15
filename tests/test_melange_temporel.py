"""Contrôle rapide de l'expérience, sans imposer le résultat scientifique."""

import runpy
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "02_melange_temporel.py"
analyser = runpy.run_path(str(SCRIPT))["analyser"]


def test_reproductibilite_et_nombre_de_melanges():
    options = dict(n=256, echelles=2, r=0.5, graine=12, progression=False)
    a = analyser(n_melanges=2, **options)
    b = analyser(n_melanges=3, **options)
    assert np.array_equal(a["serie"], b["serie"])
    assert np.array_equal(a["melanges"], b["melanges"][:2])
    assert a["melanges"].shape == (2, 2)
    assert np.allclose(a["ecart_type"], a["melanges"].std(axis=0, ddof=1))
