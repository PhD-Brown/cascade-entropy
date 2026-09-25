"""Contrôle rapide de l'expérience, sans imposer le résultat scientifique."""

import numpy as np

from cascade_entropy.controles import analyser


def test_reproductibilite_et_nombre_de_melanges():
    """
    Vérifie que la génération de mélanges temporels est reproductible pour une graine fixée,
    et que l'analyse avec un nombre de mélanges plus grand conserve exactement les deux
    premiers résultats déjà calculés. On contrôle aussi que l'écart-type retourné correspond
    bien à l'écart-type empirique des mélanges.
    """
    options = dict(n=256, echelles=2, r=0.5, graine=12, progression=False)
    a = analyser(n_melanges=2, **options)
    b = analyser(n_melanges=3, **options)
    assert np.array_equal(a["serie"], b["serie"])
    assert np.array_equal(a["melanges"], b["melanges"][:2])
    assert a["melanges"].shape == (2, 2)
    assert np.allclose(a["ecart_type"], a["melanges"].std(axis=0, ddof=1))
