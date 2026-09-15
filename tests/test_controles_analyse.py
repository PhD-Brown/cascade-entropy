"""Cas limites et comptes indépendants pour les outils d'analyse."""

import numpy as np
import pytest

from cascade_entropy.entropie import permutation, echantillon, granulariser, multiechelle
from cascade_entropy.indicateurs import hurst, charge_moyenne, charge_maximale


@pytest.mark.parametrize("fonction", [permutation, echantillon, multiechelle,
                                    hurst, charge_moyenne, charge_maximale])
@pytest.mark.parametrize("invalide", [np.nan, np.inf, -np.inf])
def test_donnees_non_finies_rejetees(fonction, invalide):
    serie = np.arange(128, dtype=float)
    serie[17] = invalide
    with pytest.raises(ValueError, match="NaN|infinies"):
        fonction(serie)


def test_egalites_departagees_par_ordre_temporel():
    assert permutation(np.ones(10)) == 0
    # Motifs stables de [0,0,1,0,0] : 012, 021, 120, équiprobables.
    assert permutation([0, 0, 1, 0, 0]) == pytest.approx(np.log(3) / np.log(6))


def test_sampen_compte_independant():
    x = np.array([0., 0., 1., 0., 0., 1., 0., 1.])
    m, seuil = 2, 0.5
    b = a = 0
    for i in range(len(x) - m):
        for j in range(i + 1, len(x) - m):
            if all(abs(x[i + k] - x[j + k]) <= seuil for k in range(m)):
                b += 1
                a += abs(x[i + m] - x[j + m]) <= seuil
    assert 0 < a < b
    assert echantillon(x, m=m, r=seuil, ecart_reference=1) == pytest.approx(-np.log(a / b))


def test_mse_reference_fixe_et_reste_ignore():
    x = np.random.default_rng(7).normal(size=256)
    tau, valeurs = multiechelle(x, echelles=3, r=0.5)
    attendu = echantillon(x[:255].reshape(-1, 3).mean(axis=1),
                         r=0.5, ecart_reference=x.std())
    assert tau[-1] == 3
    assert valeurs[-1] == pytest.approx(attendu)


@pytest.mark.parametrize("fonction,options", [
    (permutation, {"dimension": 1}), (permutation, {"delai": 0}),
    (echantillon, {"m": 0}), (echantillon, {"r": -1}),
    (granulariser, {"echelle": 0}), (multiechelle, {"echelles": 0}),
])
def test_parametres_invalides(fonction, options):
    with pytest.raises(ValueError):
        fonction(np.arange(128.), **options)
