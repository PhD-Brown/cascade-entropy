"""
Cas limites et comptes indépendants pour les outils d'analyse.
"""

import numpy as np
import pytest

from cascade_entropy.entropie import permutation, echantillon, granulariser, multiechelle
from cascade_entropy.indicateurs import hurst, charge_moyenne, charge_maximale


@pytest.mark.parametrize("fonction", [permutation, echantillon, multiechelle,
                                    hurst, charge_moyenne, charge_maximale])
@pytest.mark.parametrize("invalide", [np.nan, np.inf, -np.inf])
def test_donnees_non_finies_rejetees(fonction, invalide):
    """
    Vérifie que toutes les fonctions d'analyse rejettent explicitement une série
    contenant au moins une valeur NaN ou infinie, en levant ValueError.
    Cela empêche les calculs d'entropie/indicateurs sur des données invalides.
    """
    serie = np.arange(128, dtype=float)
    serie[17] = invalide
    with pytest.raises(ValueError, match="NaN|infinies"):
        fonction(serie)


def test_egalites_departagees_par_ordre_temporel():
    """
    Vérifie qu'une série constante a une permutation nulle : il n'y a pas
    d'information dans l'ordre des valeurs puisqu'elles sont toutes identiques.
    Pour [0,0,1,0,0], les motifs de longueur 3 peuvent être ordonnés de 6 façons,
    mais seules 3 motifs distincts sont comparables dans l'ordre temporel.
    Le score attendu correspond au rapport log(3)/log(6), qui sépare les égalités
    selon l'ordre temporel plutôt que par simple présence de valeurs.
    """
    assert permutation(np.ones(10)) == 0
    # Motifs stables de [0,0,1,0,0] : 012, 021, 120, équiprobables.
    assert permutation([0, 0, 1, 0, 0]) == pytest.approx(np.log(3) / np.log(6))


def test_sampen_compte_independant():
    """
    Vérifie la définition de la Sample Entropy sur un exemple explicite.
    On compte à la main les paires de motifs comparables (b) et celles qui
    restent comparables après un pas supplémentaire (a). La valeur théorique
    attendue est -log(a/b), ce que la fonction echantillon doit reproduire.
    """
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
    """
    Vérifie que la multiscale entropy est calculée sur des moyennes de blocs,
    avec la bonne échelle, et que le dernier bloc incomplet est ignoré.
    La valeur attendue est recalculée manuellement sur x[:255].reshape(-1, 3).mean(axis=1)
    en utilisant l'écart-type de la série comme référence.
    """
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
    """
    Vérifie que les paramètres hors limites ne sont pas acceptés : la fonction
    doit lever ValueError pour éviter des calculs mathématiquement invalides.
    """
    with pytest.raises(ValueError):
        fonction(np.arange(128.), **options)
