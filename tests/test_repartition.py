"""Validation de l'entropie de répartition et des indicateurs de référence.

L'entropie de répartition porte sur un instant, pas sur une dynamique : elle
mesure la concentration de la puissance entre les lignes. C'est l'observable
retenue dans SO2 parce qu'elle est disponible à chaque pas de temps et ne
comporte pas de zéros exacts, contrairement au délestage.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.entropie import (
    entropie_repartition,
    nombre_effectif,
    shannon,
)
from cascade_entropy.indicateurs import (
    distribution_nulle_hurst,
    exposant_loi_de_puissance,
    intervalle_nul,
    survie,
)

GRAINE = 20260915


# --------------------------------------------------------------------------
# Entropie de répartition
# --------------------------------------------------------------------------


def test_shannon_uniforme_vaut_log_n():
    for n in (2, 5, 64):
        assert shannon(np.full(n, 1.0 / n)) == pytest.approx(np.log(n))


def test_shannon_distribution_certaine_est_nulle():
    p = np.zeros(8)
    p[3] = 1.0
    assert shannon(p) == pytest.approx(0.0)


def test_shannon_rejette_une_somme_differente_de_un():
    with pytest.raises(ValueError):
        shannon(np.full(4, 0.1))


def test_nombre_effectif_egale_le_compte_si_reparti_egalement():
    """Dix lignes portant la même charge donnent exactement dix lignes effectives."""
    for n in (3, 10, 45):
        assert nombre_effectif(np.ones(n)) == pytest.approx(n)


def test_nombre_effectif_vaut_un_si_tout_passe_par_un_element():
    valeurs = np.zeros(10)
    valeurs[0] = 42.0
    assert nombre_effectif(valeurs) == pytest.approx(1.0)


def test_concentration_reduit_le_nombre_effectif():
    """Même puissance totale, répartition différente, fragilité différente."""
    reparti = nombre_effectif(np.full(5, 20.0))
    concentre = nombre_effectif(np.array([80.0, 5.0, 5.0, 5.0, 5.0]))
    assert reparti == pytest.approx(5.0)
    assert concentre < reparti / 2


def test_repartition_insensible_au_signe_des_flux():
    """Un flux négatif indique le sens de circulation, pas une charge négative."""
    flux = np.array([3.0, -7.0, 2.0, -1.0])
    assert entropie_repartition(flux) == pytest.approx(entropie_repartition(np.abs(flux)))


def test_repartition_invariante_par_changement_d_unite():
    valeurs = np.array([3.0, 7.0, 2.0, 1.0])
    assert entropie_repartition(valeurs) == pytest.approx(entropie_repartition(1000 * valeurs))


def test_repartition_normalisee_entre_zero_et_un():
    rng = np.random.default_rng(GRAINE)
    valeurs = rng.uniform(0.1, 5.0, 45)
    h = entropie_repartition(valeurs, normaliser=True)
    assert 0.0 < h < 1.0
    assert entropie_repartition(np.ones(45), normaliser=True) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Distribution nulle du Hurst
# --------------------------------------------------------------------------


def test_distribution_nulle_ne_contient_pas_un_demi():
    """Le biais de R/S déplace la référence : 0,5 n'est pas la bonne cible."""
    rng = np.random.default_rng(GRAINE)
    moyenne, bas, haut = intervalle_nul(4096, n_tirages=60, rng=rng)
    assert moyenne > 0.5
    assert bas > 0.5          # même la borne basse dépasse la valeur théorique
    assert haut - bas < 0.15  # l'intervalle reste utilisable comme référence


def test_distribution_nulle_reproductible():
    a = distribution_nulle_hurst(2048, 5, np.random.default_rng(3))
    b = distribution_nulle_hurst(2048, 5, np.random.default_rng(3))
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# Queue en loi de puissance
# --------------------------------------------------------------------------


def loi_de_puissance(n, alpha, rng):
    """Tirage exact d'une loi de puissance de densité x**(-alpha) sur [1, +inf)."""
    return (1.0 - rng.random(n)) ** (-1.0 / (alpha - 1.0))


@pytest.mark.parametrize("alpha_vrai", [1.6, 2.0, 2.5])
def test_exposant_retrouve_sur_loi_de_puissance_connue(alpha_vrai):
    """L'exposant 1,6 est celui rapporté pour le modèle OPA sur un arbre."""
    rng = np.random.default_rng(GRAINE)
    estime, incertitude, n = exposant_loi_de_puissance(loi_de_puissance(20000, alpha_vrai, rng))
    assert abs(estime - alpha_vrai) < 4 * incertitude
    assert n == 20000


def test_exposant_rejette_un_echantillon_trop_petit():
    with pytest.raises(ValueError):
        exposant_loi_de_puissance(np.array([1.0, 2.0, 3.0]))


def test_survie_decroissante_et_bornee():
    rng = np.random.default_rng(GRAINE)
    valeurs, probabilites = survie(loi_de_puissance(500, 2.0, rng))
    assert np.all(np.diff(valeurs) >= 0)
    assert np.all(np.diff(probabilites) <= 0)
    assert probabilites.max() == pytest.approx(1.0)
    assert probabilites.min() > 0
