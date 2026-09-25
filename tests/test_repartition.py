"""
Validation de l'entropie de répartition et des indicateurs de référence.

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
    """
    Vérifie la propriété fondamentale de l'entropie de Shannon : une distribution
    uniforme sur n catégories a une entropie égale à log(n).
    """
    for n in (2, 5, 64):
        assert shannon(np.full(n, 1.0 / n)) == pytest.approx(np.log(n))


def test_shannon_distribution_certaine_est_nulle():
    """
    Vérifie qu'une distribution entièrement concentrée sur une seule valeur a une
    entropie nulle, car il n'y a aucune incertitude sur le résultat.
    """
    p = np.zeros(8)
    p[3] = 1.0
    assert shannon(p) == pytest.approx(0.0)


def test_shannon_rejette_une_somme_differente_de_un():
    """
    Vérifie que la fonction refuse une distribution de probabilités qui ne somme pas à 1,
    car elle ne représente pas une loi de probabilité valide.
    """
    with pytest.raises(ValueError):
        shannon(np.full(4, 0.1))


def test_nombre_effectif_egale_le_compte_si_reparti_egalement():
    """
    Vérifie que le nombre effectif vaut exactement le nombre de lignes quand la charge est
    répartie uniformément entre elles.
    """
    for n in (3, 10, 45):
        assert nombre_effectif(np.ones(n)) == pytest.approx(n)


def test_nombre_effectif_vaut_un_si_tout_passe_par_un_element():
    """
    Vérifie qu'une concentration totale sur une seule ligne donne un nombre effectif égal à 1,
    ce qui correspond à une répartition complètement centralisée.
    """
    valeurs = np.zeros(10)
    valeurs[0] = 42.0
    assert nombre_effectif(valeurs) == pytest.approx(1.0)


def test_concentration_reduit_le_nombre_effectif():
    """
    Vérifie qu'une même puissance totale répartie de façon plus concentrée réduit le nombre
    effectif, ce qui traduit une plus grande fragilité du réseau.
    """
    reparti = nombre_effectif(np.full(5, 20.0))
    concentre = nombre_effectif(np.array([80.0, 5.0, 5.0, 5.0, 5.0]))
    assert reparti == pytest.approx(5.0)
    assert concentre < reparti / 2


def test_repartition_insensible_au_signe_des_flux():
    """
    Vérifie que l'entropie de répartition dépend de l'amplitude des flux, pas de leur sens.
    Un flux négatif indique simplement la direction de circulation, sans changer la concentration.
    """
    flux = np.array([3.0, -7.0, 2.0, -1.0])
    assert entropie_repartition(flux) == pytest.approx(entropie_repartition(np.abs(flux)))


def test_repartition_invariante_par_changement_d_unite():
    """
    Vérifie que le changement d'unité de mesure ne modifie pas l'entropie de répartition,
    puisqu'il s'agit d'une mesure de structure relative, pas d'une quantité absolue.
    """
    valeurs = np.array([3.0, 7.0, 2.0, 1.0])
    assert entropie_repartition(valeurs) == pytest.approx(entropie_repartition(1000 * valeurs))


def test_repartition_normalisee_entre_zero_et_un():
    """
    Vérifie que la version normalisée de l'entropie est comprise entre 0 et 1,
    avec la valeur 1 pour une répartition uniforme parfaite.
    """
    rng = np.random.default_rng(GRAINE)
    valeurs = rng.uniform(0.1, 5.0, 45)
    h = entropie_repartition(valeurs, normaliser=True)
    assert 0.0 < h < 1.0
    assert entropie_repartition(np.ones(45), normaliser=True) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Distribution nulle du Hurst
# --------------------------------------------------------------------------


def test_distribution_nulle_ne_contient_pas_un_demi():
    """
    Vérifie que la référence nulle du Hurst ne passe pas par 0,5 : le biais de R/S déplace
    la distribution de façon systématique vers des valeurs supérieures, et l'intervalle
    restant utile est bien défini pour servir de point de comparaison.
    """
    rng = np.random.default_rng(GRAINE)
    moyenne, bas, haut = intervalle_nul(4096, n_tirages=60, rng=rng)
    assert moyenne > 0.5
    assert bas > 0.5          # même la borne basse dépasse la valeur théorique
    assert haut - bas < 0.15  # l'intervalle reste utilisable comme référence


def test_distribution_nulle_reproductible():
    """
    Vérifie la reproductibilité de la distribution nulle du Hurst avec une graine fixée.
    Deux tirages identiques doivent donner exactement les mêmes valeurs.
    """
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
    """
    Vérifie que l'estimateur de l'exposant de loi de puissance retrouve bien la valeur
    connue utilisée pour générer l'échantillon, à tolérance près donnée par l'incertitude.
    """
    rng = np.random.default_rng(GRAINE)
    estime, incertitude, n = exposant_loi_de_puissance(loi_de_puissance(20000, alpha_vrai, rng))
    assert abs(estime - alpha_vrai) < 4 * incertitude
    assert n == 20000


def test_exposant_rejette_un_echantillon_trop_petit():
    """
    Vérifie que l'estimateur rejette un échantillon trop petit, car une estimation fiable
    d'une loi de puissance exige une taille d'échantillon suffisante.
    """
    with pytest.raises(ValueError):
        exposant_loi_de_puissance(np.array([1.0, 2.0, 3.0]))


def test_survie_decroissante_et_bornee():
    """
    Vérifie les propriétés de la fonction de survie : elle doit décroître avec la valeur,
    rester bornée entre 0 et 1, et représenter bien la queue de distribution.
    """
    rng = np.random.default_rng(GRAINE)
    valeurs, probabilites = survie(loi_de_puissance(500, 2.0, rng))
    assert np.all(np.diff(valeurs) >= 0)
    assert np.all(np.diff(probabilites) <= 0)
    assert probabilites.max() == pytest.approx(1.0)
    assert probabilites.min() > 0
