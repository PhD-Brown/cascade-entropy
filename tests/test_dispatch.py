"""Validation du module `dispatch`.

Critères du plan de travail : sur un réseau réduit résolu à la main, les flux
doivent coïncider ; à faible charge, la solution ne doit comporter ni délestage ni
ligne saturée.

S'y ajoute une vérification exacte au-delà de la capacité de production : la
fraction délestée doit valoir (r - 1)/r pour une demande de r fois la capacité,
puisque toute la puissance disponible est servie et que le reste est coupé.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.dispatch import resoudre, demande_uniforme
from cascade_entropy.reseau import Reseau, arbre, limites_par_niveau, matrice_de_flux

GRAINE = 20260915


def chaine(limites=(10.0, 10.0)) -> Reseau:
    """0 -- 1 -- 2, générateur en 0, charges en 1 et 2. Résoluble à la main."""
    return Reseau(
        n_noeuds=3,
        lignes=np.array([[0, 1], [1, 2]]),
        reactances=np.array([1.0, 1.0]),
        generateurs=np.array([0]),
        charges=np.array([1, 2]),
        limites=np.array(limites, dtype=float),
        reference=0,
    )


def arbre_calibre():
    """Réseau à 46 nœuds, capacités mises à l'échelle pour saturer vers 1,45."""
    reseau = arbre(4)
    A = matrice_de_flux(reseau)
    p_c = 2623.9
    p_max = np.full(reseau.generateurs.size, p_c / reseau.generateurs.size)
    base = limites_par_niveau(reseau)
    facteur = resoudre(reseau, demande_uniforme(reseau, 1.45 * p_c), limites=base,
                       puissance_max=p_max, A=A).taux_maximal
    return reseau, A, base * facteur, p_max, p_c


# --------------------------------------------------------------------------
# Cas résolubles à la main
# --------------------------------------------------------------------------


def test_chaine_resolue_a_la_main():
    """Deux charges d'une unité : la première ligne porte 2, la seconde 1."""
    solution = resoudre(chaine(), demande=np.array([1.0, 1.0]),
                        puissance_max=np.array([10.0]))
    assert np.allclose(solution.flux, [2.0, 1.0])
    assert np.allclose(solution.charge_servie, [1.0, 1.0])
    assert solution.delestage_total == pytest.approx(0.0)


def test_limite_de_ligne_force_le_delestage():
    """La ligne amont plafonnée à 1,5 impose de couper 0,5 en aval."""
    solution = resoudre(chaine(limites=(1.5, 10.0)), demande=np.array([1.0, 1.0]),
                        puissance_max=np.array([10.0]))
    assert solution.flux[0] == pytest.approx(1.5)
    assert solution.delestage_total == pytest.approx(0.5)
    assert solution.taux_maximal == pytest.approx(1.0)
    assert solution.lignes_saturees.tolist() == [0]


def test_capacite_de_production_insuffisante():
    solution = resoudre(chaine(), demande=np.array([1.0, 1.0]),
                        puissance_max=np.array([1.2]))
    assert solution.delestage_total == pytest.approx(0.8)
    assert solution.production.sum() == pytest.approx(1.2)


# --------------------------------------------------------------------------
# Invariants physiques, valables pour toute solution
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0.3, 0.9, 1.2, 1.6])
def test_injections_equilibrees(ratio):
    reseau, A, limites, p_max, p_c = arbre_calibre()
    solution = resoudre(reseau, demande_uniforme(reseau, ratio * p_c),
                        limites=limites, puissance_max=p_max, A=A)
    assert solution.injections.sum() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("ratio", [0.3, 0.9, 1.2, 1.6])
def test_limites_de_transport_jamais_depassees(ratio):
    """Le dispatch ne produit jamais de ligne au-dessus de sa limite.

    C'est la propriété qui définit la surcharge dans le modèle de cascade : une
    ligne ne peut qu'être saturée, jamais en dépassement.
    """
    reseau, A, limites, p_max, p_c = arbre_calibre()
    solution = resoudre(reseau, demande_uniforme(reseau, ratio * p_c),
                        limites=limites, puissance_max=p_max, A=A)
    assert np.all(solution.taux_de_charge <= 1.0 + 1e-9)


@pytest.mark.parametrize("ratio", [0.3, 0.9, 1.2, 1.6])
def test_charge_servie_jamais_superieure_a_la_demande(ratio):
    reseau, A, limites, p_max, p_c = arbre_calibre()
    demande = demande_uniforme(reseau, ratio * p_c)
    solution = resoudre(reseau, demande, limites=limites, puissance_max=p_max, A=A)
    assert np.all(solution.charge_servie <= demande + 1e-9)
    assert np.all(solution.delestage >= -1e-9)


# --------------------------------------------------------------------------
# Transition à la limite de production
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0.3, 0.6, 0.9, 1.0])
def test_aucun_delestage_sous_la_capacite(ratio):
    """Critère du plan : à faible charge, aucun délestage."""
    reseau, A, limites, p_max, p_c = arbre_calibre()
    solution = resoudre(reseau, demande_uniforme(reseau, ratio * p_c),
                        limites=limites, puissance_max=p_max, A=A)
    assert solution.delestage_total == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("ratio", [1.1, 1.3, 1.6, 2.0])
def test_delestage_exact_au_dela_de_la_capacite(ratio):
    """Au-delà de la capacité totale, tout l'excédent de demande est coupé."""
    reseau, A, limites, p_max, p_c = arbre_calibre()
    solution = resoudre(reseau, demande_uniforme(reseau, ratio * p_c),
                        limites=limites, puissance_max=p_max, A=A)
    fraction = solution.delestage_total / (ratio * p_c)
    assert fraction == pytest.approx((ratio - 1.0) / ratio, abs=1e-3)


def test_puissance_servie_plafonne_a_la_capacite():
    """Le plateau de puissance servie est la signature de la première transition."""
    reseau, A, limites, p_max, p_c = arbre_calibre()
    servies = [
        resoudre(reseau, demande_uniforme(reseau, r * p_c), limites=limites,
                 puissance_max=p_max, A=A).charge_servie.sum()
        for r in (1.2, 1.5, 2.0)
    ]
    assert np.allclose(servies, p_c, rtol=1e-3)


def test_taux_maximal_continue_de_croitre_apres_la_premiere_transition():
    """La seconde transition existe malgré une puissance servie constante.

    Au-delà de la limite de production, le total servi ne bouge plus, mais le
    délestage n'est pas uniforme : la répartition change et certaines lignes se
    chargent davantage. C'est le mécanisme qui produit la seconde transition.
    """
    reseau, A, limites, p_max, p_c = arbre_calibre()
    taux = [
        resoudre(reseau, demande_uniforme(reseau, r * p_c), limites=limites,
                 puissance_max=p_max, A=A).taux_maximal
        for r in (1.05, 1.2, 1.4)
    ]
    assert taux[0] < taux[1] < taux[2]


# --------------------------------------------------------------------------
# Robustesse
# --------------------------------------------------------------------------


def test_reproductible():
    reseau, A, limites, p_max, p_c = arbre_calibre()
    demande = demande_uniforme(reseau, 1.2 * p_c)
    a = resoudre(reseau, demande, limites=limites, puissance_max=p_max, A=A)
    b = resoudre(reseau, demande, limites=limites, puissance_max=p_max, A=A)
    assert np.array_equal(a.flux, b.flux)


def test_demande_negative_rejetee():
    with pytest.raises(ValueError):
        resoudre(chaine(), demande=np.array([-1.0, 1.0]), puissance_max=np.array([10.0]))


def test_demande_de_mauvaise_dimension_rejetee():
    with pytest.raises(ValueError):
        resoudre(chaine(), demande=np.array([1.0]), puissance_max=np.array([10.0]))


def test_demande_uniforme_somme_au_total():
    reseau = arbre(4)
    demande = demande_uniforme(reseau, 100.0)
    assert demande.sum() == pytest.approx(100.0)
    assert demande.size == reseau.charges.size
