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
    """
    Vérifie la résolution manuelle d'un réseau simple en chaîne.
    Avec deux charges de 1 unité et une seule source, le flux attendu est 2 sur la
    première ligne et 1 sur la seconde, sans délestage ni saturation.
    """
    solution = resoudre(chaine(), demande=np.array([1.0, 1.0]),
                        puissance_max=np.array([10.0]))
    assert np.allclose(solution.flux, [2.0, 1.0])
    assert np.allclose(solution.charge_servie, [1.0, 1.0])
    assert solution.delestage_total == pytest.approx(0.0)


def test_limite_de_ligne_force_le_delestage():
    """
    Vérifie qu'une limite de ligne imposée sur le tronçon amont force un délestage
    en aval. On s'attend à ce que la ligne saturée porte exactement la limite et que
    le délestage total compense le surplus.
    """
    solution = resoudre(chaine(limites=(1.5, 10.0)), demande=np.array([1.0, 1.0]),
                        puissance_max=np.array([10.0]))
    assert solution.flux[0] == pytest.approx(1.5)
    assert solution.delestage_total == pytest.approx(0.5)
    assert solution.taux_maximal == pytest.approx(1.0)
    assert solution.lignes_saturees.tolist() == [0]


def test_capacite_de_production_insuffisante():
    """
    Vérifie le cas où la production disponible est trop faible pour couvrir la demande.
    Le délestage total correspond alors à l'écart entre la demande totale et la capacité
    de production disponible.
    """
    solution = resoudre(chaine(), demande=np.array([1.0, 1.0]),
                        puissance_max=np.array([1.2]))
    assert solution.delestage_total == pytest.approx(0.8)
    assert solution.production.sum() == pytest.approx(1.2)


# --------------------------------------------------------------------------
# Invariants physiques, valables pour toute solution
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0.3, 0.9, 1.2, 1.6])
def test_injections_equilibrees(ratio):
    """
    Vérifie le principe d'équilibre du réseau : la somme des injections est nulle,
    ce qui traduit le fait que la production et la consommation se compensent.
    """
    reseau, A, limites, p_max, p_c = arbre_calibre()
    solution = resoudre(reseau, demande_uniforme(reseau, ratio * p_c),
                        limites=limites, puissance_max=p_max, A=A)
    assert solution.injections.sum() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("ratio", [0.3, 0.9, 1.2, 1.6])
def test_limites_de_transport_jamais_depassees(ratio):
    """
    Vérifie que la solution respecte toujours les limites de transport des lignes.
    Une ligne ne peut pas dépasser sa capacité : elle est au pire saturée, jamais
    en surcharge dépassant la borne physique.
    """
    reseau, A, limites, p_max, p_c = arbre_calibre()
    solution = resoudre(reseau, demande_uniforme(reseau, ratio * p_c),
                        limites=limites, puissance_max=p_max, A=A)
    assert np.all(solution.taux_de_charge <= 1.0 + 1e-9)


@pytest.mark.parametrize("ratio", [0.3, 0.9, 1.2, 1.6])
def test_charge_servie_jamais_superieure_a_la_demande(ratio):
    """
    Vérifie les bornes physiques de la demande servie : la charge livrée ne peut pas
    dépasser la demande totale, et le délestage reste toujours positif ou nul.
    """
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
    """
    Vérifie le critère du plan de travail : à faible charge, le système doit servir
    toute la demande sans délestage, tant que la production et les limites de transport
    restent suffisantes.
    """
    reseau, A, limites, p_max, p_c = arbre_calibre()
    solution = resoudre(reseau, demande_uniforme(reseau, ratio * p_c),
                        limites=limites, puissance_max=p_max, A=A)
    assert solution.delestage_total == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("ratio", [1.1, 1.3, 1.6, 2.0])
def test_delestage_exact_au_dela_de_la_capacite(ratio):
    """
    Vérifie que, au-delà de la capacité totale, le délestage représente exactement
    la fraction excédentaire de la demande. L'ensemble de la puissance disponible est
    servi et le reste est coupé.
    """
    reseau, A, limites, p_max, p_c = arbre_calibre()
    solution = resoudre(reseau, demande_uniforme(reseau, ratio * p_c),
                        limites=limites, puissance_max=p_max, A=A)
    fraction = solution.delestage_total / (ratio * p_c)
    assert fraction == pytest.approx((ratio - 1.0) / ratio, abs=1e-3)


def test_puissance_servie_plafonne_a_la_capacite():
    """
    Vérifie la première transition du système : au-delà d'un certain seuil, la puissance
    servie se stabilise à la capacité de production, ce qui marque le plafonnement.
    """
    reseau, A, limites, p_max, p_c = arbre_calibre()
    servies = [
        resoudre(reseau, demande_uniforme(reseau, r * p_c), limites=limites,
                 puissance_max=p_max, A=A).charge_servie.sum()
        for r in (1.2, 1.5, 2.0)
    ]
    assert np.allclose(servies, p_c, rtol=1e-3)


def test_taux_maximal_continue_de_croitre_apres_la_premiere_transition():
    """
    Vérifie la seconde transition du système : même si la puissance servie plafonne,
    le taux maximal de charge continue d'augmenter, ce qui montre que le réseau se
    charge davantage en redistribution malgré un total servi constant.
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
    """
    Vérifie la reproductibilité du solveur sur une même instance et une même demande.
    Deux exécutions identiques doivent conduire aux mêmes flux et au même résultat.
    """
    reseau, A, limites, p_max, p_c = arbre_calibre()
    demande = demande_uniforme(reseau, 1.2 * p_c)
    a = resoudre(reseau, demande, limites=limites, puissance_max=p_max, A=A)
    b = resoudre(reseau, demande, limites=limites, puissance_max=p_max, A=A)
    assert np.array_equal(a.flux, b.flux)


def test_demande_negative_rejetee():
    """
    Vérifie que la fonction rejette une demande négative, car une consommation négative
    n'a pas de sens physique dans ce modèle de réseau électrique.
    """
    with pytest.raises(ValueError):
        resoudre(chaine(), demande=np.array([-1.0, 1.0]), puissance_max=np.array([10.0]))


def test_demande_de_mauvaise_dimension_rejetee():
    """
    Vérifie que les demandes de mauvaise taille sont rejetées. La dimension doit matcher
    le nombre de charges du réseau pour que le calcul soit bien posé.
    """
    with pytest.raises(ValueError):
        resoudre(chaine(), demande=np.array([1.0]), puissance_max=np.array([10.0]))


def test_demande_uniforme_somme_au_total():
    """
    Vérifie que la demande uniforme répartie sur les charges a bien une somme totale égale
    à la valeur demandée, et qu'elle a la bonne dimension.
    """
    reseau = arbre(4)
    demande = demande_uniforme(reseau, 100.0)
    assert demande.sum() == pytest.approx(100.0)
    assert demande.size == reseau.charges.size
