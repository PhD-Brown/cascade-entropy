"""
Validation du module `reseau`.

Critères du plan de travail : somme des injections nulle, réseau connexe, matrice
de flux de la bonne dimension.

S'y ajoute une vérification analytique propre aux arbres. Dans un arbre, le
chemin entre deux nœuds est unique : le flux d'une ligne est donc entièrement
déterminé par les injections du sous-arbre qu'elle alimente, indépendamment des
réactances. C'est un résultat exact, qui permet de valider le calcul des flux
sans recourir à une autre implémentation.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.reseau import (
    Reseau,
    arbre,
    degres,
    est_connexe,
    flux,
    limites_depuis_cas_de_base,
    matrice_de_flux,
    matrice_incidence,
    taux_de_charge,
)


# --------------------------------------------------------------------------
# Réseaux d'essai
# --------------------------------------------------------------------------


def chaine_trois_noeuds(reactances=(1.0, 1.0)) -> Reseau:
    """Trois nœuds en série : 0 - 1 - 2. Cas résoluble à la main."""
    return Reseau(
        n_noeuds=3,
        lignes=np.array([[0, 1], [1, 2]]),
        reactances=np.array(reactances, dtype=float),
        generateurs=np.array([0]),
        charges=np.array([1, 2]),
        reference=0,
    )


def sous_arbre(reseau: Reseau, racine: int) -> set[int]:
    """Nœuds situés sous `racine`, l'arbre étant orienté depuis le nœud 0."""
    enfants: dict[int, list[int]] = {}
    for parent, enfant in reseau.lignes:
        enfants.setdefault(int(parent), []).append(int(enfant))

    vus, pile = set(), [racine]
    while pile:
        courant = pile.pop()
        vus.add(courant)
        pile.extend(enfants.get(courant, []))
    return vus


# --------------------------------------------------------------------------
# Topologie
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "generations, attendu", [(1, 4), (2, 10), (3, 22), (4, 46), (5, 94), (6, 190)]
)
def test_tailles_des_arbres(generations, attendu):
    """
    Vérifie que la construction des arbres reproduit bien les tailles attendues selon la
    littérature de référence, ce qui permet de valider la génération du réseau.
    """
    assert arbre(generations).n_noeuds == attendu


def test_arbre_connexe():
    """
    Vérifie qu'un arbre généré par la fonction est bien connexe, c'est-à-dire qu'il relie
    tous les nœuds du réseau sans composante isolée.
    """
    assert est_connexe(arbre(4))


def test_arbre_a_le_bon_nombre_de_lignes():
    """
    Vérifie que, pour un arbre, le nombre de lignes est exactement n-1, propriété fondamentale
    des graphes arborescents.
    """
    reseau = arbre(4)
    assert reseau.n_lignes == reseau.n_noeuds - 1


def test_noeuds_internes_de_degre_trois():
    """
    Vérifie la topologie du modèle retenu : les nœuds internes ont degré 3, la racine aussi,
    et les feuilles sont de degré 1, ce qui correspond à l'arbre de référence.
    """
    reseau = arbre(4)
    internes = (reseau.niveaux > 0) & (reseau.niveaux < reseau.niveaux.max())
    assert np.all(degres(reseau)[internes] == 3)
    assert degres(reseau)[0] == 3            # racine
    assert np.all(degres(reseau)[reseau.niveaux == reseau.niveaux.max()] == 1)  # feuilles


def test_generateurs_et_charges_partitionnent_les_noeuds():
    """
    Vérifie que les générateurs et les charges partitionnent bien l'ensemble des nœuds,
    sans recouvrement et avec les générateurs placés au bon niveau hiérarchique.
    """
    reseau = arbre(4, niveau_generateurs=3)
    assert set(reseau.generateurs) | set(reseau.charges) == set(range(reseau.n_noeuds))
    assert not set(reseau.generateurs) & set(reseau.charges)
    assert np.all(reseau.niveaux[reseau.generateurs] == 3)


def test_reseau_deconnecte_detecte():
    """
    Vérifie que le test de connexité détecte bien un réseau partiellement isolé, même si un
    générateur et des charges existent encore dans le graphe.
    """
    isole = Reseau(
        n_noeuds=3,
        lignes=np.array([[0, 1]]),
        reactances=np.array([1.0]),
        generateurs=np.array([0]),
        charges=np.array([1, 2]),
    )
    assert not est_connexe(isole)


# --------------------------------------------------------------------------
# Flux
# --------------------------------------------------------------------------


def test_matrice_de_flux_de_bonne_dimension():
    """
    Vérifie que la matrice de flux a bien la dimension attendue : nombre de lignes par
    nombre de nœuds moins un, puisque les flux sont définis sur les branches de l'arbre.
    """
    reseau = arbre(4)
    A = matrice_de_flux(reseau)
    assert A.shape == (reseau.n_lignes, reseau.n_noeuds - 1)


def test_chaine_resolue_a_la_main():
    """
    Vérifie la résolution analytique d'un cas simple en chaîne : une injection en 0 et une
    consommation en 2 impose un flux unitaire dans chaque ligne du chemin.
    """
    reseau = chaine_trois_noeuds()
    f = flux(reseau, np.array([1.0, 0.0, -1.0]))
    assert np.allclose(f, [1.0, 1.0])


def test_chaine_avec_soutirage_intermediaire():
    """
    Vérifie le cas où une consommation intermédiaire existe. Le flux entre la source et le
    nœud central doit couvrir la puissance nette des nœuds en aval.
    """
    reseau = chaine_trois_noeuds()
    f = flux(reseau, np.array([2.0, -1.0, -1.0]))
    assert np.allclose(f, [2.0, 1.0])


def test_flux_independants_des_reactances_sur_un_arbre():
    """
    Vérifie le résultat exact pour un arbre : comme le chemin entre deux nœuds est unique,
    les réactances ne changent pas le flux, seules les limites de transport peuvent le restreindre.
    """
    injections = np.array([1.0, 0.0, -1.0])
    egales = flux(chaine_trois_noeuds((1.0, 1.0)), injections)
    inegales = flux(chaine_trois_noeuds((0.3, 7.0)), injections)
    assert np.allclose(egales, inegales)


def test_conservation_sur_arbre_complet():
    """
    Vérifie la conservation de la puissance sur un arbre complet. Chaque ligne doit transporter
    exactement la puissance nette du sous-arbre qu'elle sépare, ce qui est une propriété exacte.
    """
    reseau = arbre(4, niveau_generateurs=3)
    rng = np.random.default_rng(20260915)

    injections = np.zeros(reseau.n_noeuds)
    injections[reseau.charges] = -rng.uniform(0.5, 1.5, size=reseau.charges.size)
    injections[reseau.generateurs] = (
        -injections[reseau.charges].sum() / reseau.generateurs.size
    )

    f = flux(reseau, injections)
    for ligne, (_, enfant) in enumerate(reseau.lignes):
        aval = sorted(sous_arbre(reseau, int(enfant)))
        assert np.isclose(f[ligne], -injections[aval].sum(), atol=1e-9)


def test_injections_desequilibrees_rejetees():
    """
    Vérifie que le calcul de flux rejette une injection déséquilibrée, car de la somme des
    injections doit être nulle pour satisfaire la conservation de la puissance.
    """
    with pytest.raises(ValueError):
        flux(chaine_trois_noeuds(), np.array([1.0, 0.0, 0.0]))


def test_matrice_incidence_coherente():
    """
    Vérifie que la matrice d'incidence a bien une structure de graphe : chaque ligne contient
    un +1 et un -1, et la somme sur chaque ligne est nulle.
    """
    reseau = arbre(3)
    M = matrice_incidence(reseau)
    assert M.shape == (reseau.n_lignes, reseau.n_noeuds)
    assert np.all(M.sum(axis=1) == 0)        # +1 et -1 par ligne
    assert np.all(np.abs(M).sum(axis=1) == 2)


# --------------------------------------------------------------------------
# Limites et taux de charge
# --------------------------------------------------------------------------


def test_cas_de_base_sans_ligne_saturee():
    """
    Vérifie le critère du plan de travail : à faible charge, aucune ligne ne doit être saturée,
    et le taux de charge maximal reste strictement inférieur à 1 pour un cas de base bien choisi.
    """
    reseau = arbre(4, niveau_generateurs=3)
    rng = np.random.default_rng(1)

    injections = np.zeros(reseau.n_noeuds)
    injections[reseau.charges] = -rng.uniform(0.5, 1.5, size=reseau.charges.size)
    injections[reseau.generateurs] = (
        -injections[reseau.charges].sum() / reseau.generateurs.size
    )

    limites = limites_depuis_cas_de_base(reseau, injections, marge=1.5)
    taux = taux_de_charge(flux(reseau, injections), limites)
    assert taux.max() < 1.0
    assert np.isclose(taux[np.argmax(np.abs(flux(reseau, injections)))], 1 / 1.5)
