"""Validation du module `cascade`.

Critères de la docstring du module : sans probabilité d'avarie, `journee()` doit
coïncider avec un simple dispatch ; à graine fixée, tout est reproductible ; une
ligne hors service n'est jamais retenue comme candidate ; la boucle termine.

Deux cas se résolvent à la main sur la chaîne à 3 nœuds (générateur en 0, charges
en 1 et 2). Ils ne dépendent pas de la dégénérescence du dispatch : la ligne 0
porte toute la puissance servie, quelle que soit la répartition entre les deux
charges.

Ces tests ne démontrent pas la validité physique de la cascade : sur un arbre,
une avarie isole un sous-arbre au lieu de redistribuer les flux.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.cascade import journee
from cascade_entropy.dispatch import SEUIL_SATURATION, demande_uniforme, resoudre
from cascade_entropy.reseau import Reseau, arbre, limites_par_niveau, matrice_de_flux

GRAINE = 20260915


def chaine(limites=(1.5, 10.0)) -> Reseau:
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
    return reseau, base * facteur, p_max, p_c


# --------------------------------------------------------------------------
# Sans avarie : équivalence avec un simple dispatch
# --------------------------------------------------------------------------


def test_sans_avarie_egale_un_dispatch_direct():
    """
    Vérifie le critère central : avec p0 = 0 et p1 = 0, la journée coïncide avec
    `resoudre()` sur la demande tirée ce jour-là, sans aucune avarie, même quand la
    demande fluctue et que des lignes sont saturées.
    """
    reseau, limites, p_max, p_c = arbre_calibre()
    moyenne = demande_uniforme(reseau, 1.6 * p_c)
    resultat = journee(reseau, moyenne, p_max, p0=0.0, p1=0.0, g=0.3,
                       rng=np.random.default_rng(GRAINE), limites=limites)
    direct = resoudre(reseau, resultat.demande, limites=limites, puissance_max=p_max)

    assert resultat.lignes_tombees.size == 0
    assert not resultat.hors_service.any()
    assert resultat.n_iterations == 1
    assert np.array_equal(resultat.solution.flux, direct.flux)
    assert resultat.delestage_total == direct.delestage_total


def test_g_nul_ne_fait_pas_fluctuer_la_demande():
    """Vérifie que g = 0 laisse la demande moyenne intacte."""
    reseau, limites, p_max, p_c = arbre_calibre()
    moyenne = demande_uniforme(reseau, 0.8 * p_c)
    resultat = journee(reseau, moyenne, p_max, 0.0, 0.0, 0.0,
                       np.random.default_rng(GRAINE), limites=limites)
    assert np.array_equal(resultat.demande, moyenne)


def test_fluctuation_dans_l_intervalle_1_moins_g_1_plus_g():
    """Vérifie que chaque charge reste dans [(1 - g), (1 + g)] × la demande moyenne."""
    reseau, limites, p_max, p_c = arbre_calibre()
    moyenne = demande_uniforme(reseau, 0.8 * p_c)
    g = 0.9
    resultat = journee(reseau, moyenne, p_max, 0.0, 0.0, g,
                       np.random.default_rng(GRAINE), limites=limites)
    assert np.all(resultat.demande >= (1 - g) * moyenne)
    assert np.all(resultat.demande <= (1 + g) * moyenne)
    assert not np.array_equal(resultat.demande, moyenne)


# --------------------------------------------------------------------------
# Cas résolus à la main
# --------------------------------------------------------------------------


def test_chaine_ligne_saturee_tombe_et_ne_retombe_pas():
    """
    Ligne 0 limitée à 1,5 pour une demande de 2 : elle sature. Avec p1 = 1 elle
    tombe, le générateur est isolé et tout est délesté (2, à 1,5e-6 près, qui
    passent encore par la limite dégradée). Après sa chute, son taux de charge
    apparent vaut 1 : elle ne doit pas être retenue une seconde fois.
    """
    resultat = journee(chaine(), np.array([1.0, 1.0]), np.array([10.0]),
                       p0=0.0, p1=1.0, g=0.0, rng=np.random.default_rng(GRAINE))
    assert [v.tolist() for v in resultat.avaries_par_surcharge] == [[0]]
    assert resultat.hors_service.tolist() == [True, False]
    assert resultat.n_iterations == 2
    assert resultat.delestage_total == pytest.approx(2.0, abs=1e-5)
    # Le taux apparent de la ligne morte est ~1 ; M_max l'ignore.
    assert resultat.solution.taux_de_charge[0] >= SEUIL_SATURATION
    assert resultat.taux_maximal < SEUIL_SATURATION


def test_chaine_p0_egal_1_toutes_les_lignes_tombent():
    """Vérifie que p0 = 1 fait tomber toutes les lignes avant le dispatch."""
    resultat = journee(chaine(), np.array([1.0, 1.0]), np.array([10.0]),
                       p0=1.0, p1=1.0, g=0.0, rng=np.random.default_rng(GRAINE))
    assert resultat.avaries_accidentelles.tolist() == [0, 1]
    assert resultat.avaries_par_surcharge == []
    assert resultat.n_iterations == 1
    assert resultat.delestage_total == pytest.approx(2.0, abs=1e-4)
    assert resultat.taux_maximal == 0.0


def test_chaine_sous_la_limite_aucune_avarie_meme_avec_p1_egal_1():
    """Vérifie qu'aucune ligne ne tombe sans saturation, même avec p1 = 1."""
    resultat = journee(chaine(limites=(10.0, 10.0)), np.array([1.0, 1.0]),
                       np.array([10.0]), p0=0.0, p1=1.0, g=0.0,
                       rng=np.random.default_rng(GRAINE))
    assert resultat.lignes_tombees.size == 0
    assert resultat.delestage_total == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Cascade complète sur l'arbre calibré
# --------------------------------------------------------------------------


def test_cascade_termine_sans_doublon_et_sans_candidate_restante():
    """
    Vérifie, avec p1 = 1 sur un arbre surchargé, que la boucle termine en au plus
    n_lignes + 1 dispatchs, qu'aucune ligne ne tombe deux fois et que, à l'arrêt,
    plus aucune ligne en service n'est saturée.
    """
    reseau, limites, p_max, p_c = arbre_calibre()
    moyenne = demande_uniforme(reseau, 1.6 * p_c)
    resultat = journee(reseau, moyenne, p_max, p0=0.0, p1=1.0, g=0.0,
                       rng=np.random.default_rng(GRAINE), limites=limites)

    tombees = resultat.lignes_tombees
    assert len(resultat.avaries_par_surcharge) >= 1
    assert len(set(tombees.tolist())) == tombees.size
    assert resultat.n_iterations == 1 + len(resultat.avaries_par_surcharge)
    assert resultat.n_iterations <= reseau.n_lignes + 1
    assert resultat.hors_service.sum() == tombees.size
    saturee = resultat.solution.taux_de_charge >= SEUIL_SATURATION
    assert not np.any(saturee & ~resultat.hors_service)


def test_reproductible_a_graine_fixee():
    """
    Vérifie que deux journées de même graine ont la même demande, la même séquence
    d'avaries, le même nombre d'itérations et le même délestage final.
    """
    reseau, limites, p_max, p_c = arbre_calibre()
    moyenne = demande_uniforme(reseau, 1.5 * p_c)

    def simuler():
        return journee(reseau, moyenne, p_max, p0=0.02, p1=0.5, g=0.3,
                       rng=np.random.default_rng(GRAINE), limites=limites)

    a, b = simuler(), simuler()
    assert np.array_equal(a.demande, b.demande)
    assert np.array_equal(a.avaries_accidentelles, b.avaries_accidentelles)
    assert len(a.avaries_par_surcharge) == len(b.avaries_par_surcharge)
    for va, vb in zip(a.avaries_par_surcharge, b.avaries_par_surcharge):
        assert np.array_equal(va, vb)
    assert a.n_iterations == b.n_iterations
    assert a.delestage_total == b.delestage_total


def test_reseau_et_limites_de_l_appelant_non_modifies():
    """Vérifie que la cascade ne mute ni le réseau ni les limites fournis."""
    reseau, limites, p_max, p_c = arbre_calibre()
    reactances, limites_avant = reseau.reactances.copy(), limites.copy()
    journee(reseau, demande_uniforme(reseau, 1.6 * p_c), p_max, p0=0.1, p1=1.0,
            g=0.0, rng=np.random.default_rng(GRAINE), limites=limites)
    assert np.array_equal(reseau.reactances, reactances)
    assert np.array_equal(limites, limites_avant)


# --------------------------------------------------------------------------
# Validation des entrées
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nom", ["p0", "p1", "g"])
@pytest.mark.parametrize("valeur", [-0.1, 1.5, float("nan"), True, "0.5", None])
def test_probabilite_invalide_rejetee(nom, valeur):
    """Vérifie que p0, p1 et g hors de [0, 1] (ou non numériques) sont rejetés."""
    parametres = {"p0": 0.0, "p1": 0.0, "g": 0.0, nom: valeur}
    with pytest.raises(ValueError):
        journee(chaine(), np.array([1.0, 1.0]), np.array([10.0]),
                rng=np.random.default_rng(GRAINE), **parametres)


def test_demande_de_mauvaise_dimension_ou_negative_rejetee():
    """Vérifie que la demande doit avoir une valeur positive par nœud de charge."""
    rng = np.random.default_rng(GRAINE)
    with pytest.raises(ValueError):
        journee(chaine(), np.array([1.0]), np.array([10.0]), 0.0, 0.0, 0.0, rng)
    with pytest.raises(ValueError):
        journee(chaine(), np.array([-1.0, 1.0]), np.array([10.0]), 0.0, 0.0, 0.0, rng)


def test_limites_absentes_ou_de_mauvaise_dimension_rejetees():
    """Vérifie qu'un réseau sans limites, ou des limites mal dimensionnées, sont rejetés."""
    rng = np.random.default_rng(GRAINE)
    sans_limites = chaine()
    sans_limites.limites = None
    with pytest.raises(ValueError):
        journee(sans_limites, np.array([1.0, 1.0]), np.array([10.0]), 0.0, 0.0, 0.0, rng)
    with pytest.raises(ValueError):
        journee(chaine(), np.array([1.0, 1.0]), np.array([10.0]), 0.0, 0.0, 0.0, rng,
                limites=np.array([1.0]))


def test_generateur_aleatoire_obligatoire():
    """Vérifie qu'un état aléatoire global (ou un entier) n'est pas accepté."""
    with pytest.raises(TypeError):
        journee(chaine(), np.array([1.0, 1.0]), np.array([10.0]), 0.0, 0.0, 0.0,
                rng=GRAINE)


# --------------------------------------------------------------------------
# Premier dispatch, conservé pour l'observation avant cascade
# --------------------------------------------------------------------------


def test_solution_initiale_est_le_dispatch_avant_toute_avarie_par_surcharge():
    """
    Sur la chaîne, le premier dispatch délestait 0,5 avec la ligne 0 à sa limite ;
    après la chute de cette ligne, le délestage final vaut 2. La solution initiale
    doit garder le premier état, et M_max initial doit être celui de ce dispatch.
    """
    resultat = journee(chaine(), np.array([1.0, 1.0]), np.array([10.0]),
                       p0=0.0, p1=1.0, g=0.0, rng=np.random.default_rng(GRAINE))
    assert resultat.solution_initiale.delestage_total == pytest.approx(0.5)
    assert resultat.solution_initiale is not resultat.solution
    assert resultat.taux_maximal_initial == pytest.approx(1.0)
    assert not resultat.hors_service_initial.any()


def test_solution_initiale_sans_avarie_est_la_solution_finale():
    """Sans avarie, le premier et le dernier dispatch sont un seul et même objet."""
    reseau, limites, p_max, p_c = arbre_calibre()
    resultat = journee(reseau, demande_uniforme(reseau, 0.8 * p_c), p_max, 0.0, 0.0, 0.3,
                       np.random.default_rng(GRAINE), limites=limites)
    assert resultat.solution_initiale is resultat.solution


def test_hors_service_initial_ne_contient_que_les_avaries_accidentelles():
    """
    Avec p0 = 1, toutes les lignes sont mortes dès le premier dispatch ; avec p1 = 1 et
    p0 = 0, la ligne qui tombe ensuite par surcharge n'y figure pas encore.
    """
    tout = journee(chaine(), np.array([1.0, 1.0]), np.array([10.0]), 1.0, 0.0, 0.0,
                   np.random.default_rng(GRAINE))
    assert tout.hors_service_initial.tolist() == [True, True]
    assert tout.taux_maximal_initial == 0.0

    cascade = journee(chaine(), np.array([1.0, 1.0]), np.array([10.0]), 0.0, 1.0, 0.0,
                      np.random.default_rng(GRAINE))
    assert cascade.hors_service.tolist() == [True, False]
    assert cascade.hors_service_initial.tolist() == [False, False]
