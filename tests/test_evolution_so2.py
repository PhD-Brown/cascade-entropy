"""Extensions SO2 de `evolution` : observables, fluctuations régionales, G, reprise.

Les valeurs par défaut doivent reproduire exactement le comportement antérieur
(couvert par `test_evolution.py`) ; on vérifie ici les nouveautés :

- observables par jour cohérentes avec la `Journee` sous-jacente ;
- fluctuation régionale : un facteur commun par région, dans [1 − g, 1 + g] ;
- `marge_pour_G` : Éq. (5) de Carreras et al. (2004) ;
- reprise : 2n jours d'un coup == n jours + n jours à partir de l'état renvoyé.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.carreras import configuration_arbre, groupes_regions
from cascade_entropy.dispatch import DEPARTAGE_EXTERIEUR_DABORD
from cascade_entropy.evolution import (
    EtatEvolution,
    Parametres,
    etat_initial,
    marge_pour_G,
    puissance_max_pour_marge,
    series,
    simuler,
)

GRAINE = 20261005


@pytest.fixture(scope="module")
def cfg46():
    return configuration_arbre(46)


def _parametres(cfg, **changements):
    demande = 0.8 * cfg.p_c
    base = dict(
        mode="auto_organise", demande_initiale=demande,
        puissance_max=puissance_max_pour_marge(cfg.reseau, demande, 0.3),
        g=0.9, p0=1e-3, p1=1.0, limites=cfg.limites, marge_seuil=0.27,
        mu=1.05, departage=DEPARTAGE_EXTERIEUR_DABORD,
    )
    base.update(changements)
    return Parametres(**base)


# --------------------------------------------------------------------------
# Observables
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fluctuation", ["noeud", "regionale"])
def test_observables_coherentes(cfg46, fluctuation):
    s = series(simuler(cfg46.reseau, 400, _parametres(cfg46, fluctuation=fluctuation),
                       np.random.default_rng(GRAINE)))
    np.testing.assert_allclose(s["fraction_delestee"],
                               s["delestage_total"] / s["demande_totale"])
    assert np.all((s["fraction_delestee"] >= -1e-12) & (s["fraction_delestee"] <= 1 + 1e-12))
    np.testing.assert_array_equal(s["n_lignes_tombees"],
                                  s["n_avaries_surcharge"] + s["n_avaries_p0"])
    jours = s["jour"]
    lambda_jour = 1.018 ** (1 / 365)
    np.testing.assert_allclose(s["demande_moyenne"],
                               0.8 * cfg46.p_c * np.exp((lambda_jour - 1) * jours))
    # Des avaries des deux types doivent apparaître sur 400 jours avec p0 = 1e-3.
    assert s["n_avaries_p0"].sum() > 0


def test_sans_p1_aucune_avarie_par_surcharge(cfg46):
    s = series(simuler(cfg46.reseau, 200, _parametres(cfg46, p1=0.0),
                       np.random.default_rng(GRAINE)))
    assert s["n_avaries_surcharge"].sum() == 0


# --------------------------------------------------------------------------
# Fluctuation régionale
# --------------------------------------------------------------------------


def test_fluctuation_regionale_facteur_commun_par_region(cfg46):
    """Mode independant, p0 = p1 = 0 : la demande tirée se lit directement."""
    p = _parametres(cfg46, mode="independant", fluctuation="regionale",
                    p0=0.0, p1=0.0, n_regions=3)
    rng = np.random.default_rng(GRAINE)
    jours = simuler(cfg46.reseau, 50, p, rng)
    groupes = groupes_regions(cfg46.reseau, 3)
    # Rejoue les tirages pour reconstruire la demande de chaque jour.
    rng2 = np.random.default_rng(GRAINE)
    base = p.demande_initiale / cfg46.n_charges
    for j in jours:
        facteurs = rng2.uniform(1 - p.g, 1 + p.g, size=3)
        rng2.uniform(1.0, 1.0, size=cfg46.n_charges)  # tirage neutre de journee (g = 0)
        rng2.random(cfg46.reseau.n_lignes)           # avaries p0 (p0 = 0)
        attendu = base * facteurs[groupes]
        assert j.demande_totale == pytest.approx(attendu.sum(), rel=1e-12)
        assert np.all((facteurs >= 1 - p.g) & (facteurs <= 1 + p.g))


def test_fluctuation_regionale_une_region_fait_varier_toute_la_demande(cfg46):
    p = _parametres(cfg46, mode="independant", fluctuation="regionale",
                    p0=0.0, p1=0.0, n_regions=1)
    s = series(simuler(cfg46.reseau, 300, p, np.random.default_rng(GRAINE)))
    rel = s["demande_totale"] / p.demande_initiale
    # Un seul facteur U[0.1, 1.9] : écart-type 0.9/sqrt(3) ≈ 0.52.
    assert rel.std() == pytest.approx(0.9 / np.sqrt(3), rel=0.15)
    assert rel.min() >= 0.1 - 1e-12 and rel.max() <= 1.9 + 1e-12


@pytest.mark.parametrize("changement,motif", [
    ({"fluctuation": "region"}, "fluctuation"),
    ({"fluctuation": "regionale", "g": 1.5}, "g"),
    ({"fluctuation": "regionale", "n_regions": 2}, "N_F"),
    ({"fluctuation": "regionale", "n_regions": 3.0}, "n_regions"),
])
def test_parametres_de_fluctuation_invalides(cfg46, changement, motif):
    with pytest.raises(ValueError, match=motif):
        simuler(cfg46.reseau, 2, _parametres(cfg46, **changement),
                np.random.default_rng(GRAINE))


# --------------------------------------------------------------------------
# Paramètre G (Éq. 5)
# --------------------------------------------------------------------------


def test_marge_pour_G():
    assert marge_pour_G(1.0, 0.9) == pytest.approx(0.9)
    assert marge_pour_G(0.25, 0.6) == pytest.approx(0.15)
    assert marge_pour_G(0.0, 0.5) == 0.0
    for G, g in [(-0.1, 0.5), (1.0, 0.0), (1.0, 1.2), (True, 0.5)]:
        with pytest.raises(ValueError):
            marge_pour_G(G, g)


# --------------------------------------------------------------------------
# Reprise d'une simulation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fluctuation", ["noeud", "regionale"])
def test_reprise_identique_a_un_run_continu(cfg46, fluctuation):
    p = _parametres(cfg46, fluctuation=fluctuation)
    continu = series(simuler(cfg46.reseau, 600, p, np.random.default_rng(GRAINE)))

    rng = np.random.default_rng(GRAINE)
    jours_a, etat = simuler(cfg46.reseau, 250, p, rng, retourner_etat=True)
    # Restauration explicite de l'état aléatoire, comme le fera un script.
    rng_b = np.random.default_rng()
    rng_b.bit_generator.state = rng.bit_generator.state
    jours_b = simuler(cfg46.reseau, 350, p, rng_b, etat=etat)
    morceaux = series(jours_a + jours_b)

    assert etat.jour == 250
    for cle in continu:
        np.testing.assert_array_equal(continu[cle], morceaux[cle], err_msg=cle)


def test_etat_final_et_etat_non_modifie(cfg46):
    p = _parametres(cfg46)
    depart = etat_initial(cfg46.reseau, p)
    copie = EtatEvolution(depart.jour, depart.limites.copy(), depart.puissance_max.copy())
    _, fin = simuler(cfg46.reseau, 300, p, np.random.default_rng(GRAINE),
                     etat=depart, retourner_etat=True)
    np.testing.assert_array_equal(depart.limites, copie.limites)
    np.testing.assert_array_equal(depart.puissance_max, copie.puissance_max)
    assert fin.jour == 300
    assert np.all(fin.limites >= depart.limites)
    assert np.all(fin.puissance_max >= depart.puissance_max)


def test_etat_incompatible_refuse(cfg46):
    p = _parametres(cfg46)
    mauvais = EtatEvolution(0, np.ones(3), np.ones(12))
    with pytest.raises(ValueError, match="état"):
        simuler(cfg46.reseau, 2, p, np.random.default_rng(GRAINE), etat=mauvais)
    with pytest.raises(TypeError):
        simuler(cfg46.reseau, 2, p, np.random.default_rng(GRAINE), etat={"jour": 0})
