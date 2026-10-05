"""Stratégies de secours du solveur (`dispatch.STRATEGIES_SOLVEUR`).

Le programme linéaire du dispatch est toujours admissible : un échec de HiGHS
est numérique. On vérifie que :

- chaque stratégie, seule, donne le même optimum « exterieur_dabord »
  (optimum unique) que l'appel historique ;
- un échec simulé du premier essai bascule sur le secours, sans changer la
  solution, et le signale (`Solution.strategie`, `Journee.n_secours`,
  série `n_secours_solveur`) ;
- une solution de secours non admissible est refusée ;
- si tout échoue, `EchecDispatch` transporte le problème complet ;
- le script SO2 relit les anciens checkpoints (sans `n_secours_solveur`) et
  sauvegarde le programme fautif en cas d'échec.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade_entropy import dispatch
from cascade_entropy.carreras import configuration_arbre
from cascade_entropy.cascade import journee
from cascade_entropy.dispatch import (
    DEPARTAGE_EXTERIEUR_DABORD,
    DEPARTAGE_HIGHS,
    STRATEGIE_HISTORIQUE,
    STRATEGIES_SOLVEUR,
    EchecDispatch,
    resoudre,
)

GRAINE = 20261005
RACINE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg46():
    return configuration_arbre(46)


def _cas_avec_delestage(cfg):
    """Demande uniforme au-delà du seuil de transport : délestage certain."""
    demande = np.full(cfg.n_charges, 2.5 * cfg.p_c / cfg.n_charges)
    return demande


def _echec_premiers(n_echecs: int):
    """Remplace linprog : les `n_echecs` premiers appels échouent comme HiGHS."""
    vrai = dispatch.linprog
    compteur = {"n": 0}

    def faux(*args, **kwargs):
        compteur["n"] += 1
        if compteur["n"] <= n_echecs:
            return SimpleNamespace(
                success=False, x=None,
                message="The HiGHS status code was not recognized. (HiGHS Status 15)")
        return vrai(*args, **kwargs)

    return faux, compteur


# --------------------------------------------------------------------------
# Équivalence des stratégies
# --------------------------------------------------------------------------


@pytest.mark.parametrize("strategie", STRATEGIES_SOLVEUR, ids=lambda s: s[0])
def test_chaque_strategie_donne_le_meme_optimum(cfg46, monkeypatch, strategie):
    demande = _cas_avec_delestage(cfg46)
    reference = resoudre(cfg46.reseau, demande, limites=cfg46.limites,
                         puissance_max=cfg46.puissance_max,
                         departage=DEPARTAGE_EXTERIEUR_DABORD)
    assert reference.delestage_total > 0
    monkeypatch.setattr(dispatch, "STRATEGIES_SOLVEUR", (strategie,))
    seule = resoudre(cfg46.reseau, demande, limites=cfg46.limites,
                     puissance_max=cfg46.puissance_max,
                     departage=DEPARTAGE_EXTERIEUR_DABORD)
    echelle = demande.sum()
    np.testing.assert_allclose(seule.charge_servie, reference.charge_servie,
                               atol=1e-6 * echelle)
    np.testing.assert_allclose(seule.flux, reference.flux, atol=1e-6 * echelle)
    assert seule.strategie == strategie[0]


def test_strategie_historique_en_premier_et_inchangee():
    nom, methode, echelle, options = STRATEGIES_SOLVEUR[0]
    assert nom == STRATEGIE_HISTORIQUE == "highs"
    assert (methode, echelle, options) == ("highs", False, {})


def test_solution_normale_sans_secours(cfg46):
    sol = resoudre(cfg46.reseau, _cas_avec_delestage(cfg46), limites=cfg46.limites,
                   puissance_max=cfg46.puissance_max)
    assert sol.strategie == STRATEGIE_HISTORIQUE
    assert not sol.secours


# --------------------------------------------------------------------------
# Bascule sur le secours
# --------------------------------------------------------------------------


@pytest.mark.parametrize("departage", [DEPARTAGE_HIGHS, DEPARTAGE_EXTERIEUR_DABORD])
def test_echec_du_premier_essai_bascule_sur_le_secours(cfg46, monkeypatch, departage):
    demande = _cas_avec_delestage(cfg46)
    reference = resoudre(cfg46.reseau, demande, limites=cfg46.limites,
                         puissance_max=cfg46.puissance_max, departage=departage)
    faux, compteur = _echec_premiers(1)
    monkeypatch.setattr(dispatch, "linprog", faux)
    sol = resoudre(cfg46.reseau, demande, limites=cfg46.limites,
                   puissance_max=cfg46.puissance_max, departage=departage)
    assert sol.secours and sol.strategie == "highs_echelle"
    # La puissance servie (seule grandeur unique pour "highs") est inchangée.
    assert sol.delestage_total == pytest.approx(reference.delestage_total,
                                                rel=1e-9, abs=1e-6)
    if departage == DEPARTAGE_EXTERIEUR_DABORD:
        np.testing.assert_allclose(sol.charge_servie, reference.charge_servie,
                                   atol=1e-6 * demande.sum())


def test_echec_etape_2_seule_signale_le_secours(cfg46, monkeypatch):
    """Étape 1 réussie, étape 2 en échec au premier essai."""
    demande = _cas_avec_delestage(cfg46)
    vrai = dispatch.linprog
    appels = {"n": 0}

    def faux(*args, **kwargs):
        appels["n"] += 1
        if appels["n"] == 2:  # premier essai de l'étape 2
            return SimpleNamespace(success=False, x=None, message="Status 15")
        return vrai(*args, **kwargs)

    monkeypatch.setattr(dispatch, "linprog", faux)
    sol = resoudre(cfg46.reseau, demande, limites=cfg46.limites,
                   puissance_max=cfg46.puissance_max,
                   departage=DEPARTAGE_EXTERIEUR_DABORD)
    assert sol.strategie == "highs_echelle"


def test_solution_de_secours_non_admissible_refusee(cfg46, monkeypatch):
    demande = _cas_avec_delestage(cfg46)
    vrai = dispatch.linprog
    appels = {"n": 0}

    def faux(*args, **kwargs):
        appels["n"] += 1
        if appels["n"] == 1:
            return SimpleNamespace(success=False, x=None, message="Status 15")
        r = vrai(*args, **kwargs)
        if appels["n"] == 2:  # « succès » trompeur : solution hors bornes
            return SimpleNamespace(success=True, x=r.x + 10.0, message="ok")
        return r

    monkeypatch.setattr(dispatch, "linprog", faux)
    sol = resoudre(cfg46.reseau, demande, limites=cfg46.limites,
                   puissance_max=cfg46.puissance_max)
    assert sol.strategie == "simplexe_dual"


def test_tout_echoue_leve_echec_dispatch_avec_le_probleme(cfg46, monkeypatch):
    faux, _ = _echec_premiers(10 ** 6)
    monkeypatch.setattr(dispatch, "linprog", faux)
    with pytest.raises(EchecDispatch) as info:
        resoudre(cfg46.reseau, _cas_avec_delestage(cfg46), limites=cfg46.limites,
                 puissance_max=cfg46.puissance_max)
    e = info.value
    assert isinstance(e, RuntimeError)
    assert len(e.messages) == len(STRATEGIES_SOLVEUR)
    assert set(e.probleme) == {"c", "A_ub", "b_ub", "A_eq", "b_eq",
                               "borne_inf", "borne_sup", "etape"}
    n_var = cfg46.reseau.generateurs.size + cfg46.reseau.charges.size
    assert e.probleme["A_ub"].shape[1] == n_var
    assert str(e.probleme["etape"]) == "étape 1"


def test_journee_compte_les_secours(cfg46, monkeypatch):
    moyenne = np.full(cfg46.n_charges, 0.9 * cfg46.p_c / cfg46.n_charges)
    normale = journee(cfg46.reseau, moyenne, cfg46.puissance_max, 0.0, 0.0, 0.5,
                      np.random.default_rng(GRAINE), limites=cfg46.limites)
    assert normale.n_secours == 0
    faux, _ = _echec_premiers(1)
    monkeypatch.setattr(dispatch, "linprog", faux)
    secours = journee(cfg46.reseau, moyenne, cfg46.puissance_max, 0.0, 0.0, 0.5,
                      np.random.default_rng(GRAINE), limites=cfg46.limites)
    assert secours.n_secours == 1
    assert secours.delestage_total == pytest.approx(normale.delestage_total, abs=1e-6)


# --------------------------------------------------------------------------
# Script SO2 : compatibilité et sauvegarde de l'échec
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def script07():
    sys.path.insert(0, str(RACINE))
    spec = importlib.util.spec_from_file_location(
        "script07", RACINE / "scripts" / "07_dynamique_lente.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ancien_checkpoint_complete_par_des_zeros(script07, tmp_path):
    n = 7
    anciennes = {cle: np.arange(n) for cle in script07.CLES_SERIES
                 if cle != "n_secours_solveur"}
    np.savez(tmp_path / "ancien.npz", **anciennes)
    with np.load(tmp_path / "ancien.npz") as d:
        lues = script07._lire_series(d)
    np.testing.assert_array_equal(lues["n_secours_solveur"], np.zeros(n, dtype=int))
    np.testing.assert_array_equal(lues["jour"], np.arange(n))


def test_checkpoint_incomplet_refuse(script07, tmp_path):
    np.savez(tmp_path / "trou.npz", jour=np.arange(3))
    with np.load(tmp_path / "trou.npz") as d, pytest.raises(KeyError):
        script07._lire_series(d)


def test_simuler_cas_sauvegarde_le_programme_fautif(script07, tmp_path, monkeypatch):
    args = script07.parser().parse_args(
        ["run", "--tailles", "46", "--G", "1.0", "--jours", "40", "--bloc", "20",
         "--transitoire", "0"])
    config = script07.construire_config(args)
    vrai = script07.simuler
    appels = {"n": 0}

    def faux(*a, **k):
        appels["n"] += 1
        if appels["n"] == 2:
            raise EchecDispatch("échec simulé", {"c": np.ones(3),
                                                 "etape": np.array("étape 1")},
                                ["highs : Status 15"])
        return vrai(*a, **k)

    monkeypatch.setattr(script07, "simuler", faux)
    with pytest.raises(RuntimeError, match="bloc commençant au jour 20") as info:
        script07.simuler_cas(str(tmp_path), config, 46, 1.0)
    assert not isinstance(info.value, EchecDispatch)  # picklable entre processus
    sauves = list((tmp_path / "echecs").glob("*.npz"))
    assert len(sauves) == 1 and sauves[0].name.endswith("_bloc_20.npz")
    with np.load(sauves[0]) as d:
        np.testing.assert_array_equal(d["c"], np.ones(3))
    # Le checkpoint du premier bloc est intact et reprenable.
    cps = [p for p in tmp_path.glob("cas_*.npz")]
    assert len(cps) == 1
    with np.load(cps[0]) as d:
        assert int(d["etat_jour"]) == 20
        assert d["n_secours_solveur"].size == 20
