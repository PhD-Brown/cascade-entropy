"""Fig. 12–13 de Carreras (2002) : seuil de transport et avarie isolée.

- `seuil_transport` redonne les r_T(N) du tableau SO1 (section B) ;
- `delestage_avarie_unique` donne une masse ∝ 1/s : une densité en s^-2,
  la prédiction analytique citée par Carreras pour le régime sous-critique.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.carreras import (
    configuration_arbre,
    delestage_avarie_unique,
    seuil_transport,
)
from cascade_entropy.reseau import Reseau

R_T_SO1 = {46: 1.992155, 94: 1.601537, 190: 1.489931, 382: 1.445289}


@pytest.mark.parametrize("n_noeuds", sorted(R_T_SO1))
def test_seuil_transport_tableau_so1(n_noeuds):
    assert seuil_transport(configuration_arbre(n_noeuds)) == pytest.approx(
        R_T_SO1[n_noeuds], abs=1e-6)


def test_avarie_unique_382_masses_en_un_sur_s():
    cfg = configuration_arbre(382)
    s = delestage_avarie_unique(cfg)
    tailles, nombres = np.unique(np.rint(s * cfg.n_charges).astype(int),
                                 return_counts=True)
    np.testing.assert_array_equal(tailles, [1, 3, 7, 15])
    np.testing.assert_array_equal(nombres, [192, 96, 48, 24])
    # Le nombre de lignes est divisé par 2 quand la taille double (à +1 près) :
    # masse ∝ 1/s, donc densité ∝ s^-2 en classes logarithmiques.
    pente = np.polyfit(np.log(tailles + 1), np.log(nombres), 1)[0]
    assert pente == pytest.approx(-1.0, abs=1e-12)


@pytest.mark.parametrize("n_noeuds,attendu", [(46, {1: 24}), (94, {1: 48, 3: 24})])
def test_avarie_unique_petits_arbres(n_noeuds, attendu):
    cfg = configuration_arbre(n_noeuds)
    t, k = np.unique(np.rint(delestage_avarie_unique(cfg) * cfg.n_charges).astype(int),
                     return_counts=True)
    assert dict(zip(t.tolist(), k.tolist())) == attendu


def test_reseau_sans_niveaux_refuse():
    cfg = configuration_arbre(46)
    r = cfg.reseau
    sans = Reseau(r.n_noeuds, r.lignes, r.reactances, r.generateurs, r.charges,
                  limites=r.limites, reference=r.reference, niveaux=None)
    from dataclasses import replace
    with pytest.raises(ValueError, match="niveaux"):
        seuil_transport(replace(cfg, reseau=sans))


# --------------------------------------------------------------------------
# Bilan par îlots : le dispatch sous le seuil critique
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_noeuds", [94, 382])
def test_bilan_ilots_egal_au_dispatch_sous_le_seuil(n_noeuds):
    """γ ρ < 0.99 : aucune ligne ne sature, le LP se réduit au bilan par îlot."""
    from cascade_entropy.carreras import delestage_bilan_ilots, demande_regionale
    from cascade_entropy.cascade import FACTEUR_LIMITE_AVARIE, journee
    from cascade_entropy.dispatch import DEPARTAGE_EXTERIEUR_DABORD

    cfg = configuration_arbre(n_noeuds)
    ratio = 0.45 * seuil_transport(cfg)          # ρ = 0.45 : γ ρ = 0.855
    rng = np.random.default_rng(20261005)
    vus = {"avarie": 0, "generation": 0}
    for _ in range(300):
        demande, _, _ = demande_regionale(cfg, ratio, rng, gamma=1.9, n_regions=3)
        # p0 élevé pour voir beaucoup d'îlots en peu de tirages.
        j = journee(cfg.reseau, demande, cfg.puissance_max, p0=0.01, p1=1.0, g=0.0,
                    rng=rng, limites=cfg.limites, departage=DEPARTAGE_EXTERIEUR_DABORD)
        assert j.avaries_par_surcharge == []      # aucune cascade sous le seuil
        attendu = delestage_bilan_ilots(cfg, j.demande, j.hors_service)
        # Seul écart permis : le flux résiduel des lignes mortes (limite × 1e-6).
        residuel = FACTEUR_LIMITE_AVARIE * cfg.limites[j.hors_service].sum()
        assert abs(j.delestage_total - attendu) <= residuel + 1e-6 * attendu + 1e-9
        vus["avarie"] += int(j.hors_service.any() and attendu > 0)
        vus["generation"] += int(j.demande.sum() > cfg.p_c)
    assert vus["avarie"] > 10 and vus["generation"] > 0


def test_bilan_ilots_cas_simples():
    from cascade_entropy.carreras import delestage_bilan_ilots
    cfg = configuration_arbre(46)
    aucune = np.zeros(cfg.reseau.n_lignes, dtype=bool)
    d = np.full(cfg.n_charges, 0.5 * cfg.p_c / cfg.n_charges)
    assert delestage_bilan_ilots(cfg, d, aucune) == 0.0
    assert delestage_bilan_ilots(cfg, 3 * d, aucune) == pytest.approx(0.5 * cfg.p_c)
    with pytest.raises(ValueError):
        delestage_bilan_ilots(cfg, d[:-1], aucune)
    with pytest.raises(ValueError):
        delestage_bilan_ilots(cfg, d, aucune[:-1])
