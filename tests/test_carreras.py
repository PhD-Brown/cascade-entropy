import numpy as np

from cascade_entropy.carreras import (
    CAPACITES_TABLE,
    P_G_TABLE,
    P_L_TABLE,
    configuration_arbre,
    demande_regionale,
    groupes_regions,
)
from cascade_entropy.dispatch import demande_uniforme, resoudre


def test_pc_est_somme_des_12_generateurs():
    for n_noeuds in (46, 94, 190, 382):
        cfg = configuration_arbre(n_noeuds)
        assert cfg.n_generateurs == 12
        assert np.all(cfg.puissance_max == P_G_TABLE)
        assert np.isclose(cfg.p_c, 12 * P_G_TABLE)


def test_table_i_limites_exactes_46_94():
    cfg46 = configuration_arbre(46)
    cfg94 = configuration_arbre(94)

    assert set(np.unique(cfg46.limites)) == set(CAPACITES_TABLE[:4])
    assert set(np.unique(cfg94.limites)) == set(CAPACITES_TABLE[:5])


def test_trois_regions_presque_equilibrees():
    cfg46 = configuration_arbre(46)
    cfg94 = configuration_arbre(94)

    c46 = np.bincount(groupes_regions(cfg46.reseau, 3), minlength=3)
    c94 = np.bincount(groupes_regions(cfg94.reseau, 3), minlength=3)

    assert sorted(c46.tolist()) == [11, 11, 12]
    assert sorted(c94.tolist()) == [27, 27, 28]


def test_gamma_un_donne_demande_non_fluctuante():
    cfg = configuration_arbre(46)
    demande, facteurs, groupes = demande_regionale(
        cfg, 0.77, np.random.default_rng(1), gamma=1.0, n_regions=3
    )

    assert np.allclose(facteurs, 1.0)
    assert np.allclose(demande, demande[0])
    assert np.isclose(demande.sum(), 0.77 * cfg.p_c)
    assert np.unique(groupes).size == 3


def test_charge_dune_region_partage_le_meme_facteur():
    cfg = configuration_arbre(94)
    ratio = 0.77
    demande, facteurs, groupes = demande_regionale(
        cfg, ratio, np.random.default_rng(123), gamma=1.9, n_regions=3
    )

    base = ratio * cfg.p_c / cfg.n_charges
    ratios = demande / base
    for region in range(3):
        assert np.allclose(ratios[groupes == region], facteurs[region])

# ---------------------------------------------------------------------------
# Validations EXTERNES : ces tests comparent le modèle à des valeurs publiées
# par Carreras et al. (2002), et non à une hypothèse codée par nous.
# ---------------------------------------------------------------------------
 
def _niveaux_lignes(reseau):
    return np.asarray(reseau.niveaux[reseau.lignes[:, 1]], dtype=int)
 
 
def test_table_i_382_reproduit_figures_3_4():
    """
    Lecture littérale de la Table I (chaque charge = |P_L| = 74) sur le 382 :
    Fig. 3 : lignes extérieures à M = 0.601 ;
    Fig. 4 : 10 générateurs au maximum, 1 partiel, 1 quasi nul.
 
    La structure "k pleins + 1 partiel" est imposée par la géométrie du
    programme linéaire (sommet optimal) ; la dégénérescence ne choisit que
    QUEL générateur est partiel, d'où un test sur les comptes et non sur les
    indices.
    """
    cfg = configuration_arbre(382)
    demande = np.full(cfg.n_charges, abs(P_L_TABLE), dtype=float)
    sol = resoudre(
        cfg.reseau,
        demande,
        limites=cfg.limites,
        puissance_max=cfg.puissance_max,
    )
 
    m_ext = sol.taux_de_charge[_niveaux_lignes(cfg.reseau) > 3]
    assert np.isclose(m_ext.mean(), 0.601, rtol=5e-3)
    assert np.ptp(m_ext) < 1e-3
 
    frac = sol.production / cfg.puissance_max
    assert np.sum(frac >= 0.99) == 10
    assert np.sum(frac <= 0.01) == 1
    assert np.sum((frac > 0.01) & (frac < 0.99)) == 1
 
 
def test_seuil_transport_382_egal_prediction_analytique():
    """
    Seconde transition (Fig. 5) : le premier M_max = 1 du dispatch déterministe
    doit coïncider avec la prédiction analytique
        r_T = F_max * N_L / (n_sous_arbre * P_C)
    sur la ligne extérieure limitante (niveau 4, 15 nœuds en aval sur le 382),
    soit r_T ≈ 1.4453, à comparer au 1.45 publié.
    """
    cfg = configuration_arbre(382)
    r_t = CAPACITES_TABLE[3] * cfg.n_charges / (15 * cfg.p_c)
 
    def mmax(ratio):
        d = demande_uniforme(cfg.reseau, ratio * cfg.p_c)
        return resoudre(
            cfg.reseau, d, limites=cfg.limites, puissance_max=cfg.puissance_max
        ).taux_maximal
 
    assert mmax(0.999 * r_t) < 1.0 - 1e-6
    assert mmax(1.001 * r_t) >= 1.0 - 1e-6
    assert abs(r_t - 1.45) / 1.45 < 0.005