import numpy as np

from cascade_entropy.carreras import (
    CAPACITES_TABLE,
    P_G_TABLE,
    configuration_arbre,
    demande_regionale,
    groupes_regions,
)


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
