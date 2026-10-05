"""Validation de l'option `departage` (dispatch, cascade, évolution, Carreras).

Deux familles de vérifications :

1. Cohérence interne — la règle « exterieur_dabord » ne change que la
   localisation du délestage : même puissance servie, même coût, solution
   identique quand rien n'est délesté, aucun tirage aléatoire consommé, entrées
   invalides refusées, comportement historique (« highs ») inchangé.

2. Validation externe — avec cette règle, le modèle reproduit deux
   comportements publiés par Carreras et al. (2002) :
   - entre les deux transitions, ce sont les charges de la couronne extérieure
     qui sont délestées (Sec. IV) ;
   - la Fig. 10 alterne des bandes ordonnées dont les frontières sont
     prédites analytiquement par `carreras.bandes_ordonnees()`.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy import evolution
from cascade_entropy.carreras import bandes_ordonnees, configuration_arbre
from cascade_entropy.cascade import journee
from cascade_entropy.dispatch import (
    DEPARTAGE_EXTERIEUR_DABORD,
    DEPARTAGE_HIGHS,
    DEPARTAGES,
    demande_uniforme,
    priorites_exterieur_dabord,
    resoudre,
    valider_departage,
)
from cascade_entropy.reseau import Reseau

EXT = DEPARTAGE_EXTERIEUR_DABORD
GRAINE = 20261005


def _dispatch(cfg, ratio, departage):
    demande = demande_uniforme(cfg.reseau, ratio * cfg.p_c)
    return demande, resoudre(
        cfg.reseau, demande, limites=cfg.limites,
        puissance_max=cfg.puissance_max, departage=departage,
    )


def _journee(cfg, ratio, departage):
    return journee(
        cfg.reseau, demande_uniforme(cfg.reseau, ratio * cfg.p_c),
        cfg.puissance_max, p0=0.0, p1=1.0, g=0.0,
        rng=np.random.default_rng(GRAINE), limites=cfg.limites,
        departage=departage,
    )


@pytest.fixture(scope="module")
def cfg94():
    return configuration_arbre(94)


@pytest.fixture(scope="module")
def cfg382():
    return configuration_arbre(382)


# --------------------------------------------------------------------------
# Validation des entrées
# --------------------------------------------------------------------------


def test_departages_disponibles():
    assert DEPARTAGES == (DEPARTAGE_HIGHS, EXT)
    for nom in DEPARTAGES:
        assert valider_departage(nom) == nom


@pytest.mark.parametrize("valeur", ["HIGHS", "exterieur", "", None, 1, True])
def test_departage_invalide_refuse_partout(cfg94, valeur):
    with pytest.raises(ValueError, match="departage"):
        valider_departage(valeur)
    with pytest.raises(ValueError, match="departage"):
        _dispatch(cfg94, 0.8, valeur)
    with pytest.raises(ValueError, match="departage"):
        _journee(cfg94, 0.8, valeur)


def test_exterieur_dabord_exige_les_niveaux():
    """Sans `niveaux`, « extérieur » n'est pas défini : refus explicite."""
    chaine = Reseau(
        n_noeuds=3, lignes=np.array([[0, 1], [1, 2]]),
        reactances=np.ones(2), generateurs=np.array([0]),
        charges=np.array([1, 2]), limites=np.array([10.0, 10.0]),
    )
    demande = np.array([4.0, 4.0])  # 8 demandés pour 5 disponibles : délestage
    resoudre(chaine, demande, puissance_max=np.array([5.0]))  # highs : accepté
    with pytest.raises(ValueError, match="niveaux"):
        resoudre(chaine, demande, puissance_max=np.array([5.0]), departage=EXT)


def test_priorites_ordre_strict_des_niveaux_puis_des_indices(cfg94):
    poids = priorites_exterieur_dabord(cfg94.reseau)
    niveaux = cfg94.reseau.niveaux[cfg94.reseau.charges]
    noeuds = cfg94.reseau.charges
    # Tout poids d'un niveau dépasse tout poids d'un niveau plus profond.
    for k in np.unique(niveaux)[:-1]:
        plus_profond = niveaux > k
        assert poids[niveaux == k].min() > poids[plus_profond].max()
    # Au sein d'un niveau : poids tous distincts, décroissants avec l'indice.
    for k in np.unique(niveaux):
        dans = niveaux == k
        ordre = np.argsort(noeuds[dans])
        assert np.all(np.diff(poids[dans][ordre]) < 0)
    assert poids.min() > 1.0 - 1e-12


def test_front_de_delestage_contigu(cfg382):
    """Les charges extérieures coupées sont celles de plus grands indices."""
    demande, sol = _dispatch(cfg382, 1.3, EXT)
    reseau = cfg382.reseau
    exterieures = reseau.niveaux[reseau.charges] == reseau.niveaux.max()
    coupees = (sol.charge_servie / demande < 1 - 1e-6) & exterieures
    servies = (sol.charge_servie / demande > 1 - 1e-6) & exterieures
    assert reseau.charges[coupees].min() > reseau.charges[servies].max() - 1


# --------------------------------------------------------------------------
# Cohérence interne : seule la localisation du délestage change
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0.5, 0.95])
def test_sans_delestage_les_deux_regles_sont_identiques(cfg94, ratio):
    _, a = _dispatch(cfg94, ratio, DEPARTAGE_HIGHS)
    _, b = _dispatch(cfg94, ratio, EXT)
    assert a.delestage_total == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_array_equal(a.production, b.production)
    np.testing.assert_array_equal(a.charge_servie, b.charge_servie)
    np.testing.assert_array_equal(a.flux, b.flux)


@pytest.mark.parametrize("ratio", [1.05, 1.3, 1.55, 2.5])
def test_meme_puissance_servie_et_meme_cout(cfg94, ratio):
    demande, a = _dispatch(cfg94, ratio, DEPARTAGE_HIGHS)
    _, b = _dispatch(cfg94, ratio, EXT)
    tol = 1e-6 * demande.sum()
    assert b.charge_servie.sum() == pytest.approx(a.charge_servie.sum(), abs=tol)
    assert b.cout == pytest.approx(a.cout, rel=1e-8)
    # Contraintes physiques respectées par la solution départagée.
    assert np.all(b.taux_de_charge <= 1.0 + 1e-7)
    assert np.all(b.production <= cfg94.puissance_max + 1e-7)
    assert np.all(b.charge_servie <= demande + 1e-7)
    assert np.all(b.charge_servie >= -1e-9)
    assert b.production.sum() == pytest.approx(b.charge_servie.sum(), rel=1e-9)


def test_solution_porte_sa_regle(cfg94):
    assert _dispatch(cfg94, 1.2, DEPARTAGE_HIGHS)[1].departage == DEPARTAGE_HIGHS
    assert _dispatch(cfg94, 1.2, EXT)[1].departage == EXT


def test_defaut_inchange(cfg94):
    """Sans argument, `resoudre` et `journee` restent sur le comportement historique."""
    demande = demande_uniforme(cfg94.reseau, 1.3 * cfg94.p_c)
    defaut = resoudre(cfg94.reseau, demande, limites=cfg94.limites,
                      puissance_max=cfg94.puissance_max)
    explicite = _dispatch(cfg94, 1.3, DEPARTAGE_HIGHS)[1]
    np.testing.assert_array_equal(defaut.charge_servie, explicite.charge_servie)
    assert defaut.departage == DEPARTAGE_HIGHS


def test_departage_ne_consomme_aucun_tirage(cfg94):
    """À graine égale, les deux règles voient la même demande et les mêmes p0."""
    resultats = []
    for regle in DEPARTAGES:
        rng = np.random.default_rng(GRAINE)
        j = journee(cfg94.reseau, demande_uniforme(cfg94.reseau, 0.9 * cfg94.p_c),
                    cfg94.puissance_max, p0=0.05, p1=1.0, g=0.3, rng=rng,
                    limites=cfg94.limites, departage=regle)
        resultats.append((j.demande, j.avaries_accidentelles, rng.random()))
    (d1, a1, x1), (d2, a2, x2) = resultats
    np.testing.assert_array_equal(d1, d2)
    np.testing.assert_array_equal(a1, a2)
    assert x1 == x2


# --------------------------------------------------------------------------
# Validation externe : Carreras et al. (2002)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [1.1, 1.3, 1.43])
def test_couronne_exterieure_delestee_en_premier(cfg382, ratio):
    """
    Sec. IV : « The nodes in the outermost ring of the network are
    progressively blacked out ». Entre les deux transitions, toute charge
    intérieure est servie en entier et le délestage porte sur le niveau 7.
    """
    demande, sol = _dispatch(cfg382, ratio, EXT)
    niveaux = cfg382.reseau.niveaux[cfg382.reseau.charges]
    fraction = sol.charge_servie / demande
    assert np.all(fraction[niveaux < niveaux.max()] > 1 - 1e-6)
    assert np.any(fraction[niveaux == niveaux.max()] < 1 - 1e-6)
    # La fraction délestée reste celle imposée par la génération : 1 − 1/r.
    assert sol.delestage_total / demande.sum() == pytest.approx(1 - 1 / ratio, abs=1e-6)


def test_highs_garde_la_couronne_servie(cfg382):
    """Témoin : le choix HiGHS déleste l'intérieur (point ✗ du tableau SO1)."""
    demande, sol = _dispatch(cfg382, 1.3, DEPARTAGE_HIGHS)
    niveaux = cfg382.reseau.niveaux[cfg382.reseau.charges]
    fraction = sol.charge_servie / demande
    assert np.all(fraction[niveaux == niveaux.max()] > 1 - 1e-6)


def test_bandes_ordonnees_analytiques_382(cfg382):
    """Frontières de la Fig. 10 : arbres équivalents de 382, 190, 94, 46 nœuds."""
    bandes = bandes_ordonnees(cfg382)
    attendu = [(382, 1.0, 1.44529), (190, 2.07865, 3.09705),
               (94, 4.51220, 7.22645), (46, 10.88235, 21.67934)]
    assert [b["equivalent_n_noeuds"] for b in bandes] == [a[0] for a in attendu]
    for b, (_, debut, fin) in zip(bandes, attendu):
        assert b["r_debut"] == pytest.approx(debut, abs=1e-4)
        assert b["r_fin"] == pytest.approx(fin, abs=1e-4)
        assert b["r_debut"] < b["r_fin_critere"] < b["r_fin"]


@pytest.mark.parametrize("ratio,bande", [(2.2, 190), (3.0, 190), (4.6, 94), (7.1, 94)])
def test_exterieur_dabord_reproduit_les_bandes_ordonnees(cfg382, ratio, bande):
    """Dans une bande prédite, la cascade déterministe ne fait tomber aucune ligne."""
    j = _journee(cfg382, ratio, EXT)
    assert j.lignes_tombees.size == 0
    # Le réseau servi est bien l'arbre restreint : charges plus profondes coupées.
    profondeur = {190: 6, 94: 5}[bande]
    niveaux = cfg382.reseau.niveaux[cfg382.reseau.charges]
    fraction = j.solution.charge_servie / j.demande
    assert np.all(fraction[niveaux > profondeur] < 1e-6)


@pytest.mark.parametrize("ratio", [1.8, 3.3, 7.4])
def test_hors_bande_la_cascade_reprend(cfg382, ratio):
    """Entre deux bandes ordonnées, des lignes saturent et tombent."""
    assert _journee(cfg382, ratio, EXT).lignes_tombees.size > 0


def test_highs_n_a_pas_de_bande_ordonnee(cfg382):
    """Témoin : avec HiGHS, effondrement au milieu de la bande 190 prédite."""
    assert _journee(cfg382, 2.5, DEPARTAGE_HIGHS).lignes_tombees.size > 300


# --------------------------------------------------------------------------
# Propagation dans `evolution`
# --------------------------------------------------------------------------


def _parametres(cfg, **changements):
    base = dict(mode="independant", demande_initiale=0.9 * cfg.p_c,
                puissance_max=cfg.puissance_max, g=0.3, p0=0.0, p1=1.0,
                limites=cfg.limites)
    base.update(changements)
    return evolution.Parametres(**base)


def test_evolution_defaut_highs(cfg94):
    assert _parametres(cfg94).departage == DEPARTAGE_HIGHS


def test_evolution_refuse_un_departage_invalide(cfg94):
    with pytest.raises(ValueError, match="departage"):
        evolution.simuler(cfg94.reseau, 3, _parametres(cfg94, departage="xyz"),
                          np.random.default_rng(GRAINE))


def test_evolution_transmet_le_departage(cfg94):
    """Même demande jour par jour ; le délestage total ne dépend pas de la règle."""
    series = {}
    for regle in DEPARTAGES:
        jours = evolution.simuler(cfg94.reseau, 30, _parametres(cfg94, p1=0.0, departage=regle),
                                  np.random.default_rng(GRAINE))
        series[regle] = np.array([j.delestage_total for j in jours])
    np.testing.assert_allclose(series[DEPARTAGE_HIGHS], series[EXT],
                               atol=1e-6 * cfg94.p_c)
