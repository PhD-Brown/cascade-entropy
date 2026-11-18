"""Validation des options de cascade et rejeu du protocole explicite de la PR."""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cascade_entropy.cascade_n1 import simuler_cascade_n1, simuler_tous_retraits_n1
from cascade_entropy.n1 import redispatch_exact
from cascade_entropy.reseau import Reseau


@pytest.fixture
def radial():
    r = Reseau(3, np.array([[0, 2], [1, 2]]), np.array([.4, .7]), np.array([0, 1]),
               np.array([2]), limites=np.array([2., 2.]))
    return r, np.array([2.]), np.array([2., 2.])


@pytest.fixture(params=[False, True], ids=["un_retrait", "tous_retraits"])
def lancer(radial, request):
    def executer(**options):
        if request.param:
            return simuler_tous_retraits_n1(*radial, p1=1., **options)
        return [simuler_cascade_n1(*radial, 0, p1=1., **options)]
    return executer


@pytest.mark.parametrize("option", [{}, {"departage_intact": None}], ids=["omis", "none"])
def test_redispatch_exige_regle_intacte(lancer, option):
    with pytest.raises(ValueError, match="explicitement.*departage_intact.*highs ou min_charge_max"):
        lancer(departage="min_redispatch", **option)


@pytest.mark.parametrize("regle", ["highs", "min_charge_max"])
def test_regle_intacte_reservee_au_redispatch(lancer, regle):
    with pytest.raises(ValueError, match="departage_intact.*réservé.*min_redispatch"):
        lancer(departage=regle, departage_intact="highs")


@pytest.mark.parametrize("regle", ["highs", "min_charge_max"])
def test_regle_intacte_explicite_rejoue_le_protocole_initial(radial, lancer, regle):
    # Dans a2f2d90, le retrait sature l'autre ligne puis le second retrait isole la charge.
    r, d, pmax = radial
    reference = redispatch_exact(r, d, pmax, np.zeros(2, bool), departage=regle)
    for resultat in lancer(departage="min_redispatch", departage_intact=regle):
        ligne = resultat.ligne_initiale
        assert len(resultat.vagues_secondaires) == 1
        assert_array_equal(resultat.vagues_secondaires[0], [1 - ligne])
        premier = redispatch_exact(r, d, pmax, np.arange(2) == ligne,
            departage="min_redispatch", production_precedente=reference.production)
        final = redispatch_exact(r, d, pmax, np.ones(2, bool),
            departage="min_redispatch", production_precedente=premier.production)
        assert len(resultat.etats) == 2
        for actuel, attendu in zip((resultat.etat_reference, *resultat.etats), (reference, premier, final)):
            for champ in ("flux", "production", "charge_servie", "delestage", "hors_service"):
                assert_array_equal(getattr(actuel, champ), getattr(attendu, champ))
