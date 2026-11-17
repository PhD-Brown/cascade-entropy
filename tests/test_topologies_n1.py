"""Tests de la phase B : topologies purement synthétiques, sans banc restreint."""
from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.topologies_n1 import (
    Topologie, anneau_cordes, arbre_temoin, charger_json, composantes,
    geometrique, gilbert, grille, ponts, sauver_json, statistiques, vers_reseau,
)
from cascade_entropy.reseau import flux, matrice_incidence


def verifier_topologie(t, n):
    assert t.n_noeuds == n
    assert t.lignes.shape[1] == 2
    assert np.all(t.lignes[:,0] < t.lignes[:,1])
    assert not np.any(np.all(t.lignes[1:] == t.lignes[:-1],axis=1))
    assert statistiques(t)["connexe"]
    assert statistiques(t)["rang_cyclomatique"] == t.n_lignes-n+1


@pytest.mark.parametrize("fonction", [
    lambda:anneau_cordes(50,25,20261113),
    lambda:grille(5,10),
    lambda:gilbert(50,0.10,20261113),
    lambda:geometrique(50,0.25,20261113),
    lambda:arbre_temoin(50,3),
])
def test_cinq_familles_connexes_sans_doublons(fonction):
    verifier_topologie(fonction(),50)


def test_anneau_nb_cordes_et_aucun_pont():
    t=anneau_cordes(50,25,4)
    assert t.n_lignes==75
    assert ponts(t)==[]
    assert statistiques(t)["degre_moyen"]==3.0


def test_anneau_sans_cordes_est_un_cycle():
    t=anneau_cordes(8,0,2)
    assert t.n_lignes==8
    assert statistiques(t)["rang_cyclomatique"]==1
    assert ponts(t)==[]


def test_grille_comptage_et_ponts():
    t=grille(5,10)
    assert t.n_lignes == 5*9+4*10 == 85
    assert t.positions.shape==(50,2)
    assert ponts(t)==[]
    assert statistiques(t)["rang_cyclomatique"]==36


def test_grille_2x2_connue():
    t=grille(2,2)
    assert t.lignes.tolist()==[[0,1],[0,2],[1,3],[2,3]]
    assert ponts(t)==[]


def test_gilbert_complete():
    t=gilbert(5,1.,10)
    assert t.n_lignes==10
    assert statistiques(t)["degre_min"]==4
    assert t.essais==1


def test_gilbert_rejet_deconnexion_pas_de_reparation():
    with pytest.raises(ValueError):
        gilbert(50,0,1)
    with pytest.raises(ValueError):
        gilbert(50,1e-12,1,max_essais=1)


def test_gilbert_reproductible_y_compris_essais():
    a=gilbert(50,.10,123)
    b=gilbert(50,.10,123)
    assert np.array_equal(a.lignes,b.lignes)
    assert a.essais==b.essais


def test_geometrique_respecte_rayons():
    t=geometrique(50,.25,4)
    assert t.positions is not None
    for i,j in t.lignes:
        assert np.linalg.norm(t.positions[i]-t.positions[j]) <= .25 +1e-12
    assert t.parametres["rayon"]==.25


def test_geometrique_reproductible_y_compris_positions():
    a=geometrique(50,.25,21)
    b=geometrique(50,.25,21)
    assert np.array_equal(a.lignes,b.lignes)
    assert np.array_equal(a.positions,b.positions)
    assert a.essais==b.essais


def test_geometrique_rejet_rayon_nul():
    with pytest.raises(ValueError):
        geometrique(50,0,3)


def test_arbre_temoin_a_uniquement_des_ponts():
    t=arbre_temoin(50,3)
    assert t.n_lignes==49
    assert len(ponts(t))==49
    assert statistiques(t)["rang_cyclomatique"]==0


def test_ponts_sur_triangle_plus_feuille():
    t=Topologie("test",4,[[0,1],[1,2],[0,2],[2,3]])
    assert t.lignes.tolist()==[[0,1],[0,2],[1,2],[2,3]]
    assert ponts(t)==[3]
    assert composantes(t)==[[0,1,2,3]]


def test_graph_deconnecte_rapporte_deux_iles():
    t=Topologie("test",4,[[0,1],[2,3]])
    assert statistiques(t)["n_composantes"]==2
    assert statistiques(t)["rang_cyclomatique"]==0
    assert len(ponts(t))==2


@pytest.mark.parametrize("mauvaises",[
    [[0,1],[1,0]],[[0,0]],[[0,3]],[[0,1.5]]
])
def test_rejette_aretes_invalides(mauvaises):
    with pytest.raises(ValueError):
        Topologie("test",3,mauvaises)


def test_export_json_et_lecture(tmp_path):
    t=geometrique(50,.25,15)
    cible=tmp_path/"topo.json"
    sauver_json(t,cible)
    assert cible.is_file()
    b=charger_json(cible)
    assert b.famille==t.famille and b.parametres==t.parametres
    assert np.array_equal(b.lignes,t.lignes)
    assert np.array_equal(b.positions,t.positions)
    assert b.graine==t.graine and b.essais==t.essais


def test_export_erreur_en_cas_de_format_invalide(tmp_path):
    f=tmp_path/"topo.json"
    f.write_text('{"format":"inconnu"}',encoding="utf-8")
    with pytest.raises(ValueError):charger_json(f)


def test_conversion_vers_reseau_conservation_dc():
    t=grille(2,3)
    r=vers_reseau(t,reactances=np.ones(t.n_lignes),
                  generateurs=np.array([0]),charges=np.array([5]),
                  limites=np.full(t.n_lignes,10.),reference=0)
    F=flux(r,np.array([1.,0,0,0,0,-1.]))
    M=matrice_incidence(r)
    assert np.allclose(M.T@F,[1,0,0,0,0,-1],atol=1e-10)


def test_conversion_parametres_electriques_explicites():
    t=grille(2,2)
    with pytest.raises(TypeError):vers_reseau(t)
    with pytest.raises(ValueError):
        vers_reseau(t,reactances=np.ones(t.n_lignes),generateurs=np.array([0]),
                   charges=np.array([0]),limites=np.ones(t.n_lignes),reference=0)
    with pytest.raises(ValueError):
        vers_reseau(t,reactances=np.ones(t.n_lignes),generateurs=np.array([0]),
                   charges=np.array([3]),limites=np.zeros(t.n_lignes),reference=0)


def test_graine_invalide_refusee():
    for fun in [lambda:gilbert(graine=-1),lambda:geometrique(graine=-1),
                lambda:anneau_cordes(graine=True)]:
        with pytest.raises(ValueError):fun()


def test_rang_cyclomatique_deconnecte():
    t=Topologie("test",6,[[0,1],[1,2],[0,2],[3,4]])
    s=statistiques(t)
    assert s["n_composantes"]==3
    assert s["rang_cyclomatique"]==1
