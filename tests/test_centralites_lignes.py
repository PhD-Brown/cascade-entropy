"""Oracles analytiques et indépendants, exclusivement sur graphes synthétiques."""

from fractions import Fraction
from itertools import combinations
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cascade_entropy.centralites_lignes import (
    EtatInitialIntact, GrapheLignes, calculer_scores, centralite_vecteur_propre,
    degre_lignes, flux_initial_absolu, intermediarite_courant,
    intermediarite_geodesique, redondance_electrique, resistance_effective,
    taux_charge_initial,
)


def graphe(n, lignes, x=None):
    return GrapheLignes(n, np.array(lignes, dtype=int).reshape(-1, 2),
                        np.ones(len(lignes)) if x is None else np.array(x))


def maille():
    # Maillage sans symétrie pondérée, avec pont vers une feuille.
    return graphe(6, [(3, 0), (1, 0), (1, 2), (2, 3), (0, 2), (2, 4), (4, 3), (4, 5)],
                  [1, 2, 3, 4, 2, 1, 3, 5])


def etat(g, flux=None, limites=None):
    return EtatInitialIntact(g, np.zeros(g.n_lignes) if flux is None else flux,
                            np.ones(g.n_lignes) if limites is None else limites,
                            np.zeros(g.n_lignes, dtype=bool), "synthetique-intact")


OPTIONS = dict(degre="somme", longueurs="reactances", normalisation_geodesique="moyenne",
               normalisation_vecteur="l2", paires_courant="uniformes",
               normalisation_courant="moyenne", convention_resistance="criticite")


def resoudre_rationnel(a, b):
    """Élimination de Gauss exacte : oracle distinct du pseudo-inverse spectral."""
    lignes = [[Fraction(x) for x in row] + [Fraction(y)] for row, y in zip(a, b)]
    for k in range(len(b)):
        pivot = next(i for i in range(k, len(b)) if lignes[i][k])
        lignes[k], lignes[pivot] = lignes[pivot], lignes[k]
        d = lignes[k][k]
        lignes[k] = [x / d for x in lignes[k]]
        for i in range(len(b)):
            if i != k:
                d = lignes[i][k]
                lignes[i] = [x - d * y for x, y in zip(lignes[i], lignes[k])]
    return [row[-1] for row in lignes]


def courant_rationnel(g, s, t, reference=0):
    n = g.n_noeuds
    b = [1 / Fraction(str(x)) for x in g.reactances]
    lap = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for (u, v), poids in zip(g.lignes, b):
        lap[u][u] += poids
        lap[v][v] += poids
        lap[u][v] -= poids
        lap[v][u] -= poids
    libres = [i for i in range(n) if i != reference]
    p = [int(i == s) - int(i == t) for i in libres]
    sol = resoudre_rationnel([[lap[i][j] for j in libres] for i in libres], p)
    theta = [Fraction(0)] * n
    for i, x in zip(libres, sol):
        theta[i] = x
    return [poids * (theta[u] - theta[v]) for (u, v), poids in zip(g.lignes, b)]


def chemins_exhaustifs(g, longueurs):
    """Énumère les chemins simples, sans Dijkstra ni récurrence de Brandes."""
    adj = [[] for _ in range(g.n_noeuds)]
    for e, (u, v) in enumerate(g.lignes):
        adj[u].append((v, e))
        adj[v].append((u, e))
    longueurs = [Fraction(str(x)) for x in longueurs]
    score = [Fraction(0)] * g.n_lignes
    for s, t in combinations(range(g.n_noeuds), 2):
        chemins = []

        def visiter(u, vus, aretes, distance):
            if u == t:
                chemins.append((distance, aretes))
                return
            for v, e in adj[u]:
                if v not in vus:
                    visiter(v, vus | {v}, aretes + [e], distance + longueurs[e])

        visiter(s, {s}, [], Fraction(0))
        dmin = min(d for d, _ in chemins)
        meilleurs = [a for d, a in chemins if d == dmin]
        for chemin in meilleurs:
            for e in chemin:
                score[e] += Fraction(1, len(meilleurs))
    return np.array(score, dtype=float)


def probabilites_arbres(g):
    """Énumère les arbres couvrants pondérés, oracle de bR sans laplacien."""
    total = Fraction(0)
    presences = [Fraction(0)] * g.n_lignes
    for selection in combinations(range(g.n_lignes), g.n_noeuds - 1):
        groupes = [{i} for i in range(g.n_noeuds)]
        poids = Fraction(1)
        for e in selection:
            u, v = g.lignes[e]
            gu = next(a for a in groupes if u in a)
            gv = next(a for a in groupes if v in a)
            if gu is gv:
                break
            groupes.remove(gu)
            groupes.remove(gv)
            groupes.append(gu | gv)
            poids /= Fraction(str(g.reactances[e]))
        else:
            total += poids
            for e in selection:
                presences[e] += poids
    return np.array([p / total for p in presences], dtype=float)


@pytest.mark.parametrize("variante,attendu", [("somme", [3, 4, 3]), ("produit", [2, 4, 2])])
def test_degre_chaine(variante, attendu):
    g = graphe(4, [(0, 1), (1, 2), (2, 3)])
    assert_array_equal(degre_lignes(g, variante=variante), attendu)


@pytest.mark.parametrize("longueurs", ["unitaires", "reactances"])
@pytest.mark.parametrize("normalisation,diviseur", [("somme", 1), ("moyenne", 6)])
def test_chaine_chemin_unique(longueurs, normalisation, diviseur):
    g = graphe(4, [(0, 1), (1, 2), (2, 3)], [2, 7, 1])
    attendu = np.array([3, 4, 3]) / diviseur
    assert_allclose(intermediarite_geodesique(g, longueurs=longueurs, normalisation=normalisation), attendu)
    assert_allclose(intermediarite_courant(g, paires="uniformes", normalisation=normalisation), attendu)
    assert_allclose(resistance_effective(g), [2, 7, 1])
    assert_allclose(redondance_electrique(g, convention="criticite"), 1)
    assert_allclose(redondance_electrique(g, convention="redondance"), 0, atol=1e-14)


def test_arbre_branche_coupes_analytiques():
    g = graphe(6, [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)], [1, 3, 2, 5, 8])
    # Chaque coupe sépare a nœuds et n-a : a(n-a) paires traversent l'arête.
    attendu = np.array([5, 5, 9, 5, 5])
    assert_allclose(intermediarite_courant(g, paires="uniformes", normalisation="somme"), attendu)
    assert_allclose(intermediarite_geodesique(g, longueurs="reactances", normalisation="somme"), attendu)
    assert_allclose(resistance_effective(g), g.reactances)
    assert_allclose(redondance_electrique(g, convention="criticite"), np.ones(5))


@pytest.mark.parametrize("n,geodesique,courant", [(3, 1, Fraction(4, 3)), (4, 2, Fraction(5, 2))])
def test_cycles_symetries_et_ex_aequo(n, geodesique, courant):
    g = graphe(n, [(i, (i + 1) % n) for i in range(n)])
    assert_allclose(degre_lignes(g, variante="somme"), 4)
    assert_allclose(degre_lignes(g, variante="produit"), 4)
    assert_allclose(intermediarite_geodesique(g, longueurs="unitaires", normalisation="somme"), geodesique)
    assert_allclose(intermediarite_courant(g, paires="uniformes", normalisation="somme"), float(courant))
    assert_allclose(centralite_vecteur_propre(g, normalisation="l2"), 1 / n)
    assert_allclose(resistance_effective(g), (n - 1) / n)
    assert_allclose(redondance_electrique(g, convention="criticite"), (n - 1) / n)


@pytest.mark.parametrize("norm,score", [("l2", 1 / 3), ("l1", 1 / 9), ("maximum", 1)])
def test_normalisations_vecteur_triangle(norm, score):
    g = graphe(3, [(0, 1), (1, 2), (2, 0)])
    assert_allclose(centralite_vecteur_propre(g, normalisation=norm), score)


def test_vecteur_chaine_bipartie_oracle_trigonometrique():
    g = graphe(4, [(0, 1), (1, 2), (2, 3)])
    c = np.sqrt(2 / 5) * np.sin(np.arange(1, 5) * np.pi / 5)
    assert_allclose(centralite_vecteur_propre(g, normalisation="l2"), c[:-1] * c[1:])


def test_regularite_ponderee_et_non_ponderee():
    aretes = [(0, 1), (1, 2), (2, 3), (3, 0)]
    g = graphe(4, aretes, [1, 0.5, 1, 0.5])  # Forces toutes égales à 3.
    assert_allclose(centralite_vecteur_propre(g, normalisation="l2"), 1 / g.reactances / 4)
    autre = graphe(4, aretes, [1, 2, 3, 4])  # Même degré, forces différentes.
    c_produits = centralite_vecteur_propre(autre, normalisation="l2") * autre.reactances
    assert np.ptp(c_produits) > 0.01


def test_triangle_pondere_chemins_multiples_et_series_parallele():
    g = graphe(3, [(0, 1), (1, 2), (0, 2)], [1, 1, 2])
    assert_allclose(intermediarite_geodesique(g, longueurs="reactances", normalisation="somme"), [1.5, 1.5, 0.5])
    assert_allclose(intermediarite_geodesique(g, longueurs="unitaires", normalisation="somme"), 1)
    assert_allclose(resistance_effective(g), [0.75, 0.75, 1])
    assert_allclose(redondance_electrique(g, convention="criticite"), [0.75, 0.75, 0.5])
    assert_allclose(intermediarite_courant(g, paires="uniformes", normalisation="moyenne"), [0.5, 0.5, 1 / 3])


@pytest.mark.parametrize("longueurs", ["unitaires", "reactances"])
def test_geodesique_oracle_exhaustif_asymetrique(longueurs):
    g = maille()
    ell = np.ones(g.n_lignes) if longueurs == "unitaires" else g.reactances
    assert_allclose(intermediarite_geodesique(g, longueurs=longueurs, normalisation="somme"),
                    chemins_exhaustifs(g, ell))


def test_partage_par_chemin_et_non_par_predecesseur():
    # Entre 0 et 6 : deux chemins via 3, un via 5, tous de longueur 3.
    g = graphe(7, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 6), (0, 4), (4, 5), (5, 6)])
    assert_allclose(intermediarite_geodesique(g, longueurs="unitaires", normalisation="somme"),
                    chemins_exhaustifs(g, np.ones(g.n_lignes)))


def test_tolerance_relative_chemins_explicitement_configurable():
    g = graphe(3, [(0, 1), (1, 2), (0, 2)], [1, 1, 2 + 1e-10])
    assert_allclose(intermediarite_geodesique(g, longueurs="reactances", normalisation="somme", egalite_rtol=0), [2, 2, 0])
    assert_allclose(intermediarite_geodesique(g, longueurs="reactances", normalisation="somme", egalite_rtol=1e-9), [1.5, 1.5, 0.5])


@pytest.mark.parametrize("reference", [0, 2, 5])
def test_courant_oracle_rationnel_et_independance_reference(reference):
    g = maille()
    courants = [courant_rationnel(g, s, t, reference) for s, t in combinations(range(6), 2)]
    attendu = np.abs(np.array(courants, dtype=float)).mean(axis=0)
    assert_allclose(intermediarite_courant(g, paires="uniformes", normalisation="moyenne"), attendu, atol=1e-14)


@pytest.mark.parametrize("normalisation", ["somme", "moyenne"])
def test_courant_generateur_charge_pondere(normalisation):
    g = maille()
    gen, conso = [0, 4], [1, 5]
    poids = np.array([[1.0, 0.0], [3.0, 2.0]])
    attendu = sum(poids[i, j] * np.abs(np.array(courant_rationnel(g, s, t), dtype=float))
                   for i, s in enumerate(gen) for j, t in enumerate(conso))
    if normalisation == "moyenne":
        attendu /= poids.sum()
    obtenu = intermediarite_courant(g, paires="generateur_charge", normalisation=normalisation,
                                  generateurs=gen, charges=conso, poids_paires=poids)
    assert_allclose(obtenu, attendu, atol=1e-14)
    uniforme = intermediarite_courant(g, paires="generateur_charge", normalisation=normalisation,
                                    generateurs=gen, charges=conso, poids_paires=np.ones((2, 2)))
    assert not np.allclose(obtenu, uniforme)
    assert not np.allclose(obtenu, intermediarite_courant(g, paires="uniformes", normalisation=normalisation))


def test_valeur_absolue_avant_somme_et_transferts_nuls_sur_certaines_lignes():
    # Sur un arbre 0-1-2 et 1-3 : la paire 0-2 n'utilise pas la feuille 3.
    g = graphe(4, [(0, 1), (1, 2), (1, 3)])
    score = intermediarite_courant(g, paires="generateur_charge", normalisation="moyenne",
                                 generateurs=[0], charges=[2], poids_paires=[[1]])
    assert_allclose(score, [1, 1, 0], atol=1e-14)
    # Les deux générateurs alimentent 1 par des courants opposés sur 1-2.
    h = graphe(3, [(0, 1), (1, 2)])
    assert_allclose(intermediarite_courant(h, paires="generateur_charge", normalisation="somme",
                                         generateurs=[0, 2], charges=[1], poids_paires=[[1], [1]]), [1, 1])


def test_resistance_oracle_arbres_couvrants_foster_et_pont():
    g = maille()
    obtenu = redondance_electrique(g, convention="criticite")
    assert_allclose(obtenu, probabilites_arbres(g), atol=1e-14)
    assert_allclose(resistance_effective(g) / g.reactances, obtenu)
    assert obtenu.sum() == pytest.approx(5)
    assert obtenu[-1] == pytest.approx(1)
    assert np.all(obtenu[:-1] < 1)
    assert_allclose(redondance_electrique(g, convention="redondance"), 1 - obtenu)


def test_vecteur_sensibilite_spectrale_locale():
    g = maille()
    a = np.zeros((g.n_noeuds, g.n_noeuds))
    for (u, v), x in zip(g.lignes, g.reactances):
        a[u, v] = a[v, u] = 1 / x
    score = centralite_vecteur_propre(g, normalisation="l2")
    assert 2 * score.sum() == pytest.approx(np.linalg.eigvalsh(a)[-1])
    # Diminution infinitésimale RELATIVE de b_e : dérivée -2 b_e c_i c_j.
    eps = 1e-5
    for e, (u, v) in enumerate(g.lignes):
        delta = np.zeros_like(a)
        delta[u, v] = delta[v, u] = eps / g.reactances[e]
        derivee = (np.linalg.eigvalsh(a - delta)[-1] - np.linalg.eigvalsh(a + delta)[-1]) / (2 * eps)
        assert derivee == pytest.approx(-2 * score[e], rel=1e-7, abs=1e-9)


@pytest.mark.parametrize("facteur", [1e-9, 7.0, 1e9])
def test_changement_echelle_reactances(facteur):
    g = maille()
    h = graphe(g.n_noeuds, g.lignes, g.reactances * facteur)
    assert_array_equal(degre_lignes(g, variante="produit"), degre_lignes(h, variante="produit"))
    assert_allclose(intermediarite_geodesique(g, longueurs="reactances", normalisation="moyenne"),
                    intermediarite_geodesique(h, longueurs="reactances", normalisation="moyenne"))
    assert_allclose(intermediarite_courant(g, paires="uniformes", normalisation="moyenne"),
                    intermediarite_courant(h, paires="uniformes", normalisation="moyenne"))
    assert_allclose(redondance_electrique(g, convention="criticite"), redondance_electrique(h, convention="criticite"))
    assert_allclose(resistance_effective(g) * facteur, resistance_effective(h))
    assert_allclose(centralite_vecteur_propre(g, normalisation="l2") / facteur,
                    centralite_vecteur_propre(h, normalisation="l2"))


def test_invariance_orientation_renumerotation_et_permutation_lignes():
    g = maille()
    ordre = np.array([4, 1, 7, 0, 5, 3, 6, 2])
    noeuds = np.array([3, 2, 5, 1, 0, 4])
    h = graphe(6, noeuds[g.lignes[ordre, ::-1]], g.reactances[ordre])
    e = etat(g, np.arange(8) - 3, np.arange(8) + 1)
    f = etat(h, -e.flux[ordre], e.limites[ordre])
    origine, permute = calculer_scores(e, **OPTIONS), calculer_scores(f, **OPTIONS)
    for nom, valeurs in origine.scores.items():
        assert_allclose(permute.scores[nom], valeurs[ordre], atol=1e-14)
    assert_array_equal(origine.lignes, g.lignes)
    assert_array_equal(origine.identifiants_lignes, np.arange(8))


def test_references_flux_nul_signe_et_capacites():
    g = graphe(4, [(0, 1), (1, 2), (2, 3)])
    e = etat(g, [0, -2, 2], [1, 4, 2])
    assert_array_equal(flux_initial_absolu(e), [0, 2, 2])
    assert_array_equal(taux_charge_initial(e), [0, 0.5, 1])
    assert_array_equal(taux_charge_initial(etat(g, [0, 0, 0])), [0, 0, 0])
    uniforme = etat(g, [-1, 2, 3], [4, 4, 4])
    assert_allclose(taux_charge_initial(uniforme), flux_initial_absolu(uniforme) / 4)
    proportionnel = etat(g, [-1, 2, 3], [2, 4, 6])
    assert_array_equal(taux_charge_initial(proportionnel), [0.5, 0.5, 0.5])


def test_sept_scores_provenance_et_non_mutation():
    g = maille()
    flux = np.arange(8, dtype=float)
    limites = np.ones(8)
    e = etat(g, flux, limites)
    avant = g.lignes.copy(), g.reactances.copy(), e.flux.copy(), e.limites.copy()
    resultat = calculer_scores(e, **OPTIONS)
    assert len(resultat.scores) == 7
    assert resultat.identifiant_etat_initial == "synthetique-intact"
    assert resultat.parametres.items() >= OPTIONS.items()
    json.dumps(resultat.parametres)
    assert_allclose(resultat.scores["degre"], degre_lignes(g, variante="somme"))
    assert_allclose(resultat.scores["intermediarite_geodesique"], intermediarite_geodesique(g, longueurs="reactances", normalisation="moyenne"))
    assert_allclose(resultat.scores["vecteur_propre"], centralite_vecteur_propre(g, normalisation="l2"))
    assert_allclose(resultat.scores["intermediarite_courant"], intermediarite_courant(g, paires="uniformes", normalisation="moyenne"))
    assert_allclose(resultat.scores["resistance_locale"], redondance_electrique(g, convention="criticite"))
    assert_allclose(resultat.scores["flux_initial_absolu"], flux_initial_absolu(e))
    assert_allclose(resultat.scores["taux_charge_initial"], taux_charge_initial(e))
    flux[:] = -100
    limites[:] = 100
    for a, b in zip(avant, (g.lignes, g.reactances, e.flux, e.limites)):
        assert_array_equal(a, b)
    for a in (g.lignes, g.reactances, e.flux, e.limites, e.hors_service):
        with pytest.raises(ValueError):
            a.flat[0] = 0


def test_agregation_gc_et_parametres_copies():
    g = maille()
    poids = [[1, 2], [0, 3]]
    options = dict(OPTIONS, degre="produit", normalisation_geodesique="somme",
                   normalisation_vecteur="l1", paires_courant="generateur_charge",
                   normalisation_courant="somme", convention_resistance="redondance",
                   generateurs=[0, 4], charges=[1, 5], poids_paires=poids)
    r = calculer_scores(etat(g), **options)
    assert_allclose(r.scores["intermediarite_courant"], intermediarite_courant(
        g, paires="generateur_charge", normalisation="somme", generateurs=[0, 4], charges=[1, 5], poids_paires=poids))
    assert_allclose(r.scores["resistance_locale"], redondance_electrique(g, convention="redondance"))
    poids[0][0] = 999
    assert r.parametres["poids_paires"][0][0] == 1


def test_deux_noeuds_cas_minimal():
    g = graphe(2, [(1, 0)], [4])
    assert_allclose(degre_lignes(g, variante="somme"), [2])
    assert_allclose(degre_lignes(g, variante="produit"), [1])
    assert_allclose(intermediarite_geodesique(g, longueurs="reactances", normalisation="moyenne"), [1])
    assert_allclose(intermediarite_courant(g, paires="uniformes", normalisation="moyenne"), [1])
    assert_allclose(centralite_vecteur_propre(g, normalisation="l2"), [1 / 8])
    assert_allclose(resistance_effective(g), [4])


@pytest.mark.parametrize("n,lignes,x", [
    (1, [], []), (True, [(0, 1)], [1]), (3, [], []),
    (3, [(0, 1)], [1]), (2, [(0, 0)], [1]), (2, [(0, 2)], [1]),
    (2, [(0, 1), (1, 0)], [1, 1]), (2, [(0.0, 1.0)], [1]),
    (2, [(0, 1)], [0]), (2, [(0, 1)], [-1]),
    (2, [(0, 1)], [np.nan]), (2, [(0, 1)], [np.inf]),
    (2, [(0, 1)], [1j]), (2, [(0, 1)], [1, 2]),
])
def test_graphes_invalides(n, lignes, x):
    with pytest.raises(ValueError):
        GrapheLignes(n, np.asarray(lignes).reshape(-1, 2), np.asarray(x))


@pytest.mark.parametrize("champ,valeur", [
    ("flux", [np.nan]), ("flux", [np.inf]), ("flux", [1j]), ("flux", [1, 2]),
    ("limites", [0]), ("limites", [-1]), ("limites", [np.inf]),
    ("hors_service", [True]), ("hors_service", [0]), ("hors_service", [False, False]),
    ("identifiant", ""),
])
def test_etat_invalide_ou_post_retrait(champ, valeur):
    args = dict(graphe=graphe(2, [(0, 1)]), flux=[0], limites=[1], hors_service=[False], identifiant="intact")
    args[champ] = valeur
    with pytest.raises(ValueError):
        EtatInitialIntact(**args)


@pytest.mark.parametrize("options", [
    dict(paires="inconnu"), dict(generateurs=[0]), dict(poids_paires=[[1]]),
    dict(paires="generateur_charge"),
    dict(paires="generateur_charge", generateurs=[0], charges=[0], poids_paires=[[1]]),
    dict(paires="generateur_charge", generateurs=[0, 0], charges=[1], poids_paires=[[1], [1]]),
    dict(paires="generateur_charge", generateurs=[0.0], charges=[1], poids_paires=[[1]]),
    dict(paires="generateur_charge", generateurs=[True], charges=[1], poids_paires=[[1]]),
    dict(paires="generateur_charge", generateurs=[6], charges=[1], poids_paires=[[1]]),
    dict(paires="generateur_charge", generateurs=[0], charges=[], poids_paires=[[]]),
    dict(paires="generateur_charge", generateurs=[0], charges=[1], poids_paires=None),
    *[dict(paires="generateur_charge", generateurs=[0], charges=[1], poids_paires=p)
      for p in ([[0]], [[-1]], [[np.nan]], [[np.inf]], [[1j]], [1], [[1, 2]])],
])
def test_paires_invalides(options):
    with pytest.raises(ValueError):
        intermediarite_courant(maille(), **dict(paires="uniformes", normalisation="moyenne") | options)


@pytest.mark.parametrize("fonction,options", [
    (degre_lignes, dict(variante="force")),
    (intermediarite_geodesique, dict(longueurs="susceptances", normalisation="somme")),
    (intermediarite_geodesique, dict(longueurs="unitaires", normalisation="autre")),
    (centralite_vecteur_propre, dict(normalisation="autre")),
    (intermediarite_courant, dict(paires="uniformes", normalisation="autre")),
    (redondance_electrique, dict(convention="globale")),
])
def test_variantes_inconnues_refusees(fonction, options):
    with pytest.raises(ValueError):
        fonction(maille(), **options)


@pytest.mark.parametrize("tol", [-1, 1, np.nan, np.inf, True, 1j])
def test_tolerance_invalide(tol):
    with pytest.raises(ValueError):
        intermediarite_geodesique(maille(), longueurs="unitaires", normalisation="somme", egalite_rtol=tol)


def test_conditionnement_numerique_signale():
    g = graphe(3, [(0, 1), (1, 2)], [1, 1e15])
    with pytest.raises(ValueError, match="conditionné"):
        resistance_effective(g)
    with pytest.raises(ValueError, match="conditionné"):
        intermediarite_courant(g, paires="uniformes", normalisation="moyenne")
    h = graphe(2, [(0, 1)], [np.nextafter(0.0, 1.0)])
    with pytest.raises(ValueError, match="Susceptances"):
        centralite_vecteur_propre(h, normalisation="l2")


def test_choix_scientifiques_obligatoires():
    with pytest.raises(TypeError):
        calculer_scores(etat(maille()))


def test_seuls_les_temoins_dependent_du_flux_et_des_capacites():
    g = maille()
    a = calculer_scores(etat(g), **OPTIONS)
    b = calculer_scores(etat(g, np.arange(8) + 1, np.arange(8) + 2), **OPTIONS)
    for nom in ("degre", "intermediarite_geodesique", "vecteur_propre",
                "intermediarite_courant", "resistance_locale"):
        assert_array_equal(a.scores[nom], b.scores[nom])
    assert not np.allclose(a.scores["flux_initial_absolu"], b.scores["flux_initial_absolu"])
    assert not np.allclose(a.scores["taux_charge_initial"], b.scores["taux_charge_initial"])


def test_objet_simulation_avec_flux_refuse_comme_etat_initial():
    # Un objet ressemblant à un résultat post-retrait ne doit pas contourner
    # le constructeur d'état intact par simple compatibilité d'attributs.
    g = maille()
    resultat = SimpleNamespace(graphe=g, flux=np.zeros(8), limites=np.ones(8))
    for fonction in (flux_initial_absolu, taux_charge_initial):
        with pytest.raises(TypeError, match="EtatInitialIntact"):
            fonction(resultat)
    with pytest.raises(TypeError, match="EtatInitialIntact"):
        calculer_scores(resultat, **OPTIONS)
    with pytest.raises(TypeError, match="GrapheLignes"):
        EtatInitialIntact(resultat, np.zeros(8), np.ones(8), np.zeros(8, dtype=bool), "invalide")


def test_ponderations_echelle_et_moyenne_stable():
    g = graphe(3, [(0, 1), (1, 2)])
    options = dict(paires="generateur_charge", generateurs=[0], charges=[1, 2])
    moyenne = intermediarite_courant(g, normalisation="moyenne", poids_paires=[[1, 2]], **options)
    assert_allclose(moyenne, [1, 2 / 3])
    assert_allclose(intermediarite_courant(g, normalisation="moyenne", poids_paires=[[1e300, 2e300]], **options), moyenne)
    assert_allclose(intermediarite_courant(g, normalisation="somme", poids_paires=[[3, 6]], **options), moyenne * 9)
    with pytest.raises(ValueError, match="Somme pondérée"):
        intermediarite_courant(g, normalisation="somme", poids_paires=[[1e308, 1e308]], **options)


def test_autres_limites_de_representation_signalees():
    g = graphe(3, [(0, 1), (1, 2)], [1e-300, 1])
    with pytest.raises(ValueError, match="Longueurs trop disparates"):
        intermediarite_geodesique(g, longueurs="reactances", normalisation="somme")
    # Deux sous-graphes identiques reliés par une susceptance presque nulle.
    h = graphe(4, [(0, 1), (1, 2), (2, 3)], [1, 1e15, 1])
    with pytest.raises(ValueError, match="principale numériquement non séparée"):
        centralite_vecteur_propre(h, normalisation="l2")
    with pytest.raises(ValueError, match="Taux de charge"):
        taux_charge_initial(etat(graphe(2, [(0, 1)]), [1e308], [1e-300]))


def test_module_sans_import_du_modele_ou_des_cascades():
    # Processus frais : la collection globale peut déjà avoir importé le fil A.
    code = '''
import importlib.abc
import sys
import numpy as np
interdits = {"reseau", "dispatch", "n1", "cascade", "cascade_n1", "carreras", "evolution"}
class Bloqueur(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("cascade_entropy.") and fullname.split(".")[1] in interdits:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Bloqueur())
from cascade_entropy.centralites_lignes import GrapheLignes, EtatInitialIntact, calculer_scores
g = GrapheLignes(2, np.array([[0, 1]]), np.ones(1))
e = EtatInitialIntact(g, [0], [1], [False], "intact")
r = calculer_scores(e, degre="somme", longueurs="unitaires", normalisation_geodesique="moyenne",
                    normalisation_vecteur="l2", paires_courant="uniformes",
                    normalisation_courant="moyenne", convention_resistance="criticite")
assert len(r.scores) == 7
assert not any("cascade_entropy." + nom in sys.modules for nom in interdits)
'''
    resultat = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert resultat.returncode == 0, resultat.stderr
