"""Validation du module `evolution`.

Critères de la docstring du module : en mode independant sans avarie, chaque jour
équivaut à un dispatch isolé ; en mode auto_organise, la capacité ne diminue jamais,
chaque mise à niveau ajoute exactement k·P̄_D(t)/N_G, une marge_seuil de -1 ne
déclenche aucune mise à niveau, la graine fixée rend tout reproductible, et le nombre
effectif reste borné par le nombre de lignes en service.

Plusieurs cas se résolvent à la main sur la chaîne à 3 nœuds (générateur en 0,
charges en 1 et 2). Un seul run de 3000 jours, partagé entre plusieurs tests, vérifie
que le mécanisme d'auto-organisation se déclenche plusieurs fois et tient la marge.

Ces tests ne démontrent pas que la dynamique reproduit Carreras et al. : 3000 jours
sans retrait de transitoire ne suffisent pas à un régime stationnaire.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.cascade import journee
from cascade_entropy.dispatch import demande_uniforme, resoudre
from cascade_entropy.entropie import nombre_effectif
from cascade_entropy.evolution import (
    JOURS_PAR_AN,
    Parametres,
    ecrire,
    marge_moyenne,
    puissance_max_pour_marge,
    series,
    simuler,
)
from cascade_entropy.reseau import Reseau, arbre, limites_par_niveau, matrice_de_flux

GRAINE = 20260915
MARGE_SEUIL = 0.10
MARGE_INITIALE = 0.11
K = 0.02


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


def parametres_chaine(**changements) -> Parametres:
    """Paramètres sur la chaîne : demande constante, sans mise à niveau par défaut."""
    base = dict(mode="auto_organise", demande_initiale=2.0,
                puissance_max=np.array([10.0]), g=0.0, p0=0.0, p1=1.0,
                lambda_=1.0, k=0.02, marge_seuil=-1.0, mu=1.4)
    base.update(changements)
    return Parametres(**base)


def parametres_arbre(mode="auto_organise", marge_initiale=MARGE_INITIALE, **changements):
    """
    Configuration de démonstration : la capacité de départ est celle de la calibration
    (p_c), et la demande initiale est choisie pour fixer la marge initiale.
    """
    reseau, limites, p_max, p_c = arbre_calibre()
    base = dict(mode=mode, demande_initiale=p_c / (1.0 + marge_initiale),
                puissance_max=p_max, g=0.9, p0=1e-3, p1=0.3, limites=limites,
                marge_seuil=MARGE_SEUIL, mu=1.05)
    base.update(changements)
    return reseau, Parametres(**base)


@pytest.fixture(scope="module")
def run_3000_jours():
    """Un run de 3000 jours en mode auto_organise, partagé par plusieurs tests."""
    reseau, parametres = parametres_arbre()
    jours = simuler(reseau, 3000, parametres, np.random.default_rng(GRAINE))
    return reseau, parametres, jours, series(jours)


def demande_du_jour(parametres: Parametres, t: np.ndarray) -> np.ndarray:
    """P̄_D(t) = P0·e^((λ-1)t) avec λ le facteur quotidien."""
    lambda_jour = parametres.lambda_ ** (1.0 / JOURS_PAR_AN)
    return parametres.demande_initiale * np.exp((lambda_jour - 1.0) * t)


# --------------------------------------------------------------------------
# Mode independant
# --------------------------------------------------------------------------


def test_independant_sans_avarie_equivaut_a_un_dispatch_isole():
    """
    Vérifie le critère central du mode independant : avec p0 = p1 = 0, chaque jour est
    un `resoudre()` isolé sur la demande tirée ce jour-là. On rejoue le même flux
    aléatoire avec `journee()` pour retrouver la demande de chaque jour. Le délestage
    vient du dispatch final, qui est ici le premier et le seul.
    """
    reseau, limites, p_max, p_c = arbre_calibre()
    parametres = Parametres(mode="independant", demande_initiale=1.2 * p_c,
                            puissance_max=p_max, g=0.3, p0=0.0, p1=0.0, limites=limites)
    jours = simuler(reseau, 20, parametres, np.random.default_rng(GRAINE))

    rng = np.random.default_rng(GRAINE)
    moyenne = demande_uniforme(reseau, 1.2 * p_c)
    for jour in jours:
        tirage = journee(reseau, moyenne, p_max, 0.0, 0.0, 0.3, rng, limites=limites)
        direct = resoudre(reseau, tirage.demande, limites=limites, puissance_max=p_max)
        assert jour.delestage_total == pytest.approx(direct.delestage_total, rel=1e-12)
        assert jour.taux_maximal == pytest.approx(direct.taux_maximal, rel=1e-12)
        assert jour.nombre_effectif == pytest.approx(nombre_effectif(direct.flux), rel=1e-12)
        assert jour.n_lignes_en_service == reseau.n_lignes
    assert any(j.delestage_total > 1.0 for j in jours)


def test_independant_ignore_la_dynamique_lente():
    """
    Vérifie que le mode independant garde un réseau fixe : lambda_, k, marge_seuil et mu
    n'ont aucun effet, et les champs propres à l'auto-organisation sont absents.
    """
    reseau, parametres = parametres_arbre(mode="independant", marge_initiale=0.0)
    _, autres = parametres_arbre(mode="independant", marge_initiale=0.0, lambda_=3.0,
                                 k=0.5, marge_seuil=9.0, mu=3.0)
    a = simuler(reseau, 30, parametres, np.random.default_rng(GRAINE))
    b = simuler(reseau, 30, autres, np.random.default_rng(GRAINE))
    assert series(a).keys() == series(b).keys()
    for cle, valeurs in series(a).items():
        assert np.array_equal(valeurs, series(b)[cle], equal_nan=True)
    assert all(j.capacite_totale_generateurs is None and j.marge_moyenne is None
               and j.generateurs_ameliores is None and j.lignes_renforcees is None
               for j in a)
    assert "capacite_totale_generateurs" not in series(a)


# --------------------------------------------------------------------------
# Mode auto_organise : mises à niveau des générateurs
# --------------------------------------------------------------------------


def test_marge_seuil_moins_un_aucune_mise_a_niveau_meme_sous_la_demande():
    """
    Avec marge_seuil = -1, la condition (c) ne se déclenche jamais, même quand la
    demande dépasse la capacité installée (marge négative). Avec marge_seuil = 0 dans
    la même situation, elle se déclenche : c'est pourquoi le critère est écrit avec -1.
    """
    reseau, parametres = parametres_arbre(marge_initiale=-0.3, lambda_=1.5, marge_seuil=-1.0)
    jours = simuler(reseau, 200, parametres, np.random.default_rng(GRAINE))
    s = series(jours)
    assert s["n_mises_a_niveau"].sum() == 0
    assert np.all(s["capacite_totale_generateurs"] == s["capacite_totale_generateurs"][0])
    assert np.all(s["marge_moyenne"] < 0)

    _, zero = parametres_arbre(marge_initiale=-0.3, lambda_=1.5, marge_seuil=0.0)
    assert series(simuler(reseau, 200, zero, np.random.default_rng(GRAINE)))[
        "n_mises_a_niveau"].sum() > 0


def test_capacite_croissante_et_increment_exact(run_3000_jours):
    """
    Vérifie que la capacité totale ne diminue jamais et que chaque mise à niveau ajoute
    exactement k·P̄_D(t)/N_G, avec P̄_D(t) la demande totale (et non la capacité).
    La capacité enregistrée au jour t est celle en vigueur ce jour-là : la différence
    avec le jour suivant est donc l'effet des mises à niveau du jour t.
    """
    reseau, parametres, jours, s = run_3000_jours
    capacite = s["capacite_totale_generateurs"]
    assert np.all(np.diff(capacite) >= 0)

    demande = demande_du_jour(parametres, s["jour"].astype(float))
    attendu = s["n_mises_a_niveau"][:-1] * parametres.k * demande[:-1] / reseau.generateurs.size
    assert np.diff(capacite) == pytest.approx(attendu, rel=1e-9, abs=1e-9)


def test_mecanisme_se_declenche_plusieurs_fois_et_tient_la_marge(run_3000_jours):
    """
    Vérifie, avec une marge initiale à 0,11 pour un seuil de 0,10, que le mécanisme
    se déclenche plusieurs fois sur 3000 jours et que la marge reste ensuite autour
    du seuil : la demande la fait baisser, les mises à niveau la ramènent. Elle
    descend au plus de 1e-3 sous le seuil, car la demande croît entre deux mises à
    niveau.
    """
    reseau, parametres, jours, s = run_3000_jours
    marge = s["marge_moyenne"]
    assert 0.0 < marge[0] - MARGE_SEUIL < 0.011
    assert s["n_mises_a_niveau"].sum() >= 50

    premier = int(np.argmax(s["n_mises_a_niveau"] > 0))
    assert premier > 1
    assert np.all(marge[:premier + 1] >= MARGE_SEUIL - 1e-3)
    assert marge[premier + 1:].min() >= MARGE_SEUIL - 1e-3
    assert marge[premier + 1:].max() <= MARGE_SEUIL + 0.01
    # Une seule mise à niveau par jour suffit ici : l'ajout est de 0,17 % de la demande.
    assert s["n_mises_a_niveau"].max() == 1


def test_generateurs_ameliores_sont_des_generateurs(run_3000_jours):
    """Vérifie que seuls des nœuds de `reseau.generateurs` sont améliorés."""
    reseau, parametres, jours, s = run_3000_jours
    ameliores = np.concatenate([j.generateurs_ameliores for j in jours])
    assert ameliores.size == s["n_mises_a_niveau"].sum()
    assert set(ameliores.tolist()) <= set(reseau.generateurs.tolist())
    assert len(set(ameliores.tolist())) > 1


def test_condition_b_bloque_les_mises_a_niveau_sans_boucle_infinie():
    """
    Sur la chaîne, la ligne 0 limite le générateur à 1,55. Avec un ajout de 0,1, la
    capacité monte de 1,0 à 1,5, puis plus rien : aucune mise à niveau ne peut dépasser
    ce que les lignes incidentes évacuent, et la simulation termine malgré une marge
    toujours sous le seuil.
    """
    reseau = chaine(limites=(1.55, 10.0))
    parametres = parametres_chaine(demande_initiale=1.0, puissance_max=np.array([1.0]),
                                   p1=0.0, k=0.1, marge_seuil=0.9)
    jours = simuler(reseau, 12, parametres, np.random.default_rng(GRAINE))
    s = series(jours)
    assert s["capacite_totale_generateurs"].max() < 1.55
    assert s["capacite_totale_generateurs"][-1] == pytest.approx(1.5)
    assert s["n_mises_a_niveau"].sum() == 5
    assert s["n_mises_a_niveau"][-3:].sum() == 0
    assert s["marge_moyenne"][-1] < 0.9


def test_auto_organise_reproductible_a_graine_fixee():
    """
    Vérifie que la séquence de générateurs mis à niveau, les capacités et le délestage
    sont identiques à graine fixée, et que le choix des générateurs dépend de la graine.
    """
    reseau, parametres = parametres_arbre(marge_initiale=0.1005)

    def lancer(graine):
        return simuler(reseau, 300, parametres, np.random.default_rng(graine))

    a, b, autre = lancer(GRAINE), lancer(GRAINE), lancer(GRAINE + 1)
    suite = lambda jours: [j.generateurs_ameliores.tolist() for j in jours]
    assert suite(a) == suite(b)
    assert any(suite(a))
    for cle, valeurs in series(a).items():
        assert np.array_equal(valeurs, series(b)[cle], equal_nan=True)
    assert suite(a) != suite(autre)


# --------------------------------------------------------------------------
# Mode auto_organise : renforcement des lignes par mu
# --------------------------------------------------------------------------


def test_mu_renforce_la_ligne_tombee_par_surcharge_et_arrete_la_cascade():
    """
    Sur la chaîne, la ligne 0 (limite 1,5) sature pour une demande de 2 et, avec p1 = 1,
    tombe : blackout de 2. Avec mu = 1,4, sa limite passe à 2,1 : le lendemain son taux
    est 2/2,1 = 0,95, sous le seuil de saturation, donc plus de blackout. Le premier
    jour montre aussi les deux dispatchs : M_max du premier (1,0), délestage du dernier.
    """
    jours = simuler(chaine(), 3, parametres_chaine(mu=1.4), np.random.default_rng(GRAINE))
    assert jours[0].lignes_renforcees.tolist() == [0]
    assert jours[0].taux_maximal == pytest.approx(1.0)
    assert jours[0].delestage_total == pytest.approx(2.0, abs=1e-5)
    for jour in jours[1:]:
        assert jour.lignes_renforcees.size == 0
        assert jour.delestage_total == pytest.approx(0.0, abs=1e-6)


def test_mu_faible_la_ligne_retombe_et_est_renforcee_a_nouveau():
    """
    Avec mu = 1,2, la limite passe de 1,5 à 1,8 : encore sous la demande de 2, donc la
    ligne sature et retombe le deuxième jour. Elle passe alors à 2,16, au-dessus de 2
    (taux 0,93) : le troisième jour, plus de blackout ni de renforcement.
    """
    jours = simuler(chaine(), 3, parametres_chaine(mu=1.2), np.random.default_rng(GRAINE))
    assert [j.lignes_renforcees.tolist() for j in jours] == [[0], [0], []]
    assert [j.delestage_total for j in jours] == pytest.approx([2.0, 2.0, 0.0], abs=1e-5)


def test_ligne_tombee_par_accident_n_est_pas_renforcee():
    """
    Avec p0 = 1, toutes les lignes tombent par accident : blackout total, mais aucune
    avarie par surcharge, donc aucun renforcement.
    """
    jours = simuler(chaine(), 2, parametres_chaine(p0=1.0, p1=0.0),
                    np.random.default_rng(GRAINE))
    assert all(j.delestage_total == pytest.approx(2.0, abs=1e-4) for j in jours)
    assert all(j.lignes_renforcees.size == 0 for j in jours)
    assert all(j.n_lignes_en_service == 0 for j in jours)
    assert all(np.isnan(j.nombre_effectif) for j in jours)


def test_parametres_et_reseau_non_modifies():
    """Vérifie que la simulation ne modifie ni les tableaux de `Parametres` ni le réseau."""
    reseau = chaine()
    parametres = parametres_chaine(mu=1.4, marge_seuil=0.9, k=0.1)
    puissance, limites = parametres.puissance_max.copy(), reseau.limites.copy()
    reactances = reseau.reactances.copy()
    jours = simuler(reseau, 5, parametres, np.random.default_rng(GRAINE))
    assert any(j.lignes_renforcees.size for j in jours)
    assert np.array_equal(parametres.puissance_max, puissance)
    assert np.array_equal(reseau.limites, limites)
    assert np.array_equal(reseau.reactances, reactances)


# --------------------------------------------------------------------------
# Nombre effectif
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["independant", "auto_organise"])
def test_nombre_effectif_borne_par_les_lignes_en_service(mode):
    """
    Vérifie, dans les deux modes, que le nombre effectif, quand il est défini, reste
    entre 1 et le nombre de lignes en service au premier dispatch. Avec p0 = 0,02, des
    lignes tombent par accident avant le dispatch, ce qui exerce la borne.
    """
    reseau, parametres = parametres_arbre(mode=mode, p0=0.02)
    jours = simuler(reseau, 150, parametres, np.random.default_rng(GRAINE))
    definis = [j for j in jours if not np.isnan(j.nombre_effectif)]
    assert len(definis) > 100
    assert any(j.n_lignes_en_service < reseau.n_lignes for j in jours)
    for j in definis:
        assert 1.0 - 1e-9 <= j.nombre_effectif <= j.n_lignes_en_service + 1e-9
        assert j.n_lignes_en_service <= reseau.n_lignes


def test_nombre_effectif_borne_sur_le_run_long(run_3000_jours):
    """Même borne, sur les 3000 jours du run partagé."""
    reseau, parametres, jours, s = run_3000_jours
    n_eff = s["nombre_effectif"]
    definis = ~np.isnan(n_eff)
    assert definis.mean() > 0.99
    assert np.all(n_eff[definis] >= 1.0 - 1e-9)
    assert np.all(n_eff[definis] <= s["n_lignes_en_service"][definis] + 1e-9)


# --------------------------------------------------------------------------
# Utilitaires, séries et écriture
# --------------------------------------------------------------------------


def test_puissance_max_pour_marge_donne_la_marge_voulue():
    """Vérifie que la capacité de départ uniforme produit exactement la marge visée."""
    reseau = arbre(4)
    for marge in (0.0, 0.11, 0.5):
        p_max = puissance_max_pour_marge(reseau, 1000.0, marge)
        assert p_max.shape == reseau.generateurs.shape
        assert marge_moyenne(p_max, 1000.0) == pytest.approx(marge)
    with pytest.raises(ValueError):
        puissance_max_pour_marge(reseau, 1000.0, -1.0)
    with pytest.raises(ValueError):
        puissance_max_pour_marge(reseau, 0.0, 0.1)


def test_series_et_ecrire_aller_retour(tmp_path):
    """Vérifie que le fichier .npz relu contient exactement les séries, une valeur par jour."""
    reseau, parametres = parametres_arbre(marge_initiale=0.1005)
    jours = simuler(reseau, 40, parametres, np.random.default_rng(GRAINE))
    s = series(jours)
    chemin = tmp_path / "sous" / "dossier" / "evolution.npz"
    ecrire(s, chemin)
    with np.load(chemin) as relu:
        assert set(relu.files) == set(s)
        for cle in s:
            assert np.array_equal(relu[cle], s[cle], equal_nan=True)
            assert relu[cle].shape == (40,)
    assert s["jour"].tolist() == list(range(1, 41))


def test_ecrire_rejette_les_entrees_invalides(tmp_path):
    """Vérifie le rejet d'un dictionnaire vide, de longueurs inégales et d'un mauvais suffixe."""
    with pytest.raises(ValueError):
        ecrire({}, tmp_path / "a.npz")
    with pytest.raises(ValueError):
        ecrire({"x": np.zeros(3), "y": np.zeros(4)}, tmp_path / "a.npz")
    with pytest.raises(ValueError):
        ecrire({"x": np.zeros(3)}, tmp_path / "a.txt")
    with pytest.raises(ValueError):
        series([])


# --------------------------------------------------------------------------
# Validation des entrées
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_jours", [0, -1, True, 2.5, "10", None])
def test_n_jours_invalide_rejete(n_jours):
    """Vérifie que le nombre de jours doit être un entier >= 1 (les booléens sont exclus)."""
    with pytest.raises(ValueError):
        simuler(chaine(), n_jours, parametres_chaine(), np.random.default_rng(GRAINE))


@pytest.mark.parametrize("changement", [
    {"mode": "autre"},
    {"demande_initiale": 0.0},
    {"demande_initiale": -1.0},
    {"demande_initiale": float("nan")},
    {"demande_initiale": True},
    {"puissance_max": np.array([10.0, 10.0])},
    {"puissance_max": np.array([-1.0])},
    {"lambda_": 0.0},
    {"lambda_": float("inf")},
    {"k": 0.0},
    {"k": -0.1},
    {"mu": 0.99},
    {"mu": None},
    {"marge_seuil": None},
    {"marge_seuil": float("nan")},
])
def test_parametres_invalides_rejetes(changement):
    """Vérifie le rejet des paramètres mal formés ou manquants en mode auto_organise."""
    with pytest.raises(ValueError):
        simuler(chaine(), 3, parametres_chaine(**changement), np.random.default_rng(GRAINE))


def test_mu_et_marge_seuil_facultatifs_en_mode_independant():
    """En mode independant, `mu` et `marge_seuil` ne sont pas requis."""
    parametres = parametres_chaine(mode="independant", mu=None, marge_seuil=None, p1=0.0)
    assert len(simuler(chaine(), 3, parametres, np.random.default_rng(GRAINE))) == 3


def test_probabilites_invalides_rejetees_par_journee():
    """Vérifie que p0, p1 et g hors de [0, 1] sont rejetés dès le premier jour."""
    for nom in ("p0", "p1", "g"):
        with pytest.raises(ValueError):
            simuler(chaine(), 3, parametres_chaine(**{nom: 1.5}), np.random.default_rng(GRAINE))


def test_generateur_aleatoire_et_limites_obligatoires():
    """Vérifie qu'un rng non `Generator` et un réseau sans limites sont rejetés."""
    with pytest.raises(TypeError):
        simuler(chaine(), 3, parametres_chaine(), GRAINE)
    sans_limites = chaine()
    sans_limites.limites = None
    with pytest.raises(ValueError):
        simuler(sans_limites, 3, parametres_chaine(), np.random.default_rng(GRAINE))
