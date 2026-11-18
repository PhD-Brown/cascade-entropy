"""Non-unicité et saturations évitables des dispatchs DC N−1.

Une saturation est évitable individuellement si une autre solution à service
maximal la place sous le seuil. Cela ne garantit ni l'évitement simultané des
autres saturations, ni une trajectoire réalisable avec rampes ou délais.
Aucune fonction de ce module n'écrit sur disque.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Real

import numpy as np

from .dispatch import SEUIL_SATURATION
from ._optimisation_dispatch_n1 import construire_ilot, face_optimale, minimiser_charge_max, resoudre_pl
from .n1 import ResultatRedispatchN1, redispatch_exact
from .reseau import Reseau, flux as flux_dc


@dataclass
class VariabiliteFlux:
    """Plages sur la face primaire (pas sur la face de l'objectif secondaire).

    Un flux est déclaré non unique si sa plage dépasse
    ``tolerance_flux[l] = 1e-6 * max(1, C_l)`` en unités de puissance.
    Les lignes retirées ont NaN dans tous les vecteurs réels. Les îlots sans
    production ou demande ont des flux nuls et un minimum maximal nul.
    """

    flux_min: np.ndarray
    flux_max: np.ndarray
    taux_min_possible: np.ndarray
    charge_max_minimale: np.ndarray
    tolerance_flux: np.ndarray
    n_flux_non_uniques: int
    hors_service: np.ndarray
    ilots: list[np.ndarray]
    statut_ilots: list[str]


@dataclass
class DiagnosticVague:
    """Attribution des seules lignes effectivement tombées, avant leur retrait."""

    numero: int
    lignes: list[int]
    evitables: list[int]
    forcees: list[int]
    deja_saturees_intact: list[int]


@dataclass
class DiagnosticCascade:
    """Totaux secondaires ; le déclencheur imposé n'entre jamais au dénominateur.

    Pour une cascade, la part vaut 0 ou 1 ; None sans vague secondaire.
    La moyenne de cet indicateur parmi les cascades AVEC vague donne la part
    dont la première vague ne contient que des saturations évitables.
    """

    vagues: list[DiagnosticVague]
    n_evitables: int
    n_forcees: int
    part_premiere_vague_uniquement_evitable: float | None


@dataclass
class EtatProrata:
    """Référence sans optimisation ; une infaisabilité est conservée et signalée."""

    resultat: ResultatRedispatchN1
    faisable: bool
    violations: tuple[str, ...]


def _seuil(seuil):
    if isinstance(seuil, (bool, np.bool_)) or not isinstance(seuil, Real):
        raise ValueError("Le seuil doit être un réel fini strictement positif.")
    if not np.isfinite(seuil) or seuil <= 0:
        raise ValueError("Le seuil doit être un réel fini strictement positif.")
    return float(seuil)


def lignes_saturees(resultat: ResultatRedispatchN1, seuil=SEUIL_SATURATION):
    """Renvoie (indices, taux) des lignes actives à un taux >= seuil."""
    seuil = _seuil(seuil)
    if not isinstance(resultat, ResultatRedispatchN1):
        raise TypeError("Un ResultatRedispatchN1 est requis.")
    masque = np.asarray(resultat.hors_service)
    taux = np.asarray(resultat.taux_de_charge)
    if masque.ndim != 1 or masque.dtype.kind != "b" or taux.shape != masque.shape:
        raise ValueError("Masque et taux incohérents.")
    if not np.all(np.isfinite(taux[~masque])) or np.any(taux[~masque] < 0):
        raise ValueError("Les taux actifs doivent être finis et non négatifs.")
    ids = np.flatnonzero((~masque) & (taux >= seuil))
    return ids, taux[ids].copy()


def _indices(reseau, noeuds, masque):
    g = np.flatnonzero(np.isin(reseau.generateurs, noeuds))
    c = np.flatnonzero(np.isin(reseau.charges, noeuds))
    l = np.flatnonzero((~masque) & np.isin(reseau.lignes[:, 0], noeuds))
    return g, c, l


@lru_cache(maxsize=1024)
def _plages_ilot(n, reference, lignes, reactances, generateurs, charges,
                 limites, demande, pmax, servie, departage, ciblees):
    """Réutilise les PL d'îlots identiques ; clés et valeurs sont immuables.

    Tous les paramètres physiques et le service cible figurent dans la clé.
    Le cache borné ne contient aucun objet mutable retourné à l'appelant et
    n'effectue aucune écriture sur disque. Un échec de solveur n'est pas caché.
    """
    ilot = Reseau(n, np.array(lignes, dtype=int).reshape(-1, 2), np.array(reactances),
                  np.array(generateurs, dtype=int), np.array(charges, dtype=int),
                  limites=np.array(limites), reference=reference)
    C = ilot.limites
    probleme = face_optimale(ilot, np.array(demande), np.array(pmax), servie)
    M, A, b, E, e, inf, sup = probleme
    _, t, _ = minimiser_charge_max(probleme, C)
    if departage == "min_charge_max":
        A = np.vstack([A, M / C[:, None], -M / C[:, None]])
        b = np.r_[b, np.full(2 * len(C), t + 1e-7)]
    lo, hi = [], []
    for j in ciblees:
        x, _ = resoudre_pl(M[j], A, b, E, e, inf, sup, f"flux minimal local {j}")
        y, _ = resoudre_pl(-M[j], A, b, E, e, inf, sup, f"flux maximal local {j}")
        lo.append(float(M[j] @ x))
        hi.append(float(M[j] @ y))
    return tuple(lo), tuple(hi), t


def variabilite_flux(reseau, demande, puissance_max, hors_service, *, limites=None,
                     departage="highs"):
    """Mesure min/max de chaque flux et le meilleur taux maximal par îlot.

    S* vient du dispatch historique ; le service reste >= S* −
    TOLERANCE_SERVIE * max(1, demande totale de l'îlot). Deux PL par ligne,
    plus un minimax, sont nécessaires. Les erreurs numériques remontent.
    Par défaut, seul l'optimum primaire est imposé. L'option diagnostique
    ``departage="min_charge_max"`` restreint aussi les flux au taux maximal
    minimal (à 1e-7 près en taux, tolérance du solveur), pour mesurer la
    non-unicité résiduelle après ce départage. Aucun précédent de production
    n'est imposé ; cette option ne décrit pas la face min_redispatch.
    """
    return _variabilite_flux(reseau, demande, puissance_max, hors_service,
                            limites=limites, departage=departage)


def _variabilite_flux(reseau, demande, puissance_max, hors_service, *, limites=None,
                      departage="highs", indices=None):
    """Même problème physique ; limite seulement les objectifs de flux calculés.

    Le diagnostic de cascade demande les seules lignes effectivement tombées.
    Toutes les lignes actives restent dans les contraintes de chaque PL.
    Le résultat partiel est interne et n'est pas exposé par l'API publique.
    """
    if not isinstance(departage, str) or departage not in ("highs", "min_charge_max"):
        raise ValueError("Le diagnostic accepte highs ou min_charge_max.")
    base = redispatch_exact(reseau, demande, puissance_max, hors_service, limites=limites)
    d = np.asarray(demande, dtype=float)
    p = np.asarray(puissance_max, dtype=float)
    C = np.asarray(reseau.limites if limites is None else limites, dtype=float)
    lo = np.full(reseau.n_lignes, np.nan)
    hi = lo.copy()
    taux = lo.copy()
    tol_flux = lo.copy()
    tol_flux[~base.hors_service] = 1e-6 * np.maximum(1.0, C[~base.hors_service])
    maxima = np.zeros(len(base.ilots))
    for k, noeuds in enumerate(base.ilots):
        g, c, l = _indices(reseau, noeuds, base.hors_service)
        ciblees = np.arange(len(l)) if indices is None else np.flatnonzero(np.isin(l, indices))
        mesurees = l[ciblees]
        if not len(mesurees):
            continue
        if base.statut_ilots[k] != "resolu":
            lo[mesurees] = hi[mesurees] = taux[mesurees] = 0.0
            continue
        ilot = construire_ilot(reseau, noeuds, g, c, l, C)
        lo[mesurees], hi[mesurees], maxima[k] = _plages_ilot(
            ilot.n_noeuds, ilot.reference, tuple(map(tuple, ilot.lignes)),
            tuple(ilot.reactances), tuple(ilot.generateurs), tuple(ilot.charges),
            tuple(ilot.limites), tuple(d[c]), tuple(p[g]),
            float(base.charge_servie[c].sum()), departage, tuple(ciblees),
        )
        zero_possible = (lo[mesurees] <= 0) & (hi[mesurees] >= 0)
        taux[mesurees] = np.where(zero_possible, 0.0,
            np.minimum(np.abs(lo[mesurees]), np.abs(hi[mesurees])) / C[mesurees])
    return VariabiliteFlux(lo, hi, taux, maxima, tol_flux,
                          int(np.count_nonzero(hi - lo > tol_flux)),
                          base.hors_service.copy(), [a.copy() for a in base.ilots],
                          base.statut_ilots.copy())


def classer_saturations(resultat, variabilite, seuil=SEUIL_SATURATION):
    """Sépare les saturations évitables individuellement et les forcées.

    La comparaison est stricte : taux_min_possible < seuil. Au voisinage
    numérique du seuil, la classification doit être interprétée avec prudence.
    Les deux arguments doivent décrire le même problème physique.
    """
    return _classer_saturations(resultat, variabilite, seuil)


def _classer_saturations(resultat, variabilite, seuil=SEUIL_SATURATION, *, indices=None):
    """Noyau de classement, ciblé en interne pour l'attribution des vagues."""
    ids, _ = lignes_saturees(resultat, seuil)
    if indices is not None:
        ids = np.intersect1d(ids, indices)
    if not isinstance(variabilite, VariabiliteFlux):
        raise TypeError("Une VariabiliteFlux est requise.")
    if (not np.array_equal(resultat.hors_service, variabilite.hors_service)
            or variabilite.taux_min_possible.shape != resultat.hors_service.shape
            or len(resultat.ilots) != len(variabilite.ilots)
            or any(not np.array_equal(a, b) for a, b in zip(resultat.ilots, variabilite.ilots))):
        raise ValueError("Le résultat et la variabilité ne décrivent pas le même état.")
    verifier = np.flatnonzero(~resultat.hors_service) if indices is None else ids
    if not np.all(np.isfinite(variabilite.taux_min_possible[verifier])):
        raise ValueError("Minimum de taux non fini sur une ligne active.")
    evitables = variabilite.taux_min_possible[ids] < seuil
    return ids[evitables].tolist(), ids[~evitables].tolist()


def diagnostiquer_cascade(resultat_cascade, reseau, puissance_max, *, limites=None):
    """Classe chaque vague dans l'état qui la précède, sans attribuer le retrait.

    Les effectifs sont des attributions conditionnelles par état, pas la preuve
    contrefactuelle que supprimer une saturation sauverait toute la cascade.
    """
    from .cascade_n1 import ResultatCascadeN1

    if not isinstance(resultat_cascade, ResultatCascadeN1):
        raise TypeError("Un ResultatCascadeN1 est requis.")
    r = resultat_cascade
    if len(r.etats) != len(r.vagues_secondaires) + 1:
        raise ValueError("Historique incomplet : un état requis avant chaque vague.")
    intact = set(lignes_saturees(r.etat_reference)[0].tolist())
    vagues = []
    for numero, (etat, lignes) in enumerate(zip(r.etats, r.vagues_secondaires), 1):
        lignes = np.asarray(lignes)
        if (lignes.ndim != 1 or lignes.dtype.kind not in "iu" or len(lignes) == 0
                or len(np.unique(lignes)) != len(lignes)
                or np.any(lignes < 0) or np.any(lignes >= reseau.n_lignes)):
            raise ValueError("Indices de vague invalides.")
        v = _variabilite_flux(reseau, r.demande, puissance_max, etat.hors_service,
                             limites=limites, indices=lignes)
        ev, fo = _classer_saturations(etat, v, indices=lignes)
        tombees = set(lignes.tolist())
        if not tombees <= set(ev + fo):
            raise ValueError("Une ligne tombée n'était pas saturée dans l'état précédent.")
        vagues.append(DiagnosticVague(numero, lignes.tolist(), sorted(tombees & set(ev)),
                                     sorted(tombees & set(fo)), sorted(tombees & intact)))
    ne = sum(len(v.evitables) for v in vagues)
    nf = sum(len(v.forcees) for v in vagues)
    part = None if not vagues else float(not vagues[0].forcees)
    return DiagnosticCascade(vagues, ne, nf, part)


def etat_intact_prorata(reseau, demande, puissance_max, *, limites=None):
    """Référence DC sans PL, au prorata des Pmax dans chaque îlot intact.

    S est la demande totale de l'îlot, servie intégralement : aucun délestage
    n'est introduit pour rendre ce témoin faisable. Bornes de production et
    limites sont contrôlées (tolérance 1e-8 * max(1, grandeur)). Une impossibilité
    de bilan sans générateur laisse les flux de cet îlot à NaN et est signalée.
    Cette référence n'est jamais utilisée comme règle de cascade.
    """
    # Validation identique au moteur, sans appeler le solveur.
    from .n1 import composantes_actives

    d = np.asarray(demande, dtype=float)
    p = np.asarray(puissance_max, dtype=float)
    C = np.asarray(reseau.limites if limites is None else limites, dtype=float)
    if (d.shape != (len(reseau.charges),) or p.shape != (len(reseau.generateurs),)
            or C.shape != (reseau.n_lignes,) or not np.all(np.isfinite(d))
            or not np.all(np.isfinite(p)) or not np.all(np.isfinite(C))
            or np.any(d < 0) or np.any(p < 0) or np.any(C <= 0)
            or np.intersect1d(reseau.generateurs, reseau.charges).size):
        raise ValueError("Demandes, Pmax ou limites invalides.")
    masque = np.zeros(reseau.n_lignes, dtype=bool)
    ilots = composantes_actives(reseau, masque)
    production = np.zeros(len(p))
    F = np.zeros(reseau.n_lignes)
    statuts, violations = [], []
    for k, noeuds in enumerate(ilots):
        g, c, l = _indices(reseau, noeuds, masque)
        S, P = float(d[c].sum()), float(p[g].sum())
        if not np.isfinite(S) or not np.isfinite(P):
            raise ValueError("Puissance totale non finie.")
        statuts.append("sans_generateur" if not len(g) else "sans_demande" if S == 0 else "resolu")
        if S == 0:
            continue
        if P == 0:
            violations.append(f"ilot_{k}_bilan_impossible")
            F[l] = np.nan
            continue
        production[g] = S * (p[g] / P)
        if S > P + 1e-8 * max(1.0, P):
            violations.append(f"ilot_{k}_production_insuffisante")
        ilot = construire_ilot(reseau, noeuds, g, c, l, C)
        injections = np.zeros(len(noeuds))
        injections[ilot.generateurs] = production[g]
        injections[ilot.charges] = -d[c]
        F[l] = flux_dc(ilot, injections)
    if np.any(np.abs(F) > C + 1e-8 * np.maximum(1.0, C)):
        violations.append("limites_transport")
    sol = ResultatRedispatchN1(masque, ilots, production, d.copy(), np.zeros(len(d)),
                               F, np.abs(F) / C, statuts)
    return EtatProrata(sol, not violations, tuple(violations))
