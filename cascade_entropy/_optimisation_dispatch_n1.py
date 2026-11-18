"""Programmes linéaires secondaires N−1, sans changer le dispatch historique."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .dispatch import TOLERANCE_SERVIE, _linprog_robuste, _matrices
from .reseau import Reseau

DEPARTAGES = ("highs", "min_charge_max", "min_redispatch")


def resoudre_pl(c, A, b, E, e, inf, sup, etape):
    """Résout un nouveau PL avec précision renforcée, puis les secours existants.

    Les tolérances primale et duale valent 1e-10. Les secours résolvent toujours
    le même objectif secondaire, jamais le seul objectif primaire. Une solution
    n'est acceptée qu'après contrôle ; seules les erreurs de bornes inférieures
    à 1e-9 fois l'échelle sont ramenées exactement dans les bornes. Les bilans
    et contraintes sont ensuite contrôlés sur le vecteur ainsi normalisé.
    """
    echelle = max(1.0, float(np.max(np.abs(sup[np.isfinite(sup)]))))
    tol = 1e-9 * echelle

    def controler(x):
        if (not np.all(np.isfinite(x)) or np.any(x < inf - tol) or np.any(x > sup + tol)):
            return None
        x = np.clip(x, inf, sup)
        if np.any(A @ x > b + tol) or np.any(np.abs(E @ x - e) > tol):
            return None
        return x

    messages = []
    for nom, methode, facteur, presolve, facteur_cout in (
        ("highs", "highs", 1.0, True, 1.0),
        ("simplexe_precis_n1", "highs-ds", echelle, False, 1.0),
        ("interieur_precis_n1", "highs-ipm", echelle, False, 1.0),
        ("interieur_cout_echelle_n1", "highs-ipm", 1.0, False, 0.01),
    ):
        sol = linprog(c * facteur_cout, A_ub=A, b_ub=b / facteur, A_eq=E, b_eq=e / facteur,
                      bounds=np.c_[inf / facteur, sup / facteur], method=methode,
                      options={"primal_feasibility_tolerance": 1e-10,
                               "dual_feasibility_tolerance": 1e-10,
                               "presolve": presolve})
        if sol.success:
            x = controler(sol.x * facteur)
            if x is not None:
                return x, nom
            messages.append(f"{nom}: solution hors bornes ou non admissible")
        else:
            messages.append(f"{nom}: échec du solveur")
    x, strategie = _linprog_robuste(c, A, b, E, e, inf, sup, etape)
    x = controler(x)
    if x is None:
        raise RuntimeError(f"Solution hors bornes ou non admissible après {etape} : {messages}")
    return x, "secours_n1_" + strategie


def valider_options(departage, production_precedente, taille):
    """Valide une règle et la production antérieure dans l'ordre des générateurs."""
    if not isinstance(departage, str) or departage not in DEPARTAGES:
        raise ValueError(f"departage doit appartenir à {DEPARTAGES}.")
    if departage != "min_redispatch":
        if production_precedente is not None:
            raise ValueError("production_precedente est réservée à min_redispatch.")
        return None
    p = np.asarray(production_precedente, dtype=float)
    if p.shape != (taille,) or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("min_redispatch exige une production_precedente finie, non négative par générateur.")
    # Une ancienne production peut dépasser la nouvelle capacité disponible.
    return p.copy()


def construire_ilot(reseau, noeuds, ids_g, ids_c, ids_l, limites):
    """Construit une copie locale en conservant l'ordre des lignes et des rôles."""
    locaux = np.full(reseau.n_noeuds, -1, dtype=int)
    locaux[noeuds] = np.arange(len(noeuds))
    return Reseau(
        int(len(noeuds)), locaux[reseau.lignes[ids_l]].copy(),
        reseau.reactances[ids_l].copy(), locaux[reseau.generateurs[ids_g]],
        locaux[reseau.charges[ids_c]], limites=limites[ids_l].copy(),
        reference=max(0, int(locaux[reseau.reference])),
    )


def face_optimale(ilot, demande, pmax, servie):
    """Contraintes DC avec service >= S* − tol, et bornes physiques inchangées.

    La tolérance est celle du moteur : TOLERANCE_SERVIE * max(1, somme d).
    La face est donc un voisinage numérique de la face optimale primaire.
    """
    _, _, M, ng, nc = _matrices(ilot, None)
    n = ng + nc
    service = np.r_[np.zeros(ng), -np.ones(nc)]
    A = np.vstack([M, -M, service])
    tol = TOLERANCE_SERVIE * max(1.0, float(demande.sum()))
    b = np.r_[ilot.limites, ilot.limites, -max(0.0, servie - tol)]
    E = np.r_[np.ones(ng), -np.ones(nc)][None, :]
    return M, A, b, E, np.zeros(1), np.zeros(n), np.r_[pmax, demande]


def minimiser_charge_max(probleme, limites):
    """Minimise t avec |F| <= t C ; d'autres flux peuvent rester non uniques."""
    M, A, b, E, e, inf, sup = probleme
    n = len(inf)
    cout = np.r_[np.zeros(n), 1.0]
    # Garder aussi les limites physiques : t ne permet jamais de les dépasser.
    A2 = np.vstack([np.c_[A, np.zeros(len(b))],
                    np.c_[M / limites[:, None], -np.ones(len(limites))],
                    np.c_[-M / limites[:, None], -np.ones(len(limites))]])
    b2 = np.r_[b, np.zeros(2 * len(limites))]
    x, strategie = resoudre_pl(
        cout, A2, b2, np.c_[E, np.zeros(len(e))], e,
        np.r_[inf, 0.0], np.r_[sup, 1.0], "N−1 min_charge_max",
    )
    return x[:-1], float(x[-1]), strategie


def departager(ilot, demande, pmax, servie, regle, precedente):
    """Résout l'objectif secondaire ; aucun repli vers la règle highs."""
    probleme = face_optimale(ilot, demande, pmax, servie)
    M, A, b, E, e, inf, sup = probleme
    if regle == "min_charge_max":
        x, _, strategie = minimiser_charge_max(probleme, ilot.limites)
    else:
        ng = len(pmax)
        n = len(inf)
        G = np.eye(n)[:ng]
        A2 = np.vstack([np.c_[A, np.zeros((len(b), ng))],
                        np.c_[G, -np.eye(ng)], np.c_[-G, -np.eye(ng)]])
        x, strategie = resoudre_pl(
            np.r_[np.zeros(n), np.ones(ng)], A2, np.r_[b, precedente, -precedente],
            np.c_[E, np.zeros((len(e), ng))], e,
            np.r_[inf, np.zeros(ng)], np.r_[sup, np.full(ng, np.inf)],
            "N−1 min_redispatch",
        )
        x = x[:n]
    return x[:len(pmax)], x[len(pmax):], M @ x, strategie
