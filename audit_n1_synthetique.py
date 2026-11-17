"""Audit N-1 en lecture seule, sur un réseau synthétique (aucune donnée restreinte).

Exécution dans la racine du dépôt cascade-entropy :
    python audit_n1_synthetique.py

L'option --reference-seule exécute un calcul de contrôle indépendant du module
cascade.py. Elle sert à valider l'oracle et la topologie, PAS le moteur du dépôt.

Ce script ne modifie aucun fichier et ne lance aucun commit/push.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class Cas:
    n: int
    lignes: np.ndarray
    reactances: np.ndarray
    generateurs: np.ndarray
    charges: np.ndarray
    demande: np.ndarray
    production_max: np.ndarray
    limites: np.ndarray


def exemple() -> Cas:
    # Triangle 0--1--2--0 et feuille 3 raccordée à 2.
    # e0,e1,e2 appartiennent au cycle, e3 est un pont.
    return Cas(
        n=4,
        lignes=np.array([[0, 1], [1, 2], [2, 0], [2, 3]], dtype=int),
        reactances=np.array([1.0, 1.4, 0.8, 1.0]),
        generateurs=np.array([0], dtype=int),
        charges=np.array([1, 2, 3], dtype=int),
        demande=np.array([1.0, 1.0, 1.0]),
        production_max=np.array([4.0]),
        limites=np.full(4, 6.0),
    )


def composantes(cas: Cas, retiree: int | None) -> list[list[int]]:
    voisins = [[] for _ in range(cas.n)]
    for indice, (u, v) in enumerate(cas.lignes):
        if indice == retiree:
            continue
        voisins[int(u)].append(int(v))
        voisins[int(v)].append(int(u))
    visite = set()
    groupes = []
    for depart in range(cas.n):
        if depart in visite:
            continue
        visite.add(depart)
        file = deque([depart])
        groupe = []
        while file:
            u = file.popleft()
            groupe.append(u)
            for v in voisins[u]:
                if v not in visite:
                    visite.add(v)
                    file.append(v)
        groupes.append(sorted(groupe))
    return groupes


def _lp_flux(n: int, lignes: np.ndarray, reactances: np.ndarray,
             generateurs: np.ndarray, charges: np.ndarray,
             demande: np.ndarray, production_max: np.ndarray,
             limites: np.ndarray) -> tuple[float, np.ndarray]:
    """PL exact sur UN îlot connexe, sans lien fictif entre îlots.

    Valeur de sortie : puissance servie et flux de ses lignes actives.
    La fonction objectif reproduit la préférence du dépôt : +1 production,
    -100 charge servie. Toutes les limites sont non contraignantes dans le banc.
    """
    ng, nc, m = len(generateurs), len(charges), len(lignes)
    if ng == 0 or nc == 0:
        return 0.0, np.zeros(m)
    selection = np.zeros((n, ng + nc))
    selection[generateurs, np.arange(ng)] = 1.0
    selection[charges, ng + np.arange(nc)] = -1.0
    M = np.zeros((m, n))
    if m:
        idx = np.arange(m)
        M[idx, lignes[:, 0]] = 1.0
        M[idx, lignes[:, 1]] = -1.0
    b = 1.0 / reactances
    B = M.T @ (b[:, None] * M)
    autres = np.arange(1, n)
    if n > 1:
        flux_par_variable = ((b[:, None] * M[:, autres])
                             @ np.linalg.solve(B[np.ix_(autres, autres)],
                                               selection[autres]))
    else:
        flux_par_variable = np.zeros((m, ng + nc))
    objectif = np.r_[np.ones(ng), -100.0 * np.ones(nc)]
    egalite = np.r_[np.ones(ng), -np.ones(nc)][None, :]
    inegalites = np.vstack([flux_par_variable, -flux_par_variable])
    bornes = [(0., float(z)) for z in np.r_[production_max, demande]]
    result = linprog(objectif, A_ub=inegalites,
                     b_ub=np.r_[limites, limites],
                     A_eq=egalite, b_eq=np.zeros(1),
                     bounds=bornes, method="highs")
    if not result.success:
        raise RuntimeError(f"L'oracle ne trouve pas de solution : {result.message}")
    return float(result.x[ng:].sum()), flux_par_variable @ result.x


def delestage_exact(cas: Cas, retiree: int | None) -> tuple[float, np.ndarray, list[list[int]]]:
    """Oracle par suppression exacte et résolution séparée des composantes."""
    groupes = composantes(cas, retiree)
    flux_globaux = np.zeros(len(cas.lignes))
    somme_servie = 0.0
    for noeuds in groupes:
        nouveau = {ancien: i for i, ancien in enumerate(noeuds)}
        indices_lignes = [i for i, (u, v) in enumerate(cas.lignes)
                          if i != retiree and int(u) in nouveau and int(v) in nouveau]
        lignes_locales = np.array([(nouveau[int(cas.lignes[i, 0])],
                                    nouveau[int(cas.lignes[i, 1])])
                                   for i in indices_lignes], dtype=int).reshape(-1, 2)
        idx_g = [i for i, nd in enumerate(cas.generateurs) if int(nd) in nouveau]
        idx_c = [i for i, nd in enumerate(cas.charges) if int(nd) in nouveau]
        servis, flux = _lp_flux(
            len(noeuds), lignes_locales, cas.reactances[indices_lignes],
            np.array([nouveau[int(cas.generateurs[i])] for i in idx_g], dtype=int),
            np.array([nouveau[int(cas.charges[i])] for i in idx_c], dtype=int),
            cas.demande[idx_c], cas.production_max[idx_g], cas.limites[indices_lignes],
        )
        somme_servie += servis
        flux_globaux[indices_lignes] = flux
    return float(cas.demande.sum() - somme_servie), flux_globaux, groupes


def approximation_reconstituee(cas: Cas, retiree: int | None) -> tuple[float, np.ndarray]:
    """Contrôle autonome de la logique ×1e6 / ×1e-6 du code public.

    Cette routine est une reconstitution indépendante de la formulation du PL,
    pas un appel au module réel cascade.py.
    """
    morte = np.zeros(len(cas.lignes), dtype=bool)
    if retiree is not None:
        morte[retiree] = True
    x = np.where(morte, cas.reactances * 1e6, cas.reactances)
    limites = np.where(morte, cas.limites * 1e-6, cas.limites)
    servis, flux = _lp_flux(cas.n, cas.lignes, x, cas.generateurs,
                            cas.charges, cas.demande, cas.production_max, limites)
    return float(cas.demande.sum() - servis), flux


def approximation_reelle(cas: Cas, retiree: int | None) -> tuple[float, np.ndarray]:
    """Appel en lecture seule à la fonction interne du moteur de cascade public."""
    from cascade_entropy.reseau import Reseau
    from cascade_entropy.cascade import _redispatch
    reseau = Reseau(
        n_noeuds=cas.n, lignes=cas.lignes, reactances=cas.reactances,
        generateurs=cas.generateurs, charges=cas.charges,
        limites=cas.limites, reference=0,
    )
    morte = np.zeros(reseau.n_lignes, dtype=bool)
    if retiree is not None:
        morte[retiree] = True
    sol = _redispatch(reseau, cas.demande, cas.limites, cas.production_max, morte)
    return sol.delestage_total, np.asarray(sol.flux)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-seule", action="store_true",
                        help="N'utilise pas cascade.py : contrôle analytique indépendant")
    args = parser.parse_args()
    cas = exemple()
    approche = approximation_reconstituee if args.reference_seule else approximation_reelle
    print("AUDIT N−1 — triangle + feuille, données 100 % synthétiques")
    print("Moteur :", "PL indépendant" if args.reference_seule else "cascade._redispatch() du dépôt")
    print("Demande initiale : 3,0 pu ; production maximale : 4,0 pu ; limites : 6,0 pu")
    print("-" * 91)
    print(f"{'retrait':<23} {'îlots':<15} {'S_exact':>11} {'S_approché':>13} {'|écart|':>12}")
    print("-" * 91)
    pires_ecarts = []
    alertes = []
    for nom, retiree in [("aucun", None), ("e0 (non-pont)", 0),
                         ("e1 (non-pont)", 1), ("e2 (non-pont)", 2),
                         ("e3 (pont)", 3)]:
        s_exact, flux_exacts, groupes = delestage_exact(cas, retiree)
        s_approx, flux_approches = approche(cas, retiree)
        ecart = abs(s_exact - s_approx)
        pires_ecarts.append(ecart)
        print(f"{nom:<23} {str(groupes):<15} {s_exact:>11.8f} "
              f"{s_approx:>13.8f} {ecart:>12.3g}")
        if retiree == 3:
            assert len(groupes) == 2, "Le retrait du pont doit créer deux îlots"
            assert np.isclose(s_exact, 1.0, atol=1e-10), "Délestage exact analytique : 1 pu"
        else:
            assert len(groupes) == 1, "Les arêtes de cycle ne doivent pas déconnecter le réseau"
            assert np.isclose(s_exact, 0., atol=1e-10), "Aucune perte sur les autres cas"
        if ecart > 1e-3:
            alertes.append((nom, ecart, s_exact, s_approx))
        if retiree is not None:
            assert abs(flux_approches[retiree]) <= 6e-6 + 1e-7, "Fuite ligne retirée > capacité affaiblie"
    print("-" * 91)
    print(f"5 cas contrôlés. Écart maximal : {max(pires_ecarts):.6g} pu")
    if alertes:
        print("ALERTE — le modèle à ligne affaiblie ne reproduit pas le retrait exact :")
        for nom, ecart, sx, sa in alertes:
            print(f"  {nom} : vrai délestage={sx:.8g} pu, "+
                  f"approximation={sa:.8g} pu, écart={ecart:.8g} pu "+
                  f"({100 * ecart / cas.demande.sum():.3f} % de la demande)")
        print("Ne pas employer cette approximation pour les classements N−1.")
    else:
        print("Aucun désaccord > 1e-3 pu sur CE petit banc seulement.")
    print("IMPORTANT : ce contrôle ne valide pas encore une cascade complète ni les réseaux à 50 nœuds.")
    print("NE PAS publier ce fichier sans une revue de provenance et des changements Git sélectionnés.")


if __name__ == "__main__":
    main()
