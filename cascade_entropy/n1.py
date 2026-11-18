"""Retrait exact d'une ligne et redispatch DC par îlot (Projet I, phase A).

Ce module ne modifie ni cascade.py ni les conventions historiques de Carreras.
Il résout uniquement l'état qui suit un ensemble de lignes hors service ; la
propagation des défaillances secondaires sera ajoutée et validée séparément.

Aucune donnée externe n'est requise. Les indices des lignes sont conservés dans
l'ordre du réseau d'origine ; une ligne retirée a un flux EXACTEMENT nul.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dispatch import resoudre
from ._optimisation_dispatch_n1 import departager, valider_options
from .reseau import Reseau


@dataclass
class ResultatRedispatchN1:
    """Grandeurs post-retrait exprimées dans les indices du réseau initial."""

    hors_service: np.ndarray
    ilots: list[np.ndarray]
    production: np.ndarray
    charge_servie: np.ndarray
    delestage: np.ndarray
    flux: np.ndarray
    taux_de_charge: np.ndarray
    statut_ilots: list[str]
    n_secours: int = 0

    @property
    def delestage_total(self) -> float:
        return float(self.delestage.sum())

    @property
    def demande_totale(self) -> float:
        return float((self.charge_servie + self.delestage).sum())

    @property
    def fraction_delestee(self) -> float:
        if self.demande_totale <= 0:
            raise ValueError("Fraction de délestage indéfinie lorsque la demande totale est nulle.")
        return self.delestage_total / self.demande_totale


def composantes_actives(reseau: Reseau, hors_service: np.ndarray) -> list[np.ndarray]:
    """Renvoie les composantes connexes des seules lignes EN SERVICE.

    Les nœuds isolés figurent explicitement dans la liste des composantes.
    Les identifiants des nœuds ne sont pas modifiés.
    """
    masque = np.asarray(hors_service)
    if masque.shape != (reseau.n_lignes,) or masque.dtype.kind != "b":
        raise ValueError("hors_service doit être un masque booléen par ligne.")
    voisins: list[list[int]] = [[] for _ in range(reseau.n_noeuds)]
    for indice, (u, v) in enumerate(reseau.lignes):
        if not masque[indice]:
            voisins[int(u)].append(int(v))
            voisins[int(v)].append(int(u))
    visite = np.zeros(reseau.n_noeuds, dtype=bool)
    ilots = []
    for origine in range(reseau.n_noeuds):
        if visite[origine]:
            continue
        pile = [origine]
        visite[origine] = True
        courant = []
        while pile:
            u = pile.pop()
            courant.append(u)
            for v in voisins[u]:
                if not visite[v]:
                    visite[v] = True
                    pile.append(v)
        ilots.append(np.array(sorted(courant), dtype=int))
    return ilots


def redispatch_exact(
    reseau: Reseau,
    demande: np.ndarray,
    puissance_max: np.ndarray,
    hors_service: np.ndarray,
    *,
    limites: np.ndarray | None = None,
    departage: str = "highs",
    production_precedente: np.ndarray | None = None,
) -> ResultatRedispatchN1:
    """Redispatch exact sur les composantes d'un réseau avec lignes retirées.

    Les entrées sont immuables pour cette fonction. Les charges d'un îlot sans
    générateur sont entièrement délestées. La production d'un îlot sans charge
    est nulle. Pour chaque îlot avec génération ET charge, on réutilise le
    programme linéaire public de :func:`cascade_entropy.dispatch.resoudre`.

    ``highs`` conserve exactement l'appel historique. ``min_charge_max``
    minimise le taux maximal à service optimal ; les autres flux peuvent rester
    non uniques. ``min_redispatch`` minimise la somme des variations absolues
    de production et exige ``production_precedente``, alignée sur les générateurs.
    Le départage scientifique définitif reste à discuter avec Patrick.
    """
    precedente = valider_options(departage, production_precedente, len(reseau.generateurs))
    if reseau.n_noeuds < 1:
        raise ValueError("Le réseau doit posséder au moins un nœud.")
    if np.intersect1d(reseau.generateurs, reseau.charges).size:
        raise ValueError("Un nœud ne peut être à la fois générateur et charge dans cette convention.")

    demande = np.asarray(demande, dtype=float)
    puissance_max = np.asarray(puissance_max, dtype=float)
    masque = np.asarray(hors_service)
    if masque.shape != (reseau.n_lignes,) or masque.dtype.kind != "b":
        raise ValueError("hors_service doit être un masque booléen par ligne.")
    if demande.shape != (len(reseau.charges),):
        raise ValueError("Une demande par nœud consommateur est requise.")
    if puissance_max.shape != (len(reseau.generateurs),):
        raise ValueError("Une capacité par générateur est requise.")
    if not np.all(np.isfinite(demande)) or np.any(demande < 0):
        raise ValueError("La demande doit être finie et positive ou nulle.")
    if not np.all(np.isfinite(puissance_max)) or np.any(puissance_max < 0):
        raise ValueError("Les capacités de génération doivent être finies et positives ou nulles.")

    limites = reseau.limites if limites is None else np.asarray(limites, dtype=float)
    if limites is None:
        raise ValueError("Les capacités des lignes doivent être fournies.")
    limites = np.asarray(limites, dtype=float)
    if limites.shape != (reseau.n_lignes,) or not np.all(np.isfinite(limites)) or np.any(limites <= 0):
        raise ValueError("Les capacités des lignes doivent être finies et strictement positives.")

    ilots = composantes_actives(reseau, masque)
    production = np.zeros(len(reseau.generateurs))
    charge_servie = np.zeros(len(reseau.charges))
    flux = np.zeros(reseau.n_lignes)
    statut_ilots = []
    n_secours = 0

    # Identifiants d'îlot pour chaque nœud : évite toute ambiguïté de correspondance.
    appartenance = np.empty(reseau.n_noeuds, dtype=int)
    for k, noeuds in enumerate(ilots):
        appartenance[noeuds] = k

    for k, noeuds in enumerate(ilots):
        indices_g = np.flatnonzero(appartenance[reseau.generateurs] == k)
        indices_c = np.flatnonzero(appartenance[reseau.charges] == k)
        indices_l = np.flatnonzero(
            (~masque)
            & (appartenance[reseau.lignes[:, 0]] == k)
            & (appartenance[reseau.lignes[:, 1]] == k)
        )

        if indices_g.size == 0:
            # L'îlot peut être entièrement déconnecté : aucun courant fictif.
            statut_ilots.append("sans_generateur")
            continue
        if indices_c.size == 0 or float(demande[indices_c].sum()) == 0.0:
            statut_ilots.append("sans_demande")
            continue

        # Un îlot connexe avec des charges et des générateurs comporte des arêtes.
        ancien_vers_local = np.full(reseau.n_noeuds, -1, dtype=int)
        ancien_vers_local[noeuds] = np.arange(noeuds.size)
        lignes_locales = ancien_vers_local[reseau.lignes[indices_l]]
        ref_local = int(ancien_vers_local[reseau.reference])
        if ref_local < 0:
            ref_local = 0
        ilot = Reseau(
            n_noeuds=int(noeuds.size),
            lignes=np.array(lignes_locales, dtype=int).reshape(-1, 2),
            reactances=np.array(reseau.reactances[indices_l], dtype=float),
            generateurs=ancien_vers_local[reseau.generateurs[indices_g]],
            charges=ancien_vers_local[reseau.charges[indices_c]],
            limites=np.array(limites[indices_l], dtype=float),
            reference=ref_local,
        )
        try:
            sol = resoudre(
                ilot,
                demande[indices_c],
                limites=limites[indices_l],
                puissance_max=puissance_max[indices_g],
                departage="highs",
            )
            if departage != "highs":
                g, s, f, strategie = departager(
                    ilot, demande[indices_c], puissance_max[indices_g],
                    float(sol.charge_servie.sum()), departage,
                    None if precedente is None else precedente[indices_g],
                )
                sol.production, sol.charge_servie, sol.flux = g, s, f
                if strategie != "highs":
                    n_secours += int(not sol.secours)
        except Exception as exc:
            raise RuntimeError(
                f"Échec de redispatch dans l'îlot {k}, nœuds {noeuds.tolist()} : {exc}"
            ) from exc
        production[indices_g] = sol.production
        charge_servie[indices_c] = sol.charge_servie
        flux[indices_l] = sol.flux
        n_secours += int(sol.secours)
        statut_ilots.append("resolu")

    delestage = demande - charge_servie
    taux_de_charge = np.full(reseau.n_lignes, np.nan)
    taux_de_charge[~masque] = np.abs(flux[~masque]) / limites[~masque]
    return ResultatRedispatchN1(
        hors_service=np.array(masque, dtype=bool, copy=True),
        ilots=ilots,
        production=production,
        charge_servie=charge_servie,
        delestage=delestage,
        flux=flux,
        taux_de_charge=taux_de_charge,
        statut_ilots=statut_ilots,
        n_secours=n_secours,
    )


def evaluer_retrait_n1(
    reseau: Reseau,
    demande: np.ndarray,
    puissance_max: np.ndarray,
    ligne_retiree: int | None,
    *,
    limites: np.ndarray | None = None,
) -> ResultatRedispatchN1:
    """Étudie une perte N−1 isolée, sans déclenchement secondaire.

    Chaque appel reconstruit son masque à partir de l'état initial intact.
    La propagation des cascades sera ajoutée dans une étape distincte.
    """
    if ligne_retiree is not None:
        if isinstance(ligne_retiree, (bool, np.bool_)) or not isinstance(ligne_retiree, (int, np.integer)):
            raise ValueError("L'indice de la ligne retirée doit être un entier.")
        if not 0 <= int(ligne_retiree) < reseau.n_lignes:
            raise ValueError("Indice de ligne hors du réseau.")
    hors_service = np.zeros(reseau.n_lignes, dtype=bool)
    if ligne_retiree is not None:
        hors_service[int(ligne_retiree)] = True
    return redispatch_exact(
        reseau, demande, puissance_max, hors_service, limites=limites
    )
