"""Cinq indicateurs de lignes et deux références avant défaillance (phase C).

Ce module travaille sur des tableaux du graphe intact, sans importer les modèles,
dispatchs ou cascades. Les variantes sont des arguments obligatoires : leur
disponibilité ne constitue pas une adoption scientifique. Voir
``docs/phase_c_centralites.md`` pour les équations, sources et limites.

Chaque tableau de scores suit EXACTEMENT l'ordre des lignes fourni, sans tri ni
départage des ex æquo. L'indice de ligne d'origine est donc son identifiant A3.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from ._validation import entier_positif, serie_finie


def _vecteur(valeurs, taille: int, nom: str, *, positif: bool = False) -> np.ndarray:
    """Réutilise les contrôles réels/finis communs et impose l'alignement."""
    a = serie_finie(valeurs).copy()
    if a.shape != (taille,) or (positif and np.any(a <= 0)):
        raise ValueError(f"{nom} : taille {taille}, valeurs {'> 0' if positif else 'finies'} requises.")
    return a


def _choix(valeur: str, permis: tuple[str, ...], nom: str) -> None:
    if valeur not in permis:
        raise ValueError(f"{nom} doit appartenir à {permis}.")


@dataclass(frozen=True, eq=False)
class GrapheLignes:
    """Copie d'un graphe intact simple, non orienté, connexe, avec n >= 2.

    ``lignes`` : entiers (m, 2) dans l'ordre D'ORIGINE, orientation arbitraire.
    ``reactances`` : m valeurs réelles finies > 0, en pu (b = 1/x).
    Les graphes vides, disjoints, boucles et lignes parallèles sont refusés ;
    aucune centralité par composante ou agrégation de lignes n'est inventée.
    Les tableaux copiés sont protégés contre les modifications ordinaires.
    """

    n_noeuds: int
    lignes: np.ndarray
    reactances: np.ndarray

    def __post_init__(self) -> None:
        n = entier_positif(self.n_noeuds, "n_noeuds", minimum=2)
        a = np.asarray(self.lignes)
        if a.ndim != 2 or a.shape[1] != 2 or not np.issubdtype(a.dtype, np.integer):
            raise ValueError("lignes doit être un tableau d'entiers (m, 2).")
        if a.size == 0 or np.any(a < 0) or np.any(a >= n) or np.any(a[:, 0] == a[:, 1]):
            raise ValueError("Graphe vide, extrémité invalide ou boucle.")
        if len({tuple(sorted(e)) for e in a.tolist()}) != len(a):
            raise ValueError("Les lignes parallèles sont hors du domaine de cette API.")
        a = a.astype(int, copy=True)
        x = _vecteur(self.reactances, len(a), "reactances", positif=True)
        voisins = [[] for _ in range(n)]
        for u, v in a:
            voisins[u].append(v)
            voisins[v].append(u)
        vus, pile = {0}, [0]
        while pile:
            for v in voisins[pile.pop()]:
                if v not in vus:
                    vus.add(v)
                    pile.append(v)
        if len(vus) != n:
            raise ValueError("Le graphe intact doit être connexe.")
        a.setflags(write=False)
        x.setflags(write=False)
        object.__setattr__(self, "n_noeuds", n)
        object.__setattr__(self, "lignes", a)
        object.__setattr__(self, "reactances", x)

    @property
    def n_lignes(self) -> int:
        """Nombre de lignes, sans unité."""
        return len(self.lignes)


@dataclass(frozen=True, eq=False)
class EtatInitialIntact:
    """État initial fourni explicitement, sans calcul ni choix de dispatch.

    ``flux`` est signé selon ``graphe.lignes`` ; ``limites`` est C > 0 dans la
    même unité (pu sur une base commune en A3). ``hors_service`` est un masque
    booléen obligatoire entièrement faux. ``identifiant`` nomme l'état pour la
    traçabilité. Un résultat de cascade n'est pas une entrée de cette API.

    Le masque permet de refuser un état déclaré post-retrait ; il ne certifie
    pas la provenance des nombres. Le producteur reste responsable de fournir
    un dispatch intact convergé, aligné avec le graphe, et de conserver ses
    injections/paramètres. Aucun filtre de délestage ou de surcharge n'est imposé.
    """

    graphe: GrapheLignes
    flux: np.ndarray
    limites: np.ndarray
    hors_service: np.ndarray
    identifiant: str

    def __post_init__(self) -> None:
        if not isinstance(self.graphe, GrapheLignes):
            raise TypeError("graphe doit être une copie validée GrapheLignes.")
        m = self.graphe.n_lignes
        masque = np.asarray(self.hors_service)
        if masque.dtype.kind != "b" or masque.shape != (m,) or np.any(masque):
            raise ValueError("L'état initial intact exige un masque booléen entièrement faux.")
        if not isinstance(self.identifiant, str) or not self.identifiant.strip():
            raise ValueError("Un identifiant non vide de l'état initial est requis.")
        for nom, a in (
            ("flux", _vecteur(self.flux, m, "flux")),
            ("limites", _vecteur(self.limites, m, "limites", positif=True)),
            ("hors_service", masque.copy()),
        ):
            a.setflags(write=False)
            object.__setattr__(self, nom, a)


def degre_lignes(graphe: GrapheLignes, *, variante: str) -> np.ndarray:
    """Somme k_i+k_j ou produit k_i*k_j (``somme``/``produit``).

    k est le nombre de lignes incidentes, sans poids. Score brut sans unité,
    sans normalisation. Une grande valeur désigne des extrémités plus connectées,
    sans affirmer une plus grande gravité de cascade.
    """
    _choix(variante, ("somme", "produit"), "variante")
    k = np.bincount(graphe.lignes.ravel(), minlength=graphe.n_noeuds).astype(float)
    u, v = graphe.lignes.T
    return k[u] + k[v] if variante == "somme" else k[u] * k[v]


def intermediarite_geodesique(
    graphe: GrapheLignes, *, longueurs: str, normalisation: str,
    egalite_rtol: float = 1e-12,
) -> np.ndarray:
    """Somme sur s<t de sigma_st(e)/sigma_st, ou moyenne par paire.

    ``longueurs`` : ``unitaires`` (nombre de sauts) ou ``reactances`` (somme x).
    ``normalisation`` : ``somme`` ou ``moyenne`` (division par n(n-1)/2).
    Le score est sans unité, même pour des longueurs en pu ; grand = davantage
    de chemins empruntent la ligne. Les chemins multiples partagent le poids
    également PAR CHEMIN, et non par branchement. Aucun échantillonnage.

    Dijkstra puis accumulation de Brandes sur le DAG des distances croissantes.
    Deux distances sont égales à une erreur relative ``egalite_rtol`` près
    (défaut numérique 1e-12, 0 pour l'égalité flottante stricte ; atol=0).
    Les x sont divisés par max(x) pour enlever leur échelle commune. Des longueurs
    trop disparates pour représenter un incrément strict sont refusées.
    """
    _choix(longueurs, ("unitaires", "reactances"), "longueurs")
    _choix(normalisation, ("somme", "moyenne"), "normalisation")
    tol = serie_finie([egalite_rtol])[0]
    if isinstance(egalite_rtol, (bool, np.bool_)) or not 0 <= tol < 1:
        raise ValueError("egalite_rtol doit être dans [0, 1[.")
    n, m = graphe.n_noeuds, graphe.n_lignes
    ell = np.ones(m) if longueurs == "unitaires" else graphe.reactances / max(graphe.reactances)
    adj = [[] for _ in range(n)]
    for e, (u, v) in enumerate(graphe.lignes):
        adj[u].append((v, e))
        adj[v].append((u, e))
    score = np.zeros(m)
    for s in range(n):
        dist = np.full(n, np.inf)
        dist[s] = 0
        tas = [(0.0, s)]
        while tas:
            d, u = heapq.heappop(tas)
            if d != dist[u]:
                continue
            for v, e in adj[u]:
                candidat = d + ell[e]
                if candidat <= d:
                    raise ValueError("Longueurs trop disparates pour les distances flottantes.")
                if candidat < dist[v]:
                    dist[v] = candidat
                    heapq.heappush(tas, (candidat, v))
        ordre = np.argsort(dist)
        pred = [[] for _ in range(n)]
        sigma = [0] * n
        sigma[s] = 1
        for v in ordre:
            for u, e in adj[v]:
                if dist[u] < dist[v] and np.isclose(dist[u] + ell[e], dist[v], rtol=tol, atol=0):
                    pred[v].append((u, e))
                    sigma[v] += sigma[u]
        delta = np.zeros(n)
        for v in ordre[::-1]:
            for u, e in pred[v]:
                part = (sigma[u] / sigma[v]) * (1 + delta[v])
                score[e] += part
                delta[u] += part
    score /= 2  # Les sources s et t ont chacune compté la paire non orientée.
    if normalisation == "moyenne":
        score /= n * (n - 1) / 2
    return score


def _susceptances(graphe: GrapheLignes) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore"):
        b = 1 / graphe.reactances
    if not np.all(np.isfinite(b)):
        raise ValueError("Susceptances non représentables en flottants.")
    return b


def centralite_vecteur_propre(graphe: GrapheLignes, *, normalisation: str) -> np.ndarray:
    """Score b_e*c_i*c_j sur A_ij=b_ij, A_ii=0, A*c=lambda_max*c.

    ``normalisation`` de c positif : ``l2`` (somme c²=1), ``l1`` (somme c=1)
    ou ``maximum`` (max c=1). Score en unités de susceptance, sans remise à
    l'échelle supplémentaire. Grand = contribution spectrale élevée, sans
    présumer le signe du lien à la gravité. Sur un graphe de force pondérée
    constante, c est uniforme ; avec l2, le score vaut b_e/n.

    Diagonalisation symétrique, y compris sur les graphes bipartis. Un vecteur
    non strictement positif ou une valeur propre principale numériquement
    non séparée sont signalés plutôt que remplacés par un score arbitraire.
    """
    _choix(normalisation, ("l2", "l1", "maximum"), "normalisation")
    b = _susceptances(graphe)
    a = np.zeros((graphe.n_noeuds, graphe.n_noeuds))
    u, v = graphe.lignes.T
    a[u, v] = a[v, u] = b / max(b)
    valeurs, vecteurs = np.linalg.eigh(a)
    if valeurs[-1] - valeurs[-2] <= 1e-12 * valeurs[-1]:
        raise ValueError("Valeur propre principale numériquement non séparée.")
    c = vecteurs[:, -1]
    c *= 1 if c.sum() > 0 else -1
    if np.any(c <= 0):
        raise ValueError("Vecteur de Perron non strictement positif numériquement.")
    if normalisation == "l1":
        c /= c.sum()
    elif normalisation == "maximum":
        c /= c.max()
    return b * c[u] * c[v]


def _operateurs_electriques(graphe: GrapheLignes):
    """Incidence, poids réduits et L+ réduit ; ne tronque aucun mode positif."""
    b = _susceptances(graphe)
    w = b / max(b)
    incidence = np.zeros((graphe.n_lignes, graphe.n_noeuds))
    incidence[np.arange(graphe.n_lignes), graphe.lignes[:, 0]] = 1
    incidence[np.arange(graphe.n_lignes), graphe.lignes[:, 1]] = -1
    lap = incidence.T @ (w[:, None] * incidence)
    valeurs, vecteurs = np.linalg.eigh(lap)
    if valeurs[1] <= 1e-12 * valeurs[-1]:
        raise ValueError("Laplacien trop mal conditionné : résistance/courant non fiables.")
    plus = (vecteurs[:, 1:] / valeurs[1:]) @ vecteurs[:, 1:].T
    return incidence, w, plus, max(b)


def _indices(indices, n: int, nom: str) -> np.ndarray:
    a = np.asarray(indices)
    if (a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) or a.size == 0
            or np.any(a < 0) or np.any(a >= n) or len(np.unique(a)) != len(a)):
        raise ValueError(f"{nom} : indices entiers uniques, non vides, dans le graphe requis.")
    return a.astype(int, copy=True)


def _paires(graphe, paires, generateurs, charges, poids_paires):
    _choix(paires, ("uniformes", "generateur_charge"), "paires")
    if paires == "uniformes":
        if any(a is not None for a in (generateurs, charges, poids_paires)):
            raise ValueError("Paires uniformes : aucun rôle ou poids supplémentaire admis.")
        s, t = np.triu_indices(graphe.n_noeuds, 1)
        return s, t, np.ones(len(s))
    g = _indices(generateurs, graphe.n_noeuds, "generateurs")
    c = _indices(charges, graphe.n_noeuds, "charges")
    if np.intersect1d(g, c).size:
        raise ValueError("Générateurs et charges doivent être disjoints (A3).")
    if poids_paires is None:
        raise ValueError("Fournir explicitement les poids générateur-charge, même uniformes.")
    poids = _vecteur(np.asarray(poids_paires).ravel(), len(g) * len(c), "poids_paires")
    if np.asarray(poids_paires).shape != (len(g), len(c)) or np.any(poids < 0) or not np.any(poids > 0):
        raise ValueError("poids_paires doit être (n_gen, n_charge), >= 0 et de total > 0.")
    return np.repeat(g, len(c)), np.tile(c, len(g)), poids


def _courant(operateurs, s, t, poids, normalisation):
    _choix(normalisation, ("somme", "moyenne"), "normalisation")
    incidence, w, plus, _ = operateurs
    reponse = w[:, None] * (incidence @ plus)
    if normalisation == "moyenne":
        poids = poids / max(poids)
        poids = poids / poids.sum()
    with np.errstate(over="ignore", invalid="ignore"):
        resultat = np.abs(reponse[:, s] - reponse[:, t]) @ poids
    if not np.all(np.isfinite(resultat)):
        raise ValueError("Somme pondérée non représentable ; revoir les poids/normalisation.")
    return resultat


def intermediarite_courant(
    graphe: GrapheLignes, *, paires: str, normalisation: str,
    generateurs=None, charges=None, poids_paires=None,
) -> np.ndarray:
    """Somme q_st*|b_e*nu_e.T*L+*(e_s-e_t)|, ou division par somme q.

    ``paires='uniformes'`` : toutes les paires s<t, q=1. Variante
    ``generateur_charge`` : produit cartésien des indices fournis (disjoints),
    poids réels explicites (n_gen, n_charge), >=0, total >0. Des uns donnent la
    version G-C uniforme ; aucune pondération par puissance n'est présumée.
    ``normalisation`` vaut ``somme`` ou ``moyenne`` pondérée.

    Chaque transfert est UNITAIRE ; valeur absolue AVANT agrégation, même si
    plusieurs transferts s'annuleraient. Moyenne sans unité ; somme dans l'unité
    des poids, à annoncer par l'appelant. Grand = participation électrique
    moyenne/sommée élevée. Ni capacités ni flux réels ni cascades n'interviennent.
    Il ne s'agit pas d'une implémentation attribuée aux variantes de Bompard/Wang.
    """
    s, t, poids = _paires(graphe, paires, generateurs, charges, poids_paires)
    return _courant(_operateurs_electriques(graphe), s, t, poids, normalisation)


def _resistance(operateurs):
    incidence, w, plus, echelle = operateurs
    r = np.einsum("ij,ij->i", incidence @ plus, incidence)
    criticite = w * r
    # Ne corrige que l'arrondi hors de [0,1] ; ne force pas les quasi-ponts à 1.
    if np.any(criticite < -1e-9) or np.any(criticite > 1 + 1e-9):
        raise ValueError("Résistance effective hors des bornes numériques attendues.")
    return r / echelle, np.clip(criticite, 0, 1)


def resistance_effective(graphe: GrapheLignes) -> np.ndarray:
    """Diagnostic R_e=nu_e.T*L+*nu_e entre extrémités, unité de x.

    Le laplacien utilise b=1/x : ce circuit auxiliaire n'utilise pas de
    résistance ohmique. Ce diagnostic n'est pas une sixième centralité.
    """
    return _resistance(_operateurs_electriques(graphe))[0]


def redondance_electrique(graphe: GrapheLignes, *, convention: str) -> np.ndarray:
    """Indicateur local ``criticite`` = b_e*R_e ou ``redondance`` = 1-b_e*R_e.

    Sans unité, dans [0,1] à l'arrondi près. Grand bR signifie moins de chemins
    électriques alternatifs ; grand 1-bR signifie davantage de redondance.
    Sur un pont bR=1 et redondance=0 ; aucune division par 1-bR, aucun infini.
    Ce sens structurel ne prouve pas un sens prédictif pour le délestage.
    """
    _choix(convention, ("criticite", "redondance"), "convention")
    _, criticite = _resistance(_operateurs_electriques(graphe))
    return criticite if convention == "criticite" else 1 - criticite


def flux_initial_absolu(etat_initial: EtatInitialIntact) -> np.ndarray:
    """|F0|, unité du flux initial (pu en A3), brut, grand = flux plus élevé."""
    _exiger_etat_initial(etat_initial)
    return np.abs(etat_initial.flux)


def taux_charge_initial(etat_initial: EtatInitialIntact) -> np.ndarray:
    """|F0|/C, sans unité, grand = moins de marge relative ; flux nul -> zéro.

    C est fourni dans l'état intact, jamais reconstruit depuis un résultat de
    cascade ; zéro, NaN et infini sont refusés par EtatInitialIntact.
    """
    _exiger_etat_initial(etat_initial)
    with np.errstate(over="ignore"):
        taux = np.abs(etat_initial.flux) / etat_initial.limites
    if not np.all(np.isfinite(taux)):
        raise ValueError("Taux de charge non représentable en flottants.")
    return taux


def _exiger_etat_initial(etat_initial) -> None:
    """Refuse un résultat de simulation ayant simplement un attribut flux."""
    if not isinstance(etat_initial, EtatInitialIntact):
        raise TypeError("Fournir explicitement un EtatInitialIntact validé.")


@dataclass
class ScoresLignes:
    """Sept colonnes alignées sur les ID A3, état nommé et conventions explicites.

    ``parametres`` conserve aussi les rôles et poids de paires copiés, pour
    reproduire une variante G-C. Aucun rang ni résultat post-défaillance.
    """

    identifiants_lignes: np.ndarray
    lignes: np.ndarray
    identifiant_etat_initial: str
    scores: dict[str, np.ndarray]
    parametres: dict


def calculer_scores(
    etat_initial: EtatInitialIntact, *, degre: str, longueurs: str,
    normalisation_geodesique: str, normalisation_vecteur: str,
    paires_courant: str, normalisation_courant: str, convention_resistance: str,
    generateurs=None, charges=None, poids_paires=None, egalite_rtol: float = 1e-12,
) -> ScoresLignes:
    """Calcule les sept scores sur UN état intact explicitement fourni.

    Les mots-clés obligatoires ont les sens définis dans les fonctions unitaires.
    L'ordre d'origine et les ex æquo sont préservés. Un seul calcul de L+ sert
    au courant et à la résistance. Les options ne fixent aucun protocole de
    campagne ; les paramètres retournés doivent accompagner les valeurs.
    """
    _exiger_etat_initial(etat_initial)
    g = etat_initial.graphe
    _choix(convention_resistance, ("criticite", "redondance"), "convention_resistance")
    s, t, poids = _paires(g, paires_courant, generateurs, charges, poids_paires)
    op = _operateurs_electriques(g)
    _, criticite = _resistance(op)
    scores = {
        "degre": degre_lignes(g, variante=degre),
        "intermediarite_geodesique": intermediarite_geodesique(
            g, longueurs=longueurs, normalisation=normalisation_geodesique, egalite_rtol=egalite_rtol),
        "vecteur_propre": centralite_vecteur_propre(g, normalisation=normalisation_vecteur),
        "intermediarite_courant": _courant(op, s, t, poids, normalisation_courant),
        "resistance_locale": criticite if convention_resistance == "criticite" else 1 - criticite,
        "flux_initial_absolu": flux_initial_absolu(etat_initial),
        "taux_charge_initial": taux_charge_initial(etat_initial),
    }
    parametres = dict(
        degre=degre, longueurs=longueurs, normalisation_geodesique=normalisation_geodesique,
        normalisation_vecteur=normalisation_vecteur, paires_courant=paires_courant,
        normalisation_courant=normalisation_courant, convention_resistance=convention_resistance,
        egalite_rtol=float(egalite_rtol), generateurs=None if generateurs is None else np.array(generateurs).tolist(),
        charges=None if charges is None else np.array(charges).tolist(),
        poids_paires=None if poids_paires is None else np.array(poids_paires).tolist(),
    )
    return ScoresLignes(np.arange(g.n_lignes), g.lignes.copy(), etat_initial.identifiant, scores, parametres)
