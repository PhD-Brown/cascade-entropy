"""Banc de topologies pour la phase B du Projet I (PHY-3202).

Réseaux 100 % synthétiques : anneau avec cordes, grille rectangulaire,
Gilbert G(n,p), graphe géométrique aléatoire G(n,r), arbre témoin.

Ce module ne fixe AUCUNE hypothèse électrique : pas de placement implicite des
producteurs, ni réactances/susceptances, ni capacités, ni taux de charge.
La conversion vers ``Reseau`` exige de fournir explicitement ces choix.

Gilbert et géométrique sont conditionnés à la connexité par rejet, jamais par
ajout caché de lignes. La graine, les paramètres et le nombre d'essais sont
conservés pour la traçabilité. Toutes les arêtes sont non orientées, uniques,
représentées par deux indices i<j et rangées en ordre lexicographique.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .reseau import Reseau


def _entier(valeur, nom: str, minimum: int) -> int:
    if isinstance(valeur, (bool, np.bool_)) or not isinstance(valeur, Integral) or valeur < minimum:
        raise ValueError(f"{nom} doit être un entier >= {minimum}.")
    return int(valeur)


def _nombre(valeur, nom: str, minimum: float = 0.0, maximum: float | None = None) -> float:
    if isinstance(valeur, (bool, np.bool_)) or not isinstance(valeur, Real):
        raise ValueError(f"{nom} doit être un nombre réel fini.")
    v = float(valeur)
    if not math.isfinite(v) or v < minimum or (maximum is not None and v > maximum):
        raise ValueError(f"{nom} doit être fini et dans l'intervalle autorisé.")
    return v


def _aretes(n: int, lignes: np.ndarray | list[tuple[int, int]]) -> np.ndarray:
    a = np.asarray(lignes)
    if a.size == 0:
        a = np.empty((0, 2), dtype=int)
    if a.ndim != 2 or a.shape[1] != 2 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("Les arêtes doivent être un tableau d'entiers (m, 2).")
    if a.size and (np.any(a < 0) or np.any(a >= n) or np.any(a[:, 0] == a[:, 1])):
        raise ValueError("Extrémité hors réseau ou boucle réflexive.")
    a = np.sort(a.astype(int, copy=True), axis=1)
    if a.size:
        a = a[np.lexsort((a[:, 1], a[:, 0]))]
        if np.any(np.all(a[1:] == a[:-1], axis=1)):
            raise ValueError("Arête répétée.")
    return a


@dataclass(eq=False)
class Topologie:
    famille: str
    n_noeuds: int
    lignes: np.ndarray
    graine: int | None = None
    parametres: dict[str, int | float | str] = field(default_factory=dict)
    essais: int = 1
    positions: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.n_noeuds = _entier(self.n_noeuds, "n_noeuds", 2)
        self.essais = _entier(self.essais, "essais", 1)
        if self.graine is not None:
            self.graine = _entier(self.graine, "graine", 0)
        if not isinstance(self.famille, str) or not self.famille:
            raise ValueError("Famille invalide.")
        self.lignes = _aretes(self.n_noeuds, self.lignes)
        self.parametres = dict(self.parametres)
        if self.positions is not None:
            p = np.asarray(self.positions, dtype=float)
            if p.shape != (self.n_noeuds, 2) or not np.all(np.isfinite(p)):
                raise ValueError("Positions attendues : tableau fini (n_noeuds, 2).")
            self.positions = p.copy()

    @property
    def n_lignes(self) -> int:
        return len(self.lignes)


def composantes(t: Topologie) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(t.n_noeuds)]
    for i, j in t.lignes:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    visite: set[int] = set()
    groupes = []
    for debut in range(t.n_noeuds):
        if debut in visite:
            continue
        pile = [debut]
        visite.add(debut)
        groupe = []
        while pile:
            v = pile.pop()
            groupe.append(v)
            for w in adj[v]:
                if w not in visite:
                    visite.add(w)
                    pile.append(w)
        groupes.append(sorted(groupe))
    return groupes


def ponts(t: Topologie) -> list[int]:
    """Indices des lignes-ponts dans l'ordre des lignes de la topologie (Tarjan)."""
    adj: list[list[tuple[int, int]]] = [[] for _ in range(t.n_noeuds)]
    for k, (a, b) in enumerate(t.lignes):
        adj[int(a)].append((int(b), k))
        adj[int(b)].append((int(a), k))
    decouverte = [-1] * t.n_noeuds
    bas = [0] * t.n_noeuds
    resultat: list[int] = []
    horloge = 0

    def visiter(u: int, lien_parent: int) -> None:
        nonlocal horloge
        decouverte[u] = bas[u] = horloge
        horloge += 1
        for v, numero in adj[u]:
            if numero == lien_parent:
                continue
            if decouverte[v] == -1:
                visiter(v, numero)
                bas[u] = min(bas[u], bas[v])
                if bas[v] > decouverte[u]:
                    resultat.append(numero)
            else:
                bas[u] = min(bas[u], decouverte[v])

    for s in range(t.n_noeuds):
        if decouverte[s] == -1:
            visiter(s, -1)
    return sorted(resultat)


def statistiques(t: Topologie) -> dict[str, int | float | bool]:
    degres = np.bincount(t.lignes.ravel(), minlength=t.n_noeuds)
    n_composantes = len(composantes(t))
    m = t.n_lignes
    n = t.n_noeuds
    return {
        "n_noeuds": n,
        "n_lignes": m,
        "connexe": n_composantes == 1,
        "n_composantes": n_composantes,
        "degre_min": int(degres.min()),
        "degre_max": int(degres.max()),
        "degre_moyen": float(2 * m / n),
        "densite": float(2 * m / (n * (n - 1))),
        "n_ponts": len(ponts(t)),
        "rang_cyclomatique": int(m - n + n_composantes),
    }


def anneau_cordes(n_noeuds: int = 50, n_cordes: int = 25, graine: int = 20261113) -> Topologie:
    n = _entier(n_noeuds, "n_noeuds", 4)
    k = _entier(n_cordes, "n_cordes", 0)
    graine = _entier(graine, "graine", 0)
    cycle = {(min(i, (i+1)%n), max(i, (i+1)%n)) for i in range(n)}
    candidates = [(i,j) for i in range(n) for j in range(i+1,n) if (i,j) not in cycle]
    if k > len(candidates):
        raise ValueError("Plus de cordes demandées que d'arêtes disponibles.")
    selection = np.random.default_rng(graine).choice(len(candidates), size=k, replace=False)
    edges = list(cycle) + [candidates[int(z)] for z in selection]
    angles = np.linspace(0.0, 2*math.pi, n, endpoint=False)
    return Topologie("anneau_cordes", n, np.asarray(edges, dtype=int), graine,
                     {"n_cordes": k}, positions=np.column_stack([np.cos(angles), np.sin(angles)]))


def grille(n_lignes: int = 5, n_colonnes: int = 10) -> Topologie:
    a = _entier(n_lignes, "n_lignes", 2)
    b = _entier(n_colonnes, "n_colonnes", 2)
    edges: list[tuple[int,int]] = []
    positions = []
    for i in range(a):
        for j in range(b):
            u = i*b + j
            positions.append((j/(b-1), i/(a-1)))
            if j+1 < b:
                edges.append((u, u+1))
            if i+1 < a:
                edges.append((u, u+b))
    return Topologie("grille", a*b, np.asarray(edges, dtype=int),
                     parametres={"n_lignes": a, "n_colonnes": b}, positions=np.asarray(positions))


def gilbert(n_noeuds: int = 50, p: float = 0.10, graine: int = 20261113,
            max_essais: int = 500) -> Topologie:
    n = _entier(n_noeuds, "n_noeuds", 2)
    p = _nombre(p, "p", 0.0, 1.0)
    graine = _entier(graine, "graine", 0)
    max_essais = _entier(max_essais, "max_essais", 1)
    if p == 0:
        raise ValueError("G(n,0) est non connexe pour n>=2.")
    rng = np.random.default_rng(graine)
    candidats = np.column_stack(np.triu_indices(n, 1))
    for essai in range(1, max_essais + 1):
        edges = candidats[rng.random(len(candidats)) < p]
        t = Topologie("gilbert", n, edges, graine, {"p": p}, essais=essai)
        if statistiques(t)["connexe"]:
            return t
    raise ValueError(f"Aucun G({n},{p}) connexe en {max_essais} essais (graine {graine}).")


def geometrique(n_noeuds: int = 50, rayon: float = 0.25, graine: int = 20261113,
                max_essais: int = 500) -> Topologie:
    n = _entier(n_noeuds, "n_noeuds", 2)
    r = _nombre(rayon, "rayon", 0.0)
    graine = _entier(graine, "graine", 0)
    max_essais = _entier(max_essais, "max_essais", 1)
    if r == 0:
        raise ValueError("Un rayon nul ne relie aucun point distinct, avec probabilité 1.")
    rng = np.random.default_rng(graine)
    a,b = np.triu_indices(n,1)
    for essai in range(1,max_essais+1):
        positions = rng.random((n,2))
        d2 = np.sum((positions[a]-positions[b])**2,axis=1)
        edges = np.column_stack((a[d2<=r*r], b[d2<=r*r]))
        t = Topologie("geometrique", n, edges, graine, {"rayon":r}, essais=essai, positions=positions)
        if statistiques(t)["connexe"]:
            return t
    raise ValueError(f"Aucun graphe géométrique connexe en {max_essais} essais (graine {graine}).")


def arbre_temoin(n_noeuds: int = 50, embranchement: int = 3) -> Topologie:
    n = _entier(n_noeuds, "n_noeuds", 2)
    k = _entier(embranchement, "embranchement", 1)
    edges = [((i-1)//k,i) for i in range(1,n)]
    return Topologie("arbre_temoin", n, np.asarray(edges), parametres={"embranchement":k})


def vers_reseau(t: Topologie, *, reactances: np.ndarray, generateurs: np.ndarray,
                charges: np.ndarray, limites: np.ndarray, reference: int) -> Reseau:
    """Convertit une topologie en Reseau après décision EXPLICITE des paramètres électriques.

    Aucune capacité, production ou charge n'est inventée ici. Une valeur de puissance
    maximale ou une demande par nœud se passe séparément au dispatch/cascade.
    """
    x = np.asarray(reactances, dtype=float)
    caps = np.asarray(limites, dtype=float)
    gen = np.asarray(generateurs)
    conso = np.asarray(charges)
    ref = _entier(reference, "reference", 0)
    if x.shape != (t.n_lignes,) or not np.all(np.isfinite(x)) or np.any(x<=0):
        raise ValueError("Une réactance finie strictement positive par ligne est requise.")
    if caps.shape != (t.n_lignes,) or not np.all(np.isfinite(caps)) or np.any(caps<=0):
        raise ValueError("Une capacité finie strictement positive par ligne est requise.")
    if ref >= t.n_noeuds:
        raise ValueError("Référence hors du réseau.")
    if not np.issubdtype(gen.dtype, np.integer) or not np.issubdtype(conso.dtype, np.integer):
        raise ValueError("Indices des générateurs et des charges entiers requis.")
    gen,conso=gen.astype(int,copy=True),conso.astype(int,copy=True)
    if gen.ndim!=1 or conso.ndim!=1 or gen.size==0 or conso.size==0:
        raise ValueError("Au moins un générateur et une charge, tableaux 1D.")
    if np.any(gen<0) or np.any(conso<0) or np.any(gen>=t.n_noeuds) or np.any(conso>=t.n_noeuds):
        raise ValueError("Indices hors du réseau.")
    if len(set(gen.tolist()))!=len(gen) or len(set(conso.tolist()))!=len(conso) or set(gen)&set(conso):
        raise ValueError("Générateurs/charges répétés ou non disjoints.")
    if ref not in gen:
        raise ValueError("La référence doit être un nœud producteur dans ce banc.")
    return Reseau(t.n_noeuds,t.lignes.copy(),x.copy(),gen,conso,caps.copy(),reference=ref)


def sauver_json(t: Topologie, chemin: str | Path) -> None:
    """Écrit une définition de topologie publique ; ne sauvegarde aucun paramètre privé."""
    p=Path(chemin)
    donnees={"format":"cascade-entropy-topologie-v1","famille":t.famille,
             "n_noeuds":t.n_noeuds,"lignes":t.lignes.tolist(),
             "graine":t.graine,"parametres":t.parametres,"essais":t.essais,
             "positions":None if t.positions is None else t.positions.tolist(),
             "statistiques":statistiques(t)}
    p.write_text(json.dumps(donnees,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def charger_json(chemin: str | Path) -> Topologie:
    d=json.loads(Path(chemin).read_text(encoding="utf-8"))
    if d.get("format")!="cascade-entropy-topologie-v1":
        raise ValueError("Format de topologie non pris en charge.")
    t=Topologie(d["famille"],d["n_noeuds"],np.asarray(d["lignes"]),
               d["graine"],d["parametres"],d["essais"],d["positions"])
    if statistiques(t)!=d["statistiques"]:
        raise ValueError("Statistiques incohérentes avec les arêtes enregistrées.")
    return t
