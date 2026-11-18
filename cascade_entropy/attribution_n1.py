"""Attribution électrique B4, entièrement synthétique et sans règle de campagne.

Ce module appartient au moteur de réseau (fil A), pas au fil d'analyse des
entropies. Il assemble les interfaces A3 existantes sans en changer les défauts.
Les nombres physiques et les stratégies sont toujours fournis par l'appelant.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from numbers import Real
from pathlib import Path
import platform

import numpy as np
import scipy

from ._validation import entier_positif, serie_finie
from ._optimisation_dispatch_n1 import valider_options
from .n1 import ResultatRedispatchN1, composantes_actives, redispatch_exact
from .reseau import Reseau, angles, matrice_incidence
from .topologies_n1 import Topologie, vers_reseau


def _reel(valeur, nom: str, *, strict: bool = False) -> float:
    if isinstance(valeur, (bool, np.bool_)) or not isinstance(valeur, Real):
        raise ValueError(f"{nom} doit être réel.")
    v = float(valeur)
    if not np.isfinite(v) or v < 0 or (strict and v == 0):
        raise ValueError(f"{nom} doit être fini et {'strictement positif' if strict else 'positif ou nul'}.")
    return v


@dataclass
class Valeurs:
    """Valeurs explicites, constantes ou uniformes, sans distribution implicite.

    ``parametres`` contient le vecteur complet en mode ``explicites``, un seul
    nombre en mode ``constante``, ou les deux bornes en mode ``uniforme``.
    L'uniforme est une option de synthèse, jamais une distribution physique
    approuvée. Une graine propre à chaque champ rend les tirages indépendants
    de l'ordre des appels ; elle est obligatoire uniquement pour l'uniforme.
    """

    mode: str
    parametres: list[float] | np.ndarray
    graine: int | None = None

    def realiser(self, taille: int) -> np.ndarray:
        """Matérialise un vecteur réel non négatif sans modifier l'entrée."""
        taille = entier_positif(taille, "taille")
        if np.asarray(self.parametres).dtype.kind == "b":
            raise ValueError("Une grandeur électrique ne peut être booléenne.")
        p = serie_finie(self.parametres)
        if np.any(p < 0):
            raise ValueError("Les valeurs et bornes doivent être positives ou nulles.")
        if self.mode == "uniforme":
            graine = entier_positif(self.graine, "graine", minimum=0)
            if p.shape != (2,) or p[1] < p[0]:
                raise ValueError("Deux bornes croissantes sont requises pour uniforme.")
            return np.random.default_rng(graine).uniform(p[0], p[1], taille)
        if self.graine is not None:
            raise ValueError("Une graine n'a de sens qu'en mode uniforme.")
        if self.mode == "constante" and p.shape == (1,):
            return np.full(taille, p[0])
        if self.mode == "explicites" and p.shape == (taille,):
            return p.copy()
        raise ValueError("Mode de valeurs inconnu ou dimension incohérente.")


@dataclass
class Placement:
    """Indices ordonnés des producteurs et charges ; nœuds neutres autorisés.

    La référence est fournie en mode explicite. Le helper aléatoire choisit
    explicitement le premier producteur trié comme référence et le documente.
    """

    generateurs: list[int] | np.ndarray
    charges: list[int] | np.ndarray
    reference: int
    provenance: dict | None = None


def placement_aleatoire(
    topologie: Topologie, *, n_generateurs: int, n_charges: int, graine: int
) -> Placement:
    """Tire sans remise des rôles disjoints, uniformément parmi les nœuds.

    Les deux groupes sont triés avant d'associer demandes et capacités.
    Ce placement est une option technique de validation, sans priorité physique.
    """
    ng = entier_positif(n_generateurs, "n_generateurs")
    nc = entier_positif(n_charges, "n_charges")
    graine = entier_positif(graine, "graine", minimum=0)
    if ng + nc > topologie.n_noeuds:
        raise ValueError("Trop de rôles pour les nœuds disponibles.")
    choix = np.random.default_rng(graine).choice(topologie.n_noeuds, ng + nc, replace=False)
    g, c = np.sort(choix[:ng]), np.sort(choix[ng:])
    return Placement(g, c, int(g[0]), {
        "strategie": "uniforme_sans_remise", "graine": graine,
        "n_generateurs": ng, "n_charges": nc, "reference": "premier_generateur_trie",
    })


@dataclass
class Capacites:
    """Règle de limites, choisie explicitement par l'appelant.

    ``independantes`` : valeurs directement affectées aux lignes.
    ``flux_reference`` : valeurs = limites du redispatch de référence ; puis
    C = max(marge * |F_reference|, plancher), avec plancher ABSOLU strictement
    positif dans l'unité de puissance d'entrée. Les limites obtenues sont
    figées, puis un nouveau redispatch donne l'état initial final. Aucune
    itération de point fixe ni égalité présumée entre les deux dispatchs.
    """

    mode: str
    valeurs: Valeurs
    marge: float | None = None
    plancher: float | None = None


@dataclass
class EtatElectrique:
    """Réseau prêt pour N−1 et contrôles de son état intact, tout en pu.

    Les réserves disponibles sont Pmax − G, sans garantie de transportabilité.
    Les réserves minimales demandées sont diagnostiques : elles ne réduisent
    pas secrètement Pmax et ne changent pas l'objectif du redispatch existant.
    ``avertissements`` ne constitue pas une règle d'admission scientifique.
    """

    reseau: Reseau
    demande: np.ndarray
    puissance_max: np.ndarray
    reserves_minimales: np.ndarray
    resultat: ResultatRedispatchN1
    injections: np.ndarray
    angles: np.ndarray
    reserves_disponibles: np.ndarray
    reserves_satisfaites: np.ndarray
    avertissements: tuple[str, ...]
    configuration: dict
    environnement: dict
    resultat_reference: ResultatRedispatchN1 | None = None

    @property
    def susceptances(self) -> np.ndarray:
        """Susceptances positives b = 1/x, dans l'ordre stable des lignes."""
        return 1.0 / self.reseau.reactances


def _controler(reseau, demande, puissance_max, sol):
    """Contrôle a posteriori du solveur : bornes, bilan nodal et loi DC.

    Une composante utilise un angle de référence local arbitraire ; les angles
    des composantes distinctes ne sont pas physiquement comparables. Les
    tolérances sont numériques (1e-7 relatif, 1e-8 pu absolu ramené à l'échelle
    de la plus grande demande si elle est sous 1 pu), pas une physique. Cela
    évite de valider un bilan entièrement faux pour une très petite demande.
    """
    tol = 1e-8 * min(1.0, float(demande.max()))

    def proche(a, b):
        return np.allclose(a, b, rtol=1e-7, atol=tol)

    def exiger(condition, message):
        if not condition:
            raise RuntimeError(f"État initial incohérent : {message}.")

    for nom, tableau, forme in (
        ("production", sol.production, puissance_max.shape),
        ("charge servie", sol.charge_servie, demande.shape),
        ("délestage", sol.delestage, demande.shape),
        ("flux", sol.flux, (reseau.n_lignes,)),
        ("taux", sol.taux_de_charge, (reseau.n_lignes,)),
    ):
        exiger(tableau.shape == forme and np.all(np.isfinite(tableau)), nom)
    exiger(sol.hors_service.shape == (reseau.n_lignes,) and not sol.hors_service.any(), "lignes retirées")
    ilots_attendus = composantes_actives(reseau, np.zeros(reseau.n_lignes, dtype=bool))
    exiger(len(sol.ilots) == len(ilots_attendus) and all(
        np.array_equal(a, b) for a, b in zip(sol.ilots, ilots_attendus)
    ), "identifiants des îlots")
    exiger(np.all(sol.production >= -tol) and np.all(sol.production <= puissance_max + tol + 1e-7 * puissance_max), "bornes de production")
    exiger(np.all(sol.charge_servie >= -tol) and np.all(sol.charge_servie <= demande + tol + 1e-7 * demande), "bornes de consommation")
    exiger(proche(sol.delestage, demande - sol.charge_servie), "décomposition de la demande")
    exiger(np.all(np.abs(sol.flux) <= reseau.limites + tol + 1e-7 * reseau.limites), "limites de transport")
    exiger(proche(sol.taux_de_charge, np.abs(sol.flux) / reseau.limites), "taux de charge")
    injection = np.zeros(reseau.n_noeuds)
    injection[reseau.generateurs] = sol.production
    injection[reseau.charges] = -sol.charge_servie
    exiger(proche(matrice_incidence(reseau).T @ sol.flux, injection), "conservation nodale")
    theta = np.zeros(reseau.n_noeuds)
    # Réutiliser le solveur d'angles existant, sur chaque composante intacte.
    for noeuds in sol.ilots:
        exiger(abs(injection[noeuds].sum()) <= tol + 1e-7 * np.abs(injection[noeuds]).sum(), "bilan par îlot")
        if noeuds.size == 1:
            continue
        locales = np.full(reseau.n_noeuds, -1, dtype=int)
        locales[noeuds] = np.arange(noeuds.size)
        ids = np.flatnonzero(np.isin(reseau.lignes[:, 0], noeuds))
        ref = int(locales[reseau.reference])
        ilot = Reseau(int(noeuds.size), locales[reseau.lignes[ids]],
                      reseau.reactances[ids], np.array([], dtype=int),
                      np.array([], dtype=int), reference=max(0, ref))
        theta[noeuds] = angles(ilot, injection[noeuds])
    flux_dc = (theta[reseau.lignes[:, 0]] - theta[reseau.lignes[:, 1]]) / reseau.reactances
    exiger(proche(flux_dc, sol.flux), "loi DC")
    exiger(len(sol.statut_ilots) == len(sol.ilots) and all(
        s in {"resolu", "sans_generateur", "sans_demande"} for s in sol.statut_ilots
    ), "statut de solveur")
    return injection, theta


def _json_copie(objet):
    """Copie JSON stricte, y compris les scalaires et tableaux NumPy."""
    def convertir(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(f"Métadonnée non sérialisable : {type(x).__name__}.")
    return json.loads(json.dumps(objet, default=convertir, allow_nan=False))


def attribuer(
    topologie: Topologie, *, placement: Placement, demande: Valeurs,
    puissance_max: Valeurs, reserves_minimales: Valeurs, reactances: Valeurs,
    capacites: Capacites, base_mva: float, unite_puissance: str,
    unite_reactance: str, departage: str = "highs",
    production_precedente: np.ndarray | None = None,
) -> EtatElectrique:
    """Attribue, redispatche l'intact et contrôle ses contraintes sans filtrage.

    Toutes les puissances d'entrée utilisent la même unité (``pu`` ou ``MW``).
    Les MW sont divisés par ``base_mva`` ; les x doivent déjà être en pu sur
    cette base. Aucune conversion d'ohms ou de base de réactance n'est devinée.
    Les exceptions de solveur remontent avec leur cause. Le délestage et les
    réserves insuffisantes restent dans le résultat et sont signalés.

    Les règles N−1 sont optionnelles. ``min_redispatch`` demande une production
    antérieure explicite, dans l'unité de puissance d'entrée et l'ordre du
    placement. En mode flux_reference, la même règle et la même production
    antérieure sont employées pour les deux dispatchs (aucun chaînage caché).
    """
    base = _reel(base_mva, "base_mva", strict=True)
    if unite_puissance not in ("pu", "MW") or unite_reactance != "pu":
        raise ValueError("Unités acceptées : puissance pu ou MW, réactance pu sur la base commune.")
    precedente = valider_options(departage, production_precedente, len(placement.generateurs))
    # Topologie est mutable : revalider une copie, sans réordonner en silence
    # des identifiants auxquels des valeurs électriques seraient déjà liées.
    copie = Topologie(**asdict(topologie))
    if not np.array_equal(copie.lignes, topologie.lignes):
        raise ValueError("Les identifiants de lignes doivent respecter l'ordre A3 de Topologie.")
    topologie = copie
    ref = entier_positif(placement.reference, "reference", minimum=0)
    facteur = base if unite_puissance == "MW" else 1.0
    if capacites.mode not in ("independantes", "flux_reference"):
        raise ValueError("Règle de capacités inconnue.")
    if capacites.mode == "independantes" and (capacites.marge is not None or capacites.plancher is not None):
        raise ValueError("Marge et plancher sont réservés au mode flux_reference.")
    if capacites.mode == "flux_reference":
        marge = _reel(capacites.marge, "marge", strict=True)
        plancher = _reel(capacites.plancher, "plancher", strict=True) / facteur
    # vers_reseau garde la responsabilité des indices, rôles et dimensions.
    reseau = vers_reseau(
        topologie, reactances=reactances.realiser(topologie.n_lignes),
        generateurs=np.asarray(placement.generateurs), charges=np.asarray(placement.charges),
        limites=capacites.valeurs.realiser(topologie.n_lignes) / facteur, reference=ref,
    )
    with np.errstate(over="ignore", divide="ignore"):
        b = 1.0 / reseau.reactances
    if not np.all(np.isfinite(b)):
        raise ValueError("Susceptances non finies : réactances trop petites.")
    d = demande.realiser(len(reseau.charges)) / facteur
    pmax = puissance_max.realiser(len(reseau.generateurs)) / facteur
    reserves = reserves_minimales.realiser(len(reseau.generateurs)) / facteur
    if np.any(reserves > pmax):
        raise ValueError("Une réserve minimale ne peut dépasser Pmax.")
    if not all(np.all(np.isfinite(v)) for v in (d, pmax, reserves)):
        raise ValueError("Conversion de puissance non finie.")
    with np.errstate(over="ignore"):
        if not all(np.isfinite(v.sum()) for v in (d, pmax, reserves)):
            raise ValueError("Puissance totale non finie.")
    # Snapshot complet pris avant le solveur ; ni topologie ni spécifications
    # n'appartiennent au résultat et aucune entrée n'est modifiée.
    configuration = _json_copie({
        "topologie": asdict(topologie), "placement": asdict(placement),
        "demande": asdict(demande), "puissance_max": asdict(puissance_max),
        "reserves_minimales": asdict(reserves_minimales), "reactances": asdict(reactances),
        "capacites": asdict(capacites), "base_mva": base,
        "unite_puissance": unite_puissance, "unite_reactance": unite_reactance,
        "departage": departage,
    })
    if precedente is not None:
        configuration["production_precedente"] = precedente.tolist()
    masque = np.zeros(reseau.n_lignes, dtype=bool)

    def resoudre_intact():
        options = {} if precedente is None else {"production_precedente": precedente / facteur}
        sol = redispatch_exact(reseau, d, pmax, masque, departage=departage, **options)
        inj, theta = _controler(reseau, d, pmax, sol)
        return sol, inj, theta

    reference = None
    if capacites.mode == "flux_reference":
        reference, _, _ = resoudre_intact()
        with np.errstate(over="ignore"):
            limites = np.maximum(marge * np.abs(reference.flux), plancher)
        if not np.all(np.isfinite(limites)) or np.any(limites <= 0):
            raise ValueError("Capacités dérivées non finies ou nulles après conversion.")
        reseau.limites = limites
    sol, injection, theta = resoudre_intact()
    disponibles = pmax - sol.production
    satisfaites = disponibles + 1e-8 + 1e-7 * pmax >= reserves
    alertes = []
    if np.any(sol.delestage > 0):
        alertes.append("delestage_initial")
    if d.sum() == 0:
        alertes.append("demande_totale_nulle_fraction_indefinie")
    if len(sol.ilots) != 1:
        alertes.append("topologie_non_connexe")
    if not satisfaites.all():
        alertes.append("reserves_minimales_non_satisfaites")
    if sol.n_secours:
        alertes.append("secours_solveur_initial")
    if reference is not None:
        alertes.append("capacites_dependantes_du_dispatch_de_reference")
        if np.any(reference.delestage > 0):
            alertes.append("delestage_reference")
        if reference.n_secours:
            alertes.append("secours_solveur_reference")
        if not np.allclose(reference.flux, sol.flux, rtol=1e-7, atol=1e-8):
            alertes.append("flux_initial_distinct_du_flux_reference")
    return EtatElectrique(
        reseau, d, pmax, reserves, sol, injection, theta, disponibles,
        satisfaites, tuple(alertes), configuration,
        {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        reference,
    )


def sauver_configuration(etat: EtatElectrique, chemin: str | Path) -> None:
    """Sauve un manifeste rejouable sans écraser un fichier existant.

    Il contient les entrées et l'environnement d'origine, pas des résultats de
    campagne. Les sorties du solveur sont toujours recalculées au rechargement.
    """
    contenu = {"format": "cascade-entropy-electrique-b4-v1",
               "configuration": etat.configuration, "environnement": etat.environnement}
    texte = json.dumps(_json_copie(contenu), ensure_ascii=False, indent=2, allow_nan=False)
    with Path(chemin).open("x", encoding="utf-8") as fichier:
        fichier.write(texte + "\n")


def charger_configuration(chemin: str | Path) -> EtatElectrique:
    """Rejoue les entrées B4 ; le résultat porte l'environnement de ce rejeu."""
    contenu = json.loads(Path(chemin).read_text(encoding="utf-8"))
    if contenu.get("format") != "cascade-entropy-electrique-b4-v1":
        raise ValueError("Format électrique B4 non pris en charge.")
    c = contenu["configuration"]
    t = Topologie(**c.pop("topologie"))
    p = Placement(**c.pop("placement"))
    cap = c.pop("capacites")
    capacites = Capacites(**{**cap, "valeurs": Valeurs(**cap["valeurs"])})
    for nom in ("demande", "puissance_max", "reserves_minimales", "reactances"):
        c[nom] = Valeurs(**c[nom])
    return attribuer(t, placement=p, capacites=capacites, **c)
