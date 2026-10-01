"""
Évolution sur plusieurs jours : échantillon i.i.d. et dynamique auto-organisée.

Quatrième module du fil A. Il orchestre `cascade.journee()` sur de nombreux
jours pour produire les séries temporelles destinées au fil B. Deux modes,
correspondant à deux usages différents et à deux papiers de Carreras et al. :

MODE "independant" — validation contre Carreras et al. (2002)
---------------------------------------------------------------
Reproduit leurs « 60 000 cases by random variation of the loads » : le réseau
et sa capacité sont fixes, chaque jour est un tirage i.i.d. (nouvelle demande
via le paramètre g, nouvelles avaries p0, cascade p1), sans aucun lien entre
les jours. C'est un échantillon Monte-Carlo à charge fixée, pas une dynamique
temporelle — à traiter et documenter comme tel.

MODE "auto_organise" — dynamique de Carreras et al. (2004)
---------------------------------------------------------------
Reproduit leur modèle complet d'auto-organisation, à échelle réduite pour le
rapport d'étape (quelques milliers de jours, transitoire non retiré — la
statistique propre sur un régime stationnaire de dizaines de milliers de
jours, comme dans le papier, est un objectif explicitement reporté au rapport
final, pas de cette version).

Six paramètres pilotent la dynamique, chacun cité de Carreras et al. (2004) :

  - `lambda_` : taux de croissance annuel de la demande, fixé à 1,8 %/an dans
    le papier (« the rate of increase in power demand, λ, which we keep fixed
    at 1.8% per year »). ATTENTION UNITÉS : comme `journee()` avance d'un jour
    à la fois, il faut convertir en facteur de croissance QUOTIDIEN avant de
    l'utiliser dans P̄_D(t) = P0·e^((λ-1)t) — sinon la demande explose en
    quelques jours au lieu de dériver sur des années. Conversion :
    `lambda_jour = lambda_annuel ** (1/365)`. `lambda_` se donne donc comme
    facteur annuel (1,018 pour 1,8 %/an), valeur par défaut.
  - `mu` : taux d'amélioration du réseau, maintenu dans [1,01 ; 1,10] dans le
    papier (« we keep μ in the range 1.01–1.1 »). Ici, `mu` multiplie les
    limites des lignes tombées par surcharge un jour de blackout (étape 5).
  - `marge_seuil` : le (ΔP/P)_c de l'équation (4) — le seuil de marge de
    génération sous lequel la réponse d'ingénierie se déclenche. C'est « le
    principal paramètre qu'ils font varier » dans leurs calculs ; à exposer
    comme LE paramètre expérimental de ce module, pas une constante cachée.
  - `p0`, `p1` : identiques à `cascade.journee()`.
  - `k` : fraction de la puissance totale P_T ajoutée à chaque mise à niveau
    d'un générateur (« a parameter that we have taken to be a few percent »),
    P_T étant la demande totale P̄_D(t) et NON la capacité installée des
    générateurs — valeur par défaut proposée : 0,02.

La condition initiale s'ajoute à ces six paramètres : `demande_initiale` (P0,
demande totale du premier jour, répartie uniformément entre les charges) et
`puissance_max` (capacité de départ de chaque générateur). Leur rapport fixe la
marge initiale. À 1,8 %/an, la demande ne croît que d'environ 16 % en 3000
jours : le mécanisme ne se déclenche que si la marge initiale dépasse
`marge_seuil` de moins de 16 % environ. `puissance_max_pour_marge()` règle la
capacité de départ pour viser une marge initiale donnée.

Mécanique d'une journée en mode auto_organise :
  1. Avancer `t` d'un jour ; calculer la demande moyenne du jour via
     `lambda_jour` (P̄_D(t) = P0·e^((λ-1)t), avec λ = `lambda_jour`) puis
     appliquer la fluctuation g comme en mode indépendant (c'est `journee()`
     lui-même qui l'applique).
  2. Appeler `cascade.journee()` normalement (p0, p1, réseau courant).
  3. Calculer la marge moyenne de génération (Éq. 4) :
     ΔP/P̄ = [Σ_{j∈générateurs} P_j^max − P̄_D(t)] / P̄_D(t).
  4. Si ΔP/P̄ < marge_seuil : tirer un générateur au hasard PARMI
     `reseau.generateurs` (pas n'importe quel nœud — dans ce code-ci, seuls
     les générateurs ont une puissance maximale à augmenter). Vérifier que la
     somme des limites des lignes incidentes à ce nœud dépasse sa puissance
     actuelle plus l'ajout prévu (condition (b) du papier, calculée par
     `reseau.capacite_incidente()`). Si la condition tient, augmenter sa
     capacité de k·(P̄_D(t)/N_G) (Éq. 3), où P̄_D(t) est la demande totale
     moyenne du jour et N_G le nombre de générateurs, puis recalculer la marge ;
     sinon, tirer un autre générateur au hasard et recommencer. Continuer
     jusqu'à ΔP/P̄ ≥ marge_seuil ou jusqu'à épuisement des générateurs
     éligibles ce jour-là (pour éviter une boucle infinie si aucun générateur
     ne satisfait jamais la condition). Le tirage est sans remise : un
     générateur est amélioré au plus une fois par jour.
  5. Si le jour a connu un blackout (délestage final non nul), multiplier par
     `mu` les limites des lignes tombées par surcharge ce jour-là. Les lignes
     tombées par p0 ne sont pas renforcées : un accident n'est pas un signe de
     surcharge. Les étapes 4 et 5 modifient le réseau pour le lendemain : la
     journée en cours a été simulée avec l'état d'avant.

Sortie, commune aux deux modes — une entrée par jour, structure
`JourEvolution` (dans l'esprit de `Solution`) :
  - `delestage_total`, `taux_maximal`, `nombre_effectif` (les trois
    observables déjà fixés). Le délestage est celui du dispatch final : c'est
    la taille du blackout du jour. `taux_maximal` et `nombre_effectif` sont
    calculés sur le PREMIER dispatch du jour, avant toute avarie par
    surcharge, sur les seules lignes encore en service à ce moment (les
    lignes tombées par p0 sont exclues). L'état final est déformé par la
    cascade : son nombre effectif suivrait mécaniquement la taille du
    blackout au lieu de décrire la répartition des flux. `nombre_effectif`
    vaut NaN quand il n'est pas défini (flux tous nuls, ou aucune ligne en
    service) ;
  - `jour` et `n_lignes_en_service` (lignes en service au premier dispatch) ;
  - en mode auto_organise seulement : `capacite_totale_generateurs` et
    `marge_moyenne`, pour pouvoir tracer la croissance de capacité et
    vérifier que le mécanisme se comporte comme attendu (analogue à la Fig. 2
    du papier — nombre de blackouts qui croît légèrement puis se stabilise) ;
    elles valent pour la journée, avant les mises à niveau de fin de journée.
    S'y ajoutent `generateurs_ameliores` et `lignes_renforcees` (indices des
    nœuds et des lignes touchés à l'issue du jour), qui rendent testable la
    séquence des mises à niveau.

Un seul `numpy.random.Generator` transmis en paramètre, jamais d'état global —
même exigence que pour `cascade.journee()`.

Échelle pour cette version : quelques milliers de jours (proposition : 3000),
sans retrait de transitoire — assumé et documenté comme démonstration du
mécanisme, pas comme reproduction statistique complète de Carreras et al.
(qui utilisent 80 000 jours en régime stationnaire après un transitoire de
20 000 jours). Le passage à cette échelle complète est noté comme objectif du
rapport final.

Critères de validation :
  - mode independant, p0 = p1 = 0 : chaque jour équivaut à un `resoudre()`
    isolé sur la demande tirée ce jour-là ;
  - mode auto_organise, marge_seuil = -1 : la capacité n'augmente jamais
    (condition (c) jamais déclenchée), puisque la marge ΔP/P̄ vaut au moins -1,
    même quand la demande dépasse la capacité installée. Avec marge_seuil = 0,
    elle se déclencherait dès que la demande dépasse la capacité ;
  - mode auto_organise : la capacité totale des générateurs ne peut que
    croître ou rester stable, jamais diminuer, et chaque mise à niveau ajoute
    exactement k·P̄_D(t)/N_G ;
  - mode auto_organise : `mu` >= 1, donc les limites des lignes ne diminuent
    jamais ; seules les lignes tombées par surcharge un jour de blackout sont
    renforcées ;
  - mode auto_organise, graine fixée : séquence de générateurs mis à niveau,
    capacités finales et série de délestage reproductibles ;
  - dans les deux modes, `nombre_effectif`, quand il est défini, reste borné
    entre 1 et `n_lignes_en_service`.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .cascade import Journee, journee
from .dispatch import demande_uniforme
from .entropie import nombre_effectif
from .reseau import Reseau, capacite_incidente

MODES = ("independant", "auto_organise")
JOURS_PAR_AN = 365
# Sous cette fraction de la demande totale, un délestage n'est que du bruit du
# solveur linéaire, pas un blackout : il ne déclenche aucun renforcement.
TOLERANCE_DELESTAGE = 1e-9


@dataclass
class Parametres:
    """
    Paramètres d'une simulation sur plusieurs jours.

    Les champs `marge_seuil` et `mu` n'ont pas de valeur par défaut : ce sont les
    paramètres expérimentaux de la dynamique, obligatoires en mode auto_organise
    et ignorés en mode independant, comme `lambda_` et `k`. `limites` vaut par
    défaut `reseau.limites`. Les tableaux fournis ne sont jamais modifiés : la
    simulation travaille sur des copies.
    """

    mode: str
    demande_initiale: float
    puissance_max: np.ndarray
    g: float
    p0: float
    p1: float
    limites: np.ndarray | None = None
    lambda_: float = 1.018
    k: float = 0.02
    marge_seuil: float | None = None
    mu: float | None = None


@dataclass
class JourEvolution:
    """
    Observables d'une journée simulée.

    Les champs propres au mode auto_organise valent `None` en mode independant.
    """

    jour: int
    delestage_total: float
    taux_maximal: float
    nombre_effectif: float
    n_lignes_en_service: int
    capacite_totale_generateurs: float | None = None
    marge_moyenne: float | None = None
    generateurs_ameliores: np.ndarray | None = None
    lignes_renforcees: np.ndarray | None = None


def marge_moyenne(puissance_max: np.ndarray, demande_totale: float) -> float:
    """
    Marge moyenne de génération ΔP/P̄ = (somme des P_j^max − P̄_D) / P̄_D (Éq. 4).

    Elle mesure de combien la capacité installée dépasse la demande moyenne, en
    fraction de celle-ci. Elle est calculée sur la demande moyenne du jour, pas
    sur la demande tirée : c'est la grandeur à laquelle réagit l'ingénierie.
    """
    return float((np.sum(puissance_max) - demande_totale) / demande_totale)


def puissance_max_pour_marge(reseau: Reseau, demande_initiale: float,
                             marge: float) -> np.ndarray:
    """
    Capacité de départ uniforme des générateurs donnant une marge initiale voulue.

    Chaque générateur reçoit la même part de (1 + marge) × demande_initiale, de
    sorte que `marge_moyenne()` vaut exactement `marge` au premier jour.
    """
    demande_initiale = _reel(demande_initiale, "demande_initiale", minimum=0.0,
                             strict=True)
    marge = _reel(marge, "marge", minimum=-1.0, strict=True)
    n_g = reseau.generateurs.size
    return np.full(n_g, (1.0 + marge) * demande_initiale / n_g)


def _reel(valeur, nom: str, minimum: float | None = None, strict: bool = False) -> float:
    """Exige un réel fini, refuse les booléens, et contrôle une borne inférieure."""
    if isinstance(valeur, (bool, np.bool_)) or not isinstance(valeur, Real):
        raise ValueError(f"{nom} doit être un nombre réel.")
    valeur = float(valeur)
    if not np.isfinite(valeur):
        raise ValueError(f"{nom} doit être fini.")
    if minimum is not None and (valeur <= minimum if strict else valeur < minimum):
        signe = ">" if strict else ">="
        raise ValueError(f"{nom} doit être {signe} {minimum}.")
    return valeur


def _verifier(reseau: Reseau, n_jours, parametres: Parametres) -> None:
    """Contrôle les entrées de `simuler()` ; p0, p1 et g sont contrôlés par `journee()`."""
    if isinstance(n_jours, (bool, np.bool_)) or not isinstance(n_jours, Integral) \
            or n_jours < 1:
        raise ValueError("n_jours doit être un entier >= 1.")
    if parametres.mode not in MODES:
        raise ValueError(f"mode doit être l'un de {MODES}.")
    _reel(parametres.demande_initiale, "demande_initiale", minimum=0.0, strict=True)

    puissance_max = np.asarray(parametres.puissance_max, dtype=float)
    if puissance_max.shape != reseau.generateurs.shape:
        raise ValueError("Une capacité de départ par générateur est requise.")
    if not np.isfinite(puissance_max).all() or np.any(puissance_max < 0):
        raise ValueError("Les capacités des générateurs doivent être finies et positives.")

    if parametres.mode == "auto_organise":
        _reel(parametres.lambda_, "lambda_", minimum=0.0, strict=True)
        _reel(parametres.k, "k", minimum=0.0, strict=True)
        if parametres.marge_seuil is None or parametres.mu is None:
            raise ValueError("marge_seuil et mu sont requis en mode auto_organise.")
        _reel(parametres.marge_seuil, "marge_seuil")
        _reel(parametres.mu, "mu", minimum=1.0)


def _observer(jour: int, resultat: Journee) -> JourEvolution:
    """
    Extrait les observables du jour.

    Le délestage vient du dispatch final (le blackout), la répartition des flux du
    premier dispatch (le réseau avant que la cascade ne le déforme), sur les seules
    lignes alors en service.
    """
    en_service = ~resultat.hors_service_initial
    flux = resultat.solution_initiale.flux[en_service]
    return JourEvolution(
        jour=jour,
        delestage_total=resultat.delestage_total,
        taux_maximal=resultat.taux_maximal_initial,
        nombre_effectif=nombre_effectif(flux) if flux.size else float("nan"),
        n_lignes_en_service=int(en_service.sum()),
    )


def _ameliorer(reseau: Reseau, limites: np.ndarray, puissance_max: np.ndarray,
               demande_totale: float, marge_seuil: float, k: float,
               rng: np.random.Generator) -> np.ndarray:
    """
    Réponse d'ingénierie : améliore des générateurs tant que la marge est sous le seuil.

    Modifie `puissance_max` sur place et retourne les nœuds améliorés, dans l'ordre.
    Les générateurs sont parcourus dans un ordre aléatoire sans remise ; celui dont
    les lignes incidentes ne peuvent pas évacuer la puissance supplémentaire est
    sauté, et la boucle s'arrête faute de générateurs éligibles.
    """
    ameliores: list[int] = []
    if marge_moyenne(puissance_max, demande_totale) >= marge_seuil:
        return np.array(ameliores, dtype=int)

    ajout = k * demande_totale / reseau.generateurs.size
    for position in rng.permutation(reseau.generateurs.size):
        noeud = int(reseau.generateurs[position])
        if capacite_incidente(reseau, limites, noeud) > puissance_max[position] + ajout:
            puissance_max[position] += ajout
            ameliores.append(noeud)
            if marge_moyenne(puissance_max, demande_totale) >= marge_seuil:
                break
    return np.array(ameliores, dtype=int)


def _renforcer(resultat: Journee, demande_totale: float, limites: np.ndarray,
               mu: float) -> np.ndarray:
    """
    Renforce de `mu` les lignes tombées par surcharge un jour de blackout.

    Modifie `limites` sur place et retourne les lignes renforcées. Les lignes
    tombées par p0 sont exclues : un accident n'indique pas une surcharge.
    """
    blackout = resultat.delestage_total > TOLERANCE_DELESTAGE * demande_totale
    if not blackout or not resultat.avaries_par_surcharge:
        return np.array([], dtype=int)
    lignes = np.concatenate(resultat.avaries_par_surcharge)
    limites[lignes] *= mu
    return lignes


def simuler(reseau: Reseau, n_jours: int, parametres: Parametres,
            rng: np.random.Generator) -> list[JourEvolution]:
    """
    Fait évoluer le réseau sur `n_jours` jours et retourne une entrée par jour.

    En mode independant, le réseau reste identique et chaque jour est un tirage
    i.i.d. autour de la demande `demande_initiale`. En mode auto_organise, la
    demande croît, les générateurs sont améliorés quand la marge passe sous le
    seuil et les lignes tombées par surcharge sont renforcées. Chaque jour repart
    avec toutes les lignes en service : les avaries ne durent qu'une journée.

    Les tirages suivent un ordre fixe, celui de `journee()` puis, en mode
    auto_organise, celui du choix des générateurs à améliorer. Le `Reseau` et les
    tableaux de `parametres` ne sont jamais modifiés.
    """
    _verifier(reseau, n_jours, parametres)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng doit être un numpy.random.Generator.")
    limites = reseau.limites if parametres.limites is None else parametres.limites
    if limites is None:
        raise ValueError("Aucune limite de ligne définie.")

    auto = parametres.mode == "auto_organise"
    # Copies : les capacités et les limites évoluent au fil des jours.
    limites = np.array(limites, dtype=float)
    puissance_max = np.array(parametres.puissance_max, dtype=float)
    demande_initiale = float(parametres.demande_initiale)
    lambda_jour = parametres.lambda_ ** (1.0 / JOURS_PAR_AN)

    jours: list[JourEvolution] = []
    for t in range(1, n_jours + 1):
        # Demande moyenne du jour : constante en mode independant, en croissance
        # exponentielle quotidienne en mode auto_organise.
        demande_totale = (demande_initiale * np.exp((lambda_jour - 1.0) * t)
                          if auto else demande_initiale)
        resultat = journee(reseau, demande_uniforme(reseau, demande_totale),
                           puissance_max, parametres.p0, parametres.p1, parametres.g,
                           rng, limites=limites)
        jour = _observer(t, resultat)

        if auto:
            jour.capacite_totale_generateurs = float(puissance_max.sum())
            jour.marge_moyenne = marge_moyenne(puissance_max, demande_totale)
            jour.generateurs_ameliores = _ameliorer(
                reseau, limites, puissance_max, demande_totale,
                parametres.marge_seuil, parametres.k, rng)
            jour.lignes_renforcees = _renforcer(resultat, demande_totale, limites,
                                                parametres.mu)
        jours.append(jour)
    return jours


def series(jours: list[JourEvolution]) -> dict[str, np.ndarray]:
    """
    Convertit la liste des jours en séries temporelles, une par observable.

    Les colonnes propres au mode auto_organise n'apparaissent que si la simulation
    était dans ce mode. Les indices des générateurs et des lignes touchés ne sont
    pas des séries de longueur fixe : on en garde le nombre par jour.
    """
    if not jours:
        raise ValueError("Aucun jour à convertir.")
    resultat = {
        "jour": np.array([j.jour for j in jours], dtype=int),
        "delestage_total": np.array([j.delestage_total for j in jours], dtype=float),
        "taux_maximal": np.array([j.taux_maximal for j in jours], dtype=float),
        "nombre_effectif": np.array([j.nombre_effectif for j in jours], dtype=float),
        "n_lignes_en_service": np.array([j.n_lignes_en_service for j in jours], dtype=int),
    }
    if jours[0].capacite_totale_generateurs is not None:
        resultat["capacite_totale_generateurs"] = np.array(
            [j.capacite_totale_generateurs for j in jours], dtype=float)
        resultat["marge_moyenne"] = np.array([j.marge_moyenne for j in jours], dtype=float)
        resultat["n_mises_a_niveau"] = np.array(
            [j.generateurs_ameliores.size for j in jours], dtype=int)
        resultat["n_lignes_renforcees"] = np.array(
            [j.lignes_renforcees.size for j in jours], dtype=int)
    return resultat


def ecrire(series: dict[str, np.ndarray], chemin: Path) -> None:
    """
    Écrit les séries dans un fichier .npz, seul lien avec le fil B.

    Toutes les séries doivent avoir la même longueur : une entrée par jour. Le
    dossier parent est créé au besoin.
    """
    if not series:
        raise ValueError("Aucune série à écrire.")
    if len({np.asarray(valeurs).shape for valeurs in series.values()}) != 1:
        raise ValueError("Toutes les séries doivent avoir la même longueur.")
    chemin = Path(chemin)
    if chemin.suffix != ".npz":
        raise ValueError("Le fichier de sortie doit avoir l'extension .npz.")
    chemin.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(chemin, **series)
