"""Indicateurs de référence.

Ces mesures servent de témoins : si les mesures entropiques ne font que
reproduire leur comportement, elles n'apportent pas d'information nouvelle.

L'exposant de Hurst est le témoin principal, puisque c'est la mesure utilisée par
Carreras et al. pour caractériser les séries temporelles de blackouts produites
par la dynamique lente de leur modèle.
"""

from __future__ import annotations

import numpy as np

from ._validation import serie_finie


def _rs_segment(segment: np.ndarray) -> float:
    """Statistique R/S d'un segment.

    R est l'étendue de la série des écarts cumulés à la moyenne, S son écart-type.
    Retourne NaN si le segment est constant (S nul), ce qui rend la statistique
    indéfinie.
    """
    ecart = segment.std(ddof=0)
    if ecart == 0:
        return np.nan
    cumul = np.cumsum(segment - segment.mean())
    etendue = cumul.max() - cumul.min()
    return etendue / ecart


def rs_par_echelle(
    serie: np.ndarray,
    echelles: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Statistique R/S moyenne pour une gamme d'échelles.

    La série est découpée en segments disjoints de longueur n, la statistique R/S
    est calculée sur chacun, puis moyennée. Les échelles par défaut sont espacées
    logarithmiquement entre 8 points et un quart de la longueur de la série, ce
    qui garantit au moins quatre segments par échelle.

    Retourne les échelles effectivement utilisées et les R/S correspondantes.
    """
    serie = serie_finie(serie)
    n_total = serie.size
    if n_total < 32:
        raise ValueError("Série trop courte pour une analyse R/S (minimum 32 points).")

    if echelles is None:
        echelles = np.unique(
            np.logspace(np.log10(8), np.log10(n_total // 4), num=20).astype(int)
        )

    tailles, valeurs = [], []
    for n in echelles:
        if n < 8 or n > n_total // 2:
            continue
        n_segments = n_total // n
        segments = serie[: n_segments * n].reshape(n_segments, n)
        rs = np.array([_rs_segment(s) for s in segments])
        rs = rs[np.isfinite(rs)]
        if rs.size == 0:
            continue
        tailles.append(n)
        valeurs.append(rs.mean())

    return np.array(tailles, dtype=float), np.array(valeurs, dtype=float)


def hurst(
    serie: np.ndarray,
    echelles: np.ndarray | None = None,
    retourner_ajustement: bool = False,
):
    """Exposant de Hurst par la méthode R/S.

    L'exposant est la pente de log(R/S) en fonction de log(n). Une valeur voisine
    de 0,5 indique une série non corrélée, une valeur supérieure une persistance,
    une valeur inférieure une antipersistance.

    Avec `retourner_ajustement`, retourne aussi les échelles et les R/S ayant servi
    à l'ajustement, pour tracer la droite de régression et juger de sa qualité.
    """
    tailles, valeurs = rs_par_echelle(serie, echelles)
    if tailles.size < 3:
        raise ValueError("Pas assez d'échelles exploitables pour estimer l'exposant.")

    pente, ordonnee = np.polyfit(np.log(tailles), np.log(valeurs), deg=1)
    if retourner_ajustement:
        return pente, tailles, valeurs, ordonnee
    return pente


def charge_moyenne(taux_de_charge: np.ndarray) -> float:
    """Taux de charge moyen des lignes pour un état du réseau."""
    return float(np.mean(serie_finie(taux_de_charge)))


def charge_maximale(taux_de_charge: np.ndarray) -> float:
    """Taux de charge maximal des lignes, noté M_max.

    Témoin trivial le plus fort du projet : c'est l'indicateur qui déclenche
    directement la cascade dans le modèle, donc toute mesure entropique doit être
    comparée à lui.
    """
    return float(np.max(serie_finie(taux_de_charge)))


# --------------------------------------------------------------------------
# Distribution nulle de l'exposant de Hurst
# --------------------------------------------------------------------------


def distribution_nulle_hurst(
    n: int,
    n_tirages: int = 200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Exposants de Hurst obtenus sur des séries non corrélées de longueur n.

    L'estimateur R/S surestime systématiquement H sur des séries finies. Comparer
    un exposant mesuré à la valeur théorique de 0,5 conduit donc à conclure à une
    persistance qui n'existe pas. La référence correcte est la distribution
    obtenue ici, à la longueur de série effectivement analysée.

    Retourne les exposants bruts, pour en tirer moyenne, écart-type ou quantiles.
    """
    if rng is None:
        rng = np.random.default_rng()
    return np.array([hurst(rng.standard_normal(n)) for _ in range(n_tirages)])


def intervalle_nul(
    n: int,
    n_tirages: int = 200,
    niveau: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Moyenne et bornes de l'intervalle de référence pour l'exposant de Hurst.

    Un exposant mesuré n'est déclaré persistant ou antipersistant que s'il sort
    de cet intervalle.
    """
    valeurs = distribution_nulle_hurst(n, n_tirages, rng)
    marge = (1.0 - niveau) / 2.0 * 100
    return (
        float(valeurs.mean()),
        float(np.percentile(valeurs, marge)),
        float(np.percentile(valeurs, 100 - marge)),
    )


# --------------------------------------------------------------------------
# Queue en loi de puissance
# --------------------------------------------------------------------------


def survie(tailles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fonction de survie empirique, P(X >= x), pour un tracé log-log.

    C'est la représentation utilisée dans la littérature de référence pour
    visualiser la queue de la distribution des tailles de blackout.
    """
    valeurs = np.sort(serie_finie(tailles))
    valeurs = valeurs[valeurs > 0]
    if valeurs.size == 0:
        raise ValueError("Aucune valeur strictement positive.")
    probabilites = 1.0 - np.arange(valeurs.size) / valeurs.size
    return valeurs, probabilites


def exposant_loi_de_puissance(
    tailles: np.ndarray,
    seuil: float | None = None,
) -> tuple[float, float, int]:
    """Exposant d'une queue en loi de puissance, par maximum de vraisemblance.

    La densité est supposée de la forme p(x) proportionnel à x**(-alpha) au-delà
    du seuil. L'estimateur du maximum de vraisemblance est plus fiable que
    l'ajustement d'une droite sur un histogramme log-log, qui pondère mal les
    événements rares.

    Le seuil vaut par défaut la plus petite valeur strictement positive. La
    convention de la littérature de référence est de rapporter alpha, l'exposant
    de la densité : la pente de la fonction de survie en log-log vaut alpha - 1.
    Le modèle OPA sur un arbre donne alpha voisin de 1,6.

    Retourne l'exposant, son incertitude et le nombre de points utilisés.
    """
    valeurs = serie_finie(tailles)
    valeurs = valeurs[valeurs > 0]
    if seuil is None:
        seuil = float(valeurs.min())
    queue = valeurs[valeurs >= seuil]
    if queue.size < 10:
        raise ValueError("Trop peu de points au-delà du seuil pour un ajustement.")

    n = queue.size
    alpha = 1.0 + n / np.sum(np.log(queue / seuil))
    incertitude = (alpha - 1.0) / np.sqrt(n)
    return float(alpha), float(incertitude), int(n)
