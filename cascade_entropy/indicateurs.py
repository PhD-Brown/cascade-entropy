"""
Indicateurs de référence.

Ce module rassemble les indicateurs qui servent de points de comparaison aux
mesures entropiques du projet.

On peut voir ces fonctions comme des "témoins" : si une mesure entropique ne fait
que retrouver les signes déjà présents dans un indicateur plus simple, elle ne
ajoute pas de nouvelle information. C'est pourquoi ce fichier contient des
indicateurs classiques de séries temporelles et de queues de distribution.

Le plus important est l'exposant de Hurst, car il mesure la persistance ou
l'antipersistance d'une série. Dans le cadre des blackouts, c'est une quantité
centrale : la littérature de référence s'en sert pour caractériser des séries
qui présentent des corrélations à long terme et des mécanismes de propagation
lents.

Autrement dit, ce module ne cherche pas à "résoudre" le problème directement ;
il cherche à fournir des signatures simples et interprétables qui permettent de
comprendre la structure des données avant d'interpréter les résultats plus
subtils produits par les mesures d'entropie.
"""

from __future__ import annotations

import numpy as np

from ._validation import serie_finie


def _rs_segment(segment: np.ndarray) -> float:
    """
    Statistique R/S d'un segment.

    La méthode R/S (range over standard deviation) est une manière de mesurer la
    mémoire d'une série temporelle.

    On construit d'abord les écarts cumulés par rapport à la moyenne :
        x_t - moyenne
    puis on cumule ces écarts pour obtenir une marche aléatoire locale. La
    quantité R mesure l'amplitude totale de cette marche, c'est-à-dire la
    différence entre son maximum et son minimum. La quantité S est l'écart-type
    du segment, qui normalise cette amplitude.

    Si le segment est constant, alors S = 0 et la statistique n'a plus de sens.
    On renvoie NaN dans ce cas afin de retirer proprement ce segment de
    l'analyse.
    """
    # L'écart-type du segment sert de facteur d'échelle : il mesure l'amplitude
    # typique des variations du signal.
    ecart = segment.std(ddof=0)
    if ecart == 0:
        # Un segment constant ne doit pas être utilisé, car R/S est alors
        # indéfini : la marche cumulée est plate et n'a pas de portée.
        return np.nan

    # On enlève la moyenne avant de cumuler, pour regarder seulement les
    # fluctuations autour du niveau moyen du segment.
    cumul = np.cumsum(segment - segment.mean())

    # R est la portée totale de la marche : l'amplitude du déplacement cumulatif.
    etendue = cumul.max() - cumul.min()

    # R/S compare la portée de la marche à l'écart-type du signal. Une valeur
    # grande indique des excursions persistantes ou très structurées dans le temps.
    return etendue / ecart


def rs_par_echelle(
    serie: np.ndarray,
    echelles: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Statistique R/S moyenne pour une gamme d'échelles.

    L'idée est de regarder comment la statistique R/S évolue quand on change la
    fenêtre d'observation. Si on choisit une échelle n, on découpe la série en
    blocs de longueur n, on calcule R/S sur chaque bloc, puis on en prend la
    moyenne.

    La pente de cette courbe en échelle logarithmique permet ensuite d'estimer
    l'exposant de Hurst : c'est l'idée centrale de la méthode R/S.

    Les échelles par défaut sont choisies logarithmiquement entre 8 points et un
    quart de la longueur totale. Cela évite de se placer dans des régimes trop
    petits ou trop grands où l'estimation serait peu fiable.

    Retourne les échelles effectivement utilisées et les R/S correspondantes.
    """
    # On valide d'abord la série pour s'assurer qu'elle contient bien des valeurs
    # finies, ce qui est essentiel pour toute analyse statistique.
    serie = serie_finie(serie)
    n_total = serie.size
    if n_total < 32:
        # En dessous de 32 points, il y a trop peu de données pour estimer une
        # échelle de corrélation avec une méthode R/S stable.
        raise ValueError("Série trop courte pour une analyse R/S (minimum 32 points).")

    if echelles is None:
        # np.logspace crée une grille logarithmique de longueurs d'échelle. On
        # garde des entiers et on élimine les doublons pour obtenir une liste
        # propre des échelles testées.
        echelles = np.unique(
            np.logspace(np.log10(8), np.log10(n_total // 4), num=20).astype(int)
        )

    tailles, valeurs = [], []
    for n in echelles:
        # On ignore les échelles trop petites ou trop grandes pour garder une
        # estimation suffisamment robuste et comparables entre séries.
        if n < 8 or n > n_total // 2:
            continue

        # Nombre de segments complets disponibles pour cette échelle.
        n_segments = n_total // n

        # On découpe la série en blocs disjoints de longueur n.
        segments = serie[: n_segments * n].reshape(n_segments, n)

        # Pour chaque bloc, on calcule R/S. On élimine ensuite les cas où ce
        # calcul est indéfini (segment constant, etc.).
        rs = np.array([_rs_segment(s) for s in segments])
        rs = rs[np.isfinite(rs)]
        if rs.size == 0:
            continue

        # On conserve l'échelle et la valeur moyenne du rapport R/S sur les
        # segments valides.
        tailles.append(n)
        valeurs.append(rs.mean())

    return np.array(tailles, dtype=float), np.array(valeurs, dtype=float)


def hurst(
    serie: np.ndarray,
    echelles: np.ndarray | None = None,
    retourner_ajustement: bool = False,
):
    """
    Exposant de Hurst par la méthode R/S.

    L'exposant est la pente de log(R/S) en fonction de log(n). Cette pente se
    lit comme une loi d'échelle :

        R/S ~ n^H

    donc log(R/S) = H log(n) + constante.

    Si H ≈ 0,5, la série est proche d'un bruit blanc : absence de mémoire à long
    terme. Si H > 0,5, la série est persistante : les variations positives ou
    négatives tendent à se prolonger. Si H < 0,5, la série est antipersistante :
    les changements se compensent plus souvent.

    Avec `retourner_ajustement`, retourne aussi les échelles et les R/S ayant servi
    à l'ajustement, pour tracer la droite de régression et juger de sa qualité.
    """
    # On calcule le rapport R/S en fonction de l'échelle choisie.
    tailles, valeurs = rs_par_echelle(serie, echelles)
    if tailles.size < 3:
        # Il faut au moins quelques échelles pour construire une droite de
        # régression stable.
        raise ValueError("Pas assez d'échelles exploitables pour estimer l'exposant.")

    # La pente de la régression log-log donne H. C'est en pratique la mesure de
    # la persistance de la série.
    pente, ordonnee = np.polyfit(np.log(tailles), np.log(valeurs), deg=1)
    if retourner_ajustement:
        return pente, tailles, valeurs, ordonnee
    return pente


def charge_moyenne(taux_de_charge: np.ndarray) -> float:
    """
    Taux de charge moyen des lignes pour un état du réseau.

    C'est simplement la moyenne arithmétique des charges sur toutes les lignes.
    Cette valeur donne un aperçu global du niveau de sollicitation du réseau,
    sans dire où se situent les points critiques.
    """
    return float(np.mean(serie_finie(taux_de_charge)))


def charge_maximale(taux_de_charge: np.ndarray) -> float:
    """
    Taux de charge maximal des lignes, noté M_max.

    C'est un indicateur très simple mais très puissant : si une ligne atteint une
    charge élevée, elle devient susceptible de provoquer un dépassement de
    capacité et d'initier ou d'alimenter une cascade.

    Dans le projet, c'est l'un des "témoins" les plus importants. Toute mesure
    entropique ou statistique est d'autant plus pertinente qu'elle est comparée à
    ce seuil de stress local ou global.
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
    """
    Exposants de Hurst obtenus sur des séries non corrélées de longueur n.

    L'estimateur R/S surestime systématiquement H sur des séries finies. En
    d'autres termes, un bruit blanc sur une série courte peut apparaître comme un
    signal persistant si on compare simplement sa valeur à 0,5.

    C'est exactement pourquoi ce code simule un grand nombre de séries aléatoires
    gaussiennes de longueur n et recalcule H sur chacune d'elles : on obtient une
    distribution de référence réaliste pour cette taille de série.

    La bonne interprétation est donc locale : pour la longueur donnée, on compare
    la valeur observée à la distribution nulles, et non à 0,5 de manière brute.

    Retourne les exposants bruts, pour en tirer moyenne, écart-type ou quantiles.
    """
    if rng is None:
        # On crée un générateur aléatoire NumPy, ce qui permet d'avoir des tirages
        # reproductibles si l'appelant fournit explicitement un rng.
        rng = np.random.default_rng()

    # On simule 200 séries de bruit blanc, puis on mesure H sur chacune. La
    # liste obtenue est la distribution de référence du test nul.
    return np.array([hurst(rng.standard_normal(n)) for _ in range(n_tirages)])


def intervalle_nul(
    n: int,
    n_tirages: int = 200,
    niveau: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Moyenne et bornes de l'intervalle de référence pour l'exposant de Hurst.

    Une fois la distribution nulle construite, on calcule sa moyenne et des
    quantiles. Si l'exposant observé tombe en dehors de ces bornes, cela suggère
    que la série ne ressemble pas à un bruit blanc pour cette longueur de
    fenêtre ; on peut alors parler de persistance ou d'antipersistance au sens
    statistique.

    Exemple : si le niveau vaut 0,95, on prend le centile 2,5% et le centile
    97,5% de la distribution nulle. Une mesure en dehors de cet intervalle est
    alors considérée comme suffisamment rare pour être interprétée comme un
    signal significatif.
    """
    valeurs = distribution_nulle_hurst(n, n_tirages, rng)

    # On calcule la marge de confiance à deux extrémités de la distribution.
    marge = (1.0 - niveau) / 2.0 * 100

    # Retourne : moyenne de la distribution nulle, borne basse, borne haute.
    return (
        float(valeurs.mean()),
        float(np.percentile(valeurs, marge)),
        float(np.percentile(valeurs, 100 - marge)),
    )


# --------------------------------------------------------------------------
# Queue en loi de puissance
# --------------------------------------------------------------------------


def survie(tailles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fonction de survie empirique, P(X >= x), pour un tracé log-log.

    Dans une distribution de tailles d'événements (par exemple des blackouts ou
    des pics de charge), on s'intéresse souvent à la probabilité qu'une valeur
    soit supérieure ou égale à x. C'est la fonction de survie.

    Sur un graphique log-log, la queue d'une loi de puissance apparaît comme une
    droite, ce qui est un signe typique de phénomène critique ou de distribution
    à queue lourde. C'est la raison pour laquelle cette fonction est si centrale
    dans les analyses de fiabilité, d'avalanches et de réseaux complexes.
    """
    valeurs = np.sort(serie_finie(tailles))

    # On ne conserve que les valeurs strictement positives, car une taille ou une
    # amplitude négative n'a pas de sens dans cette mécanique de queue de
    # distribution.
    valeurs = valeurs[valeurs > 0]
    if valeurs.size == 0:
        raise ValueError("Aucune valeur strictement positive.")

    # P(X >= x) est approchée empiriquement par le rapport entre le nombre de
    # valeurs supérieures ou égales à x et le nombre total d'observations.
    probabilites = 1.0 - np.arange(valeurs.size) / valeurs.size
    return valeurs, probabilites


def exposant_loi_de_puissance(
    tailles: np.ndarray,
    seuil: float | None = None,
) -> tuple[float, float, int]:
    """
    Exposant d'une queue en loi de puissance, par maximum de vraisemblance.

    La queue de distribution est supposée suivre une loi de puissance :

        p(x) ~ x^(-alpha)

    pour x >= seuil. Ici, alpha est l'exposant du comportement de queue. Une
    queue plus lourde correspond à un alpha plus faible ; une queue plus légère
    correspond à un alpha plus grand.

    L'estimation par maximum de vraisemblance est plus fiable qu'un simple
    ajustement sur un histogramme log-log, car elle prend mieux en compte les
    événements rares qui dominent souvent la queue de distribution.

    Le seuil vaut par défaut la plus petite valeur strictement positive. La
    convention de la littérature de référence est de rapporter alpha, l'exposant
    de la densité : la pente de la fonction de survie en log-log vaut alpha - 1.
    Le modèle OPA sur un arbre donne alpha voisin de 1,6.

    Retourne l'exposant, son incertitude et le nombre de points utilisés.
    """
    valeurs = serie_finie(tailles)

    # On travaille uniquement sur les tailles strictement positives, car la
    # queue de distribution concerne des amplitudes ou des intensités non nulles.
    valeurs = valeurs[valeurs > 0]
    if seuil is None:
        # Le seuil minimal est une première approximation utile pour découper la
        # partie de queue qu'on cherche à caractériser.
        seuil = float(valeurs.min())
    queue = valeurs[valeurs >= seuil]
    if queue.size < 10:
        raise ValueError("Trop peu de points au-delà du seuil pour un ajustement.")

    n = queue.size

    # Formule du maximum de vraisemblance pour une loi de puissance tronquée à
    # partir du seuil : alpha = 1 + n / sum(log(x/seuil)).
    alpha = 1.0 + n / np.sum(np.log(queue / seuil))

    # L'incertitude est estimée à partir de la variance asymptotique de
    # l'estimateur ; elle diminue quand le nombre de données augmente.
    incertitude = (alpha - 1.0) / np.sqrt(n)
    return float(alpha), float(incertitude), int(n)
