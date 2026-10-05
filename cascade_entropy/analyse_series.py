"""
Analyse statistique des longues séries de blackouts (fil B, SO2).

Ce module ne connaît rien du réseau : il ne reçoit que des tableaux NumPy lus
sur disque. Il rassemble les outils de la mission 3 de SO2, c'est-à-dire de la
comparaison avec Carreras et al. (2004) :

1. **R/S par plage d'échelles.** Carreras ajuste R/S sur deux plages : courte
   (« a few days and a few years ») et longue (600 < t < 10^5). Une série peut
   changer de pente entre les deux (cassure). `pentes_rs` donne une pente par
   plage à partir d'une seule courbe R/S.

2. **Référence par mélange.** R/S est biaisé aux petites échelles : même sur un
   bruit blanc, la pente dépasse 0.5 (Anis et Lloyd, 1976). Pour savoir si un H
   mesuré traduit une vraie mémoire, on le compare à la même mesure sur des
   copies mélangées de la série. Le mélange garde la distribution des valeurs
   mais détruit tout ordre temporel. `references_melange` et
   `resume_reference` construisent cet intervalle de référence.

3. **Période dominante.** Un cycle déterministe (renforcement des lignes de μ,
   rattrapé par la croissance λ de la demande) crée une cassure dans R/S.
   `periode_dominante` le repère par le périodogramme.

4. **Queue de la distribution des tailles.** La Table I de Carreras donne un
   indice de décroissance et une « étendue » de la loi de puissance, sans
   définir la méthode. `pdf_logarithmique` et `plage_loi_puissance` en donnent
   une opérationnalisation explicite et déterministe : la plus longue plage
   contiguë, en classes logarithmiques, où log(pdf) est linéaire en log(taille)
   avec R² ≥ r2_min. Le choix est le nôtre et doit être annoncé comme tel.

Critères de validation (tests/test_analyse_series.py) : `rs_vectorise` égale
`indicateurs.rs_par_echelle` ; sur un bruit blanc, la pente mesurée tombe dans
l'intervalle de mélange ; sur un bruit corrélé, elle en sort ; un sinus de
période 1000 est retrouvé à 1000 ; un échantillon de Pareto redonne son indice.
"""

from __future__ import annotations

from numbers import Real

import numpy as np

from ._validation import entier_positif, serie_finie

# Plus petite échelle admise par R/S (même règle que indicateurs.rs_par_echelle).
ECHELLE_MIN = 8


def _reel_positif(valeur, nom: str) -> float:
    if isinstance(valeur, (bool, np.bool_)) or not isinstance(valeur, Real):
        raise ValueError(f"{nom} doit être un nombre réel > 0.")
    valeur = float(valeur)
    if not np.isfinite(valeur) or valeur <= 0:
        raise ValueError(f"{nom} doit être un nombre réel > 0.")
    return valeur


def _plages_valides(plages) -> np.ndarray:
    """Exige une liste de couples (début, fin) avec 0 < début < fin."""
    plages = np.asarray(plages, dtype=float)
    if plages.ndim != 2 or plages.shape[1] != 2 or plages.shape[0] == 0:
        raise ValueError("plages doit être une liste non vide de couples (début, fin).")
    if not np.all(np.isfinite(plages)) or np.any(plages[:, 0] <= 0) \
            or np.any(plages[:, 1] <= plages[:, 0]):
        raise ValueError("Chaque plage doit vérifier 0 < début < fin.")
    return plages


def echelles_log(n_total: int, n_echelles: int = 40) -> np.ndarray:
    """
    Grille logarithmique d'échelles entières, de 8 au quart de la série.

    Le quart est la borne usuelle : au-delà, il reste moins de quatre blocs
    pour moyenner R/S et la statistique devient trop bruitée.
    """
    n_total = entier_positif(n_total, "n_total", minimum=4 * ECHELLE_MIN)
    n_echelles = entier_positif(n_echelles, "n_echelles", minimum=3)
    return np.unique(np.rint(np.logspace(np.log10(ECHELLE_MIN), np.log10(n_total // 4),
                                         n_echelles)).astype(int))


def rs_vectorise(serie, echelles=None) -> tuple[np.ndarray, np.ndarray]:
    """
    R/S moyen par échelle : même définition que `indicateurs.rs_par_echelle`.

    Pour chaque échelle n, la série est coupée en blocs disjoints de longueur n.
    Sur chaque bloc : R = étendue des écarts cumulés à la moyenne du bloc,
    S = écart-type du bloc (ddof = 0), puis R/S est moyenné sur les blocs non
    constants. Les échelles < 8 ou > N/2 sont ignorées.

    Seule l'implémentation change : tous les blocs d'une échelle sont traités
    d'un coup (tableau m × n) au lieu d'une boucle Python, ce qui rend
    possibles les dizaines de mélanges nécessaires à l'intervalle de référence.
    """
    serie = serie_finie(serie)
    n_total = serie.size
    if n_total < 4 * ECHELLE_MIN:
        raise ValueError("Série trop courte pour une analyse R/S (minimum 32 points).")
    echelles = echelles_log(n_total, 20) if echelles is None else np.asarray(echelles)

    tailles, valeurs = [], []
    for n in echelles:
        n = int(n)
        if n < ECHELLE_MIN or n > n_total // 2:
            continue
        m = n_total // n
        blocs = serie[: m * n].reshape(m, n)
        ecart = blocs.std(axis=1)
        cumul = np.cumsum(blocs - blocs.mean(axis=1, keepdims=True), axis=1)
        etendue = cumul.max(axis=1) - cumul.min(axis=1)
        valides = ecart > 0
        if not np.any(valides):
            continue
        tailles.append(n)
        valeurs.append(float(np.mean(etendue[valides] / ecart[valides])))
    return np.array(tailles, dtype=float), np.array(valeurs, dtype=float)


def pentes_depuis_courbe(tailles, valeurs, plages, points_min: int = 3) -> np.ndarray:
    """
    Pente de log(R/S) en fonction de log(n) sur chaque plage, à partir d'une courbe.

    `tailles` et `valeurs` sont la sortie de `rs_vectorise`. Une plage qui
    contient moins de `points_min` échelles exploitables (par exemple parce
    que sa fin dépasse N/4) renvoie NaN plutôt qu'une pente estimée sur deux
    points. Séparer le calcul de la courbe de son ajustement permet de tracer
    la courbe et d'en tirer plusieurs pentes sans la recalculer.
    """
    plages = _plages_valides(plages)
    points_min = entier_positif(points_min, "points_min", minimum=2)
    tailles = np.asarray(tailles, dtype=float)
    valeurs = np.asarray(valeurs, dtype=float)
    if tailles.shape != valeurs.shape or tailles.ndim != 1:
        raise ValueError("tailles et valeurs doivent être deux tableaux 1D de même taille.")
    pentes = np.full(plages.shape[0], np.nan)
    for i, (debut, fin) in enumerate(plages):
        dedans = (tailles >= debut) & (tailles <= fin) & (valeurs > 0)
        if np.count_nonzero(dedans) >= points_min:
            pentes[i] = np.polyfit(np.log(tailles[dedans]), np.log(valeurs[dedans]), 1)[0]
    return pentes


def pentes_rs(serie, plages, echelles=None, points_min: int = 3) -> np.ndarray:
    """
    Pente de log(R/S) en fonction de log(n) sur chaque plage [début, fin].

    Une seule courbe R/S est calculée (`rs_vectorise`), puis ajustée
    séparément sur chaque plage (`pentes_depuis_courbe`).
    """
    plages = _plages_valides(plages)
    serie = serie_finie(serie)
    if echelles is None:
        echelles = echelles_log(serie.size)
    tailles, valeurs = rs_vectorise(serie, echelles)
    return pentes_depuis_courbe(tailles, valeurs, plages, points_min)


def references_melange(serie, plages, n_melanges: int,
                       rng: np.random.Generator, echelles=None,
                       points_min: int = 3) -> np.ndarray:
    """
    Pentes R/S de `n_melanges` copies mélangées de la série (une ligne par copie).

    Le mélange (permutation aléatoire des valeurs) conserve la distribution
    exacte de la série, y compris ses nombreux zéros, mais détruit tout ordre
    temporel. La dispersion des pentes obtenues est donc ce que R/S donne
    pour CETTE distribution sans aucune mémoire : biais aux petites échelles
    compris.
    """
    n_melanges = entier_positif(n_melanges, "n_melanges", minimum=1)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng doit être un numpy.random.Generator.")
    serie = serie_finie(serie)
    plages = _plages_valides(plages)
    if echelles is None:
        echelles = echelles_log(serie.size)
    return np.array([pentes_rs(rng.permutation(serie), plages, echelles, points_min)
                     for _ in range(n_melanges)])


def resume_reference(references, niveau: float = 0.95):
    """
    Médiane et intervalle central de niveau `niveau` des pentes de mélange.

    Retourne trois tableaux (médiane, borne basse, borne haute), un élément
    par plage. Les NaN (plages non estimables) sont ignorés ; une plage sans
    aucune valeur donne NaN.
    """
    references = np.asarray(references, dtype=float)
    if references.ndim != 2 or references.shape[0] == 0:
        raise ValueError("references doit être un tableau (n_melanges, n_plages).")
    if isinstance(niveau, (bool, np.bool_)) or not isinstance(niveau, Real) \
            or not 0 < niveau < 1:
        raise ValueError("niveau doit être dans ]0, 1[.")
    alpha = (1 - float(niveau)) / 2
    mediane = np.full(references.shape[1], np.nan)
    bas, haut = mediane.copy(), mediane.copy()
    for j in range(references.shape[1]):
        col = references[:, j][np.isfinite(references[:, j])]
        if col.size:
            mediane[j], bas[j], haut[j] = np.quantile(col, [0.5, alpha, 1 - alpha])
    return mediane, bas, haut


def periode_dominante(serie, periode_min: float, periode_max: float,
                      largeur_relative: float = 0.1) -> tuple[float, float]:
    """
    Période du plus haut pic du périodogramme entre `periode_min` et `periode_max`.

    Retourne (période, fraction), où fraction est la part de la puissance de
    la bande [periode_min, periode_max] contenue à ± `largeur_relative` de la
    période du pic. Une fraction proche de 0 signifie qu'aucun cycle ne se
    détache ; une série constante renvoie (NaN, 0).

    La résolution en période vaut ≈ P² / N : pour N = 100 000 jours, une
    période de 1000 jours est connue à ± 10 jours environ.
    """
    serie = serie_finie(serie)
    periode_min = _reel_positif(periode_min, "periode_min")
    periode_max = _reel_positif(periode_max, "periode_max")
    largeur_relative = _reel_positif(largeur_relative, "largeur_relative")
    if periode_min < 2 or periode_max <= periode_min:
        raise ValueError("Il faut 2 <= periode_min < periode_max.")
    if periode_max > serie.size:
        raise ValueError("periode_max ne peut dépasser la longueur de la série.")
    centree = serie - serie.mean()
    puissance = np.abs(np.fft.rfft(centree)) ** 2
    frequences = np.fft.rfftfreq(serie.size)
    bande = (frequences >= 1 / periode_max) & (frequences <= 1 / periode_min)
    if not np.any(bande) or puissance[bande].sum() == 0:
        return float("nan"), 0.0
    i = np.argmax(np.where(bande, puissance, -1.0))
    periode = 1.0 / frequences[i]
    with np.errstate(divide="ignore"):
        periodes = np.where(frequences > 0, 1.0 / frequences, np.inf)
    pres = bande & (np.abs(periodes - periode) <= largeur_relative * periode)
    return float(periode), float(puissance[pres].sum() / puissance[bande].sum())


def pdf_logarithmique(valeurs, n_classes: int = 25):
    """
    Densité de probabilité en classes logarithmiques des valeurs > 0.

    Retourne (centres géométriques, densité, effectifs) des classes non vides.
    La densité est normalisée sur les valeurs > 0 (les zéros, jours sans
    blackout, ne font pas partie de la distribution des tailles).
    """
    valeurs = serie_finie(valeurs)
    n_classes = entier_positif(n_classes, "n_classes", minimum=3)
    positives = valeurs[valeurs > 0]
    if positives.size < 2 or positives.min() == positives.max():
        raise ValueError("Il faut au moins deux valeurs > 0 distinctes.")
    bornes = np.logspace(np.log10(positives.min()), np.log10(positives.max()),
                         n_classes + 1)
    # 10**log10(x) peut différer de x d'un ulp : sans ces deux affectations,
    # la plus petite ou la plus grande valeur tombe parfois hors de toutes les
    # classes (et le plus grand blackout disparaît de la queue).
    bornes[0], bornes[-1] = positives.min(), positives.max()
    effectifs, _ = np.histogram(positives, bornes)
    centres = np.sqrt(bornes[1:] * bornes[:-1])
    densite = effectifs / np.diff(bornes) / positives.size
    garde = effectifs > 0
    return centres[garde], densite[garde], effectifs[garde]


def plage_loi_puissance(centres, densite, effectifs, r2_min: float = 0.98,
                        points_min: int = 4, effectif_min: int = 5,
                        pente_max: float | None = None) -> dict:
    """
    Plus longue plage contiguë où log(densité) est linéaire en log(taille).

    On ne garde que les classes d'effectif ≥ `effectif_min` (les classes
    presque vides ont une densité très bruitée). Parmi toutes les plages
    contiguës d'au moins `points_min` classes dont l'ajustement linéaire
    log-log a un R² ≥ `r2_min`, on retient celle qui couvre le plus grand
    rapport taille_max / taille_min (l'« étendue »), puis, à égalité, celle
    de meilleur R². Avec `pente_max` (par exemple 0), seules les plages de
    pente < `pente_max` sont admises : pour une QUEUE de distribution, une
    plage croissante n'a pas de sens.

    Retourne un dict : pente (indice de décroissance de la densité),
    x_debut, x_fin, etendue (= x_fin / x_debut), r2, n_classes. Tout vaut NaN
    (n_classes = 0) si aucune plage ne satisfait les critères.
    """
    centres = serie_finie(centres)
    densite = serie_finie(densite)
    effectifs = serie_finie(effectifs)
    if not (centres.size == densite.size == effectifs.size):
        raise ValueError("centres, densite et effectifs doivent avoir la même taille.")
    if isinstance(r2_min, (bool, np.bool_)) or not isinstance(r2_min, Real) \
            or not 0 < r2_min <= 1:
        raise ValueError("r2_min doit être dans ]0, 1].")
    points_min = entier_positif(points_min, "points_min", minimum=3)
    effectif_min = entier_positif(effectif_min, "effectif_min", minimum=1)
    if pente_max is not None:
        if isinstance(pente_max, (bool, np.bool_)) or not isinstance(pente_max, Real) \
                or not np.isfinite(pente_max):
            raise ValueError("pente_max doit être un réel fini ou None.")

    garde = (effectifs >= effectif_min) & (densite > 0) & (centres > 0)
    x, y = np.log(centres[garde]), np.log(densite[garde])
    meilleur = {"pente": float("nan"), "x_debut": float("nan"), "x_fin": float("nan"),
                "etendue": float("nan"), "r2": float("nan"), "n_classes": 0}
    cle_meilleure = (-np.inf, -np.inf)
    for i in range(x.size):
        for j in range(i + points_min, x.size + 1):
            xs, ys = x[i:j], y[i:j]
            pente, ordonnee = np.polyfit(xs, ys, 1)
            residus = ys - (pente * xs + ordonnee)
            total = np.sum((ys - ys.mean()) ** 2)
            r2 = 1.0 - np.sum(residus ** 2) / total if total > 0 else 0.0
            if r2 < r2_min or (pente_max is not None and pente >= pente_max):
                continue
            cle = (xs[-1] - xs[0], r2)
            if cle > cle_meilleure:
                cle_meilleure = cle
                meilleur = {"pente": float(pente), "x_debut": float(np.exp(xs[0])),
                            "x_fin": float(np.exp(xs[-1])),
                            "etendue": float(np.exp(xs[-1] - xs[0])),
                            "r2": float(r2), "n_classes": int(j - i)}
    return meilleur
