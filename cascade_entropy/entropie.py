"""
Mesures informationnelles.

Ce module constitue le fil B du projet : il reçoit des séries numériques et
calcule des mesures d'information sur leur organisation. Il reste volontairement
indépendant du fil A : ces fonctions ne savent pas si les données proviennent
d'un réseau, d'un blackout ou d'une simulation synthétique.

Cette séparation est importante. Elle permet d'utiliser exactement les mêmes
mesures sur des données de natures différentes et de comparer leurs résultats
sans mélanger la production des données avec leur analyse.

Le module contient trois familles de mesures, qui répondent à trois questions
différentes :

1. L'entropie de permutation caractérise la complexité à partir de l'ordre
    relatif des valeurs successives, en ignorant complètement leurs amplitudes.

2. L'entropie d'échantillon mesure la régularité : deux segments qui se
    ressemblent sur m points continuent-ils à se ressembler au point suivant ?

3. L'entropie multiéchelle applique la précédente à des versions granularisées de
    la série. Elle permet ainsi de distinguer une irrégularité très locale d'une
    complexité qui reste visible à plusieurs échelles d'observation.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import factorial, log

import numpy as np

from ._validation import entier_positif, serie_finie

TAILLE_BLOC = 512

# Méthodes de comptage des paires de SampEn. « blocs » est le calcul historique
# (comparaison directe, par blocs de TAILLE_BLOC lignes, coût n²) ; « arbre »
# compte les mêmes paires avec un arbre k-d (scipy.spatial.cKDTree, distance de
# Tchebychev), sans jamais construire la matrice des distances. Les deux
# donnent exactement les mêmes comptes (tests/test_entropie_arbre.py) ; seul
# l'arbre reste praticable sur les 100 000 jours d'un cas de SO2.
METHODES_SAMPEN = ("blocs", "arbre")


def shannon(probabilites: np.ndarray) -> float:
    """
    Entropie de Shannon d'une distribution de probabilité, en nats.

    Cette fonction est la base mathématique commune à plusieurs mesures du
    module. Pour une distribution p, elle applique la formule :

        H(p) = - somme(p_i * log(p_i))

    Elle mesure le degré d'incertitude : plus la distribution est uniforme, plus
    l'entropie est élevée ; plus elle est concentrée sur quelques états, plus elle
    est faible. Le résultat est exprimé en nats, car le logarithme utilisé est le
    logarithme naturel.

    Une distribution parfaitement uniforme sur K états a une entropie maximale
    égale à log(K). Une distribution concentrée sur un seul état a une entropie
    nulle.
    """
    # On valide d'abord le tableau afin de refuser les NaN, les infinis et les
    # valeurs qui ne peuvent pas représenter des probabilités.
    p = serie_finie(probabilites)
    if np.any(p < 0):
        raise ValueError("Une probabilité ne peut pas être négative.")
    if not np.isclose(p.sum(), 1.0, atol=1e-9):
        raise ValueError(f"Les probabilités ne somment pas à 1 (somme = {p.sum():.3e}).")

    # Les termes p_i = 0 ne contribuent pas à l'entropie, car on définit
    # mathématiquement la limite 0 * log(0) comme étant égale à 0.
    non_nuls = p[p > 0]

    # Le signe moins transforme la somme des p_i log(p_i), négative, en une
    # quantité positive ou nulle.
    return float(-np.sum(non_nuls * np.log(non_nuls)))


def entropie_repartition(valeurs: np.ndarray, normaliser: bool = False) -> float:
    """
    Entropie de la répartition d'une quantité positive entre plusieurs éléments.

    Ici, on transforme les valeurs en parts relatives du total :

        p_i = |valeur_i| / somme(|valeur_j|)

    On obtient donc une distribution de probabilité qui décrit la manière dont
    la quantité totale est répartie entre les éléments. Si la puissance est
    répartie uniformément, l'entropie est maximale ; si elle est concentrée sur
    quelques lignes, l'entropie est faible.

    C'est une mesure d'espace, pas de temps. Elle décrit à un instant donné la
    forme de la répartition sur le réseau, et non son évolution dans le temps.
    """
    # La valeur absolue permet de mesurer une répartition d'amplitudes, même si
    # les données originales sont signées, comme des flux orientés.
    valeurs = np.abs(serie_finie(valeurs))
    total = valeurs.sum()
    if total == 0:
        # Une répartition entièrement nulle ne possède aucune part relative :
        # l'entropie n'est donc pas définie.
        return np.nan

    # Chaque élément devient une probabilité représentant sa part du total.
    h = shannon(valeurs / total)

    # La normalisation par log(K) ramène l'entropie dans [0, 1], où K est le
    # nombre d'éléments. Sans normalisation, on conserve l'entropie en nats.
    return h / np.log(valeurs.size) if normaliser else h


def nombre_effectif(valeurs: np.ndarray) -> float:
    """
    Nombre effectif d'éléments portant une répartition, exp(H).

    Ce nombre est la reformulation exponentielle de l'entropie. Si dix lignes
    portent exactement la même puissance, alors exp(H) vaut 10. Si une seule
    ligne porte tout, le nombre effectif vaut 1. C'est donc une mesure très
    intuitive : elle estime combien d'éléments participent réellement à la
    répartition, même lorsque les valeurs ne sont pas parfaitement égales.
    """
    # On commence par calculer l'entropie spatiale de la répartition.
    h = entropie_repartition(valeurs)

    # L'exponentielle inverse le logarithme présent dans la définition de H et
    # convertit une quantité abstraite en un nombre effectif d'éléments.
    return float(np.exp(h)) if np.isfinite(h) else np.nan


def permutation(
    serie: np.ndarray,
    dimension: int = 3,
    delai: int = 1,
    normaliser: bool = True,
) -> float:
    """
    Entropie de permutation d'une série.

    L'idée est de ne regarder que l'ordre relatif des valeurs dans des fenêtres
    glissantes. Par exemple, si une fenêtre vaut [3, 1, 2], son ordre croissant
    est donné par [1, 2, 0] : la plus petite valeur se trouve à la position 1,
    puis la valeur intermédiaire à la position 2, et enfin la plus grande à la
    position 0.

    La valeur exacte des observations est donc volontairement oubliée. Les
    fenêtres [30, 10, 20] et [3, 1, 2] produisent le même motif ordinal.

    Ensuite, on compte les motifs ordonnaux observés et on applique l'entropie de
    Shannon. Une série très régulière produira peu de motifs, tandis qu'un bruit
    non corrélé produira presque tous les ordres possibles avec quasi-uniformité.
    """
    # La validation garantit que les fenêtres ne contiennent ni NaN ni infini.
    serie = serie_finie(serie)
    dimension = entier_positif(dimension, "dimension", minimum=2)
    delai = entier_positif(delai, "delai")

    # Le nombre de fenêtres dépend de la dimension et du délai : la dernière
    # fenêtre doit encore tenir entièrement dans la série.
    n_fenetres = serie.size - (dimension - 1) * delai
    if n_fenetres < 2:
        raise ValueError("Série trop courte pour cette dimension et ce délai.")

    # Ces indices décrivent les positions observées dans une fenêtre. Avec un
    # délai de 2 et une dimension de 3, on compare par exemple t, t+2 et t+4.
    indices = np.arange(dimension) * delai

    # L'indexation avancée construit toutes les fenêtres simultanément, sans
    # boucle Python explicite sur chaque position de départ.
    fenetres = serie[np.arange(n_fenetres)[:, None] + indices]

    # Pour chaque fenêtre, argsort renvoie les indices qui classent les valeurs
    # par ordre croissant : c'est précisément le motif ordinal recherché.
    ordres = np.argsort(fenetres, axis=1, kind="stable")

    # On regroupe les motifs identiques et on compte leur fréquence d'apparition.
    _, effectifs = np.unique(ordres, axis=0, return_counts=True)

    # Les effectifs sont convertis en probabilités avant d'appliquer Shannon.
    p = effectifs / effectifs.sum()
    h = -np.sum(p * np.log(p))

    # Il existe dimension! ordres possibles. La normalisation par log(dimension!)
    # exprime l'entropie comme une fraction de son maximum théorique.
    return h / log(factorial(dimension)) if normaliser else h


def _compter_paires(modeles: np.ndarray, r: float) -> tuple[int, int]:
    """
    Compte les paires de motifs proches selon la distance de Tchebychev.

    Chaque ligne de `modeles` contient un motif de longueur m+1. On compare la
    partie de longueur m de chaque motif à celle de tous les autres motifs.
    Ensuite, on vérifie séparément le point supplémentaire pour obtenir le compte
    des correspondances de longueur m+1.

    La distance de Tchebychev est utilisée : deux motifs sont proches si la plus
    grande différence entre leurs coordonnées ne dépasse pas r. Cette règle est
    plus stricte qu'une moyenne des différences, car une seule coordonnée très
    éloignée suffit à invalider la comparaison.

    Le calcul est fait par blocs pour rester efficace sans construire une matrice
    de distances complète sur tout l'échantillon.
    """
    # n est le nombre de motifs et largeur vaut m+1.
    n, largeur = modeles.shape
    m = largeur - 1

    # `court` contient les m premiers points de chaque motif ; `supplement`
    # contient le point qui permettra de tester le prolongement du motif.
    court, supplement = modeles[:, :m], modeles[:, m]

    total_m = 0
    total_m1 = 0
    for debut in range(0, n, TAILLE_BLOC):
        # Le traitement par blocs limite la mémoire nécessaire. On évite ainsi
        # de créer d'un seul coup une très grande matrice n x n.
        bloc = court[debut : debut + TAILLE_BLOC]

        # Broadcasting NumPy : chaque motif du bloc est comparé à chaque motif
        # global, coordonnée par coordonnée. Le maximum donne la distance de
        # Tchebychev entre deux motifs de longueur m.
        distances = np.abs(bloc[:, None, :] - court[None, :, :]).max(axis=2)
        proches_m = distances <= r

        supp_bloc = supplement[debut : debut + TAILLE_BLOC]

        # Pour passer de m à m+1 points, le point supplémentaire doit lui aussi
        # être suffisamment proche.
        ecart_supp = np.abs(supp_bloc[:, None] - supplement[None, :])
        proches_m1 = proches_m & (ecart_supp <= r)

        # Chaque masque contient des booléens ; leur somme compte les paires
        # considérées comme proches.
        total_m += int(proches_m.sum())
        total_m1 += int(proches_m1.sum())

    # Chaque motif est forcément proche de lui-même. La Sample Entropy exclut
    # ces auto-comparaisons, d'où la soustraction de n dans les deux comptes.
    return total_m - n, total_m1 - n


def _compter_paires_arbre(modeles: np.ndarray, r: float) -> tuple[int, int]:
    """
    Mêmes comptes que `_compter_paires`, obtenus avec deux arbres k-d.

    `cKDTree.count_neighbors(arbre, r, p=inf)` compte toutes les paires
    ordonnées (i, j), i = j compris, dont la distance de Tchebychev est ≤ r :
    c'est exactement la somme de la matrice booléenne de `_compter_paires`.
    Un premier arbre est construit sur les m premières coordonnées (comptes de
    longueur m), un second sur les m + 1 coordonnées (comptes de longueur
    m + 1, puisque la distance en m + 1 est le maximum de la distance en m et
    de l'écart sur le point supplémentaire). On retire ensuite les n
    auto-comparaisons, comme dans le calcul par blocs.

    L'arbre regroupe les points voisins et compte d'un coup les paires de
    boîtes entièrement à moins de r l'une de l'autre : la matrice n × n n'est
    jamais formée. Sur 100 000 points, le calcul prend quelques secondes au
    lieu de plusieurs dizaines de minutes.
    """
    # Import local : SciPy n'est requis que si l'on choisit cette méthode.
    from scipy.spatial import cKDTree

    n, largeur = modeles.shape
    m = largeur - 1
    court = cKDTree(modeles[:, :m])
    complet = cKDTree(modeles)
    total_m = int(court.count_neighbors(court, r, p=np.inf))
    total_m1 = int(complet.count_neighbors(complet, r, p=np.inf))
    return total_m - n, total_m1 - n


def _methode_valide(methode) -> str:
    """Refuse toute méthode de comptage autre que celles de METHODES_SAMPEN."""
    if not isinstance(methode, str) or methode not in METHODES_SAMPEN:
        raise ValueError(f"methode doit valoir {' ou '.join(METHODES_SAMPEN)}.")
    return methode


def echantillon(
    serie: np.ndarray,
    m: int = 2,
    r: float = 0.15,
    ecart_reference: float | None = None,
    methode: str = "blocs",
) -> float:
    """
    Entropie d'échantillon (SampEn).

    La SampEn compare la probabilité qu'un motif de longueur m soit suivi d'un
    motif de longueur m+1 semblable. En pratique, on compte les paires de motifs
    proches, puis on regarde quelle proportion reste proche après l'ajout d'un
    point supplémentaire :

        SampEn = -log(compte_m1 / compte_m)

    Une petite valeur signifie que les motifs qui se ressemblaient restent souvent
    semblables : la série est régulière. Une grande valeur signifie que leur
    évolution diverge rapidement : la série est plus irrégulière.

    `r` est donné en fraction de l'écart-type. Cela donne une tolérance de
    similarité, ce qui rend la mesure robuste à de petites variations.

    `methode` choisit le comptage des paires : « blocs » (défaut, calcul
    historique, coût en n²) ou « arbre » (arbre k-d, mêmes comptes, praticable
    sur 10^5 points). Le résultat est identique ; seul le temps de calcul change.
    """
    # On vérifie les données et les paramètres avant de construire les motifs.
    serie = serie_finie(serie)
    methode = _methode_valide(methode)
    m = entier_positif(m, "m")
    if not np.isfinite(r) or r <= 0:
        raise ValueError("r doit être fini et strictement positif.")
    if ecart_reference is not None and (
        not np.isfinite(ecart_reference) or ecart_reference < 0
    ):
        raise ValueError("ecart_reference doit être fini et positif ou nul.")
    n = serie.size
    if n < m + 2:
        raise ValueError("Série trop courte pour cette dimension d'immersion.")

    # Par défaut, la tolérance est proportionnelle à la dispersion de la série.
    # L'appelant peut fournir une référence commune pour comparer plusieurs
    # séries ou plusieurs échelles avec le même seuil absolu.
    ecart = serie.std(ddof=0) if ecart_reference is None else ecart_reference
    if ecart == 0:
        # Une série constante ne permet pas de distinguer des motifs différents.
        return np.nan
    tolerance = r * ecart

    # Une série de longueur n contient n-m motifs de longueur m+1 : chaque motif
    # commence à une position différente et doit tenir jusqu'à son dernier point.
    n_modeles = n - m
    indices = np.arange(m + 1)
    modeles = serie[np.arange(n_modeles)[:, None] + indices]

    # On compte les correspondances de longueur m puis de longueur m+1.
    compter = _compter_paires if methode == "blocs" else _compter_paires_arbre
    compte_m, compte_m1 = compter(modeles, tolerance)
    if compte_m <= 0 or compte_m1 <= 0:
        # Sans paire comparable, le logarithme du rapport ne peut pas être
        # calculé de manière interprétable.
        return np.nan
    return -log(compte_m1 / compte_m)


def granulariser(serie: np.ndarray, echelle: int) -> np.ndarray:
    """
    Moyennes sur des fenêtres disjointes de longueur `echelle`.

    Cette étape est au cœur de la multiéchelle : on remplace des blocs de points
    par leur moyenne, ce qui fait apparaître le comportement à une échelle plus
    grossière. À l'échelle 1, la série est inchangée. À l'échelle 2, les points
    sont regroupés deux par deux, puis moyennés ; à l'échelle 10, ce sont des
    blocs de dix points qui sont remplacés par une seule valeur.

    Les points qui ne remplissent pas un bloc complet à la fin de la série sont
    ignorés. Cela garantit que toutes les valeurs produites sont des moyennes de
    blocs de même taille.
    """
    # Validation de la série et de l'échelle demandée.
    serie = serie_finie(serie)
    echelle = entier_positif(echelle, "echelle")

    # Seuls les blocs complets sont retenus. La division entière indique combien
    # de blocs de longueur `echelle` peuvent être formés.
    n = serie.size // echelle
    if n == 0:
        raise ValueError("Échelle plus grande que la série.")

    # On retire l'éventuel reste, on donne au tableau une forme (nombre de blocs,
    # taille d'un bloc), puis on calcule une moyenne par ligne.
    return serie[: n * echelle].reshape(n, echelle).mean(axis=1)


def multiechelle(
    serie: np.ndarray,
    echelles: int | Sequence[int] = 20,
    m: int = 2,
    r: float = 0.15,
    methode: str = "blocs",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Entropie multiéchelle : SampEn calculée sur plusieurs échelles.

    L'idée est de voir comment la régularité d'une série évolue quand on la lisse.
    À chaque échelle, la série est d'abord granularisée par moyennes de blocs,
    puis sa SampEn est calculée. On obtient donc une courbe et non une seule
    valeur : l'axe horizontal indique la résolution d'observation et l'axe
    vertical la régularité mesurée à cette résolution.

    Un bruit blanc très irrégulier à petite échelle peut devenir beaucoup plus
    régulier à grande échelle, tandis qu'un signal à longue mémoire garde une
    structure plus stable. C'est précisément cette différence qui motive la
    multiéchelle.
    """
    # Les validations sont faites ici afin de détecter rapidement les paramètres
    # invalides avant de lancer plusieurs calculs de SampEn.
    serie = serie_finie(serie)
    methode = _methode_valide(methode)
    liste = None
    if isinstance(echelles, Sequence) and not isinstance(echelles, str):
        liste = [entier_positif(e, "echelle") for e in echelles]
        if not liste or any(b <= a for a, b in zip(liste, liste[1:])):
            raise ValueError("La liste d'échelles doit être non vide et strictement croissante.")
    else:
        echelles = entier_positif(echelles, "echelles")
    m = entier_positif(m, "m")
    if not np.isfinite(r) or r <= 0:
        raise ValueError("r doit être fini et strictement positif.")

    if liste is not None:
        # Liste explicite : une valeur par échelle demandée, NaN si trop courte.
        ecart = serie.std(ddof=0)
        valeurs = []
        for echelle in liste:
            if serie.size // echelle < 10 * (m + 1):
                valeurs.append(np.nan)
                continue
            granulee = granulariser(serie, echelle)
            valeurs.append(echantillon(granulee, m=m, r=r, ecart_reference=ecart,
                                       methode=methode))
        return np.array(liste, dtype=float), np.array(valeurs, dtype=float)

    # Cette dispersion sert de référence commune à toutes les séries granularisées.
    # Ainsi, la tolérance r * ecart ne change pas artificiellement uniquement
    # parce que la série a été moyennée.
    ecart = serie.std(ddof=0)

    liste_echelles, valeurs = [], []
    for echelle in range(1, echelles + 1):
        # Chaque passage remplace la série par une version observée à une
        # résolution plus grossière.
        granulee = granulariser(serie, echelle)

        # SampEn a besoin d'un nombre suffisant de motifs. Dès que la série
        # granularisée devient trop courte, les échelles suivantes seraient elles
        # aussi trop courtes : on peut arrêter la boucle.
        if granulee.size < 10 * (m + 1):
            break
        liste_echelles.append(echelle)
        valeurs.append(echantillon(granulee, m=m, r=r, ecart_reference=ecart,
                                   methode=methode))

    # Les listes sont converties en tableaux NumPy pour faciliter les tracés et
    # les calculs ultérieurs sur la courbe multiéchelle.
    return np.array(liste_echelles, dtype=float), np.array(valeurs, dtype=float)
