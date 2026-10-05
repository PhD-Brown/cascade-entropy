"""
Dispatch de la production sous contraintes.

Deuxième module du fil A. Étant donné une demande, il décide quel générateur
produit combien et quelle charge est servie, en respectant les capacités de
production et les limites de transport.

Le problème est linéaire. Les variables de décision sont la production de chaque
générateur et la puissance effectivement servie à chaque charge. Le coût vaut une
unité par unité produite et moins W unités par unité servie, avec W grand : servir
la charge rapporte donc cent fois plus que produire ne coûte, ce qui revient à
dire « sers le maximum, et si c'est impossible, coupe le moins possible ».

    minimiser   sum_i G_i  -  W sum_j L_j

    sous        0 <= G_i <= P_i^max          capacités de production
                0 <= L_j <= d_j              on ne sert jamais plus que demandé
                sum_i G_i = sum_j L_j        équilibre global
                |F_l| <= F_l^max             limites de transport

Les flux se déduisent des injections par la matrice de flux du module `reseau`,
ce qui rend les contraintes de transport linéaires en les variables de décision.

Point essentiel pour la suite : le dispatch ne produit jamais de ligne au-dessus
de sa limite. Une ligne peut au mieux être saturée, c'est-à-dire collée contre sa
contrainte. C'est cette saturation, et non un dépassement, qui déclenchera
l'avarie dans le module `cascade`.

Départage des optima dégénérés
------------------------------
Toutes les charges ont la même priorité dans le coût : dès qu'un délestage est
nécessaire, des solutions différentes ont exactement le même coût optimal
(couper la charge A ou la charge B revient au même). Le paramètre `departage`
de `resoudre()` choisit explicitement parmi ces optima :

- ``"highs"`` (défaut) : la solution que renvoie le solveur HiGHS, sans règle
  de sélection. Comportement historique du projet, inchangé.
- ``"exterieur_dabord"`` : programme linéaire en deux étapes.
  1. Le problème ci-dessus donne la puissance servie maximale S*.
  2. À puissance servie fixée (S >= S* − tolérance), on maximise
     sum_j w_j L_j, où w_j décroît avec la profondeur topologique du nœud
     (`Reseau.niveaux`) : les charges les plus extérieures sont délestées
     en premier.
  La puissance servie, donc le coût physique, est identique à celle de
  ``"highs"`` ; seule change la localisation du délestage. C'est la règle qui
  reproduit le récit de Carreras et al. (2002) : « The nodes in the outermost
  ring of the network are progressively blacked out », et les bandes ordonnées
  de leur Fig. 10. C'est une hypothèse de réplication, pas une loi physique.

Au sein d'un même niveau, les charges sont physiquement équivalentes ; un petit
terme de rang (voir `priorites_exterieur_dabord`) coupe d'abord les indices les
plus élevés. L'optimum devient unique et ne dépend plus du solveur.

Robustesse numérique
--------------------
Le problème est toujours admissible (tout à zéro respecte chaque contrainte) :
un échec du solveur est donc numérique, jamais physique. Sur de longues
dynamiques lentes, HiGHS peut en rencontrer (statut « Unknown ») quand les
échelles divergent : demande multipliée par des centaines au fil des siècles
simulés, limites des lignes mortes réduites d'un facteur 1e6. `resoudre`
essaie alors, dans l'ordre de `STRATEGIES_SOLVEUR`, des formulations
équivalentes du même programme linéaire : même problème mis à l'échelle
(toutes les puissances divisées par la demande totale), puis simplexe dual et
point intérieur sans prérésolution. Une stratégie de secours n'est acceptée
que si sa solution respecte toutes les contraintes du problème d'origine
(`_verifier_admissible`). Le premier essai est l'appel historique : quand il
réussit, rien ne change, bit pour bit. `Solution.strategie` indique la
stratégie retenue ; si aucune ne réussit, `EchecDispatch` transporte le
problème complet pour qu'on puisse le sauvegarder et l'auditer.

Critères de validation : sur un réseau réduit résolu à la main, les flux doivent
coïncider ; à faible charge, la solution ne doit comporter ni délestage ni ligne
saturée ; au-delà de la capacité totale de production, le délestage doit égaler
exactement l'excédent de demande.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .reseau import Reseau, matrice_de_flux

POIDS_DELESTAGE = 100.0
SEUIL_SATURATION = 0.99

DEPARTAGE_HIGHS = "highs"
DEPARTAGE_EXTERIEUR_DABORD = "exterieur_dabord"
DEPARTAGES = (DEPARTAGE_HIGHS, DEPARTAGE_EXTERIEUR_DABORD)

# Tolérance relative sur la puissance servie entre les deux étapes du départage.
# Elle doit rester au-dessus de la tolérance de faisabilité de HiGHS (~1e-7 en
# absolu) tout en étant négligeable devant toute charge physique.
TOLERANCE_SERVIE = 1e-9

# Stratégies essayées dans l'ordre : (nom, méthode linprog, mise à l'échelle,
# options HiGHS). La première est l'appel historique, inchangé.
STRATEGIES_SOLVEUR = (
    ("highs", "highs", False, {}),
    ("highs_echelle", "highs", True, {}),
    ("simplexe_dual", "highs-ds", True, {"presolve": False}),
    ("point_interieur", "highs-ipm", True, {"presolve": False}),
)
STRATEGIE_HISTORIQUE = STRATEGIES_SOLVEUR[0][0]

# Tolérance relative (à l'échelle du problème) de la vérification
# d'admissibilité d'une solution de secours.
TOLERANCE_ADMISSIBILITE = 1e-7


class EchecDispatch(RuntimeError):
    """
    Aucune stratégie de `STRATEGIES_SOLVEUR` n'a résolu le programme linéaire.

    `probleme` contient les tableaux complets du programme (c, A_ub, b_ub,
    A_eq, b_eq, bornes inférieures et supérieures, étape) : l'appelant peut
    les sauvegarder avec `numpy.savez` pour rejouer et auditer l'échec.
    `messages` liste le message de chaque tentative.
    """

    def __init__(self, message: str, probleme: dict, messages: list[str]):
        super().__init__(message)
        self.probleme = probleme
        self.messages = messages


@dataclass
class Solution:
    """
    Représente la solution optimiser d'un dispatch.

    On garde à la fois les grandeurs physiques du réseau (flux, injections,
    charge servie) et les grandeurs utiles pour la cascade (délestage, taux de
    charge). C'est la structure de sortie qui sera réutilisée ensuite dans les
    modules de contrôle et d'analyse.
    """

    injections: np.ndarray
    production: np.ndarray
    charge_servie: np.ndarray
    delestage: np.ndarray
    flux: np.ndarray
    taux_de_charge: np.ndarray
    cout: float
    departage: str = DEPARTAGE_HIGHS
    # Stratégie de `STRATEGIES_SOLVEUR` qui a produit la solution (la plus
    # « tardive » des deux étapes pour "exterieur_dabord").
    strategie: str = STRATEGIE_HISTORIQUE

    @property
    def secours(self) -> bool:
        """Vrai si une stratégie de secours a été nécessaire."""
        return self.strategie != STRATEGIE_HISTORIQUE

    @property
    def delestage_total(self) -> float:
        """Somme du délestage sur toutes les charges."""
        return float(self.delestage.sum())

    @property
    def taux_maximal(self) -> float:
        """M_max, indicateur témoin principal du projet."""
        return float(self.taux_de_charge.max())

    @property
    def lignes_saturees(self) -> np.ndarray:
        """
        Indices des lignes à moins de 1 % de leur limite.

        Ce sont elles, et elles seules, qui peuvent tomber lors d'une cascade.
        """
        return np.flatnonzero(self.taux_de_charge >= SEUIL_SATURATION)


def _matrices(reseau: Reseau, A: np.ndarray | None):
    """
    Construit la correspondance entre variables de décision et flux.

    Les variables de décision sont rangées dans l'ordre :
    1. production de chaque générateur,
    2. puissance servie à chaque charge.

    La matrice `selection` transforme ce vecteur de décision en injections du
    réseau : production positive sur les nœuds générateurs, consommation négative
    sur les nœuds de charge. Ensuite, on applique la matrice de flux pour obtenir
    les flux sur les lignes.
    """
    if A is None:
        A = matrice_de_flux(reseau)

    n_g, n_c = reseau.generateurs.size, reseau.charges.size
    selection = np.zeros((reseau.n_noeuds, n_g + n_c))
    selection[reseau.generateurs, np.arange(n_g)] = 1.0
    selection[reseau.charges, n_g + np.arange(n_c)] = -1.0

    autres = np.array([i for i in range(reseau.n_noeuds) if i != reseau.reference])
    # On applique A aux injections hors référence pour obtenir les flux de ligne.
    return A, selection, A @ selection[autres], n_g, n_c


def _verifier_admissible(x, A_ub, b_ub, A_eq, b_eq, inf, sup, echelle) -> bool:
    """Vérifie qu'un vecteur respecte toutes les contraintes, à tolérance relative."""
    tol = TOLERANCE_ADMISSIBILITE * echelle
    if x is None or not np.all(np.isfinite(x)):
        return False
    return bool(np.all(x >= inf - tol) and np.all(x <= sup + tol)
                and np.all(A_ub @ x <= b_ub + tol)
                and np.all(np.abs(A_eq @ x - b_eq) <= tol))


def _linprog_robuste(c, A_ub, b_ub, A_eq, b_eq, inf, sup, etape: str):
    """
    Résout min c·x sous A_ub x <= b_ub, A_eq x = b_eq, inf <= x <= sup.

    Essaie les stratégies de `STRATEGIES_SOLVEUR` dans l'ordre et renvoie
    (x, nom de la stratégie). Mise à l'échelle : toutes les contraintes portent
    sur des puissances ; on divise seconds membres et bornes par
    s = max(1, max |sup|), on résout en y = x / s, puis x = s·y. L'objectif
    est linéaire : l'optimum est le même, seule l'arithmétique change.
    Lève `EchecDispatch` si aucune stratégie ne donne une solution admissible.
    """
    finies = np.abs(sup[np.isfinite(sup)])
    echelle = max(1.0, float(finies.max()) if finies.size else 1.0)
    messages = []
    for nom, methode, a_echelle, options in STRATEGIES_SOLVEUR:
        s = echelle if a_echelle else 1.0
        resultat = linprog(c, A_ub=A_ub, b_ub=b_ub / s, A_eq=A_eq, b_eq=b_eq / s,
                           bounds=np.column_stack([inf / s, sup / s]),
                           method=methode, options=options or None)
        if resultat.success:
            x = resultat.x * s
            # Le premier essai est l'appel historique : on ne le soumet à
            # aucune vérification supplémentaire, pour ne rien changer.
            if nom == STRATEGIE_HISTORIQUE or _verifier_admissible(
                    x, A_ub, b_ub, A_eq, b_eq, inf, sup, echelle):
                return x, nom
            messages.append(f"{nom} : solution non admissible au contrôle")
        else:
            messages.append(f"{nom} : {resultat.message}")
    probleme = {"c": c, "A_ub": A_ub, "b_ub": b_ub, "A_eq": A_eq, "b_eq": b_eq,
                "borne_inf": inf, "borne_sup": sup, "etape": np.array(etape)}
    raise EchecDispatch(
        f"Le dispatch a échoué ({etape}) avec toutes les stratégies : "
        + " | ".join(messages), probleme, messages)


def valider_departage(departage) -> str:
    """
    Vérifie le nom de la règle de départage et le renvoie.

    Refuse tout ce qui n'est pas exactement l'une des chaînes de `DEPARTAGES`,
    pour qu'une faute de frappe ne retombe jamais silencieusement sur HiGHS.
    """
    if not isinstance(departage, str) or departage not in DEPARTAGES:
        raise ValueError(
            f"departage doit être l'une des valeurs {DEPARTAGES}, "
            f"pas {departage!r}."
        )
    return departage


def priorites_exterieur_dabord(reseau: Reseau) -> np.ndarray:
    """
    Poids de service w_j de chaque charge pour la règle ``"exterieur_dabord"``.

        w_j = 1 + (niveau_max − niveau_j) + 0.5 × (1 − rang_j / n_niveau)

    - Terme entier : chaque niveau plus proche de la racine vaut une unité de
      plus. Maximiser sum_j w_j L_j à puissance servie fixée sert donc les
      charges intérieures en premier et coupe les extérieures en premier.
    - Terme de rang, < 0.5 : au sein d'un niveau, la charge de plus petit
      indice de nœud est servie en premier, donc les indices les plus élevés
      sont coupés en premier. Il ne peut jamais inverser l'ordre des niveaux.
      Il rend l'optimum unique (plus aucune dépendance au solveur) et donne
      des fronts de délestage contigus, orientés comme dans les Fig. 9-10 de
      Carreras (front partant des numéros de ligne les plus élevés). Ce
      terme est un choix d'étiquetage : dans un arbre symétrique, toutes les
      charges d'un même niveau sont physiquement équivalentes.

    Exige `reseau.niveaux` (fourni par `reseau.arbre()`). Un réseau sans niveaux
    n'a pas de notion d'« extérieur » définie : on refuse plutôt que d'en
    inventer une.
    """
    if reseau.niveaux is None:
        raise ValueError(
            "Le départage 'exterieur_dabord' exige reseau.niveaux "
            "(réseau construit par reseau.arbre())."
        )
    noeuds = np.asarray(reseau.charges)
    niveaux = np.asarray(reseau.niveaux, dtype=float)[noeuds]
    poids = 1.0 + (niveaux.max() - niveaux)
    for niveau in np.unique(niveaux):
        dans = np.flatnonzero(niveaux == niveau)
        rang = np.argsort(np.argsort(noeuds[dans], kind="stable"), kind="stable")
        poids[dans] += 0.5 * (1.0 - rang / dans.size)
    return poids


def _solution(reseau: Reseau, x: np.ndarray, flux_par_variable: np.ndarray,
              demande: np.ndarray, limites: np.ndarray, n_g: int,
              poids_delestage: float, departage: str,
              strategie: str = STRATEGIE_HISTORIQUE) -> Solution:
    """Construit la `Solution` à partir du vecteur de décision optimal."""
    production = x[:n_g]
    charge_servie = x[n_g:]

    # On reconstruit les injections au format attendu par le module réseau.
    injections = np.zeros(reseau.n_noeuds)
    injections[reseau.generateurs] = production
    injections[reseau.charges] = -charge_servie

    # Le flux est simplement la matrice de transfert appliquée au vecteur de décision.
    flux = flux_par_variable @ x

    return Solution(
        injections=injections,
        production=production,
        charge_servie=charge_servie,
        delestage=demande - charge_servie,
        flux=flux,
        taux_de_charge=np.abs(flux) / limites,
        # Coût physique (étape 1), recalculé sur la solution retenue : il est
        # identique pour les deux règles de départage, à la tolérance près.
        cout=float(production.sum() - poids_delestage * charge_servie.sum()),
        departage=departage,
        strategie=strategie,
    )


def resoudre(
    reseau: Reseau,
    demande: np.ndarray,
    limites: np.ndarray | None = None,
    puissance_max: np.ndarray | None = None,
    poids_delestage: float = POIDS_DELESTAGE,
    A: np.ndarray | None = None,
    departage: str = DEPARTAGE_HIGHS,
) -> Solution:
    """
    Résout le dispatch pour une demande donnée.

    La logique générale est la suivante :
    - on choisit la production à chaque générateur,
    - on choisit la puissance effectivement servie à chaque charge,
    - on impose que la production totale soit égale à la charge totale servie,
    - on impose que chaque flux respecte sa limite de transport,
    - on minimise le coût économique de l'opération, avec un lourd pénalisation
      du délestage pour servir le plus possible.

    L'algorithme est un programme linéaire, ce qui est parfaitement adapté ici
    puisque les contraintes sont affines et la fonction objectif est linéaire.

    `departage` choisit parmi les optima de même coût lorsqu'un délestage est
    nécessaire (voir la docstring du module) :
    ``"highs"`` (défaut, historique) ou ``"exterieur_dabord"``. Sans délestage,
    les deux règles renvoient exactement la même solution : la seconde étape
    n'est alors pas exécutée.
    """
    departage = valider_departage(departage)
    demande = np.asarray(demande, dtype=float)
    if demande.shape != reseau.charges.shape:
        raise ValueError("Une valeur de demande par nœud de charge est requise.")
    if np.any(demande < 0):
        raise ValueError("La demande doit être positive.")

    # Si aucune limite n'est fournie, on prend celle du réseau par défaut.
    limites = reseau.limites if limites is None else np.asarray(limites, dtype=float)
    if limites is None:
        raise ValueError("Aucune limite de ligne définie.")
    if puissance_max is None:
        raise ValueError("Les capacités de production doivent être fournies.")
    puissance_max = np.asarray(puissance_max, dtype=float)

    A, selection, flux_par_variable, n_g, n_c = _matrices(reseau, A)

    # Coût : produire coûte 1, servir la demande rapporte W, donc le solveur préfère
    # servir autant que possible et ne délester que le strict nécessaire.
    cout = np.concatenate([np.ones(n_g), -poids_delestage * np.ones(n_c)])

    # Les contraintes de transport se traduisent par des bornes sur les flux :
    # -F_max <= F <= F_max, soit deux inégalités de la forme A x <= b.
    A_ub = np.vstack([flux_par_variable, -flux_par_variable])
    b_ub = np.concatenate([limites, limites])

    # Équilibre global : la puissance produite doit compenser exactement la charge servie.
    A_eq = np.concatenate([np.ones(n_g), -np.ones(n_c)])[None, :]
    b_eq = np.zeros(1)

    # Chaque variable de décision est bornée par 0 et par sa limite physique :
    # production <= puissance_max, charge_servie <= demande.
    inf = np.zeros(n_g + n_c)
    sup = np.concatenate([puissance_max, demande])

    x, strategie = _linprog_robuste(cout, A_ub, b_ub, A_eq, b_eq, inf, sup,
                                    "étape 1")

    if departage == DEPARTAGE_EXTERIEUR_DABORD:
        servie_max = float(x[n_g:].sum())
        tolerance = TOLERANCE_SERVIE * max(1.0, float(demande.sum()))
        # Étape 2 seulement s'il y a un délestage : sinon toutes les charges
        # sont servies en entier et il n'y a rien à départager.
        if float(demande.sum()) - servie_max > tolerance:
            poids = priorites_exterieur_dabord(reseau)
            cout_2 = np.concatenate([np.zeros(n_g), -poids])
            # Puissance servie maintenue à son maximum : -sum L <= -(S* - tol).
            ligne_servie = np.concatenate([np.zeros(n_g), -np.ones(n_c)])[None, :]
            A_ub_2 = np.vstack([A_ub, ligne_servie])
            b_ub_2 = np.concatenate([b_ub, [-(servie_max - tolerance)]])
            # La solution de l'étape 1 est admissible pour l'étape 2 : un échec
            # ici est numérique, jamais un choix ; les mêmes secours s'appliquent.
            x, strategie_2 = _linprog_robuste(cout_2, A_ub_2, b_ub_2, A_eq, b_eq,
                                              inf, sup, "étape 2")
            if strategie_2 != STRATEGIE_HISTORIQUE:
                strategie = strategie_2

    return _solution(reseau, x, flux_par_variable, demande, limites, n_g,
                     poids_delestage, departage, strategie)


def demande_uniforme(reseau: Reseau, total: float,
                     poids: np.ndarray | None = None) -> np.ndarray:
    """
    Répartit une demande totale entre les nœuds de charge.

    Cette aide est utilisée pour créer des scénarios de balayage de charge : on
    garde le même profil de demande, mais on ajuste le niveau global. On obtient
    ainsi une famille de demandes uniformes, faciles à comparer entre elles.
    """
    if poids is None:
        # Cas le plus simple : chaque charge reçoit la même quantité.
        poids = np.ones(reseau.charges.size)
    poids = np.asarray(poids, dtype=float)
    if np.any(poids < 0) or poids.sum() == 0:
        raise ValueError("Les poids doivent être positifs et de somme non nulle.")
    # La demande est d'abord distribuée selon les poids, puis normalisée pour
    # faire exactement le total voulu.
    return total * poids / poids.sum()
