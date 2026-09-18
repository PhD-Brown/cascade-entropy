# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Laboratoire 01 — Outils d'analyse et modèle de réseau
#
# **Projet PHY-3202 — Détection entropique des pannes en cascade**
# Alex Baker, superviseur : Patrick Desrosiers
#
# ---
#
# Ce carnet fait la démonstration de ce qui est en place et validé à ce stade du
# projet. Il ne contient aucune logique de calcul : tout vient du paquet
# `cascade_entropy`, couvert par la suite de tests. Le carnet raconte, le paquet
# calcule.
#
# **Ce qui est démontré ici**
#
# 1. Les séries de validation et leur comportement attendu
# 2. L'entropie de permutation, et ce qu'elle distingue
# 3. Le croisement multiéchelle — pourquoi le projet utilise MSE plutôt que
#    l'entropie brute
# 4. L'exposant de Hurst, et un biais qui change l'interprétation
# 5. Un piège identifié avant qu'il ne fausse les résultats
# 6. Le réseau en arbre et le calcul des flux
#
# **Ce qui n'est pas encore là** : le dispatch sous contraintes, la cascade et la
# dynamique lente. Ce sont les trois prochains modules.

# %%
import numpy as np

from cascade_entropy import figures
from cascade_entropy.entropie import echantillon, multiechelle, permutation
from cascade_entropy.indicateurs import hurst
from cascade_entropy.reseau import (
    arbre,
    degres,
    flux,
    limites_depuis_cas_de_base,
    matrice_de_flux,
    taux_de_charge,
)
from cascade_entropy.synthetiques import bruit_puissance, periodique

figures.appliquer_style()

# Toute valeur de ce carnet est reproductible à partir de cette graine.
GRAINE = 20260915
N = 8192

rng = np.random.default_rng(GRAINE)

print(f"numpy {np.__version__} — graine {GRAINE} — séries de {N} points")

# %% [markdown]
# ---
# ## 1. Les séries de validation
#
# Aucune mesure ne peut être appliquée à des données de réseau avant d'avoir été
# vérifiée sur des séries dont on connaît déjà la réponse attendue. Quatre
# séries suffisent à couvrir la gamme utile : un signal parfaitement régulier, un
# bruit sans aucune corrélation, un bruit corrélé à longue portée, et un cas
# fortement corrélé.

# %%
series = {
    "périodique": periodique(N, periode=50),
    "non corrélé": bruit_puissance(N, beta=0.0, rng=rng),
    "corrélé 1/f": bruit_puissance(N, beta=1.0, rng=rng),
    "brun (1/f²)": bruit_puissance(N, beta=2.0, rng=rng),
}

figures.galerie_series(series, n_points=400,
                       titre="Séries de validation — 400 premiers points");

# %% [markdown]
# ---
# ## 2. Entropie de permutation
#
# La mesure remplace chaque fenêtre de trois valeurs successives par l'ordre de
# ses éléments, puis applique l'entropie de Shannon à la distribution des ordres
# observés. Les amplitudes sont complètement ignorées : seule compte la forme.
#
# Normalisée, elle vaut 0 pour une série parfaitement monotone et tend vers 1
# pour une série sans structure.

# %%
pe = {nom: permutation(serie) for nom, serie in series.items()}

figures.barres_comparatives(
    pe, ylabel="entropie de permutation normalisée",
    titre="La mesure sépare nettement le régulier du désordonné", reference=1.0
);

# %% [markdown]
# Le signal périodique n'utilise que quelques motifs ordinaux, d'où sa valeur
# basse. Le bruit non corrélé les utilise tous de façon équiprobable et atteint
# presque le maximum. Le bruit brun, malgré son apparence très irrégulière,
# descend sensiblement : ses variations successives sont corrélées, donc certains
# motifs sont plus fréquents que d'autres.

# %% [markdown]
# ---
# ## 3. Le croisement multiéchelle
#
# C'est le résultat qui justifie un choix du plan de travail.
#
# L'entropie d'échantillon mesure la régularité d'une série : la probabilité que
# deux segments qui se ressemblaient sur $m$ points se ressemblent encore au
# point suivant. Appliquée telle quelle, elle attribue sa valeur la plus élevée
# au bruit non corrélé, ce qui suggérerait que le bruit blanc est le signal le
# plus complexe qui soit.
#
# L'entropie multiéchelle répète la mesure sur des versions granularisées de la
# série. Le résultat inverse la conclusion.

# %%
ECHELLES = 15
N_REALISATIONS = 3

def courbe(beta):
    """Moyenne et dispersion de MSE sur plusieurs réalisations."""
    resultats = []
    for _ in range(N_REALISATIONS):
        tau, valeurs = multiechelle(bruit_puissance(N, beta=beta, rng=rng),
                                    echelles=ECHELLES)
        resultats.append(valeurs)
    empilees = np.vstack(resultats)
    return tau, empilees.mean(axis=0), empilees.std(axis=0, ddof=1)

tau, moy_blanc, ec_blanc = courbe(0.0)
_, moy_rose, ec_rose = courbe(1.0)

figures.courbes_multiechelle(
    tau,
    {"non corrélé": moy_blanc, "corrélé 1/f": moy_rose},
    {"non corrélé": ec_blanc, "corrélé 1/f": ec_rose},
    titre=f"Entropie multiéchelle — {N_REALISATIONS} réalisations de {N} points",
);

print(f"échelle  1 : non corrélé {moy_blanc[0]:.3f}   1/f {moy_rose[0]:.3f}")
print(f"échelle {int(tau[-1]):2d} : non corrélé {moy_blanc[-1]:.3f}   1/f {moy_rose[-1]:.3f}")

# %% [markdown]
# **Lecture.** À l'échelle 1, le bruit non corrélé domine. Après granularisation,
# il s'effondre tandis que le bruit corrélé conserve son entropie sur toutes les
# échelles. Les deux courbes se croisent.
#
# Le signal le plus irrégulier de tous est donc celui dont la complexité chute le
# plus vite avec l'échelle. Irrégularité et complexité sont deux propriétés
# distinctes, et c'est exactement pourquoi le sous-objectif SO3 prévoit du
# multiéchelle plutôt que de l'entropie brute. Ce point n'a plus besoin d'être
# justifié par citation : il est reproduit ici.

# %% [markdown]
# ---
# ## 4. L'exposant de Hurst, et un biais qui compte
#
# Le Hurst est l'indicateur témoin principal du projet, puisque c'est la mesure
# utilisée par Carreras et al. sur les séries de blackouts que le modèle
# produira. Une valeur voisine de 0,5 indique une série non corrélée, une valeur
# supérieure une persistance.

# %%
pente, tailles, rs, ordonnee = hurst(series["non corrélé"], retourner_ajustement=True)
figures.ajustement_rs(tailles, rs, pente, ordonnee,
                      titre="Ajustement R/S sur un bruit non corrélé");

# %%
N_NUL = 200
nuls = np.array([hurst(bruit_puissance(N, 0.0, rng)) for _ in range(N_NUL)])
mesure_1f = hurst(series["corrélé 1/f"])

figures.distribution_nulle(
    nuls, mesure=mesure_1f, xlabel="exposant de Hurst",
    titre=f"Distribution nulle sur {N_NUL} bruits non corrélés de {N} points",
);

print(f"bruit non corrélé : H = {nuls.mean():.3f} ± {nuls.std(ddof=1):.3f}")
print(f"intervalle à 95 % : [{np.percentile(nuls, 2.5):.3f}, {np.percentile(nuls, 97.5):.3f}]")
print(f"bruit 1/f         : H = {mesure_1f:.3f}")

# %% [markdown]
# **Le point à retenir.** Sur des séries dont on sait qu'elles n'ont aucune
# corrélation, l'estimateur ne retourne pas 0,5 mais une valeur sensiblement plus
# élevée. C'est le biais connu de la méthode R/S sur des séries finies.
#
# Or Carreras rapporte $H = 0{,}55 \pm 0{,}02$ aux échelles courtes sur ses séries
# de blackouts, qu'il interprète comme une faible persistance. Les deux valeurs se
# recouvrent.
#
# Cela ne signifie pas que son résultat est faux : ses séries sont beaucoup plus
# longues et son biais n'est pas nécessairement le même. Mais avant de conclure à
# une persistance sur nos propres séries, il faudra comparer à la distribution
# nulle tracée ci-dessus plutôt qu'à la valeur théorique de 0,5. C'est la logique
# du témoin, appliquée cette fois au témoin lui-même.

# %% [markdown]
# ---
# ## 5. Un piège identifié à l'avance
#
# Les mesures départagent les valeurs égales par ordre temporel. Sur une série
# comportant beaucoup de valeurs identiques, cette convention finit par dominer le
# résultat.
#
# La question n'est pas théorique : la série de délestage quotidien produite par
# la dynamique lente sera exactement de cette forme, puisque la plupart des jours
# ne comportent aucun blackout et valent donc zéro exact.

# %%
print(f"{'zéros':>7} {'PE':>8} {'PE + bruit 1e-9':>17} {'SampEn':>9}")
print("-" * 45)

resultats_zeros = {}
for fraction in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]:
    serie = np.zeros(N)
    actifs = rng.random(N) > fraction
    serie[actifs] = rng.lognormal(0, 1, actifs.sum())

    pe_brut = permutation(serie)
    pe_bruite = permutation(serie + rng.normal(0, 1e-9, N))
    se = echantillon(serie)
    resultats_zeros[f"{fraction:.0%}"] = pe_brut
    print(f"{fraction:>6.0%} {pe_brut:>8.4f} {pe_bruite:>17.4f} {se:>9.4f}")

# %%
figures.barres_comparatives(
    resultats_zeros, ylabel="entropie de permutation",
    titre="La mesure suit la fraction de zéros, pas la dynamique", reference=1.0
);

# %% [markdown]
# Même série, même information : un bruit de $10^{-9}$, sans aucune signification
# physique, ramène la mesure au voisinage de 1.
#
# **Conséquence pour le projet.** Une entropie de permutation calculée sur le
# délestage quotidien mesurerait la fréquence des zéros plutôt que la dynamique du
# réseau. Elle produirait une belle courbe en fonction du paramètre de contrôle,
# entièrement explicable par un indicateur trivial — soit précisément le mode
# d'échec que SO3 cherche à détecter, mais provenant de l'outil et non du système.
#
# **Décision.** Aucun bruit ajouté aux données. L'analyse par entropie de
# permutation et SampEn portera sur la distribution des taux de charge des lignes,
# disponible à chaque pas de temps et dépourvue de zéros exacts. Le délestage et
# le nombre de lignes en panne restent les observables du Hurst et des
# statistiques de taille de blackout.
#
# L'exposant de Hurst n'est pas affecté de la même façon : R/S repose sur des
# sommes cumulées et un écart-type, que les égalités ne perturbent pas.

# %% [markdown]
# ---
# ## 6. Le réseau et les flux
#
# Premier module du fil A. La construction suit celle de Carreras : la racine
# porte trois branches, chaque nœud du bord en reçoit deux à chaque génération,
# et les générateurs sont placés au troisième niveau.

# %%
for generations in range(3, 8):
    reseau = arbre(generations)
    print(f"{generations} générations : {reseau.n_noeuds:>3} nœuds, "
          f"{reseau.n_lignes:>3} lignes, {reseau.generateurs.size:>2} générateurs")

# %% [markdown]
# Ces tailles — 22, 46, 94, 190, 382 — sont exactement celles utilisées dans la
# littérature de référence. Le réseau à 46 nœuds sert de cas de travail.

# %%
reseau = arbre(4)

demande = np.zeros(reseau.n_noeuds)
demande[reseau.charges] = -rng.uniform(0.5, 1.5, reseau.charges.size)
demande[reseau.generateurs] = -demande[reseau.charges].sum() / reseau.generateurs.size

A = matrice_de_flux(reseau)
f = flux(reseau, demande, A)
limites = limites_depuis_cas_de_base(reseau, demande, marge=1.5)
taux = taux_de_charge(f, limites)

figures.reseau_en_arbre(reseau, taux=taux,
                        titre="Réseau à 46 nœuds — taux de charge au cas de base");

print(f"demande totale       : {-demande[reseau.charges].sum():.1f}")
print(f"degré moyen          : {degres(reseau).mean():.2f}")
print(f"taux de charge max   : {taux.max():.3f}")
print(f"taux de charge moyen : {taux.mean():.3f}")

# %% [markdown]
# **Point ouvert.** Les capacités de lignes utilisées ici viennent d'une marge
# uniforme appliquée au cas de base, ce qui donne à toutes les lignes le même
# taux de charge — une situation dégénérée où le taux maximal se confond avec le
# taux moyen. Le modèle de référence utilise plutôt des capacités décroissant
# avec la profondeur de l'arbre, disponibles dans `limites_par_niveau`. Le choix
# définitif sera tranché en écrivant le dispatch, puisque c'est lui qui
# détermine réellement la répartition de la production.

# %% [markdown]
# **Validation analytique.** Dans un arbre, le chemin entre deux nœuds est unique.
# Retirer une ligne sépare donc le réseau en deux, et toute la puissance échangée
# entre les deux parties transite forcément par cette ligne. Le flux de chaque
# ligne doit donc égaler exactement la puissance nette de son sous-arbre,
# indépendamment des réactances.
#
# C'est vérifié sur les 45 lignes par la suite de tests. Le calcul des flux est
# donc validé contre les mathématiques, et non contre une autre implémentation.

# %% [markdown]
# ---
# ## Où en est le projet
#
# **En place et validé** — les deux modules du fil B, `entropie` et
# `indicateurs`, ainsi que le premier module du fil A, `reseau`. Soixante tests,
# environ vingt secondes. Résultats identiques entre Linux et Windows.
#
# **Deux constats qui orientent la suite** — le biais de l'estimateur R/S impose
# de comparer à une distribution nulle plutôt qu'à 0,5 ; et le traitement des
# égalités impose de choisir l'observable avec soin plutôt que d'appliquer les
# mesures au délestage brut.
#
# **Prochains modules** — `dispatch` (résolution sous contraintes), `cascade`
# (boucle d'avaries), puis `evolution` (dynamique lente), qui produira les séries
# temporelles sur lesquelles tout ce qui précède sera appliqué.
