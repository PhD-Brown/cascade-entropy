"""
Validation quantitative de SO1 — Carreras et al. (2002).

Ce script est un AUDIT : il ne modifie aucun paramètre du modèle et n'effectue
aucun ajustement. Il demande au modèle tel qu'il est codé de produire des
nombres, puis les compare aux références publiées.

Contenu
-------
1. Fig. 3-4 (382 nœuds) : comparaison de deux lectures du papier
     A) l'étiquette publiée P_D/P_C = 0.30 ;
     B) la Table I lue littéralement : chaque charge vaut |P_L| = 74.

2. Seuil déterministe de transport r_T(N) pour N = 46, 94, 190, 382 :
     - mesure numérique (balayage grossier + bissection sur le dispatch) ;
     - prédiction analytique indépendante, ligne par ligne ;
     - écart entre les deux.

3. Ratios du mini-scan finite-size à distance relative commune
       rho = r / r_T(N),
   avec un diagnostic du régime de génération : probabilité que la demande
   totale fluctuée dépasse P_C (délestage imposé par la génération, et non
   par le transport).

4. Bandes ordonnées de la Fig. 10 (382 nœuds) : frontières analytiques
   (`carreras.bandes_ordonnees`) et vérification au milieu de chaque bande
   avec les deux règles de départage. Attendu : bande présente (aucune
   avarie) avec « exterieur_dabord », absente avec « highs ».

Sorties : data/05_reproduction_carreras/validation_so1/
    audit_figures_3_4.json
    seuils_transport.csv
    cibles_rho.csv
    commandes_mini_scan.txt
    bandes_fig10.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cascade_entropy.carreras import (
    GAMMA_TABLE,
    bandes_ordonnees,
    P_G_TABLE,
    P_L_TABLE,
    configuration_arbre,
    groupes_regions,
)
from cascade_entropy.chemins import DOSSIER_DATA
from cascade_entropy.cascade import journee
from cascade_entropy.dispatch import (
    DEPARTAGES,
    SEUIL_SATURATION,
    demande_uniforme,
    resoudre,
)


TAILLES = (46, 94, 190, 382)

# Série principale du mini-scan. Le 46 nœuds est conservé mais signalé : sous la
# Table I littérale, son seuil de transport fluctué tombe au-delà de la limite
# de génération (voir le diagnostic P(P_D > P_C) dans la sortie).
TAILLES_PRINCIPALES = (94, 190, 382)

# Valeur publiée pour les lignes extérieures de la Fig. 3.
CIBLE_M_EXTERIEUR = 0.601

# Deux définitions du seuil :
#   0.99 : critère de saturation effectivement utilisé par cascade.py ;
#   1.00 : limite physique M = 1 (à la tolérance numérique du solveur près).
CIBLE_M_CASCADE = SEUIL_SATURATION
CIBLE_M_PHYSIQUE = 1.0 - 1e-6

# Distances relatives au seuil. 1/gamma = cas où le facteur régional maximal
# amène exactement une région au seuil déterministe (estimation, pas un point
# critique démontré). 0.60 encadre le seuil par le haut.
RHO_CIBLES = (0.50, 1.0 / GAMMA_TABLE, 0.55, 0.60)

# Tirages Monte-Carlo pour P(P_D > P_C) — aucun dispatch, très rapide.
N_TIRAGES_GENERATION = 200_000
GRAINE = 20261003

DOSSIER_SORTIE = DOSSIER_DATA / "05_reproduction_carreras" / "validation_so1"


# ===========================================================================
# Utilitaires
# ===========================================================================

def niveau_lignes(reseau) -> np.ndarray:
    """Niveau d'une ligne = niveau du nœud enfant (arbre() oriente parent -> enfant)."""
    if reseau.niveaux is None:
        raise ValueError("Le réseau ne contient pas ses niveaux.")
    return np.asarray(reseau.niveaux[reseau.lignes[:, 1]], dtype=int)


def masque_lignes_exterieures(reseau) -> np.ndarray:
    """Lignes situées au-delà du niveau des générateurs (niveau 3)."""
    return niveau_lignes(reseau) > 3


def taille_sous_arbres(reseau) -> np.ndarray:
    """Nombre de nœuds dans le sous-arbre alimenté par chaque ligne."""
    enfants: dict[int, list[int]] = {i: [] for i in range(reseau.n_noeuds)}
    for a, b in reseau.lignes:
        enfants[int(a)].append(int(b))

    taille = np.ones(reseau.n_noeuds, dtype=int)
    # Niveaux décroissants : chaque enfant est traité avant son parent.
    for noeud in np.argsort(-reseau.niveaux, kind="stable"):
        for enfant in enfants[int(noeud)]:
            taille[noeud] += taille[enfant]
    return taille[reseau.lignes[:, 1]]


def solution_ratio(cfg, ratio: float):
    """Dispatch déterministe à demande uniforme (gamma = 1)."""
    demande = demande_uniforme(cfg.reseau, float(ratio) * cfg.p_c)
    return resoudre(
        cfg.reseau,
        demande,
        limites=cfg.limites,
        puissance_max=cfg.puissance_max,
    )


def erreur_relative_pct(valeur: float, reference: float) -> float:
    return 100.0 * abs(valeur - reference) / abs(reference)


# ===========================================================================
# 1. Fig. 3-4
# ===========================================================================

def _compter_generateurs(frac: np.ndarray):
    pleins = int(np.sum(frac >= 0.99))
    nuls = int(np.sum(frac <= 0.01))
    partiels = frac[(frac > 0.01) & (frac < 0.99)]
    return pleins, int(partiels.size), nuls, partiels


def audit_figures_3_4() -> dict:
    cfg = configuration_arbre(382)
    masque_ext = masque_lignes_exterieures(cfg.reseau)

    # Cas A : étiquette publiée.
    sol_030 = solution_ratio(cfg, 0.30)
    m_ext_030 = sol_030.taux_de_charge[masque_ext]
    frac_030 = sol_030.production / cfg.puissance_max

    # Cas B : Table I, chaque charge = |P_L|.
    demande_table = np.full(cfg.n_charges, abs(P_L_TABLE), dtype=float)
    pd_table = float(demande_table.sum())
    sol_table = resoudre(
        cfg.reseau,
        demande_table,
        limites=cfg.limites,
        puissance_max=cfg.puissance_max,
    )
    m_ext_table = sol_table.taux_de_charge[masque_ext]
    frac_table = sol_table.production / cfg.puissance_max

    p030, n030, z030, _ = _compter_generateurs(frac_030)
    ptab, ntab, ztab, partiels_tab = _compter_generateurs(frac_table)

    return {
        "n_noeuds": 382,
        "n_charges": cfg.n_charges,
        "P_G": float(P_G_TABLE),
        "P_C": float(cfg.p_c),
        "M_reference": CIBLE_M_EXTERIEUR,

        "ratio_papier": 0.30,
        "M_ext_moyen_r030": float(m_ext_030.mean()),
        "M_ext_std_r030": float(m_ext_030.std()),
        "erreur_M_r030_pct": erreur_relative_pct(float(m_ext_030.mean()), CIBLE_M_EXTERIEUR),
        "generateurs_pleins_partiels_nuls_r030": [p030, n030, z030],

        "P_L_table": float(P_L_TABLE),
        "P_D_table": pd_table,
        "ratio_table": pd_table / cfg.p_c,
        "M_ext_moyen_table": float(m_ext_table.mean()),
        "M_ext_std_table": float(m_ext_table.std()),
        "erreur_M_table_pct": erreur_relative_pct(float(m_ext_table.mean()), CIBLE_M_EXTERIEUR),
        "generateurs_pleins_partiels_nuls_table": [ptab, ntab, ztab],
        "fractions_partielles_table": partiels_tab.tolist(),
        "demande_table_en_unites_PG": pd_table / P_G_TABLE,

        "production_fraction_r030": frac_030.tolist(),
        "production_fraction_table": frac_table.tolist(),
    }


# ===========================================================================
# 2. Seuil de transport r_T(N)
# ===========================================================================

def mmax_initial(cfg, ratio: float) -> float:
    """M_max du dispatch déterministe, sans aucune avarie."""
    return float(solution_ratio(cfg, ratio).taux_maximal)


def premier_franchissement(
    cfg,
    cible: float,
    *,
    ratio_min: float = 0.20,
    ratio_max: float = 2.50,
    pas_scan: float = 0.01,
    iterations: int = 40,
) -> float:
    """
    Premier r tel que M_max(r) >= cible.

    Balayage grossier (pas 0.01) puis bissection dans le premier intervalle.
    La bissection suppose M_max monotone dans cet intervalle ; la comparaison
    avec la prédiction analytique sert justement de garde-fou.
    """
    ratios = np.arange(ratio_min, ratio_max + pas_scan / 2.0, pas_scan)
    r_prec = float(ratios[0])
    if mmax_initial(cfg, r_prec) >= cible:
        return r_prec

    for r in ratios[1:]:
        r = float(r)
        if mmax_initial(cfg, r) >= cible:
            bas, haut = r_prec, r
            for _ in range(iterations):
                milieu = 0.5 * (bas + haut)
                if mmax_initial(cfg, milieu) >= cible:
                    haut = milieu
                else:
                    bas = milieu
            return haut
        r_prec = r
    return float("nan")


def seuil_analytique(cfg) -> tuple[float, int, int]:
    """
    Prédiction analytique, indépendante du solveur.

    Pour une ligne extérieure (niveau > 3), le flux vaut la demande totale de
    son sous-arbre lorsque celui-ci est entièrement servi :
        F = n_sous_arbre * r * P_C / N_L.
    Elle sature quand F = F_max, d'où
        r_ligne = F_max * N_L / (n_sous_arbre * P_C),
    et r_T est le minimum sur les lignes extérieures.

    Retourne (r_T, niveau de la ligne limitante, taille de son sous-arbre).
    """
    niveaux = niveau_lignes(cfg.reseau)
    tailles = taille_sous_arbres(cfg.reseau)

    r_lignes = cfg.limites * cfg.n_charges / (tailles * cfg.p_c)
    r_lignes = np.where(niveaux > 3, r_lignes, np.inf)
    idx = int(np.argmin(r_lignes))
    return float(r_lignes[idx]), int(niveaux[idx]), int(tailles[idx])


def audit_seuils_transport() -> list[dict]:
    rows = []
    for n_noeuds in TAILLES:
        print(f"  seuils — {n_noeuds} nœuds...", flush=True)
        cfg = configuration_arbre(n_noeuds)

        r_99 = premier_franchissement(cfg, CIBLE_M_CASCADE)
        r_1 = premier_franchissement(cfg, CIBLE_M_PHYSIQUE)
        r_ana, niveau, n_sous = seuil_analytique(cfg)

        rows.append({
            "n_noeuds": n_noeuds,
            "n_charges": cfg.n_charges,
            "ratio_table": cfg.ratio_table,
            "r_M099_numerique": r_99,
            "r_M099_analytique": CIBLE_M_CASCADE * r_ana,
            "r_T_numerique": r_1,
            "r_T_analytique": r_ana,
            "ecart_r_T_pct": erreur_relative_pct(r_1, r_ana),
            "niveau_limitant": niveau,
            "noeuds_sous_arbre_limitant": n_sous,
            "r_T_sur_gamma": r_1 / GAMMA_TABLE,
            "gamma": GAMMA_TABLE,
        })
    return rows


# ===========================================================================
# 3. Cibles du mini-scan et diagnostic de génération
# ===========================================================================

def probabilite_depassement_generation(
    n_noeuds: int,
    ratio: float,
    rng: np.random.Generator,
) -> float:
    """
    P(P_D réel > P_C) sous les fluctuations régionales (N_F = 3, gamma = 1.9).

    Chaque région f contient une fraction w_f des charges et reçoit un facteur
    r_f ~ U[2-gamma, gamma] (même hypothèse que carreras.facteurs_regionaux).
    La demande réelle vaut ratio * P_C * sum_f w_f r_f. Aucun dispatch : c'est
    la fraction des jours où la génération seule impose un délestage,
    indépendamment du transport.
    """
    cfg = configuration_arbre(n_noeuds)
    groupes = groupes_regions(cfg.reseau, 3)
    poids = np.bincount(groupes) / groupes.size

    facteurs = rng.uniform(
        2.0 - GAMMA_TABLE,
        GAMMA_TABLE,
        size=(N_TIRAGES_GENERATION, poids.size),
    )
    total_relatif = ratio * (facteurs @ poids)
    return float(np.mean(total_relatif > 1.0))


def construire_cibles_rho(seuils: list[dict]) -> list[dict]:
    rng = np.random.default_rng(GRAINE)
    rows = []
    for row in seuils:
        n_noeuds = int(row["n_noeuds"])
        r_t = float(row["r_T_numerique"])
        for rho in RHO_CIBLES:
            ratio = float(rho * r_t)
            rows.append({
                "n_noeuds": n_noeuds,
                "serie": "principale" if n_noeuds in TAILLES_PRINCIPALES else "signalee",
                "rho": float(rho),
                "r_T": r_t,
                "ratio_PD_PC": ratio,
                "moyenne_au_dessus_de_PC": ratio > 1.0,
                "P_PD_sup_PC": probabilite_depassement_generation(n_noeuds, ratio, rng),
            })
    return rows


def commandes_mini_scan(cibles: list[dict]) -> list[str]:
    """Une commande (une seule ligne, compatible PowerShell) par taille."""
    lignes = []
    for n_noeuds in TAILLES:
        ratios = [c["ratio_PD_PC"] for c in cibles if c["n_noeuds"] == n_noeuds]
        texte = " ".join(f"{r:.6f}" for r in ratios)
        lignes.append(
            "python scripts/05_reproduction_carreras.py scan "
            f"--tailles {n_noeuds} --ratios {texte} "
            "--n 1000 --chunk-size 100 --min-tail 30 --workers 8 "
            f"--run-id validation-rho-N{n_noeuds}"
        )
    return lignes


# ===========================================================================
# 4. Bandes ordonnées de la Fig. 10
# ===========================================================================

def audit_bandes_fig10() -> list[dict]:
    """
    Pour chaque bande prédite sur le 382, simule une journée déterministe
    (p0 = 0, p1 = 1, sans fluctuation) au milieu de la bande, avec chaque
    règle de départage, et compte les lignes tombées.
    """
    cfg = configuration_arbre(382)
    rows = []
    for bande in bandes_ordonnees(cfg):
        milieu = 0.5 * (bande["r_debut"] + bande["r_fin_critere"])
        row = {
            "arbre_equivalent": bande["equivalent_n_noeuds"],
            "r_debut": bande["r_debut"],
            "r_fin": bande["r_fin"],
            "r_fin_critere": bande["r_fin_critere"],
            "r_milieu": milieu,
        }
        for regle in DEPARTAGES:
            j = journee(
                cfg.reseau, demande_uniforme(cfg.reseau, milieu * cfg.p_c),
                cfg.puissance_max, p0=0.0, p1=1.0, g=0.0,
                rng=np.random.default_rng(GRAINE), limites=cfg.limites,
                departage=regle,
            )
            row[f"lignes_tombees_{regle}"] = int(j.lignes_tombees.size)
        rows.append(row)
    return rows


# ===========================================================================
# Sauvegarde et affichage
# ===========================================================================

def ecrire_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def afficher_fig34(a: dict) -> None:
    print("\n" + "=" * 76)
    print("1. FIGURES 3-4 — AUDIT INDÉPENDANT (382 nœuds)")
    print("=" * 76)
    print(f"P_C = {a['P_C']:.4f}    référence M_ext (Carreras) = {a['M_reference']:.3f}")

    print("\nA) Étiquette publiée P_D/P_C = 0.30")
    print(f"   M_ext = {a['M_ext_moyen_r030']:.6f} ± {a['M_ext_std_r030']:.1e}"
          f"   écart = {a['erreur_M_r030_pct']:.2f} %")
    p, n, z = a["generateurs_pleins_partiels_nuls_r030"]
    print(f"   dispatch : {p} pleins · {n} partiel(s) · {z} nuls")

    print("\nB) Table I : chaque charge = |P_L| = 74")
    print(f"   P_D = {a['P_D_table']:.1f}   P_D/P_C = {a['ratio_table']:.6f}"
          f"   (= {a['demande_table_en_unites_PG']:.4f} P_G)")
    print(f"   M_ext = {a['M_ext_moyen_table']:.6f} ± {a['M_ext_std_table']:.1e}"
          f"   écart = {a['erreur_M_table_pct']:.3f} %")
    p, n, z = a["generateurs_pleins_partiels_nuls_table"]
    partiels = ", ".join(f"{x:.4f}" for x in a["fractions_partielles_table"])
    print(f"   dispatch : {p} pleins · {n} partiel(s) [{partiels}] · {z} nuls")
    print("   (Carreras Fig. 4 : 10 au maximum, le 11e réduit, le 12e quasi nul)")


def afficher_seuils(seuils: list[dict]) -> None:
    print("\n" + "=" * 76)
    print("2. SEUILS DÉTERMINISTES DE TRANSPORT")
    print("=" * 76)
    print(f"{'N':>5} {'r(M=.99)':>10} {'r_T num':>10} {'r_T ana':>10} "
          f"{'écart %':>9} {'niveau':>7} {'n_sous':>7} {'r_T/γ':>8}")
    for r in seuils:
        print(f"{r['n_noeuds']:5d} {r['r_M099_numerique']:10.6f} "
              f"{r['r_T_numerique']:10.6f} {r['r_T_analytique']:10.6f} "
              f"{r['ecart_r_T_pct']:9.2e} {r['niveau_limitant']:7d} "
              f"{r['noeuds_sous_arbre_limitant']:7d} {r['r_T_sur_gamma']:8.4f}")
    print("\nr_T ana = F_max * N_L / (n_sous * P_C) sur la ligne extérieure limitante.")
    print("r(M=.99) = 0.99 * r_T : écart de définition de 1 %, pas une incertitude.")


def afficher_cibles(cibles: list[dict]) -> None:
    print("\n" + "=" * 76)
    print("3. CIBLES rho = r / r_T  ET RÉGIME DE GÉNÉRATION")
    print("=" * 76)
    print(f"{'N':>5} {'série':>11} {'rho':>7} {'r = P_D/P_C':>12} "
          f"{'moy > P_C':>10} {'P(P_D>P_C)':>11}")
    for c in cibles:
        print(f"{c['n_noeuds']:5d} {c['serie']:>11} {c['rho']:7.4f} "
              f"{c['ratio_PD_PC']:12.6f} {str(c['moyenne_au_dessus_de_PC']):>10} "
              f"{c['P_PD_sup_PC']:11.3f}")
    print("\nP(P_D>P_C) = fraction des jours où la génération seule impose un délestage.")
    print("Si elle est grande, la taille n'est pas la seule variable qui change.")


def afficher_bandes(rows: list[dict]) -> None:
    print("\n" + "=" * 76)
    print("4. BANDES ORDONNÉES DE LA FIG. 10 (382 nœuds, p1 = 1)")
    print("=" * 76)
    print(f"{'arbre':>6} {'r_début':>9} {'r_fin(0.99)':>12} {'milieu':>8} "
          + " ".join(f"{'tombées ' + r:>26}" for r in DEPARTAGES))
    for r in rows:
        print(f"{r['arbre_equivalent']:6d} {r['r_debut']:9.4f} "
              f"{r['r_fin_critere']:12.4f} {r['r_milieu']:8.3f} "
              + " ".join(f"{r['lignes_tombees_' + d]:26d}" for d in DEPARTAGES))
    print("\nBande présente = 0 ligne tombée au milieu. Frontières lues sur la")
    print("Fig. 10 de Carreras (± 0.05) : 1.45 · 2.07–3.10 · 4.46–7.18.")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

    fig34 = audit_figures_3_4()
    afficher_fig34(fig34)
    (DOSSIER_SORTIE / "audit_figures_3_4.json").write_text(
        json.dumps(fig34, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nCalcul des seuils de transport (≈ 30 s)...")
    seuils = audit_seuils_transport()
    afficher_seuils(seuils)
    ecrire_csv(DOSSIER_SORTIE / "seuils_transport.csv", seuils)

    cibles = construire_cibles_rho(seuils)
    afficher_cibles(cibles)
    ecrire_csv(DOSSIER_SORTIE / "cibles_rho.csv", cibles)

    commandes = commandes_mini_scan(cibles)
    (DOSSIER_SORTIE / "commandes_mini_scan.txt").write_text(
        "\n".join(commandes) + "\n", encoding="utf-8"
    )

    bandes = audit_bandes_fig10()
    afficher_bandes(bandes)
    ecrire_csv(DOSSIER_SORTIE / "bandes_fig10.csv", bandes)

    print("\n" + "=" * 76)
    print("COMMANDES DU MINI-SCAN (à ne lancer qu'après lecture des tableaux)")
    print("=" * 76)
    for ligne in commandes:
        print("\n" + ligne)

    print("\nFichiers produits dans :", DOSSIER_SORTIE)


if __name__ == "__main__":
    main()
