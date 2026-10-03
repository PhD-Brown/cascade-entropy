"""
Diagnostics déterministes Carreras et al. (2002).

Ce script est volontairement séparé de `05_reproduction_carreras.py` pendant
qu'une production lourde peut encore être en cours. Il n'altère aucun fichier
de production existant et sert à préparer la réplication des figures 3 à 11 de
Carreras et al. (Chaos 12, 985-994, 2002).

Une fois validé, son contenu peut être fusionné dans
`scripts/05_reproduction_carreras.py` sous un sous-commande `deterministe`.

Sorties principales
-------------------
data/05_reproduction_carreras/deterministe/
    scan_382.csv
    scan_382.npz
    scan_etendu_382.csv
    scan_etendu_382.npz
    sensibilite_p1_382.csv
    metadata_deterministe_382.json

figures/05_reproduction_carreras/deterministe/
    05_carreras_fig03_M_r030.png/.pdf
    05_carreras_fig04_generateurs_r030.png/.pdf
    05_carreras_fig05_transitions_382.png/.pdf
    05_carreras_fig06_M_r104.png/.pdf
    05_carreras_fig07_M_r145.png/.pdf
    05_carreras_fig08_M_r173.png/.pdf
    05_carreras_fig09_carte_M.png/.pdf
    05_carreras_fig10_carte_M_etendue.png/.pdf
    05_carreras_fig11_sensibilite_p1.png/.pdf

Usage
-----
Réplication standard des figures déterministes de Carreras :

    python scripts/05_diagnostics_carreras.py

Balayage plus fin :

    python scripts/05_diagnostics_carreras.py --pas 0.0025

Sans la sensibilité à p1 :

    python scripts/05_diagnostics_carreras.py --sans-p1

Autre taille, pour audit uniquement :

    python scripts/05_diagnostics_carreras.py --taille 190

Important
---------
- Le balayage principal n'a AUCUNE fluctuation de demande.
- p0=0.
- p1=1 pour les figures 3-10, comme dans la discussion déterministe de
  Carreras autour des figures 9-11.
- Pour les lignes tombées, M_ij n'est pas physiquement défini dans notre
  représentation numérique (réactance et limite artificiellement dégradées).
  On conserve donc deux matrices :
    * M_final_service : NaN pour les lignes hors service ;
    * M_final_carreras : 0 pour les lignes hors service, uniquement pour une
      visualisation analogue aux figures du papier.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cascade_entropy.carreras import P_G_TABLE, configuration_arbre
from cascade_entropy.cascade import journee
from cascade_entropy.chemins import DOSSIER_DATA, DOSSIER_FIGURES
from cascade_entropy.dispatch import SEUIL_SATURATION, demande_uniforme


DOSSIER_DATA_DET = DOSSIER_DATA / "05_reproduction_carreras" / "deterministe"
DOSSIER_FIG_DET = DOSSIER_FIGURES / "05_reproduction_carreras" / "deterministe"

GRAINE_DEFAUT = 20261005
RATIOS_CIBLES = (0.30, 1.04, 1.45, 1.73)

# Valeurs explicitement utilisées/discutées dans l'article.
CIBLE_GENERATION = 1.00
CIBLE_TRANSPORT = 1.45
CIBLE_M_EXTERIEUR_R030 = 0.601


@dataclass
class AuditDeterministe:
    taille: int
    n_lignes: int
    n_generateurs: int
    n_charges: int
    p_c: float
    seuil_generation_observe: float | None
    seuil_transport_initial_observe: float | None
    seuil_premiere_avarie_observe: float | None
    ratio_plus_grand_saut_shed: float | None
    taille_plus_grand_saut_shed: float | None
    m_exterieur_moyen_r030: float | None
    m_exterieur_std_r030: float | None
    generateurs_a_99pct_r030: int | None
    generateurs_sous_1pct_r030: int | None


def _ratios(minimum: float, maximum: float, pas: float) -> np.ndarray:
    """Grille inclusive et stable numériquement."""
    if minimum <= 0 or maximum <= minimum or pas <= 0:
        raise ValueError("Intervalle de ratios invalide.")
    n = int(np.floor((maximum - minimum) / pas + 0.5))
    valeurs = minimum + pas * np.arange(n + 1)
    if valeurs[-1] < maximum - pas * 1e-6:
        valeurs = np.append(valeurs, maximum)
    return np.round(valeurs, 10)


def _ecrire_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Aucune ligne à écrire.")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sauvegarder(fig, nom: str) -> None:
    DOSSIER_FIG_DET.mkdir(parents=True, exist_ok=True)
    fig.savefig(DOSSIER_FIG_DET / f"{nom}.png", dpi=190, bbox_inches="tight")
    fig.savefig(DOSSIER_FIG_DET / f"{nom}.pdf", bbox_inches="tight")
    plt.close(fig)


def _niveau_lignes(reseau) -> np.ndarray:
    """
    Niveau topologique de chaque ligne = niveau du nœud enfant.

    `arbre()` oriente les lignes du parent vers l'enfant. Cette quantité permet
    d'identifier les lignes situées à l'extérieur du niveau des générateurs sans
    dépendre de la numérotation exacte des lignes.
    """
    if reseau.niveaux is None:
        return np.full(reseau.n_lignes, -1, dtype=int)
    return np.asarray([reseau.niveaux[int(j)] for _, j in reseau.lignes], dtype=int)


def _ligne_exterieure_generateurs(reseau) -> np.ndarray:
    """
    Masque des lignes situées au-delà du niveau des générateurs.

    Les générateurs Carreras sont au niveau 3. Les lignes de niveau >3 relient
    donc les charges externes, là où le papier rapporte M_ij ≈ 0.601 à r=0.3.
    """
    return _niveau_lignes(reseau) > 3


def simuler_point(cfg, ratio: float, *, p1: float, graine: int):
    """
    Une réalisation à demande uniforme, sans fluctuation et sans avarie p0.

    Pour p1=0 ou 1, le résultat est déterministe. Pour 0<p1<1, `graine`
    rend le tirage reproductible.
    """
    demande = demande_uniforme(cfg.reseau, ratio * cfg.p_c)
    return journee(
        cfg.reseau,
        demande,
        cfg.puissance_max,
        p0=0.0,
        p1=p1,
        g=0.0,
        rng=np.random.default_rng(graine),
        limites=cfg.limites,
    )


def _mesures_point(cfg, ratio: float, resultat) -> tuple[dict, dict[str, np.ndarray]]:
    """
    Transforme une `Journee` en métriques scalaires + tableaux détaillés.

    Les grandeurs initiales décrivent le premier dispatch, avant toute chute
    par surcharge. Les grandeurs finales décrivent l'état convergé de la cascade.
    """
    pd = float(resultat.demande.sum())

    sol_i = resultat.solution_initiale
    sol_f = resultat.solution

    ps_i = float(sol_i.delestage_total)
    ps_f = float(sol_f.delestage_total)

    servie_i = float(sol_i.charge_servie.sum())
    servie_f = float(sol_f.charge_servie.sum())

    hors_i = resultat.hors_service_initial.copy()
    hors_f = resultat.hors_service.copy()

    m_i = sol_i.taux_de_charge.astype(float, copy=True)
    m_f_service = sol_f.taux_de_charge.astype(float, copy=True)
    m_f_service[hors_f] = np.nan

    # Version strictement visuelle : une ligne tombée n'est plus en service,
    # donc on l'affiche à zéro comme absence de transport. La matrice physique
    # avec NaN reste sauvegardée séparément.
    m_f_carreras = np.nan_to_num(m_f_service, nan=0.0)

    flux_i = sol_i.flux.astype(float, copy=True)
    flux_f_service = sol_f.flux.astype(float, copy=True)
    flux_f_service[hors_f] = np.nan

    row = {
        "ratio_PD_PC": float(ratio),
        "P_D": pd,
        "P_C": float(cfg.p_c),

        "P_shed_initial": ps_i,
        "P_shed_final": ps_f,
        "P_served_initial": servie_i,
        "P_served_final": servie_f,

        # Observable explicitement définie dans le texte de Carreras.
        "shed_sur_demande_initial": ps_i / pd if pd > 0 else 0.0,
        "shed_sur_demande_final": ps_f / pd if pd > 0 else 0.0,

        # Conservée aussi pour audit de la légende/normalisation historique.
        "shed_sur_servie_initial": ps_i / servie_i if servie_i > 0 else np.nan,
        "shed_sur_servie_final": ps_f / servie_f if servie_f > 0 else np.nan,

        "served_sur_PC_initial": servie_i / cfg.p_c,
        "served_sur_PC_final": servie_f / cfg.p_c,

        "Mmax_initial": float(resultat.taux_maximal_initial),
        "Mmax_final_service": float(resultat.taux_maximal),

        "n_lignes_saturees_initial": int(
            np.sum((m_i >= SEUIL_SATURATION) & ~hors_i)
        ),
        "n_lignes_tombees": int(resultat.lignes_tombees.size),
        "n_avaries_p0": int(resultat.avaries_accidentelles.size),
        "n_avaries_surcharge": int(
            sum(v.size for v in resultat.avaries_par_surcharge)
        ),
        "n_vagues_surcharge": int(len(resultat.avaries_par_surcharge)),
        "n_iterations": int(resultat.n_iterations),

        "production_totale_initiale": float(sol_i.production.sum()),
        "production_totale_finale": float(sol_f.production.sum()),
        "n_generateurs_99pct_initial": int(
            np.sum(sol_i.production >= 0.99 * cfg.puissance_max)
        ),
        "n_generateurs_1pct_initial": int(
            np.sum(sol_i.production <= 0.01 * cfg.puissance_max)
        ),

        "n_lignes_service_final": int((~hors_f).sum()),
    }

    arrays = {
        "M_initial": m_i,
        "M_final_service": m_f_service,
        "M_final_carreras": m_f_carreras,
        "flux_initial": flux_i,
        "flux_final_service": flux_f_service,
        "production_initiale": sol_i.production.astype(float, copy=True),
        "production_finale": sol_f.production.astype(float, copy=True),
        "charge_servie_initiale": sol_i.charge_servie.astype(float, copy=True),
        "charge_servie_finale": sol_f.charge_servie.astype(float, copy=True),
        "delestage_initial": sol_i.delestage.astype(float, copy=True),
        "delestage_final": sol_f.delestage.astype(float, copy=True),
        "hors_service_initial": hors_i,
        "hors_service_final": hors_f,
    }
    return row, arrays


def balayer(cfg, ratios: np.ndarray, *, p1: float, graine: int):
    """
    Balaye P_D/P_C et conserve TOUTES les grandeurs utiles à Carreras.

    Pour p1=1, la graine n'influence pas le résultat mais reste fournie pour
    maintenir une interface unique.
    """
    rows: list[dict] = []
    champs: dict[str, list[np.ndarray]] | None = None

    graines = np.random.SeedSequence(graine).spawn(len(ratios))

    for i, (ratio, seq) in enumerate(zip(ratios, graines), 1):
        seed_i = int(seq.generate_state(1, dtype=np.uint64)[0])
        resultat = simuler_point(cfg, float(ratio), p1=p1, graine=seed_i)
        row, arrays = _mesures_point(cfg, float(ratio), resultat)
        rows.append(row)

        if champs is None:
            champs = {k: [] for k in arrays}
        for k, v in arrays.items():
            champs[k].append(v)

        if i == 1 or i == len(ratios) or i % max(1, len(ratios) // 10) == 0:
            print(
                f"[{i:4d}/{len(ratios):4d}] r={ratio:.4f} | "
                f"shed={row['shed_sur_demande_final']:.3f} | "
                f"Mmax0={row['Mmax_initial']:.3f} | "
                f"outages={row['n_lignes_tombees']}"
            )

    assert champs is not None
    matrice = {k: np.stack(v, axis=0) for k, v in champs.items()}
    return rows, matrice


def _index_ratio(ratios: np.ndarray, cible: float) -> int:
    return int(np.argmin(np.abs(ratios - cible)))


def _audit(cfg, ratios, rows, arrays) -> AuditDeterministe:
    shed = np.array([r["shed_sur_demande_final"] for r in rows], dtype=float)
    mmax0 = np.array([r["Mmax_initial"] for r in rows], dtype=float)
    outages = np.array([r["n_lignes_tombees"] for r in rows], dtype=int)

    idx_g = np.flatnonzero(shed > 1e-8)
    idx_t = np.flatnonzero(mmax0 >= SEUIL_SATURATION)
    idx_o = np.flatnonzero(outages > 0)

    jumps = np.diff(shed)
    if jumps.size:
        j = int(np.nanargmax(jumps))
        r_jump = float(ratios[j + 1])
        d_jump = float(jumps[j])
    else:
        r_jump = None
        d_jump = None

    idx03 = _index_ratio(ratios, 0.30)
    masque_ext = _ligne_exterieure_generateurs(cfg.reseau)
    m03 = arrays["M_initial"][idx03, masque_ext]
    prod03 = arrays["production_initiale"][idx03]
    frac_prod03 = prod03 / cfg.puissance_max

    return AuditDeterministe(
        taille=cfg.n_noeuds,
        n_lignes=cfg.reseau.n_lignes,
        n_generateurs=cfg.n_generateurs,
        n_charges=cfg.n_charges,
        p_c=float(cfg.p_c),
        seuil_generation_observe=float(ratios[idx_g[0]]) if idx_g.size else None,
        seuil_transport_initial_observe=float(ratios[idx_t[0]]) if idx_t.size else None,
        seuil_premiere_avarie_observe=float(ratios[idx_o[0]]) if idx_o.size else None,
        ratio_plus_grand_saut_shed=r_jump,
        taille_plus_grand_saut_shed=d_jump,
        m_exterieur_moyen_r030=float(np.nanmean(m03)) if m03.size else None,
        m_exterieur_std_r030=float(np.nanstd(m03)) if m03.size else None,
        generateurs_a_99pct_r030=int(np.sum(frac_prod03 >= 0.99)),
        generateurs_sous_1pct_r030=int(np.sum(frac_prod03 <= 0.01)),
    )


def _save_npz(path: Path, cfg, ratios, rows, arrays) -> None:
    payload = {
        "ratios": ratios,
        "niveau_lignes": _niveau_lignes(cfg.reseau),
        "limites": cfg.limites,
        "puissance_max": cfg.puissance_max,
    }
    payload.update(arrays)

    # Les métriques scalaires importantes sont aussi dupliquées en vecteurs
    # pour une exploitation rapide depuis le notebook.
    for cle in (
        "P_D",
        "P_shed_initial",
        "P_shed_final",
        "P_served_initial",
        "P_served_final",
        "shed_sur_demande_initial",
        "shed_sur_demande_final",
        "shed_sur_servie_initial",
        "shed_sur_servie_final",
        "Mmax_initial",
        "Mmax_final_service",
        "n_lignes_saturees_initial",
        "n_lignes_tombees",
        "n_vagues_surcharge",
        "n_iterations",
    ):
        payload[cle] = np.asarray([r[cle] for r in rows])

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def figure_M_lignes(ratios, arrays, cible: float, numero: int, titre: str) -> None:
    idx = _index_ratio(ratios, cible)
    ratio = float(ratios[idx])
    m = arrays["M_final_carreras"][idx]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(np.arange(m.size), m, linewidth=1.0)
    ax.axhline(1.0, linestyle="--", linewidth=0.9)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("numéro de ligne")
    ax.set_ylabel(r"$M_{ij}=|F_{ij}|/F_{ij}^{\max}$")
    ax.set_title(f"{titre} — $P_D/P_C={ratio:.3f}$")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _sauvegarder(fig, f"05_carreras_fig{numero:02d}_M_r{int(round(cible*100)):03d}")


def figure_generateurs(cfg, ratios, arrays) -> None:
    idx = _index_ratio(ratios, 0.30)
    ratio = float(ratios[idx])
    frac = arrays["production_initiale"][idx] / cfg.puissance_max

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(np.arange(1, frac.size + 1), frac)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(np.arange(1, frac.size + 1))
    ax.set_xlabel("générateur")
    ax.set_ylabel(r"$P_i/P_i^{\max}$")
    ax.set_title(
        "Répartition de la production entre les 12 générateurs "
        f"— $P_D/P_C={ratio:.3f}$"
    )
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    _sauvegarder(fig, "05_carreras_fig04_generateurs_r030")


def figure_transitions(ratios, rows, taille: int) -> None:
    shed = np.array([r["shed_sur_demande_final"] for r in rows], dtype=float)
    outages = np.array([r["n_lignes_tombees"] for r in rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(7.7, 4.9))
    ax2 = ax1.twinx()

    ax1.plot(ratios, shed, linewidth=1.5, label=r"$P_{\rm shed}/P_D$")
    ax2.plot(
        ratios, outages, linewidth=1.2, linestyle="--",
        label="nombre de lignes tombées"
    )

    ax1.axvline(CIBLE_GENERATION, linestyle=":", linewidth=1.0)
    ax1.axvline(CIBLE_TRANSPORT, linestyle=":", linewidth=1.0)

    ax1.set_xlabel(r"$P_D/P_C$")
    ax1.set_ylabel(r"$P_{\rm shed}/P_D$")
    ax2.set_ylabel("nombre de lignes tombées")
    ax1.set_title(
        f"Transitions déterministes — arbre {taille} nœuds, $p_1=1$"
    )
    ax1.grid(alpha=0.2)

    lignes = ax1.get_lines()[:1] + ax2.get_lines()[:1]
    ax1.legend(lignes, [l.get_label() for l in lignes], frameon=False, loc="upper left")

    fig.tight_layout()
    _sauvegarder(fig, f"05_carreras_fig05_transitions_{taille}")


def figure_carte(ratios, arrays, numero: int, nom: str, titre: str) -> None:
    m = arrays["M_final_carreras"]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    im = ax.imshow(
        m,
        aspect="auto",
        interpolation="nearest",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        extent=[0, m.shape[1] - 1, ratios[-1], ratios[0]],
        origin="upper",
    )
    ax.set_xlabel("numéro de ligne")
    ax.set_ylabel(r"$P_D/P_C$")
    ax.set_title(titre)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$M_{ij}$")
    fig.tight_layout()
    _sauvegarder(fig, f"05_carreras_fig{numero:02d}_{nom}")


def sensibilite_p1(
    cfg,
    ratios: np.ndarray,
    valeurs_p1: tuple[float, ...],
    repetitions: int,
    graine: int,
):
    """
    Sensibilité de la transition au paramètre p1.

    p1=0 et p1=1 sont déterministes. Pour les valeurs intermédiaires, on
    effectue `repetitions` réalisations indépendantes par ratio et on conserve
    moyenne, écart-type et quantiles 10/50/90. Cela fournit plus d'information
    que la seule courbe historique, tout en permettant de tracer la médiane.
    """
    rows = []
    ss = np.random.SeedSequence(graine)
    graines = ss.spawn(len(valeurs_p1) * len(ratios) * repetitions)
    k = 0

    for p1 in valeurs_p1:
        for ratio in ratios:
            valeurs = []
            outages = []

            nrep = 1 if p1 in (0.0, 1.0) else repetitions
            for _ in range(nrep):
                seed_i = int(graines[k].generate_state(1, dtype=np.uint64)[0])
                k += 1
                res = simuler_point(cfg, float(ratio), p1=p1, graine=seed_i)
                pd = float(res.demande.sum())
                valeurs.append(res.delestage_total / pd if pd > 0 else 0.0)
                outages.append(res.lignes_tombees.size)

            a = np.asarray(valeurs, dtype=float)
            o = np.asarray(outages, dtype=float)
            rows.append({
                "p1": float(p1),
                "ratio_PD_PC": float(ratio),
                "n_repetitions": int(nrep),
                "shed_moyen": float(a.mean()),
                "shed_std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
                "shed_q10": float(np.quantile(a, 0.10)),
                "shed_mediane": float(np.quantile(a, 0.50)),
                "shed_q90": float(np.quantile(a, 0.90)),
                "outages_moyen": float(o.mean()),
                "outages_max": int(o.max()),
            })

    return rows


def figure_p1(rows, valeurs_p1) -> None:
    fig, ax = plt.subplots(figsize=(7.7, 4.9))

    for p1 in valeurs_p1:
        sub = [r for r in rows if np.isclose(r["p1"], p1)]
        x = np.asarray([r["ratio_PD_PC"] for r in sub])
        y = np.asarray([r["shed_mediane"] for r in sub])
        ax.plot(x, y, linewidth=1.4, label=fr"$p_1={p1:g}$")

        if p1 not in (0.0, 1.0):
            lo = np.asarray([r["shed_q10"] for r in sub])
            hi = np.asarray([r["shed_q90"] for r in sub])
            ax.fill_between(x, lo, hi, alpha=0.15)

    ax.axvline(CIBLE_TRANSPORT, linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"$P_D/P_C$")
    ax.set_ylabel(r"$P_{\rm shed}/P_D$")
    ax.set_title("Sensibilité de la transition à $p_1$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    _sauvegarder(fig, "05_carreras_fig11_sensibilite_p1")


def _metadata(args, cfg, audit: AuditDeterministe) -> dict:
    return {
        "experience": "diagnostics_deterministes_Carreras_2002",
        "taille": cfg.n_noeuds,
        "P_G_par_generateur": P_G_TABLE,
        "P_C": cfg.p_c,
        "p0_scan_deterministe": 0.0,
        "p1_figures_3_a_10": 1.0,
        "fluctuation_demande": "aucune; demande uniforme, facteur Carreras gamma=1",
        "ratio_min": args.ratio_min,
        "ratio_max": args.ratio_max,
        "pas": args.pas,
        "ratio_min_etendu": args.ratio_min_etendu,
        "ratio_max_etendu": args.ratio_max_etendu,
        "pas_etendu": args.pas_etendu,
        "ratios_cibles_figures": list(RATIOS_CIBLES),
        "p1_sensibilite": list(args.p1),
        "repetitions_p1_intermediaire": args.repetitions_p1,
        "graine": args.graine,
        "convention_lignes_tombees": (
            "M_final_service=NaN; M_final_carreras=0 uniquement pour affichage"
        ),
        "normalisations_sauvegardees": [
            "P_shed/P_D",
            "P_shed/P_served",
            "P_served/P_C",
        ],
        "audit": asdict(audit),
        "cibles_publiees_de_comparaison": {
            "transition_generation_PD_PC": 1.0,
            "transition_transport_PD_PC": 1.45,
            "M_lignes_exterieures_a_r_0_3": 0.601,
        },
        "limite_connue": (
            "Au-dessus de la seconde transition, Carreras indique que les "
            "solutions peuvent dépendre de l'implémentation numérique du "
            "solveur et de l'ordre d'application des contraintes."
        ),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--taille", type=int, choices=[46, 94, 190, 382], default=382)

    # Fig. 3-9 / 11 : plage proche des deux premières transitions.
    p.add_argument("--ratio-min", type=float, default=0.75)
    p.add_argument("--ratio-max", type=float, default=1.75)
    p.add_argument("--pas", type=float, default=0.005)

    # Fig. 10 : plage fortement étendue.
    p.add_argument("--ratio-min-etendu", type=float, default=0.75)
    p.add_argument("--ratio-max-etendu", type=float, default=8.0)
    p.add_argument("--pas-etendu", type=float, default=0.02)

    p.add_argument("--graine", type=int, default=GRAINE_DEFAUT)

    p.add_argument(
        "--p1",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 1.0],
        help="Valeurs de p1 pour la sensibilité analogue à la Fig. 11.",
    )
    p.add_argument(
        "--repetitions-p1",
        type=int,
        default=20,
        help="Répétitions par ratio pour les p1 intermédiaires.",
    )
    p.add_argument(
        "--sans-p1",
        action="store_true",
        help="Ne calcule pas la sensibilité à p1 / Fig. 11.",
    )
    return p


def main() -> None:
    args = parser().parse_args()

    if args.repetitions_p1 < 1:
        raise SystemExit("--repetitions-p1 doit être >= 1.")
    if any(not 0.0 <= p1 <= 1.0 for p1 in args.p1):
        raise SystemExit("Toutes les valeurs de --p1 doivent être dans [0,1].")

    cfg = configuration_arbre(args.taille)
    DOSSIER_DATA_DET.mkdir(parents=True, exist_ok=True)
    DOSSIER_FIG_DET.mkdir(parents=True, exist_ok=True)

    print("=== Diagnostics déterministes Carreras 2002 ===")
    print(f"taille        : {cfg.n_noeuds} nœuds")
    print(f"lignes        : {cfg.reseau.n_lignes}")
    print(f"générateurs   : {cfg.n_generateurs}")
    print(f"charges       : {cfg.n_charges}")
    print(f"P_C           : {cfg.p_c:.4f}")
    print("p0            : 0")
    print("p1 (Fig.3-10) : 1")
    print("fluctuations  : aucune")
    print()

    # ------------------------------------------------------------------
    # Balayage principal : Fig. 3 à 9.
    # On force l'inclusion des quatre ratios historiques exacts.
    # ------------------------------------------------------------------
    ratios = _ratios(args.ratio_min, args.ratio_max, args.pas)
    ratios = np.unique(np.concatenate([ratios, np.asarray(RATIOS_CIBLES)]))
    ratios.sort()

    print("--- Balayage principal ---")
    rows, arrays = balayer(
        cfg, ratios, p1=1.0, graine=args.graine
    )

    _ecrire_csv(DOSSIER_DATA_DET / f"scan_{args.taille}.csv", rows)
    _save_npz(
        DOSSIER_DATA_DET / f"scan_{args.taille}.npz",
        cfg, ratios, rows, arrays
    )

    audit = _audit(cfg, ratios, rows, arrays)

    # Figures homologues aux Fig. 3-9.
    figure_M_lignes(
        ratios, arrays, 0.30, 3,
        "Fraction de charge des lignes à faible demande"
    )
    figure_generateurs(cfg, ratios, arrays)
    figure_transitions(ratios, rows, args.taille)
    figure_M_lignes(
        ratios, arrays, 1.04, 6,
        "Lignes juste au-dessus de la limite de génération"
    )
    figure_M_lignes(
        ratios, arrays, 1.45, 7,
        "Lignes au voisinage de la limite de transport"
    )
    figure_M_lignes(
        ratios, arrays, 1.73, 8,
        "Lignes après la transition de transport"
    )
    figure_carte(
        ratios, arrays, 9, "carte_M",
        "Structure des solutions autour de la seconde transition"
    )

    # ------------------------------------------------------------------
    # Balayage étendu : Fig. 10.
    # ------------------------------------------------------------------
    print("\n--- Balayage étendu ---")
    ratios_ext = _ratios(
        args.ratio_min_etendu, args.ratio_max_etendu, args.pas_etendu
    )
    rows_ext, arrays_ext = balayer(
        cfg, ratios_ext, p1=1.0, graine=args.graine + 1
    )
    _ecrire_csv(
        DOSSIER_DATA_DET / f"scan_etendu_{args.taille}.csv",
        rows_ext
    )
    _save_npz(
        DOSSIER_DATA_DET / f"scan_etendu_{args.taille}.npz",
        cfg, ratios_ext, rows_ext, arrays_ext
    )
    figure_carte(
        ratios_ext, arrays_ext, 10, "carte_M_etendue",
        "Alternance des bandes de solutions sur une plage étendue"
    )

    # ------------------------------------------------------------------
    # Sensibilité p1 : Fig. 11.
    # ------------------------------------------------------------------
    if not args.sans_p1:
        print("\n--- Sensibilité à p1 ---")
        valeurs_p1 = tuple(float(v) for v in args.p1)
        rows_p1 = sensibilite_p1(
            cfg,
            ratios,
            valeurs_p1,
            repetitions=args.repetitions_p1,
            graine=args.graine + 2,
        )
        _ecrire_csv(
            DOSSIER_DATA_DET / f"sensibilite_p1_{args.taille}.csv",
            rows_p1
        )
        figure_p1(rows_p1, valeurs_p1)

    # ------------------------------------------------------------------
    # Métadonnées et résumé.
    # ------------------------------------------------------------------
    meta = _metadata(args, cfg, audit)
    (DOSSIER_DATA_DET / f"metadata_deterministe_{args.taille}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== Audit Carreras déterministe ===")
    print(f"seuil génération observé       : {audit.seuil_generation_observe}")
    print(f"cible publiée                  : {CIBLE_GENERATION:.2f}")
    print(f"seuil Mmax initial ≥ 0.99      : {audit.seuil_transport_initial_observe}")
    print(f"première avarie par surcharge  : {audit.seuil_premiere_avarie_observe}")
    print(f"cible seconde transition       : {CIBLE_TRANSPORT:.2f}")
    print(
        f"M extérieur moyen à r=0.30     : "
        f"{audit.m_exterieur_moyen_r030:.4f} "
        f"± {audit.m_exterieur_std_r030:.4f}"
        if audit.m_exterieur_moyen_r030 is not None else
        "M extérieur moyen à r=0.30     : n/a"
    )
    print(f"cible publiée pour M extérieur : {CIBLE_M_EXTERIEUR_R030:.3f}")
    print(
        f"générateurs ≥99% à r=0.30      : "
        f"{audit.generateurs_a_99pct_r030}/{cfg.n_generateurs}"
    )
    print(
        f"générateurs ≤1% à r=0.30       : "
        f"{audit.generateurs_sous_1pct_r030}/{cfg.n_generateurs}"
    )
    print(
        f"plus grand saut de P_shed/P_D  : "
        f"r={audit.ratio_plus_grand_saut_shed}, "
        f"Δ={audit.taille_plus_grand_saut_shed}"
    )

    print(f"\nDonnées : {DOSSIER_DATA_DET}")
    print(f"Figures : {DOSSIER_FIG_DET}")
    print(
        "\nLe run lourd déjà lancé n'est pas affecté : "
        "ce script est un sidecar indépendant."
    )


if __name__ == "__main__":
    main()
