"""Contrôles des rangs appariés et des bornes des classes de gravité."""
import csv
import hashlib
import json
from math import sqrt
from types import SimpleNamespace

import pytest

from scripts import comparer_regles_departage_n1 as script
from scripts import diagnostic_departage_n1 as diagnostic


def ecrire(chemin, lignes):
    with chemin.open("w", encoding="utf-8", newline="") as fichier:
        writer = csv.writer(fichier)
        writer.writerow(("reseau", "regle", "retrait", "fraction_incrementale"))
        writer.writerows(lignes)


def lire(chemin):
    with chemin.open(encoding="utf-8", newline="") as fichier:
        return list(csv.DictReader(fichier))


def test_comparer_scores_et_ex_aequo_a_la_main(tmp_path):
    dossier = tmp_path / "synthetique"
    dossier.mkdir()
    source = dossier / "retraits.csv"
    fractions = {"a": [0, 5e-10, .05, .30, -1.1e-9, .02],
                 "b": [.20, -5e-10, .30, .05, -2e-9, .02]}
    ecrire(source, [("r", regle, i, fraction) for regle, valeurs in fractions.items()
                     for i, fraction in reversed(list(enumerate(valeurs)))]
        + [("r2", regle, 0, 0) for regle in fractions])
    assert script.main(["--run-id", "synthetique", "--sortie", str(tmp_path)]) == 0
    ligne, constante = lire(dossier / "spearman_regles.csv")
    # Rangs traités : (2.5, 2.5, 5, 6, 1, 4) et (5, 2, 6, 4, 1, 3).
    assert float(ligne["spearman"]) == pytest.approx(11 / sqrt(17 * 17.5))
    assert float(ligne["spearman_brut"]) == pytest.approx(19 / 35)
    assert [ligne[nom] for nom in ("n_retraits", "n_nuls_a", "n_nuls_b",
            "n_ex_aequo_a", "n_ex_aequo_b", "n_ex_aequo_brut_a", "n_ex_aequo_brut_b")] == [
                "6", "2", "1", "2", "0", "0", "0"]
    assert constante["spearman"] == constante["spearman_brut"] == ""

    gravite = lire(dossier / "repartition_gravite.csv")
    assert [(g["reseau"], g["regle"]) for g in gravite] == [
        ("", "a"), ("", "b"), ("r", "a"), ("r", "b"), ("r2", "a"), ("r2", "b")]
    a = gravite[2]
    classes = ("nulle", "moins_5pct", "de_5_a_30pct", "au_moins_30pct", "residu_negatif_hors_tolerance")
    assert [int(a[f"n_{classe}"]) for classe in classes] == [2, 1, 1, 1, 1]
    assert sum(float(a[f"part_{classe}"]) for classe in classes) == pytest.approx(1.)
    assert float(a["mediane_non_nulle"]) == pytest.approx(.035)
    assert gravite[0]["n_retraits"] == "7" and gravite[0]["n_nulle"] == "3"
    assert gravite[4]["mediane_non_nulle"] == ""
    provenance = json.loads((dossier / "comparaison_metadata.json").read_text(encoding="utf-8"))
    assert provenance["sha256_source"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert provenance["parametres"]["decimales_rangs"] == 9
    assert provenance["environnement"]["scipy"]


@pytest.mark.parametrize("retrait, fraction, erreur", [
    (0, 0., "Retrait répété"), (1, 0., "Retraits différents"), (1, "nan", "fraction non finie"),
])
def test_retraits_non_appariables(tmp_path, retrait, fraction, erreur):
    source = tmp_path / "retraits.csv"
    ecrire(source, [("r", "a", 0, .5), ("r", "b", 0, 0.), ("r", "a", retrait, fraction)])
    with pytest.raises(ValueError, match=erreur):
        script.lire_retraits(source)


def test_prorata_inadmissible_sans_classification(tmp_path, monkeypatch):
    e = SimpleNamespace(reseau=SimpleNamespace(n_lignes=1), demande=None, puissance_max=None)
    prorata = SimpleNamespace(resultat=SimpleNamespace(taux_de_charge=[1.25]),
                             faisable=False, violations=("limites_transport",))
    var = SimpleNamespace(charge_max_minimale=[.6], n_flux_non_uniques=0)
    monkeypatch.setattr(diagnostic.D, "etat_intact_prorata", lambda *args: prorata)

    def classification_interdite(*args):
        pytest.fail("Le témoin infaisable ne doit pas être classé.")

    monkeypatch.setattr(diagnostic.D, "lignes_saturees", classification_interdite)
    monkeypatch.setattr(diagnostic.D, "classer_saturations", classification_interdite)
    contenu = diagnostic.mesurer("r", e, "prorata", var, var)
    diagnostic._sauver(tmp_path / "prorata.json", contenu)
    diagnostic.sauver_csv(tmp_path / "comparatif.csv", [contenu["resume"]])
    resume = json.loads((tmp_path / "prorata.json").read_text(encoding="utf-8"))["resume"]
    ligne, = lire(tmp_path / "comparatif.csv")
    assert resume["prorata_faisable"] is resume["classification_intacte_applicable"] is False
    assert resume["prorata_violations"] == ligne["prorata_violations"] == "limites_transport"
    assert ligne["classification_intacte_applicable"] == "False"
    for cle in ("n_saturees_intact", "n_evitables_intact"):
        assert resume[cle] is None and ligne[cle] == ""
