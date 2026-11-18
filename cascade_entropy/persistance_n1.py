"""Persistance non destructive des réalisations synthétiques B5/B6.

La topologie reste au format de ``topologies_n1.sauver_json``. Le manifeste
ajoute provenance, reproduction et un conteneur électrique optionnel, sans
importer les modules B4/C. Voir ``docs/persistance_n1.md`` pour le contrat v1.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import inspect
import json
import math
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import numpy as np

from ._validation import entier_positif
from . import topologies_n1 as topo


FORMAT = "cascade-entropy-realisation-v1"
GENERATEURS = {
    nom: getattr(topo, nom)
    for nom in ("anneau_cordes", "grille", "gilbert", "geometrique", "arbre_temoin")
}
STATUTS = ("demonstration", "validation", "campagne_scientifique")
PARAMETRES_REELS = {"gilbert": "p", "geometrique": "rayon"}


@dataclass
class Realisation:
    """Topologie et manifeste relus, indépendants des objets fournis à l'écriture."""

    topologie: topo.Topologie
    metadata: dict

    @property
    def id_reseau(self) -> str:
        return self.metadata["id_reseau"]


def _texte(valeur, nom):
    if not isinstance(valeur, str) or not valeur.strip():
        raise ValueError(f"{nom} doit être une chaîne non vide.")
    return valeur


def _json_strict(valeur):
    """Refuse pertes implicites de types, clés converties, NaN et infinis."""
    if valeur is None or type(valeur) in (str, bool, int):
        return
    if type(valeur) is float and math.isfinite(valeur):
        return
    if type(valeur) is list:
        for item in valeur:
            _json_strict(item)
        return
    if type(valeur) is dict and all(type(cle) is str for cle in valeur):
        for item in valeur.values():
            _json_strict(item)
        return
    raise ValueError("Valeur non JSON stricte (types natifs finis et clés texte requis).")


def _canonique(valeur):
    _json_strict(valeur)
    return json.dumps(valeur, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _empreinte(valeur):
    return hashlib.sha256(_canonique(valeur)).hexdigest()


def _cles(d, attendues, nom):
    if not isinstance(d, dict) or set(d) != set(attendues):
        raise ValueError(f"Champs invalides pour {nom} : {', '.join(attendues)} requis.")


def _sans_doublons(paires):
    d = {}
    for cle, valeur in paires:
        if cle in d:
            raise ValueError(f"Clé JSON répétée : {cle}.")
        d[cle] = valeur
    return d


def _lire_json(chemin):
    try:
        d = json.loads(Path(chemin).read_text(encoding="utf-8"),
                       object_pairs_hook=_sans_doublons)
        _json_strict(d)
        return d
    except (UnicodeError, ValueError) as erreur:
        raise ValueError(f"JSON invalide dans {chemin} : {erreur}") from erreur


def champ_numerique(valeurs, *, axes, unite, convention):
    """Décrit un scalaire/tableau fini sans perdre son dtype ni son ordre.

    Chaque axe vaut ``noeud`` ou ``ligne`` et suit les identifiants d'origine.
    Un scalaire a ``axes=[]``. Un masque nodal peut représenter des rôles sans
    imposer la représentation interne de B4. Les unités sont toujours explicites.
    """
    a = np.asarray(valeurs)
    if (a.dtype.kind not in "bifu" or a.dtype.itemsize > 8
            or not np.all(np.isfinite(a))):
        raise ValueError("Un tableau réel fini de type bool/entier/flottant <= 64 bits est requis.")
    d = {"valeurs": a.tolist(), "dtype": a.dtype.str, "axes": axes,
         "unite": unite, "convention": convention}
    tableau_champ(d)
    return d


def tableau_champ(champ):
    """Reconstruit une copie NumPy d'un champ ; refuse toute conversion avec perte."""
    _cles(champ, ("valeurs", "dtype", "axes", "unite", "convention"), "champ numérique")
    _json_strict(champ)
    _texte(champ["unite"], "unite")
    _texte(champ["convention"], "convention")
    axes = champ["axes"]
    if type(axes) is not list or any(axe not in ("noeud", "ligne") for axe in axes):
        raise ValueError("axes doit être une liste d'axes noeud/ligne.")
    try:
        dtype = np.dtype(_texte(champ["dtype"], "dtype"))
        if dtype.kind not in "bifu" or dtype.itemsize > 8:
            raise ValueError("dtype numérique <= 64 bits requis.")
        a = np.asarray(champ["valeurs"], dtype=dtype)
    except (TypeError, OverflowError, ValueError) as erreur:
        raise ValueError(f"Champ numérique invalide : {erreur}") from erreur
    if (a.ndim != len(axes) or not np.all(np.isfinite(a))
            or _canonique(a.tolist()) != _canonique(champ["valeurs"])):
        raise ValueError("Forme, valeurs ou dtype incohérents dans le champ numérique.")
    return a.copy()


def _valider_electrique(d, t):
    if d is None:
        return
    _cles(d, ("base_puissance", "conventions", "champs"), "electrique")
    _cles(d["base_puissance"], ("valeur", "unite"), "base_puissance")
    base = d["base_puissance"]
    if (type(base["valeur"]) not in (int, float) or base["valeur"] <= 0
            or not math.isfinite(base["valeur"]) or base["unite"] != "MVA"):
        raise ValueError("Base de puissance finie strictement positive, en MVA, requise.")
    for nom in ("conventions", "champs"):
        if not isinstance(d[nom], dict) or not d[nom]:
            raise ValueError(f"electrique.{nom} doit être un dictionnaire non vide.")
    for nom, champ in d["champs"].items():
        _texte(nom, "nom du champ")
        a = tableau_champ(champ)
        forme = tuple(t.n_noeuds if axe == "noeud" else t.n_lignes
                      for axe in champ["axes"])
        if a.shape != forme:
            raise ValueError(f"Forme du champ {nom} incompatible avec les identifiants : {forme}.")


def _reel_generation(famille, valeur):
    """Valide uniquement p/rayon, sans réécrire leur représentation JSON.

    Les générateurs convertissent ces deux grandeurs en float. Un entier doit
    donc y être exactement représentable ; aucun arrondi supplémentaire n'est
    admis par la persistance. Les booléens ne sont jamais des grandeurs réelles.
    """
    try:
        if type(valeur) not in (int, float):
            raise ValueError("Paramètre réel natif requis (pas un booléen).")
        reel = float(valeur)
        if not math.isfinite(reel) or reel != valeur:
            raise ValueError("Paramètre réel non représentable exactement en float fini.")
    except OverflowError as erreur:
        raise ValueError("Paramètre réel non représentable exactement en float fini.") from erreur
    if reel <= 0 or (famille == "gilbert" and reel > 1):
        raise ValueError("Paramètre réel hors du domaine du générateur connexe.")
    return reel


def _parametres_egaux(famille, a, b):
    """Égalité numérique ciblée ; tous les autres champs restent typés stricts."""
    if a.keys() != b.keys():
        return False
    return all(
        _reel_generation(famille, a[nom]) == _reel_generation(famille, b[nom])
        if nom == PARAMETRES_REELS.get(famille)
        else _canonique(a[nom]) == _canonique(b[nom])
        for nom in a
    )


def _arguments(famille, arguments):
    if famille not in GENERATEURS:
        raise ValueError(f"Famille non prise en charge : {famille}.")
    if not isinstance(arguments, dict):
        raise ValueError("Les arguments de génération doivent former un dictionnaire.")
    _json_strict(arguments)
    try:
        liaison = inspect.signature(GENERATEURS[famille]).bind(**arguments)
    except TypeError as erreur:
        raise ValueError(f"Arguments de génération invalides : {erreur}") from erreur
    liaison.apply_defaults()
    if famille in PARAMETRES_REELS:
        _reel_generation(famille, liaison.arguments[PARAMETRES_REELS[famille]])
    return dict(liaison.arguments)


def verifier_reproduction(realisation):
    """Régénère une seule topologie et exige l'identité, y compris positions/essais.

    Contrôle technique, sans flux DC ni campagne. L'écriture appelle ce contrôle ;
    la lecture reste possible avec une autre version, sans régénération implicite.
    """
    t = realisation.topologie
    r = realisation.metadata["reproduction"]
    arguments = _arguments(t.famille, r["arguments"])
    autre = GENERATEURS[t.famille](**arguments)
    if (autre.famille != t.famille or autre.n_noeuds != t.n_noeuds
            or autre.graine != t.graine or autre.essais != t.essais
            or not _parametres_egaux(t.famille, autre.parametres, t.parametres)
            or not np.array_equal(autre.lignes, t.lignes)
            or not np.array_equal(autre.positions, t.positions)):
        raise ValueError("La régénération ne reproduit pas exactement la topologie fournie.")


def _charger_topologie(chemin):
    """Réutilise le lecteur v1 et interdit sa normalisation implicite des indices."""
    d = _lire_json(chemin)
    _cles(d, ("format", "famille", "n_noeuds", "lignes", "graine", "parametres",
              "essais", "positions", "statistiques"), "topologie v1")
    try:
        t = topo.charger_json(chemin)
    except (TypeError, KeyError, OverflowError) as erreur:
        raise ValueError(f"Topologie invalide : {erreur}") from erreur
    if d["lignes"] != t.lignes.tolist():
        raise ValueError("L'ordre et l'orientation canoniques des lignes doivent être conservés.")
    if (_canonique(d["statistiques"]) != _canonique(topo.statistiques(t))
            or _canonique(d["positions"]) != _canonique(
                None if t.positions is None else t.positions.tolist())):
        raise ValueError("Statistiques ou positions non conformes au format v1.")
    if not isinstance(d["parametres"], dict):
        raise ValueError("parametres doit être un dictionnaire.")
    return t, d


def _valider_metadata(d, t, definition):
    _cles(d, ("format", "id_reseau", "statut", "noeuds", "lignes", "reproduction",
              "provenance", "parametres_experience", "electrique", "sha256_topologie",
              "sha256_metadata"), "metadata")
    if d["format"] != FORMAT:
        raise ValueError("Version du manifeste non prise en charge.")
    if d["statut"] not in STATUTS:
        raise ValueError(f"Statut requis parmi {STATUTS}.")
    if (_canonique(d["noeuds"]) != _canonique(list(range(t.n_noeuds)))
            or _canonique(d["lignes"]) != _canonique(list(range(t.n_lignes)))):
        raise ValueError("Identifiants de nœuds/lignes incohérents.")
    if d["id_reseau"] != "n1-" + _empreinte(definition):
        raise ValueError("Identifiant réseau incohérent avec la définition topologique.")
    r = d["reproduction"]
    _cles(r, ("generateur", "arguments", "alea"), "reproduction")
    if r["generateur"] != "cascade_entropy.topologies_n1." + t.famille:
        raise ValueError("Générateur incohérent avec la famille.")
    if _canonique(_arguments(t.famille, r["arguments"])) != _canonique(r["arguments"]):
        raise ValueError("Tous les arguments de génération doivent être enregistrés.")
    arguments = r["arguments"]
    attendus = dict(t.parametres)
    if t.famille != "grille":
        attendus["n_noeuds"] = t.n_noeuds
    if t.graine is not None:
        attendus["graine"] = t.graine
    for nom, valeur in attendus.items():
        if nom not in arguments or not _parametres_egaux(
                t.famille, {nom: arguments[nom]}, {nom: valeur}):
            raise ValueError(f"Argument {nom} incohérent avec la topologie.")
    if t.famille == "grille" and arguments["n_lignes"] * arguments["n_colonnes"] != t.n_noeuds:
        raise ValueError("Dimensions de grille incohérentes.")
    if "max_essais" in arguments:
        maximum = entier_positif(arguments["max_essais"], "max_essais")
        if maximum < t.essais:
            raise ValueError("max_essais inférieur au nombre d'essais enregistré.")
    _texte(r["alea"], "alea")
    p = d["provenance"]
    _cles(p, ("revision_code", "arbre_modifie", "versions", "plateforme", "sources_sha256"), "provenance")
    _texte(p["revision_code"], "revision_code")
    if type(p["arbre_modifie"]) is not bool:
        raise ValueError("arbre_modifie doit être booléen.")
    _cles(p["versions"], ("python", "numpy", "scipy", "cascade-entropy"), "versions")
    for nom, valeur in p["versions"].items():
        _texte(valeur, nom)
    _texte(p["plateforme"], "plateforme")
    sources = ("persistance_n1.py", "topologies_n1.py", "reseau.py", "_validation.py")
    _cles(p["sources_sha256"], sources, "sources_sha256")
    for valeur in p["sources_sha256"].values():
        if (not isinstance(valeur, str) or len(valeur) != 64
                or any(c not in "0123456789abcdef" for c in valeur)):
            raise ValueError("Empreinte de source SHA-256 invalide.")
    if not isinstance(d["parametres_experience"], dict):
        raise ValueError("parametres_experience doit être un dictionnaire explicite.")
    _valider_electrique(d["electrique"], t)
    if d["statut"] == "campagne_scientifique" and d["electrique"] is None:
        raise ValueError("Une réalisation de campagne doit inclure les propriétés électriques.")


def sauver_realisation(t, dossier, *, statut, arguments_generation, revision_code,
                       arbre_modifie, parametres_experience, electrique=None):
    """Sauve une réalisation dans un NOUVEAU dossier et renvoie son ID réseau.

    Les arguments effectifs (défauts inclus) et les versions sont figés. La révision
    et l'état de l'arbre de travail sont déclarés par l'appelant. Les paramètres
    d'expérience incluent les autres graines/conventions lorsqu'ils existent.
    Les validations précèdent la création du dossier cible. Celui-ci est réservé
    par mkdir exclusif, puis metadata.json est publié en dernier. Une erreur E/S
    laisse éventuellement un dossier incomplet, jamais une réalisation lisible.
    """
    if not isinstance(t, topo.Topologie):
        raise ValueError("Une Topologie est requise.")
    # Le sérialiseur existant est l'unique producteur de la définition v1.
    with TemporaryDirectory(prefix="cascade-n1-") as temporaire:
        chemin = Path(temporaire) / "topologie.json"
        topo.sauver_json(t, chemin)
        topologie, definition = _charger_topologie(chemin)
        contenu = chemin.read_bytes()
    arguments = _arguments(topologie.famille, arguments_generation)
    d = {
        "format": FORMAT,
        "id_reseau": "n1-" + _empreinte(definition),
        "statut": statut,
        "noeuds": list(range(topologie.n_noeuds)),
        "lignes": list(range(topologie.n_lignes)),
        "reproduction": {
            "generateur": "cascade_entropy.topologies_n1." + topologie.famille,
            "arguments": arguments,
            "alea": "numpy.random.default_rng/PCG64" if topologie.graine is not None
                    else "deterministe_sans_alea",
        },
        "provenance": {
            "revision_code": revision_code, "arbre_modifie": arbre_modifie,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": version("scipy"), "cascade-entropy": version("cascade-entropy")},
            "plateforme": platform.platform(),
            "sources_sha256": {
                nom: hashlib.sha256(Path(__file__).with_name(nom).read_bytes()).hexdigest()
                for nom in ("persistance_n1.py", "topologies_n1.py", "reseau.py", "_validation.py")
            },
        },
        "parametres_experience": parametres_experience,
        "electrique": electrique,
        "sha256_topologie": hashlib.sha256(contenu).hexdigest(),
        "sha256_metadata": "",
    }
    _json_strict(d)
    _valider_metadata(d, topologie, definition)
    verifier_reproduction(Realisation(topologie, d))
    d["sha256_metadata"] = _empreinte({k: v for k, v in d.items() if k != "sha256_metadata"})
    texte = json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    cible = Path(dossier)
    cible.mkdir(parents=True, exist_ok=False)
    with (cible / "topologie.json").open("xb") as fichier:
        fichier.write(contenu)
    # Le remplacement atomique empêche un lecteur de voir un manifeste tronqué.
    provisoire = cible / "metadata.json.tmp"
    with provisoire.open("x", encoding="utf-8", newline="\n") as fichier:
        fichier.write(texte)
    provisoire.replace(cible / "metadata.json")
    return d["id_reseau"]


def charger_realisation(dossier):
    """Relit et contrôle intégrité, format, indices et tableaux, sans simulation.

    Les erreurs E/S gardent leur type (FileNotFoundError, PermissionError, etc.).
    Une corruption, version inconnue ou entrée incohérente produit ValueError.
    Les empreintes détectent l'altération accidentelle ; ce ne sont pas des signatures.
    """
    cible = Path(dossier)
    d = _lire_json(cible / "metadata.json")
    if not isinstance(d, dict) or d.get("format") != FORMAT:
        raise ValueError("Version du manifeste non prise en charge.")
    attendu = _empreinte({k: v for k, v in d.items() if k != "sha256_metadata"})
    if attendu != d.get("sha256_metadata"):
        raise ValueError("Empreinte du manifeste incohérente.")
    contenu = (cible / "topologie.json").read_bytes()
    if hashlib.sha256(contenu).hexdigest() != d.get("sha256_topologie"):
        raise ValueError("Empreinte du fichier topologique incohérente.")
    t, definition = _charger_topologie(cible / "topologie.json")
    _valider_metadata(d, t, definition)
    return Realisation(t, d)
