"""Orchestration transversale B4 → disque B5/B6 → entrée intacte de C.

Ce module est extérieur aux fils A et B : aucun moteur ni module d'analyse
ne l'importe. Il réutilise leurs API et les contrôles B4, sans calcul de scores,
nouveau dispatch, cascade ou choix de paramètres scientifiques.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from . import attribution_n1 as b4
from . import persistance_n1 as p
from ._validation import entier_positif
from .centralites_lignes import EtatInitialIntact, GrapheLignes
from .n1 import ResultatRedispatchN1, composantes_actives
from .topologies_n1 import Topologie, vers_reseau


CLE = "integration_b4_c"
FORMAT = "cascade-entropy-integration-b4-c-v1"
FORMAT_V2 = "cascade-entropy-integration-b4-c-v2"
_SOURCES_DISPATCH = ("dispatch.py", "n1.py", "_optimisation_dispatch_n1.py", "attribution_n1.py")


@dataclass
class EtatInitialIntegre:
    """Valeurs relues, rôles ordonnés B4 et copie intacte indépendante pour C.

    ``etat_electrique`` permet un pilote N−1 avec ses demandes/Pmax d'origine.
    ``etat_initial`` est une copie protégée des flux et limites FINALS ; modifier
    ultérieurement le premier objet n'altère pas l'entrée prédictive de C.
    ``realisation`` conserve le manifeste et l'identité topologique B5/B6.
    """

    etat_initial: EtatInitialIntact
    etat_electrique: b4.EtatElectrique
    realisation: p.Realisation


def _egal(a, b, nom):
    if not np.array_equal(a, b):
        raise ValueError(f"Association ou valeur incohérente pour {nom}.")


def _valider(etat):
    """Contrôle le snapshot mutable B4, sans rejouer son optimisation.

    Les spécifications B4 sont matérialisées avec Valeurs.realiser ; le contrôle
    physique réutilise _controler de B4 (bornes, conservation et loi DC). Aucun
    calcul de flux/redispatch n'est réimplémenté ici.
    """
    if not isinstance(etat, b4.EtatElectrique):
        raise TypeError("Un EtatElectrique B4 est requis, jamais un résultat de cascade.")
    cfg, r = etat.configuration, etat.reseau
    try:
        if isinstance(cfg, dict) and cfg.get("departage") == "min_redispatch":
            raise ValueError("min_redispatch est réservé à la cascade : calculez l'état B4 "
                             "avec la règle departage_intact (highs ou min_charge_max).")
        entier_positif(r.reference, "reference réseau", minimum=0)
        for nom in ("lignes", "generateurs", "charges"):
            if np.asarray(getattr(r, nom)).dtype.kind not in "iu":
                raise ValueError(f"Identifiants entiers requis pour {nom}.")
        grandeurs = [r.reactances, r.limites, etat.demande, etat.puissance_max,
            etat.reserves_minimales, etat.reserves_disponibles, etat.injections, etat.angles]
        for sol in (etat.resultat, etat.resultat_reference):
            if sol is not None:
                grandeurs.extend((sol.production, sol.charge_servie, sol.delestage,
                                   sol.flux, sol.taux_de_charge))
        if any(np.asarray(a).dtype.kind not in "iuf" or not np.isfinite(a).all()
               for a in grandeurs):
            raise ValueError("Les grandeurs électriques doivent être réelles finies, non booléennes.")
        p._cles(cfg, ("topologie", "placement", "demande", "puissance_max",
            "reserves_minimales", "reactances", "capacites", "base_mva",
            "unite_puissance", "unite_reactance", "departage"), "configuration B4")
        if cfg["departage"] not in ("highs", "min_charge_max"):
            raise ValueError("Départage B4 incompatible avec l'intégration.")
        t = Topologie(**cfg["topologie"])
        if p._canonique(b4._json_copie(asdict(t))) != p._canonique(cfg["topologie"]):
            raise ValueError("La topologie B4 doit conserver les indices canoniques A3.")
        base = b4._reel(cfg["base_mva"], "base_mva", strict=True)
        if cfg["unite_puissance"] not in ("pu", "MW") or cfg["unite_reactance"] != "pu":
            raise ValueError("Unités B4 incompatibles.")
        facteur = base if cfg["unite_puissance"] == "MW" else 1.0
        placement = b4.Placement(**cfg["placement"])
        ref = entier_positif(placement.reference, "reference", minimum=0)
        attendu = vers_reseau(t,
            reactances=b4.Valeurs(**cfg["reactances"]).realiser(t.n_lignes),
            generateurs=placement.generateurs, charges=placement.charges,
            limites=r.limites, reference=ref)
        if r.n_noeuds != t.n_noeuds or r.reference != ref:
            raise ValueError("Nombre de nœuds ou référence B4 incohérent.")
        for nom in ("lignes", "reactances", "generateurs", "charges"):
            _egal(getattr(r, nom), getattr(attendu, nom), nom)
        for nom, taille in (("demande", len(r.charges)),
                            ("puissance_max", len(r.generateurs)),
                            ("reserves_minimales", len(r.generateurs))):
            valeurs = b4.Valeurs(**cfg[nom]).realiser(taille) / facteur
            _egal(getattr(etat, nom), valeurs, nom)
        if np.any(etat.reserves_minimales > etat.puissance_max):
            raise ValueError("Réserves minimales supérieures à Pmax.")
        cap = cfg["capacites"]
        p._cles(cap, ("mode", "valeurs", "marge", "plancher"), "capacites B4")
        provisoires = b4.Valeurs(**cap["valeurs"]).realiser(t.n_lignes) / facteur
        if cap["mode"] == "independantes":
            if (etat.resultat_reference is not None or cap["marge"] is not None
                    or cap["plancher"] is not None):
                raise ValueError("Référence inattendue pour capacités indépendantes.")
            _egal(r.limites, provisoires, "limites indépendantes")
        elif cap["mode"] == "flux_reference":
            if not isinstance(etat.resultat_reference, ResultatRedispatchN1):
                raise ValueError("Dispatch intact de référence B4 manquant.")
            reference = deepcopy(r)
            reference.limites = provisoires
            b4._controler(reference, etat.demande, etat.puissance_max, etat.resultat_reference)
            marge = b4._reel(cap["marge"], "marge", strict=True)
            plancher = b4._reel(cap["plancher"], "plancher", strict=True) / facteur
            _egal(r.limites, np.maximum(marge * abs(etat.resultat_reference.flux), plancher),
                  "limites figées depuis le dispatch de référence")
        else:
            raise ValueError("Stratégie de capacités B4 inconnue.")
        if not isinstance(etat.resultat, ResultatRedispatchN1):
            raise TypeError("Le résultat intact B4 doit être un ResultatRedispatchN1.")
        for sol in (etat.resultat, etat.resultat_reference):
            if sol is not None:
                entier_positif(sol.n_secours, "n_secours", minimum=0)
                if np.asarray(sol.hors_service).dtype.kind != "b":
                    raise ValueError("Les masques des dispatchs intacts doivent être booléens.")
        injection, theta = b4._controler(r, etat.demande, etat.puissance_max, etat.resultat)
        tol = 1e-8 * min(1.0, float(etat.demande.max()))
        for a, b, nom in ((etat.injections, injection, "injections"),
                          (etat.angles, theta, "angles"),
                          (etat.reserves_disponibles, etat.puissance_max - etat.resultat.production,
                           "réserves disponibles")):
            if np.shape(a) != np.shape(b) or not np.allclose(a, b, rtol=1e-7, atol=tol):
                raise ValueError(f"Grandeur B4 incohérente : {nom}.")
        satisfaites = np.asarray(etat.reserves_satisfaites)
        if satisfaites.dtype.kind != "b":
            raise ValueError("Le diagnostic de réserves doit être booléen.")
        _egal(satisfaites, etat.reserves_disponibles + 1e-8 + 1e-7 * etat.puissance_max
              >= etat.reserves_minimales, "diagnostic de réserves B4")
        if (type(etat.avertissements) is not tuple
                or any(not isinstance(x, str) or not x for x in etat.avertissements)):
            raise ValueError("Avertissements B4 invalides.")
        if np.any(etat.resultat.delestage > 0) and "delestage_initial" not in etat.avertissements:
            raise ValueError("Délestage intact non signalé dans les avertissements B4.")
        p._cles(etat.environnement, ("python", "numpy", "scipy"), "environnement B4")
        for nom, valeur in etat.environnement.items():
            p._texte(valeur, nom)
        # Le domaine connexe/simple de C est contrôlé avant toute écriture.
        graphe = GrapheLignes(r.n_noeuds, r.lignes, r.reactances)
        EtatInitialIntact(graphe, etat.resultat.flux, r.limites,
                          etat.resultat.hors_service, "validation-intact-final")
        return t
    except (KeyError, TypeError, AttributeError) as erreur:
        raise ValueError(f"Configuration ou objet B4 incompatible : {erreur}") from erreur


def _diagnostic(sol):
    if sol is None:
        return None
    return dict(statut_ilots=list(sol.statut_ilots), n_secours=sol.n_secours)


def _contrat(etat):
    format_integration = _format_integration(etat.configuration["departage"])
    contrat = dict(format=format_integration, configuration_b4=deepcopy(etat.configuration),
        environnement_b4=deepcopy(etat.environnement), avertissements=list(etat.avertissements),
        final=_diagnostic(etat.resultat), reference=_diagnostic(etat.resultat_reference))
    if format_integration == FORMAT_V2:
        contrat["sources_dispatch_sha256"] = {
            nom: sha256(Path(__file__).with_name(nom).read_bytes()).hexdigest()
            for nom in _SOURCES_DISPATCH}
    return contrat


def _format_integration(departage):
    return FORMAT if departage == "highs" else FORMAT_V2


def _electrique(etat):
    """Dilate vers les nœuds par identifiants ; les rangs conservent l'ordre G/C."""
    r, champs = etat.reseau, {}

    def ajouter(nom, valeurs, axe, unite, convention):
        champs[nom] = p.champ_numerique(valeurs, axes=axe, unite=unite, convention=convention)

    def nodal(nom, valeurs, indices):
        dense = np.zeros(r.n_noeuds, dtype=np.asarray(valeurs).dtype)
        dense[indices] = valeurs
        ajouter(nom, dense, ["noeud"], "1" if dense.dtype.kind == "b" else "pu",
                "valeur par nœud initial ; zéro hors du rôle")

    for nom, indices in (("generateur", r.generateurs), ("charge", r.charges)):
        rangs = np.full(r.n_noeuds, -1, dtype=np.int64)
        rangs[indices] = np.arange(len(indices))
        ajouter("rang_" + nom, rangs, ["noeud"], "1", "rang B4 dans le rôle ; -1 hors du rôle")
    for nom, indices in (("demande", r.charges), ("puissance_max", r.generateurs),
                         ("reserves_minimales", r.generateurs),
                         ("reserves_disponibles", r.generateurs),
                         ("reserves_satisfaites", r.generateurs)):
        nodal(nom, getattr(etat, nom), indices)
    for prefixe, sol in (("final_", etat.resultat), ("reference_", etat.resultat_reference)):
        if sol is None:
            continue
        for nom, indices in (("production", r.generateurs), ("charge_servie", r.charges),
                             ("delestage", r.charges)):
            nodal(prefixe + nom, getattr(sol, nom), indices)
        for nom, unite in (("flux", "pu"), ("taux_de_charge", "1"), ("hors_service", "1")):
            ajouter(prefixe + nom, getattr(sol, nom), ["ligne"], unite,
                    prefixe + "dispatch intact B4 ; indices A3")
    for nom, a, axe, unite in (("reactances", r.reactances, "ligne", "pu"),
        ("limites", r.limites, "ligne", "pu"), ("injections", etat.injections, "noeud", "pu"),
        ("angles", etat.angles, "noeud", "rad")):
        ajouter(nom, a, [axe], unite, "état intact final ; base commune B4")
    return dict(base_puissance={"valeur": etat.configuration["base_mva"], "unite": "MVA"},
        conventions={"format": _format_integration(etat.configuration["departage"]),
                     "orientation": "u vers v ; F=b(theta_u-theta_v)",
                     "injections": "G-L_servie", "puissances": "pu sur la base MVA déclarée"},
        champs=champs)


def sauver_etat_initial(etat: b4.EtatElectrique, dossier, *, statut, arguments_generation,
                        revision_code, arbre_modifie, parametres_experience):
    """Persiste un snapshot B4 contrôlé dans un NOUVEAU dossier B5/B6.

    La clé réservée integration_b4_c conserve configuration, graines, versions,
    avertissements et diagnostics des deux dispatchs distincts. Aucun filtre
    scientifique d'admission des intacts délestés n'est appliqué. Le retour est
    l'ID topologique ; charger_etat_initial construit ensuite l'ID d'état de C.
    """
    if not isinstance(etat, b4.EtatElectrique):
        raise TypeError("Un EtatElectrique B4 est requis.")
    if not isinstance(parametres_experience, dict) or CLE in parametres_experience:
        raise ValueError(f"Dictionnaire d'expérience requis ; clé {CLE} réservée.")
    e = deepcopy(etat)
    t = _valider(e)
    experience = deepcopy(parametres_experience)
    experience[CLE] = _contrat(e)
    return p.sauver_realisation(t, dossier, statut=statut,
        arguments_generation=arguments_generation, revision_code=revision_code,
        arbre_modifie=arbre_modifie, parametres_experience=experience, electrique=_electrique(e))


def charger_etat_initial(dossier) -> EtatInitialIntegre:
    """Relit B5/B6, valide le contrat transversal et fournit l'intact FINAL à C.

    Ne relance ni B4, ni solveur de dispatch, ni calcul de scores. L'identifiant
    est lié à l'empreinte du manifeste complet, donc distingue les variantes
    électriques d'une topologie. L'intégrité n'est pas une preuve d'authenticité.
    """
    realisation = p.charger_realisation(dossier)
    try:
        contrat = realisation.metadata["parametres_experience"][CLE]
        format_integration = contrat["format"]
        if format_integration not in (FORMAT, FORMAT_V2):
            raise ValueError("Version d'intégration B4/C non prise en charge.")
        cles = ("format", "configuration_b4", "environnement_b4", "avertissements", "final", "reference")
        if format_integration == FORMAT_V2:
            cles += ("sources_dispatch_sha256",)
        p._cles(contrat, cles, "contrat transversal")
        if format_integration == FORMAT_V2:
            sources = contrat["sources_dispatch_sha256"]
            p._cles(sources, _SOURCES_DISPATCH, "sources_dispatch_sha256")
            for valeur in sources.values():
                if (not isinstance(valeur, str) or len(valeur) != 64
                        or any(c not in "0123456789abcdef" for c in valeur)):
                    raise ValueError("Empreinte de source de dispatch SHA-256 invalide.")
        if type(contrat["avertissements"]) is not list:
            raise ValueError("Liste explicite d'avertissements B4 requise.")
        cfg = deepcopy(contrat["configuration_b4"])
        if p._canonique(b4._json_copie(asdict(realisation.topologie))) != p._canonique(cfg["topologie"]):
            raise ValueError("Définition topologique différente de la configuration B4.")
        champs = realisation.metadata["electrique"]["champs"]
        lire = lambda nom: p.tableau_champ(champs[nom])
        placement = b4.Placement(**cfg["placement"])
        r = vers_reseau(realisation.topologie, reactances=lire("reactances"),
            limites=lire("limites"), generateurs=placement.generateurs,
            charges=placement.charges, reference=placement.reference)
        ilots = composantes_actives(r, np.zeros(r.n_lignes, dtype=bool))

        def solution(prefixe, diagnostic):
            if diagnostic is None:
                return None
            p._cles(diagnostic, ("statut_ilots", "n_secours"), "diagnostic intact")
            entier_positif(diagnostic["n_secours"], "n_secours", minimum=0)
            return ResultatRedispatchN1(lire(prefixe + "hors_service"), deepcopy(ilots),
                lire(prefixe + "production")[r.generateurs],
                lire(prefixe + "charge_servie")[r.charges],
                lire(prefixe + "delestage")[r.charges],
                lire(prefixe + "flux"), lire(prefixe + "taux_de_charge"),
                deepcopy(diagnostic["statut_ilots"]), diagnostic["n_secours"])

        e = b4.EtatElectrique(r, lire("demande")[r.charges], lire("puissance_max")[r.generateurs],
            lire("reserves_minimales")[r.generateurs], solution("final_", contrat["final"]),
            lire("injections"), lire("angles"), lire("reserves_disponibles")[r.generateurs],
            lire("reserves_satisfaites")[r.generateurs], tuple(contrat["avertissements"]),
            cfg, deepcopy(contrat["environnement_b4"]), solution("reference_", contrat["reference"]))
        _valider(e)
        if format_integration != _format_integration(cfg["departage"]):
            raise ValueError("Version d'intégration B4/C incompatible avec le départage.")
        # Reconstitution du contrat : refuse les unités/axes/dtypes/rangs altérés,
        # les valeurs hors des rôles et les champs électriques manquants/inattendus.
        if p._canonique(_electrique(e)) != p._canonique(realisation.metadata["electrique"]):
            raise ValueError("Champs électriques incompatibles avec le contrat B4/C.")
        g = GrapheLignes(r.n_noeuds, r.lignes, r.reactances)
        identifiant = realisation.id_reseau + ":" + realisation.metadata["sha256_metadata"]
        initial = EtatInitialIntact(g, e.resultat.flux, r.limites, e.resultat.hors_service, identifiant)
        return EtatInitialIntegre(initial, e, realisation)
    except (KeyError, TypeError, AttributeError, IndexError) as erreur:
        raise ValueError(f"Réalisation incompatible avec l'intégration B4/C : {erreur}") from erreur
