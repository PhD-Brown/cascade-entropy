"""
Chemins canoniques du projet.

Centralise où vivent les données simulées et les figures, pour qu'un notebook
lancé depuis `notebooks/` et un script lancé depuis `scripts/` écrivent au
même endroit --- la racine du dépôt --- plutôt que chacun dans une copie
locale à son propre dossier de travail. C'est précisément ce qui a causé le
problème cette semaine : `Path("data")` dans un notebook dépend d'où Jupyter
place son répertoire de travail, qui n'est pas forcément la racine du dépôt.

À importer en tout premier, avant toute écriture de figure ou de donnée :

    from cascade_entropy.chemins import DOSSIER_DATA, DOSSIER_FIGURES
"""

from __future__ import annotations

from pathlib import Path


def racine_projet(marqueur: str = "cascade_entropy") -> Path:
    """
    Remonte depuis le dossier courant jusqu'à trouver la racine du dépôt.

    La racine est reconnue par la présence d'un dossier `cascade_entropy/`
    (le package lui-même) --- un repère stable, qui existe quel que soit le
    sous-dossier depuis lequel le notebook ou le script est lancé.
    """
    depart = Path.cwd()
    for candidat in (depart, *depart.parents):
        if (candidat / marqueur).is_dir():
            return candidat
    raise RuntimeError(
        f"Racine du projet introuvable : aucun dossier '{marqueur}' trouvé "
        f"en remontant depuis {depart}. Le notebook est-il bien lancé "
        f"quelque part à l'intérieur du dépôt cascade-entropy ?"
    )


RACINE = racine_projet()
DOSSIER_DATA = RACINE / "data"
DOSSIER_FIGURES = RACINE / "figures"

# Créés une fois pour toutes à l'import : aucun notebook n'a besoin de s'en
# souvenir ni de vérifier leur existence avant d'écrire dedans.
DOSSIER_DATA.mkdir(parents=True, exist_ok=True)
DOSSIER_FIGURES.mkdir(parents=True, exist_ok=True)
