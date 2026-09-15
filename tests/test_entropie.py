"""Validation du module `entropie`.

Critères du plan de travail : entropie de permutation basse sur un signal
périodique et proche du maximum sur un bruit non corrélé ; en multiéchelle,
croisement entre le bruit non corrélé et le bruit corrélé à longue portée.

Les séries sont volontairement courtes ici pour que la suite reste rapide. La
reproduction complète du résultat de Costa se fait par le script
`scripts/01_croisement_multiechelle.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade_entropy.entropie import (
    echantillon,
    granulariser,
    multiechelle,
    permutation,
)
from cascade_entropy.synthetiques import bruit_puissance, periodique

GRAINE = 20260915


def test_permutation_basse_sur_signal_periodique():
    """Un signal périodique n'utilise que quelques motifs ordinaux."""
    assert permutation(periodique(4096, periode=50)) < 0.6


def test_permutation_maximale_sur_bruit_non_correle():
    """Un bruit non corrélé utilise tous les motifs de façon équiprobable."""
    rng = np.random.default_rng(GRAINE)
    assert permutation(bruit_puissance(4096, 0.0, rng)) > 0.99


def test_permutation_discrimine_periodique_et_bruit():
    rng = np.random.default_rng(GRAINE)
    assert permutation(periodique(4096)) < permutation(bruit_puissance(4096, 0.0, rng))


def test_permutation_serie_trop_courte_rejetee():
    with pytest.raises(ValueError):
        permutation(np.array([1.0, 2.0]), dimension=5)


def test_granularisation_reduit_la_longueur():
    serie = np.arange(100, dtype=float)
    assert granulariser(serie, 1).size == 100
    assert granulariser(serie, 5).size == 20
    assert np.isclose(granulariser(serie, 2)[0], 0.5)


def test_sampen_plus_eleve_pour_bruit_non_correle_a_echelle_un():
    """À l'échelle 1, le bruit non corrélé est plus irrégulier que le 1/f."""
    rng = np.random.default_rng(GRAINE)
    blanc = echantillon(bruit_puissance(2048, 0.0, rng))
    rose = echantillon(bruit_puissance(2048, 1.0, rng))
    assert blanc > rose


def test_croisement_multiechelle():
    """Résultat de Costa : le bruit non corrélé s'effondre, le 1/f se maintient.

    C'est la justification méthodologique du recours au multiéchelle plutôt qu'à
    l'entropie brute : une forte irrégularité à une seule échelle n'est pas une
    forte complexité.
    """
    rng = np.random.default_rng(GRAINE)
    _, blanc = multiechelle(bruit_puissance(8192, 0.0, rng), echelles=10)
    _, rose = multiechelle(bruit_puissance(8192, 1.0, rng), echelles=10)

    assert blanc[0] > rose[0]           # à l'échelle 1, le bruit blanc domine
    assert blanc[-1] < rose[-1]         # à grande échelle, l'ordre s'inverse
    assert blanc[-1] < blanc[0] - 0.5   # le bruit blanc s'effondre
    assert abs(rose[-1] - rose[0]) < 0.4  # le 1/f se maintient
