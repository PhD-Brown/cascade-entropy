"""
Séries de substitution pour tester une mesure contre une hypothèse nulle (fil B).

Une mesure d'entropie calculée sur une série ne dit rien seule : il faut savoir
ce qu'elle vaudrait si la série n'avait PAS la structure que l'on cherche. On
construit pour cela des séries de substitution (« surrogates », Theiler et al.
1992) qui conservent certaines propriétés de l'original et en détruisent
d'autres. Deux hypothèses nulles servent au projet :

1. **Mélange** (permutation aléatoire des jours, `rng.permutation`) : conserve
   exactement la distribution des valeurs, détruit tout ordre temporel. Une
   mesure égale à celle des mélanges ne voit AUCUNE structure temporelle.
   C'est la référence déjà utilisée pour Hurst (`analyse_series`).

2. **IAAFT** (« iterative amplitude adjusted Fourier transform », Schreiber et
   Schmitz 1996, Phys. Rev. Lett. 77, 635) : conserve exactement la
   distribution des valeurs ET, à une petite tolérance près, le spectre de
   puissance (donc toute l'autocorrélation linéaire), mais tire les phases de
   Fourier au hasard. Une mesure égale à celle des substituts IAAFT ne voit
   rien de plus que le spectre : rien de plus que ce qu'exprime l'exposant de
   Hurst ou toute autre mesure de corrélation linéaire. C'est le test de la
   seconde moitié de la problématique (information « complémentaire » à Hurst).

Algorithme IAAFT. On garde les valeurs triées de l'original et le module de sa
transformée de Fourier. Partant d'un mélange, on alterne :
  (a) imposer le spectre : garder les phases de la série courante, remplacer
      les modules par ceux de l'original, revenir dans le temps ;
  (b) imposer la distribution : remplacer chaque valeur par la valeur de
      l'original de même rang.
On s'arrête quand l'étape (b) ne change plus aucun rang (point fixe), ou après
`iterations` passages. Le résultat a exactement la distribution de l'original
(étape b en dernier) ; son spectre en diffère d'une petite erreur résiduelle,
que mesure `ecart_spectral`.

Ce module ne connaît rien du réseau. Critères de validation
(tests/test_substituts.py) : distribution conservée exactement, spectre
conservé à quelques pour cent, un processus linéaire gaussien est
indiscernable de ses substituts, une dynamique déterministe non linéaire
(application logistique) en est nettement distinguée.
"""

from __future__ import annotations

import numpy as np

from ._validation import entier_positif, serie_finie

ITERATIONS_IAAFT = 100


def _verifier_rng(rng) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng doit être un numpy.random.Generator.")
    return rng


def substitut_iaaft(serie, rng: np.random.Generator,
                    iterations: int = ITERATIONS_IAAFT) -> np.ndarray:
    """
    Un substitut IAAFT : même distribution, même spectre (à peu près), phases aléatoires.

    `serie` : série réelle finie d'au moins 4 points. `rng` : générateur NumPy
    (le substitut est reproductible pour une graine donnée). `iterations` :
    nombre maximal de passages spectre puis distribution ; la boucle s'arrête
    plus tôt dès que les rangs ne changent plus.

    Une série constante est rendue telle quelle (copie) : elle n'a ni ordre ni
    spectre à randomiser.
    """
    x = serie_finie(serie)
    _verifier_rng(rng)
    iterations = entier_positif(iterations, "iterations")
    if x.size < 4:
        raise ValueError("La série doit compter au moins 4 points.")
    if np.all(x == x[0]):
        return x.copy()

    valeurs_triees = np.sort(x)
    modules = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    rangs = np.argsort(np.argsort(y, kind="stable"), kind="stable")
    for _ in range(iterations):
        # (a) Spectre de l'original, phases de la série courante.
        phases = np.angle(np.fft.rfft(y))
        y_spectre = np.fft.irfft(modules * np.exp(1j * phases), n=x.size)
        # (b) Distribution de l'original : la valeur de même rang.
        nouveaux = np.argsort(np.argsort(y_spectre, kind="stable"), kind="stable")
        y = valeurs_triees[nouveaux]
        if np.array_equal(nouveaux, rangs):
            break
        rangs = nouveaux
    return y


def ecart_spectral(reference, substitut) -> float:
    """
    Écart relatif entre les modules de Fourier de deux séries de même longueur.

    ‖ |F(substitut)| − |F(reference)| ‖₂ / ‖ |F(reference)| ‖₂, sans la
    composante de fréquence nulle (la moyenne, identique par construction pour
    un substitut IAAFT). Vaut 0 pour un spectre identique ; un mélange d'une
    série corrélée donne typiquement plusieurs dizaines de pour cent.
    """
    a, b = serie_finie(reference), serie_finie(substitut)
    if a.size != b.size:
        raise ValueError("Les deux séries doivent avoir la même longueur.")
    if a.size < 4:
        raise ValueError("Les séries doivent compter au moins 4 points.")
    fa, fb = np.abs(np.fft.rfft(a))[1:], np.abs(np.fft.rfft(b))[1:]
    norme = np.linalg.norm(fa)
    if norme == 0:
        return 0.0 if np.linalg.norm(fb) == 0 else np.inf
    return float(np.linalg.norm(fb - fa) / norme)
