# Journal de développement

Décisions techniques et résultats intermédiaires. Complète le journal de bord du
cours, qui porte sur le raisonnement scientifique.

## 2026-09-15

- Structure du dépôt, paquet installable `cascade_entropy`.
- `indicateurs.hurst` par méthode R/S. Validation : bruit non corrélé donne
  H = 0,544 ± 0,018 sur 30 réalisations de 8192 points.
- À vérifier : l'écart à 0,5 est le biais connu de R/S sur séries finies. Il se
  trouve que Carreras rapporte H = 0,55 aux échelles courtes. Avant d'interpréter
  une persistance, construire une distribution nulle par simulation ou appliquer
  la correction d'Anis-Lloyd.
