# Protocole expérimental

## Séparation des données

Les cartes sont procédurales ; les graines définissent sans ambiguïté carte et positions initiales.

- entraînement : graine training.seed, flux pseudo-aléatoire poursuivi entre épisodes ;
- validation : graines du bloc validation, réservées à la sélection de configuration ;
- test : graines du bloc evaluation, jamais utilisées pour choisir modèle ou paramètres.

Le profil benchmark prépare trois graines d’entraînement et trente cartes de test. Le profil smoke n’est qu’un contrôle de chaîne sur trois cartes et ne permet aucune conclusion robuste.

## Comparabilité

Pour une graine donnée, ALÉATOIRE, FRONTIÈRES et PPO reçoivent la même configuration du monde. Les références lisent uniquement la mémoire locale, la position propre et les informations effectivement reçues. La politique par frontières effectue un BFS dans les cellules libres connues vers une cellule libre adjacente à l’inconnu.

PPO partagé est adapté comme un lot synchronisé de trois emplacements. SB3 collecte trois transitions individuelles à chaque pas conjoint. Les trajectoires sont corrélées, mais chaque action vient de l’observation locale de son robot et du même réseau.

## Mesures

Chaque ligne CSV contient couverture finale, réussite du seuil, pas d’atteinte du seuil ou valeur vide, pas totaux, tentatives de collision, énergie consommée, déplacements revisités, paquets émis/reçus/perdus et octets simulés. Le JSON contient aussi moyenne, écart-type et taux de réussite.

## Expériences communication

Deux questions doivent rester distinctes :

1. politique entraînée sans canal contre politique entraînée avec canal, à budgets comparables ;
2. même politique entraînée avec canal, évaluée en canal normal puis perturbé.

Le mode perturbé impose au moins 50 % de pertes et trois pas de délai. Il s’agit d’une ablation du protocole programmé, pas d’une expérience de langage émergent.

## Interprétation

Le smoke test vérifie uniquement que des mises à jour de gradient changent réellement les paramètres, qu’un modèle est écrit, rechargé et produit des actions valides. Une amélioration de récompense sur quelques épisodes ou un modèle sauvegardé n’établissent ni coopération ni performance.

Le benchmark doit rapporter toutes les graines, la moyenne et la dispersion. Si PPO reste inférieur à FRONTIÈRES, le résultat doit être conservé ; les diagnostics prioritaires sont budget insuffisant, attribution du crédit collectif, représentation MLP aplatie et coefficients de récompense.

