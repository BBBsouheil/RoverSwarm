# Environnement RoverSwarm

## Monde et accessibilité

La carte vraie contient des cellules libres ou occupées. La génération procédurale est déterminée par une graine. Les robots apparaissent dans la plus grande composante connexe de cellules libres ; cette composante seule définit le dénominateur de couverture. Les cellules libres isolées ne rendent donc pas la mission impossible.

Les cartes fixes open7, rooms9 et corridor9 servent aux tests. L’état vrai reste dans RoverWorld et ne figure pas dans l’observation de la politique.

## Actions conjointes

Les actions 0 à 4 sont haut, bas, gauche, droite et attente. Toutes sont choisies sur le même état, puis résolues ensemble dans un ordre d’identifiants fixe.

- mur, obstacle ou sortie : le robot reste en place ;
- plusieurs robots vers la même case : les tentatives sont bloquées ;
- échange direct de deux positions : les deux tentatives sont bloquées ;
- entrée dans la case d’un robot immobile ou déjà bloqué : tentative bloquée ;
- un cycle de trois déplacements est autorisé.

Une collision désigne ici une tentative invalide ou conflictuelle, pas un impact physique.

## Capteur, mémoire et observation

Le capteur révèle le carré de rayon de Chebyshev configuré autour du robot. Il n’applique pas d’occlusion. La mémoire commence entièrement inconnue et n’est mise à jour que par ce capteur ou par un paquet livré.

L’observation float32 de taille fixe contient :

1. mémoire aplatie : inconnu 0, libre 0,5, obstacle 1 ;
2. masque de visibilité courante aplati : 0 ou 1 ;
3. position propre y/x normalisée ;
4. énergie normalisée et statut actif ;
5. position normalisée du dernier émetteur effectivement reçu, ou -1/-1 ;
6. fraîcheur normalisée de ce dernier message.

L’observation V2, utilisée par development_v2 à development_v4, conserve la même séparation d’information mais l’encode pour un CNN compact : cellules libres connues, obstacles connus, visibilité courante, fréquence de visite et marqueur de position propre. Neuf scalaires ajoutent énergie, statut, dernier message et état connu des quatre mouvements.

Le repère commun et la position propre exacte sont supposés connus. Aucune position distante n’est fournie sans message. La vue globale de diagnostic et render_rgb ne modifient pas la mémoire.

## Énergie et fins

Une attente coûte 0,2 unité abstraite ; une tentative de déplacement 1,0 ; une collision ajoute 0,5. Ces unités n’ont pas de prétention physique. À zéro, le robot devient inactif et conserve son emplacement jusqu’à la fin globale.

- mission réussie : couverture au moins égale au seuil, terminaison ;
- tous les robots épuisés : terminaison ;
- horizon atteint sans les deux cas précédents : troncature.

Un horizon atteint n’est jamais converti en réussite.

## Récompense

La récompense collective par pas est :

    1,00 × nouvelles cellules libres accessibles
    - 0,35 × tentatives de collision
    - 0,01

L’union de connaissance de l’équipe sert au premier terme ; une cellule découverte simultanément n’est comptée qu’une fois et ne récompense plus ensuite. Cet état global sert au calcul de récompense, jamais comme entrée de politique. Les composantes sont présentes séparément dans info.

En V4, une cellule découverte simultanément est partagée fractionnellement entre ses observateurs : la somme des crédits reste exactement égale à une cellule. Une part collective normalisée est ajoutée, ainsi que des pénalités locales de collision, revisite, attente et temps. Les agents déjà inactifs reçoivent zéro.

PPO MASQUÉ utilise uniquement la mémoire du robot pour invalider un mur déjà connu. Les cellules inconnues restent tentables. Cette couche de sûreté est mesurée séparément et ne constitue pas une planification par frontières.

## Communication

Le protocole est programmé. À la fréquence configurée, chaque émetteur sélectionne au plus K cellules connues, triées par observation la plus récente puis par coordonnées. Un lien n’existe que dans la portée de Manhattan configurée. Chaque livraison subit un délai entier et un tirage de perte depuis un générateur aléatoire séparé.

Paquet abstrait : position y/x de l’émetteur, nombre de cellules, puis y/x/valeur de chaque cellule. Comptage : 5 octets d’en-tête et 5 octets par cellule, avec coordonnées uint16 et valeurs uint8. Ce n’est pas une trame radio réelle.

Les compteurs distinguent paquets émis, reçus, perdus et octets tentés. Aucune fusion globale automatique n’a lieu.
