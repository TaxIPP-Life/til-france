# Comment faire tourner TIL

## Création des données

Il suffit de lancer Patrimoine.py, qui est stocké dans ./til_france/data/data.
Il faut avoir avant configurer le config.ini de til-core qui sera créé dans
/.config/til-core pour lui donner le chemin des données de l'enquête "Patrimoine"

## Modèle minimal

Une fois les données créées, il faut rajouter les données non créées par le script,
notamment Hyp_fecondite.csv (présent pour l'instant dans 
/til-france/til_france/param/demo) dans son dossier d'input.

Les scripts qui tournent sans erreurs une fois cela fait sont :
-