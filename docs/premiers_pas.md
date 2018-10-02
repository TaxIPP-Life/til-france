# Premiers pas en TIL-France

Ce document présente les étapes à suivre pour faire tourner les options de TIL-France
inclues dans le dépôt.

### Création des données

Pour l'instant, TIL-France s'appuie sur deux sources de données :
- l'enquête Patrimoine de 2010
- les données HSI de 2009-2010

Il faut utiliser le script `til_init.py` pour tout initialiser. 

Ce script suppose que vous avez accès aux données Patrimoine et HSI.

### Utilisation

Pour exécuter une simulation, il faut

* la définir :

```
simulation = create_til_simulation(
    input_name = 'patrimoine',
    option = option,
    output_name_suffix = 'test_institutions',
    uniform_weight = 200,
    )
```

* l'exécuter :

```
simulation.run()
```

* éventuellement, stocker les résultats hors du dossier temp

```
simulation.backup(
    option,
    erase = True
)
```

cf les scripts du dossier [test](../til_france/tests) pour des exemples.
