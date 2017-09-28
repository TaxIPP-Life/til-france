# Premiers pas en TIL-France

Ce document présente les étapes à suivre pour faire tourner les options de TIL-France
inclues dans le dépôt (ci-après 'TILF').

### Création des données

Pour l'instant, TILF s'appuie sur deux sources de données :
- l'enquête Patrimoine de 2010
- les données HSI de 2009-2010

Il faut utiliser le script `Patrimoine.py` présent dans [/til-france/til_france/data/data](../til_france/data/data)
pour mettre en forme les données Patrimoine. Il faut préalablement avoir réglé
les options de configuration de [til-core](https://github.com/TaxIPP-Life/til-core).

Pour les données HSI, il faut créer un fichier .h5 à via open-fisca-survey-manager
(une collection avec un seul survey), puis à partir du fichier créé par 
[open-fisca-survey-manager](https://github.com/openfisca/openfisca-survey-manager), utiliser le script `handicap_sante_institutions.py`
également présent dans [/til-france/til_france/data/data](../til_france/data/data)

### Paramètres

Ils sont stockés [ici](../til_france/param/demo). On peut les reconstruire (sauf la fécondité pour l'instant) avec le script [`build_parameters.py`](../til_france/scripts/build_parameters.py).

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
