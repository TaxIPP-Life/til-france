# -*- coding: utf-8 -*-
"""
Utilisation minimale de taxIPP-life pour débugger.
"""

import os
import shutil
import seaborn as sns
import sys
from til_france.tests.base import create_til_simulation
from til_france.tests.base import create_til_simulation, plot_csv
from til_france.plot.dependance import (
    plot_dependance_csv, plot_dependance_gir_csv, plot_dependance_prevalence_by_age,
    plot_dependance_incidence_by_age, plot_dependance_mortalite_by_age, plot_dependance_by_age,
    plot_dependance_by_age_separate, multi_extract_dependance_csv,
    plot_multi_prevalence_csv, plot_multi_dependance_csv
    )
from til_france.plot.population import (
    plot_population, population_diagnostic, plot_ratio_demographique,
    )

sns.set_style("whitegrid")

# fonctions de suppression temporaire de l'affichage  console


def sauvegarde_chemin_affichage_console():
    """
    NoneType -> NoneType
    Stocke l'adresse de sys.stdout dans la variable globale chemin_console
    pour pouvoir rétablir l'affichage console via censure(False)
    """
    # Si on perd la valeur originale de sys.stdout (qui est différente
    # de sys.__stdout__ avec l'idle/spyder), on ne peut plus rétablir
    # l'affichage console et il faut relancer le noyau
    global chemin_affichage_console
    chemin_affichage_console = sys.stdout

sauvegarde_chemin_affichage_console()


def censure(booleen):
    """
    bool -> Nonetype
    Supprime (True) ou restore (False) les sorties consoles
    """
    global chemin_affichage_console

    if booleen:
        sys.stdout = open(os.devnull, 'w')
    else:
        sys.stdout = chemin_affichage_console


def censure_fonction(fonction):
    """
    function -> function
    Version décorateur de censure
    """

    def enrobage(*args, **kwargs):
        censure(1)
        fonction(*args, **kwargs)
        censure(0)

    return enrobage


# Outils de test

options = ['dependance', 'dependance_aligned', 'dependance_pessimistic', 'dependance_medium']


def run(option):
    print(option)

    censure(1)

    simulation = create_til_simulation(
        input_name = 'patrimoine',
        option = option,
        output_name_suffix = 'test_institutions',
        uniform_weight = 200,
        )

    censure(0)

    print("Début simulation")

    censure(1)

    simulation.run()

    simulation.backup(
        option,
        erase = True
        )

    censure(0)

    print("Fin simulation")

    plot_population(simulation, backup = option)
    plot_dependance_csv(simulation, backup = option)
