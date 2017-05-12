# -*- coding:utf-8 -*-


import logging
import sys

import seaborn as sns
sns.set_style("whitegrid")

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


log = logging.getLogger(__name__)


option = 'dependance_RT'


def get_simulation(run = False, option = option):
    simulation = create_til_simulation(
        input_name = 'patrimoine',
        option = option,
        uniform_weight = 200,
        )

    if run:
        simulation.run()
        simulation.backup(option, erase = True)
    return simulation


def plot_results(simulation, option = option):
    plot_population(simulation, backup = option)
    plot_dependance_csv(simulation, backup = option)

    plot_dependance_by_age_separate(
        simulation,
        backup = option,
        years = [2010, 2025, 2040],
        save = True,
        age_max = 95)
    #
    Boum


    # plot_dependance_by_age(simulation, years = [2010, 2020, 2030], save = True, age_max = 100)
    # plot_dependance_prevalence_by_age(simulation, years = [2010, 2025, 2040], age_max = 100)
    #
    # plot_dependance_by_age_separate(simulation, years = [2010, 2025, 2040], save = True, age_max = 95)
    #  avec alignement 403
    #  sans alignement

    options = ['dependance', 'dependance_aligned', 'dependance_pessimistic', 'dependance_medium']

    age_max = 95
    plot_multi_dependance_csv(simulation, options = options, save_figure = True)

    # plot_multi_prevalence_csv(
    #   simulation, options = options, save_figure = True, years = years, age_max = age_max)


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG, stream = sys.stdout)
    simulation = get_simulation(run = False)
    plot_results(simulation, option = option)