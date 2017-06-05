# -*- coding:utf-8 -*-


import logging
import sys


import seaborn as sns
sns.set_style("whitegrid")

from til_france.tests.base import create_til_simulation, plot_csv
from til_france.plot.dependance import (
    extract_dependance_niveau_csv,
    multi_extract_dependance_csv,
    plot_dependance_by_age_separate,
    plot_dependance_by_age,
    plot_dependance_csv,
    plot_dependance_gir_csv,
    plot_dependance_incidence_by_age,
    plot_dependance_mortalite_by_age,
    plot_dependance_prevalence_by_age,
    plot_multi_dependance_csv,
    plot_multi_prevalence_csv,
    )

from til_france.plot.population import (
    plot_population,
    plot_ratio_demographique,
    population_diagnostic,
    )


log = logging.getLogger(__name__)


def get_simulation(run = False, option = None):
    assert option is not None
    simulation = create_til_simulation(
        input_name = 'patrimoine',
        option = option,
        uniform_weight = 200,
        )

    if run:
        simulation.run()
        simulation.backup(option, erase = True)
    return simulation


def plot_results(simulation, option = None, age_max = None, age_min = None):
    assert option is not None
    # plot_population(simulation, backup = option)
    plot_dependance_csv(simulation, backup = option, year_min = 2011)

    plot_dependance_by_age_separate(
        simulation,
        backup = option,
        years = [2011, 2025, 2040],
        save = True,
        age_max = age_max,
        age_min = age_min,
        )
    #
    Boum


def extract_dependance_niveau(simulation, option = None):
    assert option is not None
    df = extract_dependance_niveau_csv(simulation, backup = option)
    import os
    import pkg_resources
    assets_path = os.path.join(
        pkg_resources.get_distribution('til-france').location,
        'til_france',
        'model',
        'options',
        'dependance_RT',
        'assets',
        )

    df.to_csv(os.path.join(assets_path, 'dependance_niveau.csv'))
    Boum


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG, stream = sys.stdout)
    option = 'dependance_RT_paquid'
    simulation = get_simulation(run = True, option = option)
    # extract_dependance_niveau(simulation, option = option)
    plot_results(simulation, option = option, age_min = 65, age_max = 95)
