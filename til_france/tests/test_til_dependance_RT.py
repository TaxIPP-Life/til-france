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
    plot_dependance_prevalence_all_levels_by_age,
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




    bim
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
    age_min = 65
    age_max = 95

    data = plot_dependance_prevalence_all_levels_by_age(simulation, years = [2010, 2025, 2040],
                                                 age_max = age_max, age_min = age_min)

    years = [2009, 2025, 2040]
    data.columns
    import numpy as np
    data['age_group'] = np.trunc((data.age - 60) / 5)
    data.age_group = data.age_group.astype(int)

    if not age_min:
        age_min = 60
    if age_max:
        query = '(age >= @age_min) and (age <= @age_max) and (period in @years)'
    else:
        query = '(age >= @age_min) and (period in @years)'


    data.columns

    data_plot = data.query(query).query('sexe == 0')
    prevalences = [col for col in data_plot.columns if col.startswith('preval')]
    data_plot.plot(x = 'age', y = prevalences)

    BOUM
    if ax is None:
        save_figure = True
        fig, ax = plt.subplots()
    else:
        save_figure = False

    colors = [ipp_colors[cname] for cname in [
        'ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]
    color_by_period = dict(zip(data_plot['period'].unique(), colors))
    legend_handles, legend_labels = list(), list()
    for grouped in data_plot.groupby(['period', 'sexe']):
        period, sexe = grouped[0]
        linestyle = '--' if sexe == 1 else "-"
        x = grouped[1]['age']
        y = grouped[1][name]
        line, = ax.plot(x, y, linestyle = linestyle, color = color_by_period[period], )
        legend_handles.append(line)
        if english:
            sex_label = 'Men' if sexe == 0 else 'Women'
        else:
            sex_label = 'Hommes' if sexe == 0 else 'Femmes'
        legend_labels.append('{} - {}'.format(sex_label, period))

        if english:
            ax.set_xlabel("age")
        else:
            ax.set_xlabel(u"âge")
        if ylabel is None:
            ax.set_ylabel(name)
        else:
            ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(to_percent_round_formatter)

    if save_figure:
        pdf_name = pdf_name if pdf_name else name
        ax.legend(legend_handles, legend_labels, loc = 'best')
        fig.savefig(os.path.join(figures_directory, '{}.pdf'.format(pdf_name)), bbox_inches='tight')
        del fig



#    df = plot_dependance_prevalence_all_levels_by_age(simulation, years = [2011, 2025, 2040],
#                                                 age_max = age_max, age_min = age_min)
#
#
#    dependance_level_variable = 'dependance_niveau'
#    df.dtypes
#
#    df = (df
#        .groupby([dependance_level_variable, 'period', 'age', 'sexe'])['total'].sum()
#        .unstack([dependance_level_variable])
#        .fillna(0)
#        )
#    df[0] = df[-1] + df[0]
#    df.drop(-1, axis = 1, inplace = True)
#    for column in df.columns:
#        df['prevalence_{}'.format(column)] = df[column] / sum(
#            [df[i] for i in range(6)]
#            )
#
#    df = df.reset_index()
#    df.age = df.age.astype(int)
#    df.set_index(['period', 'age', 'sexe'], inplace = True)
#    df[[col for col in df if str(col).startswith('prevalence')]]
#    #
#     return df



    plot_results(simulation, option = option, age_min = 65, age_max = 95)
