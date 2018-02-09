# -*- coding:utf-8 -*-


import logging
import numpy as np
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


from til_france.tests.base import ipp_colors


colors = [ipp_colors[cname] for cname in ['ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]


log = logging.getLogger(__name__)


def get_simulation(run = False, option = None, no_output = False):
    assert option is not None
    simulation = create_til_simulation(
        input_name = 'patrimoine',
        no_output = no_output,
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


def plot_dependance_niveau_by_age_at_period(simulation, age_min = 65, age_max = None, area = True, pct = True, period = None, option = None, sexe = None):
    assert period is not None

    data = extract_dependance_niveau_csv(simulation, backup = option)

    if age_max:
        query = '(age >= @age_min) and (age <= @age_max) and (period == @period)'
    else:
        query = '(age >= @age_min) and (period == @period)'

    data_plot = data.query(query).copy()  # [['age', 'period', name, 'sexe']]

    data_plot['dependance_niveau'] = data_plot.dependance_niveau.astype('str')

    if sexe:
        data_plot = data_plot.query('sexe == @sexe')
    else:
        data_plot = data_plot.groupby(['period', 'age', 'dependance_niveau'])['total'].sum().reset_index()

    pivot_table = (data_plot[['dependance_niveau', 'total', 'age']]
        .pivot('age', 'dependance_niveau', 'total')
        .replace(0, np.nan)  # Next three lines to remove all 0 columns
        .dropna(how = 'all', axis = 1)
        .replace(np.nan, 0)
        )
    if pct:
        pivot_table = pivot_table.divide(pivot_table.sum(axis=1), axis=0)

    if area:
        pivot_table.plot.area(stacked = True, color = colors)
    else:
        pivot_table.plot.line(stacked = True, color = colors)


def get_dependance_niveau_by_period(simulation, option = None, sexe = None, age_min = 65, age_max = None):
    data = extract_dependance_niveau_csv(simulation, backup = option)
    if age_max:
        query = '(age >= @age_min) and (age <= @age_max)'
    else:
        query = '(age >= @age_min)'
    data_plot = data.query(query).copy()  # [['age', 'period', name, 'sexe']]
    data_plot['dependance_niveau'] = data_plot.dependance_niveau.astype('str')
    if sexe:
        data_plot = data_plot.query('sexe == @sexe')
    else:
        data_plot = data_plot.groupby(['period', 'age', 'dependance_niveau'])['total'].sum().reset_index()

    pivot_table = (data_plot[['period', 'dependance_niveau', 'total', 'age']]
        .groupby(['period', 'dependance_niveau'])['total'].sum().reset_index()
        .pivot('period', 'dependance_niveau', 'total')
        .replace(0, np.nan)  # Next three lines to remove all 0 columns
        .dropna(how = 'all', axis = 1)
        .replace(np.nan, 0)
        )
    return pivot_table

def plot_dependance_niveau_by_period(simulation, option = None, sexe = None, area = False, age_min = 65, age_max = 100):
    pivot_table = get_dependance_niveau_by_period(simulation, option = None, sexe = None, age_min = age_min, age_max = age_max)

    if area:
        pivot_table = pivot_table.divide(pivot_table.sum(axis=1), axis=0)
        ax = pivot_table.plot.area(stacked = True)
    else:
        ax = pivot_table.plot.line()

    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG, stream = sys.stdout)
#    options_suffixes = [0, 1]
#    for option_suffix in options_suffixes:
#        option = 'dependance_RT_paquid_{}'.format(option_suffix)
#        simulation = get_simulation(run = True, option = option)
#        plot_dependance_niveau_by_period(simulation, option = option, area = False)

    age_min = 65
    option_suffix = 1
    option = 'dependance_RT_paquid_{}'.format(option_suffix)
    simulation = get_simulation(run = False, option = option)
    df_1 = get_dependance_niveau_by_period(
        simulation,
        option = 'dependance_RT_paquid_{}'.format(option_suffix),
        age_min = age_min
        )
    option_suffix = 0
    option = 'dependance_RT_paquid_{}'.format(option_suffix)
    simulation = get_simulation(run = False, option = option)
    df_0 = get_dependance_niveau_by_period(
        simulation,
        option = 'dependance_RT_paquid_{}'.format(option_suffix),
        age_min = age_min
        )
    df_0.plot.line()

    df_1.plot.line()

    (df_1-df_0).plot.line()
    BIM
    df_1.query('period > 2010').plot.area(stacked = True)
    # plot_dependance_niveau_by_age_at_period(simulation, period = 2009)

    data = get_dependance_niveau_by_period(simulation, option = option, age_min = 64, age_max = 64)
    for period in range(2009, 2040):
        pivot_table = data.query('period == @period')
        print pivot_table #.divide(pivot_table.sum(axis=1), axis=0)
    BOUM



    BIM
    years = [2009, 2025, 2040]
    data.columns
    import numpy as np
    data['age_group'] = np.trunc((data.age - 60) / 5)/home/benjello/data/til/output/patrimoine/patrimoine_output.h5
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
