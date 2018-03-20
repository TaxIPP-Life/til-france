#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division


import logging
import numpy as np
import os
import pandas as pd
import sys

from til_core.config import Config

from til_france.model.options.dependance_RT.life_expectancy.merged_0_1_state.transition_matrices import (
    assets_path,
    get_transitions_from_formula,
    )

from til_france.model.options.dependance_RT.life_expectancy.calibration import (
    get_insee_projected_mortality,
    get_insee_projected_population,
    plot_projected_target,
    regularize
    )

from til_france.model.options.dependance_RT.life_expectancy.merged_0_1_state.calibration import (
    build_mortality_calibrated_target_from_formula,
    build_mortality_calibrated_target_from_transitions,
    correct_transitions_for_mortality,
    export_calibrated_dependance_transition,
    )

from til_france.tests.base import ipp_colors
colors = [ipp_colors[cname] for cname in ['ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]


log = logging.getLogger(__name__)


life_table_path = os.path.join(
    assets_path,
    'lifetables_period.xlsx'
    )

figures_directory = '/home/benjello/figures'


def add_65_66_population(population = None):
    assert population is not None
    assert len(population.period.unique().tolist()) == 1, 'More than one period are present: {}'.format(
        population.period.unique().tolist())
    period = population.period.unique().tolist()[0]
    initial_population = (get_initial_population()
        .query('(age in [65, 66])')
        )
    initial_population['part'] = (
        initial_population / initial_population.groupby(['age', 'sex']
        ).transform(sum))['population']
    del initial_population['population']

    population_65_66 = (get_insee_projected_population()
        .reset_index()
        .rename(columns = {'year': 'period'})
        .query('(period == @period) and (age in [65, 66])')
        .merge(initial_population, on = ['sex', 'age'], how = 'left')
        .eval('population = population * part', inplace = False)
        .drop('part', axis = 1)
        )

    population_65_66[['sex', 'age', 'period', 'initial_state', 'population']]

    completed_population = pd.concat([population_65_66, population]).sort_values(
        ['period', 'age', 'sex', 'initial_state'])

    assert completed_population.notnull().all().all(), 'Missing values are present: {}'.format(
        completed_population.loc[completed_population.isnull()])
    return completed_population


def apply_transition_matrix(population = None, transition_matrix = None):
    assert population is not None and transition_matrix is not None
    assert len(population.period.unique()) == 1
    final_population = (population
        .merge(
            transition_matrix.reset_index().drop('period', axis = 1),
            on = ['age', 'sex', 'initial_state'])
        .eval('new_population = population * calibrated_probability', inplace = False)
        .drop(['initial_state', 'calibrated_probability', 'population'], axis = 1)
        .rename(columns = {'new_population': 'population'})
        .groupby(['age', 'sex', 'period', 'final_state'])['population'].sum()
        .reset_index()
        .rename(columns = {'final_state': 'initial_state'})
        )

    simulated_mortality = (final_population
        .query('(initial_state == 5) & (age <= 120) & (age >= 65)')
        .groupby(['sex', 'age'])['population']
        .sum() / final_population
        .query('(age <= 120) & (age >= 65)')
        .groupby(['sex', 'age'])['population']
        .sum()
        ).reset_index()

    period = population.period.unique()[0]
    mortality = get_insee_projected_mortality().query('(year == @period) and (age >= 65)').reset_index().eval(
        'two_year_mortality = 1 - (1 - mortality) ** 2')

    log.debug(simulated_mortality.merge(mortality).query("sex == 'male'").head(50))
    log.debug(simulated_mortality.merge(mortality).query("sex == 'female'").head(50))

    final_population = (final_population
        .eval('age = age + 2', inplace = False)
        .eval('period = period + 2', inplace = False)
        .query('(initial_state != 5) & (age <= 120)')
        .copy()
        )
    assert final_population.age.max() <= 120
    assert final_population.age.min() >= 67
    return final_population


def check_67_and_over(population):
    period = population.period.max()
    insee_population = get_insee_projected_population()
    log.info("period {}: insee = {} vs {} = til".format(
        period,
        insee_population.query('(age >= 67) and (year == @period)')['population'].sum(),
        population.query('(age >= 67) and (period == @period)')['population'].sum()
        ))


def corrrect_transitions(transitions, probability_name = 'calibrated_probability'):
    assert probability_name in transitions.columns, "Column {} not found in tarnsitions columns {}".format(
        probability_name, transitions.columns)

    central_age = 93
    width = 2

    transitions = transitions.copy().rename(columns = {probability_name: 'calibrated_probability'})

    corrections = transitions.query('(initial_state == 3) & (final_state == 4)').copy()
    corrections.eval('factor = (1 + tanh( (age- @central_age) / @width)) / 2', inplace = True)
    corrections.eval('calibrated_probability = factor * calibrated_probability + 0 * (1 - factor)', inplace = True)

    transitions.update(corrections)

    corrections_3_3 = (1 - (
        transitions
            .query('(initial_state == 3) and (final_state != 3)')
            .groupby(['period', 'sex', 'age', 'initial_state'])['calibrated_probability'].sum()
            )).reset_index()

    corrections_3_3['final_state'] = 3
    corrections_3_3 = corrections_3_3.set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
    transitions.update(corrections_3_3)
#    transitions.query(
#        "(period == 2012) and (sex == 'male') and (age <= 100) and (initial_state == 3) and (final_state == 3)"
#        ).plot(y = ['calibrated_probability'])
    return transitions.rename(columns = {'calibrated_probability': probability_name})


def get_initial_population():
    data_by_sex = dict()
    for sex in ['male', 'female']:
        sexe = 'homme' if sex == 'male' else 'femme'
        config = Config()
        input_dir = config.get('til', 'input_dir')
        filename = os.path.join(input_dir, 'dependance_initialisation_level_merged_0_1_{}.csv'.format(sexe))
        log.info('Loading initial population dependance states from {}'.format(filename))
        df = (pd.read_csv(filename, names = ['age', 0, 2, 3, 4], skiprows = 1)
            .query('(age >= 65)')
            )
        df['age'] = df['age'].astype('int')

        df = (
            pd.melt(
                df,
                id_vars = ['age'],
                value_vars = [0, 2, 3, 4],
                var_name = 'initial_state',
                value_name = 'population'
                )
            .sort_values(['age', 'initial_state'])
            .set_index(['age', 'initial_state'])
            )
        assert (df.query('initial_state == -1')['population'] == 0).all()
        data_by_sex[sex] = (df
            .assign(sex = sex)
            )
    data = pd.concat(data_by_sex.values()).reset_index()
    return data


def get_population_by_gir(population):
    from til_france.data.data.hsm_dependance_niveau import get_hsi_hsm_dependance_gir_mapping
    gir = (pd.concat([
            get_hsi_hsm_dependance_gir_mapping(sexe = sexe)
            .assign(sex = 'male' if sexe == 'homme' else 'female')
            for sexe in ['homme', 'femme']
            ])
        .reset_index()
        .set_index(['age', 'dependance_niveau', 'sex', 'est1_gir_s'])
        .rename(columns = dict(poids = 'gir'))
        .unstack('est1_gir_s').fillna(0).reset_index()
        )

    # Joining in dependance_niveau = 0 and dependance_niveau = 1
    gir['dependance_niveau'] = gir['dependance_niveau'].where(gir['dependance_niveau'] != 1, 0)
    gir = gir.groupby(['age', 'dependance_niveau', 'sex']).sum().reset_index()
    gir.columns = [''.join([str(subcol) for subcol in col]) for col in gir.columns.values]
    gir = gir.set_index(['age', 'dependance_niveau', 'sex'])
    gir = gir.divide(gir.sum(axis=1), axis=0).reset_index()

    population = population.rename(columns = {'initial_state': 'dependance_niveau'})
    population_by_gir = population.merge(gir)
    population_by_gir = pd.melt(
        population_by_gir,
        id_vars = ['age', 'dependance_niveau', 'sex', 'period', 'population'],
        value_vars = ['gir{}'.format(i) for i in range(1, 7)],
        var_name = 'gir',
        )
    population_by_gir['population'] = population_by_gir['population'] * population_by_gir['value']
    population_by_gir['gir'] = population_by_gir['gir'].str[3]
    return population_by_gir


def life_expectancy_diagnostic(uncalibrated_transitions = None, initial_period = 2010):
    initial_population = get_initial_population()
    initial_population['period'] = initial_period
    population = initial_population.copy()
    transitions = build_mortality_calibrated_target_from_transitions(
        transitions = uncalibrated_transitions,
        period = initial_period,
        dependance_initialisation = population,
        )
    transitions = corrrect_transitions(transitions)
    delta = 1e-7
    transitions = regularize(
        transition_matrix_dataframe = transitions,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        delta = delta,
        )

    period = initial_period
    transitions_by_period = dict()

    while period <= 2050:
        period = population['period'].max()
        transitions_by_period[period] = transitions
        iterated_population = apply_transition_matrix(
            population = population.query('(period == @period)').copy(),
            transition_matrix = transitions
            )
        population = pd.concat([population, iterated_population])

    life_expectancy_path = os.path.join(figures_directory, 'life_expectancy.txt')
    with open(life_expectancy_path, "w") as text_file:
        for sex in ['male', 'female']:
            population_sex = population.query('(sex == @sex) & (period - age == 2010 - 65)').copy()
            population_sex['total'] = population_sex.query('period == 2010').population.sum()
            population_sex['probability'] = population_sex.population / population_sex.total

            text_file.write('\nsex = {}'.format(sex))
            text_file.write("\nEV : {}".format(
                population_sex
                .groupby('age')['probability'].sum()
                .reindex(range(65, 120))
                .interpolate('cubic')
                .sum() - .5
                ))
            for level in [2, 3, 4]:
                population_sex['sans_incapacite'] = population_sex['initial_state'] < level
                text_file.write("\nEVSI {} et plus: {}".format(
                    level,
                    population_sex.query('sans_incapacite').groupby('age')['probability'].sum()
                    .reindex(range(65, 120))
                    .interpolate('cubic')
                    .sum() - .5
                    ))

        text_file.write('\nInitial population')

#        for sex in ['male', 'female']:
#            population_sex = initial_population.query('(sex == @sex) and (period == 2010) and (age >= 65)').copy()
#            population_sex['total'] = population_sex.query('period == 2010').groupby('age')['population'].transform(sum)
#            population_sex['probability'] = population_sex.population / population_sex.total  # TODO à corriger de la mortalité
#
#            text_file.write('\nsex = {}'.format(sex))
#            text_file.write("\nEV : {}".format(
#                population_sex
#                .groupby('age')['probability'].sum()
#                .reindex(range(65, 120))
#                .interpolate('cubic')
#                .sum() - .5
#                ))
#            for level in [2, 3, 4]:
#                population_sex['sans_incapacite'] = population_sex['initial_state'] < level
#                text_file.write("\nEVSI {} et plus: {}".format(
#                    level,
#                    (population_sex.query('sans_incapacite').groupby('age')['probability'].sum()
#                    .sum() - .5
#                    ))


def load_and_plot_dependance_niveau_by_period(survival_gain_cast = None, mu = None, periods = None, area = False):
    suffix = survival_gain_cast
    if mu is not None:
        suffix += '_mu_{}'.format(mu)

    population = pd.read_csv(os.path.join(figures_directory, 'population_{}.csv'.format(suffix)))
    periods = range(2010, 2050, 2) if periods is None else periods
    for period in periods:
        plot_dependance_niveau_by_age(population, period, age_max = 100, area = area)


def load_and_plot_gir_projections(survival_gain_cast = None, mu = None):
    assert survival_gain_cast is not None
    suffix = survival_gain_cast
    if mu is not None:
        suffix += '_mu_{}'.format(mu)

    reference_population = (
        pd.read_csv(os.path.join(figures_directory, 'population_{}.csv'.format('homogeneous')), index_col = 0)
        .query('age >= 65')
        .copy()
        )

    population = (
        pd.read_csv(os.path.join(figures_directory, 'population_{}.csv'.format(suffix)), index_col = 0)
        .query('age >= 65')
        .copy()
        )
    population_by_gir = get_population_by_gir(population)
    reference_population_by_gir = get_population_by_gir(reference_population)

    pivot_table = population_by_gir.groupby(['period', 'gir'])['population'].sum().unstack()
    pivot_table['APA'] = sum(pivot_table[str(i)] for i in range(1, 5))
    pivot_table['GIR 1 + 2'] = sum(pivot_table[str(i)] for i in range(1, 3))
    pivot_table['GIR 3 + 4'] = sum(pivot_table[str(i)] for i in range(3, 5))

    reference_pivot_table = reference_population_by_gir.groupby(['period', 'gir'])['population'].sum().unstack()
    reference_pivot_table['APA'] = sum(reference_pivot_table[str(i)] for i in range(1, 5))

    diff_pivot_table = (pivot_table - reference_pivot_table)

    pivot_table = pivot_table.query('period > 2010 and period <= 2060') / 1e6
    pivot_table.index.name = u'Année'

    columns = ['GIR 1 + 2', 'GIR 3 + 4']
    ax = pivot_table[columns].plot.line(xlim = [2012, 2060])
    figure = ax.get_figure()
    figure_path_name = os.path.join(figures_directory, 'gir_{}'.format(suffix))
    ax.set_title('Effectifs GIR')
    ax.set_ylabel('Effectifs (millions)')
    ax.legend(frameon = True, edgecolor = 'k', framealpha = 1, title = "")
    figure_path_name = figure_path_name + '_' + suffix
    figure.savefig(figure_path_name, bbox_inches = 'tight', format = 'png')
    figure.savefig(figure_path_name + ".pdf", bbox_inches = 'tight', format = 'pdf')

    diff_pivot_table = diff_pivot_table.query('period > 2010 and period <= 2060') / 1e6
    diff_pivot_table.index.name = u'Année'

    ax = diff_pivot_table[columns].plot.line(xlim = [2012, 2060])
    ax.set_ylabel('Effectifs (millions)')
    ax.legend(frameon = True, edgecolor = 'k', framealpha = 1, title = "")
    figure = ax.get_figure()
    figure_path_name = os.path.join(figures_directory, 'diff_gir_{}'.format(suffix))
    ax.set_title('Effectifs GIR ({} - homogeneous'.format(suffix))
    figure_path_name = figure_path_name + '_' + suffix
    figure.savefig(figure_path_name, bbox_inches = 'tight', format = 'png')
    figure.savefig(figure_path_name + ".pdf", bbox_inches = 'tight', format = 'pdf')


def load_and_plot_projected_target(survival_gain_cast = None, mu = None, age_max = 100, initial_states = None,
        final_states = None):
    suffix = survival_gain_cast
    if mu is not None:
        suffix += '_mu_{}'.format(mu)
    transitions = pd.read_csv(os.path.join(figures_directory, 'transitions_{}.csv'.format(suffix)))
    plot_projected_target(
        age_min = 65,
        age_max = age_max,
        projected_target = transitions,
        years = [2012, 2020, 2030, 2050],
        probability_name = 'calibrated_probability',
        initial_states = initial_states,
        final_states = final_states,
        french = True,
        save = True,
        )
    return transitions


def load_and_plot_scenarios_difference(survival_gain_cast = None, mu = None):
    suffix = survival_gain_cast
    if mu is not None:
        suffix += '_mu_{}'.format(mu)

    population = (
        pd.read_csv(os.path.join(figures_directory, 'population_{}.csv'.format(suffix)), index_col = 0)
        .drop('sex', axis = 1)
        .query('age >= 65')
        .copy()
        )
    population_reference = (
        pd.read_csv(os.path.join(figures_directory, 'population_{}.csv'.format('homogeneous')), index_col = 0)
        .drop('sex', axis = 1)
        .query('age >= 65')
        .copy()
        )
    pivot_table = population.groupby(['period', 'initial_state'])['population'].sum().unstack()
    reference_pivot_table = population_reference.groupby(['period', 'initial_state'])['population'].sum().unstack()

    diff_pivot_table = pivot_table - reference_pivot_table
    diff_pivot_table.to_csv(os.path.join(figures_directory, 'merged_0_1_proj_{}.csv'.format(
        suffix)))
    ax = diff_pivot_table.plot.line()
    figure = ax.get_figure()
    figure.savefig(os.path.join(figures_directory, 'merged_0_1_diff_proj_{}.pdf'.format(
        suffix)), bbox_inches = 'tight')

    pct_pivot_table = pivot_table.divide(pivot_table.sum(axis=1), axis=0)
    pct_reference_pivot_table = reference_pivot_table.divide(reference_pivot_table.sum(axis=1), axis=0)
    diff_pct_pivot_table = pct_pivot_table - pct_reference_pivot_table
    ax = diff_pct_pivot_table.plot.line()
    figure = ax.get_figure()
    pct_pivot_table.to_csv(os.path.join(figures_directory, 'merged_0_1_diff_proj_pct_{}.csv'.format(
        suffix)))
    figure.savefig(os.path.join(figures_directory, 'merged_0_1_diff_proj_pct_{}.pdf'.format(suffix)), bbox_inches = 'tight')


def plot_dependance_niveau_by_age(population, period, sexe = None, area = False, pct = True, age_max = None, suffix = None):
    assert 'period' in population.columns, 'period is not present in population columns: {}'.format(population.columns)
    assert period in population.period.unique()
    data = population.query('period == @period').copy()
    if 'initial_state' in data.columns:
        data.rename(columns = {'initial_state': 'dependance_niveau'}, inplace = True)
    pivot_table = (data
        .groupby(['period', 'age', 'dependance_niveau'])['population'].sum().reset_index()
        .pivot('age', 'dependance_niveau', 'population')
        )

    if pct:
        pivot_table = pivot_table.divide(pivot_table.sum(axis = 1), axis = 0)

    # Remove all 0 columns
    pivot_table = (pivot_table
        .replace(0, np.nan)
        .dropna(how = 'all', axis = 1)
        .replace(np.nan, 0)
        )

    if age_max:
        pivot_table = pivot_table.query('age < @age_max').copy()

    xlim = [pivot_table.reset_index()['age'].min(), pivot_table.reset_index()['age'].max()]
    ylim = [0, 1]

    plot_kwargs = dict(
        color = colors,
        xlim = xlim,
        ylim = ylim,
        )
    pivot_table.index.name = u"Âge"
    pivot_table.columns = ["0-1", "2", "3", "4"]

    if area:
        ax = pivot_table.plot.area(stacked = True, **plot_kwargs)
    else:
        ax = pivot_table.plot.line(stacked = True, **plot_kwargs)

    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    figure = ax.get_figure()
    figure_path_name = os.path.join(figures_directory, 'prevalence_{}'.format(period))

    # ax.set_title(u'Année = {}'.format(period))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title = u'Niveau de \ndépendance')

    if suffix is not None:
        figure_path_name = figure_path_name + '_' + suffix
    figure.savefig(figure_path_name, bbox_inches = 'tight')
    figure.savefig(figure_path_name + ".pdf", bbox_inches = 'tight', format = "pdf")


def plot_dependance_niveau_by_period(population, period, sexe = None, area = False):
    data = population.copy()
    if 'initial_state' in data.columns:
        data.rename(columns = {'initial_state': 'dependance_niveau'}, inplace = True)
    pivot_table = (data[['period', 'dependance_niveau', 'population', 'age']]
        .groupby(['period', 'dependance_niveau'])['population'].sum().reset_index()
        .pivot('period', 'dependance_niveau', 'population')
        .replace(0, np.nan)  # Next three lines to remove all 0 columns
        .dropna(how = 'all', axis = 1)
        .replace(np.nan, 0)
        )

#    if area:
#        pivot_table = pivot_table.divide(pivot_table.sum(axis=1), axis=0)
#        ax = pivot_table.plot.area(stacked = True)
#    else:
#        ax = pivot_table.plot.line()
#
#    from matplotlib.ticker import MaxNLocator
#    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def run_calibration(uncalibrated_transitions = None, initial_population = None, initial_period = 2010, mu = None,
        survival_gain_cast = None):

    initial_population['period'] = initial_period
    population = initial_population.copy()

    uncalibrated_transitions = (uncalibrated_transitions
        .reset_index()
        .assign(period = initial_period)
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        )

    uncalibrated_transitions = corrrect_transitions(
        uncalibrated_transitions,
        probability_name = 'probability'
        )

    transitions = build_mortality_calibrated_target_from_transitions(
        transitions = uncalibrated_transitions,
        period = initial_period,
        dependance_initialisation = population,
        )

    period = initial_period

    transitions_by_period = dict()

    while period < 2060:
        print 'Running period {}'.format(period)
        period = population['period'].max()

        if period > initial_period:
            dependance_initialisation = population.query('period == @period').copy()
            # Update the transitions matrix if necessary
            if survival_gain_cast is None:
                log.info("Calibrate transitions for period = {}".format(period))
                delta = 1e-7
                transitions = regularize(
                    transition_matrix_dataframe = transitions.rename(
                        columns = {'calibrated_probability': 'probability'}),
                    by = ['period', 'sex', 'age', 'initial_state'],
                    probability = 'probability',
                    delta = delta,
                    )
                transitions = build_mortality_calibrated_target_from_transitions(
                    transitions = transitions,
                    period = period,
                    dependance_initialisation = dependance_initialisation,
                    )
                transitions_by_period[period] = transitions

            else:
                log.info('Updating period = {} transitions for mu = {} and survival_gain_cast = {}'.format(
                    period, mu, survival_gain_cast))
                transitions = correct_transitions_for_mortality(
                    transitions,
                    dependance_initialisation = dependance_initialisation,
                    mu = mu,
                    survival_gain_cast = survival_gain_cast,
                    period = period,
                    )
                transitions_by_period[period] = transitions
#            plot_dependance_niveau_by_period(population, period)
#            plot_dependance_niveau_by_age(population, period)
#        plot_projected_target(age_min = 65, projected_target = transitions, years = [period],
#            probability_name = 'calibrated_probability')
#            raw_input("Press Enter to continue...")
        # Iterate
        iterated_population = apply_transition_matrix(
            population = population.query('period == @period').copy(),
            transition_matrix = transitions
            )
        check_67_and_over(iterated_population)
        iterated_population = add_65_66_population(population = iterated_population)
        population = pd.concat([population, iterated_population])

    return population, transitions_by_period


def save_data_and_graph(uncalibrated_transitions, mu = None, survival_gain_cast = None):
    log.info("Running with survival_gain_cast = {}".format(survival_gain_cast))
    initial_period = 2010
    initial_population = get_initial_population()
    initial_population['period'] = initial_period
    population, transitions_by_period = run_calibration(
        uncalibrated_transitions = uncalibrated_transitions,
        initial_population = initial_population,
        mu = mu,
        survival_gain_cast = survival_gain_cast,
        )
    suffix = survival_gain_cast
    if mu is not None:
        suffix += '_mu_{}'.format(mu)

    population_path = os.path.join(figures_directory, 'population_{}.csv'.format(suffix))
    log.info("Saving population data to {}".format(population_path))
    population.to_csv(population_path)

    transitions = pd.concat(transitions_by_period.values())
    (transitions
        .reset_index()
        .sort_values(['period', 'sex', 'age', 'initial_state', 'final_state'])
        .to_csv(os.path.join(figures_directory, 'transitions_{}.csv'.format(suffix)))
        )

    pivot_table = population.groupby(['period', 'initial_state'])['population'].sum().unstack()
    pivot_table.to_csv(os.path.join(figures_directory, 'merged_0_1_proj_{}.csv'.format(
        suffix)))
    ax = pivot_table.plot.line()
    figure = ax.get_figure()
    figure.savefig(os.path.join(figures_directory, 'merged_0_1_proj_{}.pdf'.format(
        suffix)), bbox_inches = 'tight')

    pct_pivot_table = pivot_table.divide(pivot_table.sum(axis = 1), axis = 0)
    ax = pct_pivot_table.plot.line()
    figure = ax.get_figure()
    pct_pivot_table.to_csv(os.path.join(figures_directory, 'merged_0_1_proj_pct_{}.csv'.format(
        suffix)))
    figure.savefig(os.path.join(figures_directory, 'merged_0_1_proj_pct_{}.pdf'.format(suffix)), bbox_inches = 'tight')


if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, stream = sys.stdout)

    def run(survival_gain_casts = None):
        from til_france.data.data.hsm_dependance_niveau import create_dependance_initialisation_merged_0_1
        create_dependance_initialisation_merged_0_1(smooth = True, survey = 'both')
        formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1) ** 2) + I(((age - 80) * 0.1) ** 3)'
        uncalibrated_transitions = get_transitions_from_formula(formula = formula)
        # life_expectancy_diagnostic(uncalibrated_transitions = uncalibrated_transitions, initial_period = 2010)

        for survival_gain_cast in survival_gain_casts:
            if survival_gain_cast in ['initial_vs_others', 'autonomy_vs_disability']:
                for mu in [0, 1]:
                    save_data_and_graph(
                        uncalibrated_transitions = uncalibrated_transitions,
                        survival_gain_cast = survival_gain_cast,
                        mu = mu,
                        )
            else:
                save_data_and_graph(
                    uncalibrated_transitions = uncalibrated_transitions,
                    survival_gain_cast = survival_gain_cast,
                    )

    def plot_differences(survival_gain_casts = None):
        for survival_gain_cast in survival_gain_casts:
            if survival_gain_cast in ['initial_vs_others', 'autonomy_vs_disability']:
                for mu in [0, 1]:
                    load_and_plot_scenarios_difference(survival_gain_cast = survival_gain_cast, mu = mu)
            else:
                load_and_plot_scenarios_difference(survival_gain_cast = survival_gain_cast)

    def plot_girs(survival_gain_casts):
        for survival_gain_cast in survival_gain_casts:
            if survival_gain_cast in ['initial_vs_others', 'autonomy_vs_disability']:
                for mu in [0, 1]:
                    load_and_plot_gir_projections(survival_gain_cast = survival_gain_cast, mu = mu)
            else:
                load_and_plot_gir_projections(survival_gain_cast = survival_gain_cast)

    def load_and_plot_summary():
        renaming = {
            'homogeneous': u'gains homogènes',
            'initial_vs_others_mu_1': u'transitions ralenties',
            'initial_vs_others_mu_0': u'transitions accélérées',
            'autonomy_vs_disability_mu_1': u'gain de survie en autonomie',
            'autonomy_vs_disability_mu_0': u'gain de survie en dépendance',
            'increase_gradual_disability': u'gain de survie en dépendance modérée',
            }
        meta_dependance_niveaux = [
            [2],
            [3],
            [4],
            [2, 3, 4],
            [3, 4],
            ]

        style = {
            'homogeneous': 'r-',
            'initial_vs_others_mu_1': 'b-',
            'initial_vs_others_mu_0': 'b--',
            'autonomy_vs_disability_mu_1': 'b-',
            'autonomy_vs_disability_mu_0': 'b--',
            'increase_gradual_disability': 'c-.',
            }

        style = dict(
            (new_name, style[scenario])
            for scenario, new_name in renaming.iteritems()
            )

        survival_gain_casts_by_theme = dict(
            acceleration = [
                'homogeneous',
                'initial_vs_others',
                ],
            entry = [
                'homogeneous',
                'autonomy_vs_disability',
                'increase_gradual_disability',
                ]
            )

        for theme, survival_gain_casts in survival_gain_casts_by_theme.iteritems():
            for dependance_niveaux in meta_dependance_niveaux:
                suffixes = list()
                for survival_gain_cast in survival_gain_casts:
                    suffix = survival_gain_cast
                    if survival_gain_cast in ['initial_vs_others', 'autonomy_vs_disability']:
                        for mu in [0, 1]:
                            tmp_suffix = suffix + '_mu_{}'.format(mu)
                            suffixes.append(tmp_suffix)
                    else:
                        suffixes.append(suffix)

                population = pd.DataFrame()
                for suffix in suffixes:
                    population = pd.concat([
                        population,
                        (pd.read_csv(os.path.join(figures_directory, 'population_{}.csv'.format(suffix)), index_col = 0)
                            .drop('sex', axis = 1)
                            .query('age >= 65')
                            .query('(period > 2010) & (period <= 2060)')
                            .query('initial_state in @dependance_niveaux')
                            .assign(scenario = suffix)
                            .groupby(['period', 'scenario'])['population'].sum()
                            .reset_index())
                            .rename(columns = {'period': u'Année'})
                        ])

                pivot_table = population.pivot(index = u'Année', columns = 'scenario', values = 'population')
                pivot_table.rename(columns = renaming, inplace = True)
                if pivot_table.max().max() > 1e6:
                    pivot_table = pivot_table / 1e6
                    unite = 'Effectifs (millions)'
                else:
                    pivot_table = pivot_table / 1e3
                    unite = 'Effectifs (milliers)'

                ax = pivot_table.plot.line(
                    xlim = [2012, 2060],
                    style = style,
                    )
                ax.set_ylabel(unite)
                ax.legend(frameon = True, edgecolor = 'k', framealpha = 1)
                figure = ax.get_figure()
                figure_path_name = os.path.join(figures_directory, 'multi_scenario_{}_{}'.format(
                    theme,
                    '_'.join([str(dependance_niveau) for dependance_niveau in dependance_niveaux]))
                    )
                # ax.set_title(u'Effectifs dépendants {}'.format(dependance_niveaux))
                figure.savefig(figure_path_name, bbox_inches = 'tight')
                figure.savefig(figure_path_name + ".pdf", bbox_inches = 'tight', format = 'pdf')

    import seaborn as sns
    sns.set_style("whitegrid")
    BOUM
    load_and_plot_summary()
    BIM
    transitions = load_and_plot_projected_target(survival_gain_cast = 'homogeneous', age_max = 100, initial_states = [2],
        final_states = None)
    BAM
    survival_gain_casts = [
        'homogeneous',
        'initial_vs_others',
        'autonomy_vs_disability',
        'increase_gradual_disability',
        ]
    run(survival_gain_casts)
    plot_differences(survival_gain_casts)
    plot_girs(survival_gain_casts)

    BIM
    load_and_plot_dependance_niveau_by_period(survival_gain_cast = 'homogeneous', periods = None, area = True)

    BIM

    insee_population = get_insee_projected_population()
    insee_population.reset_index().query('(age >= 65) and (year > 2010)').groupby('year')['population'].sum().plot()
    insee_population.query('(age >= 65) and (year == 2008)').sum()
    initial_population = get_initial_population()
    initial_population['population'].sum()

    BIM


def mbj_plot(data, pdf_dump, grain):
    """
    pd.DataFrame*str*float -> None
    Plot the data à la MBJ (2017) and put in the pdf_dump directory
    """
    pdf_path = os.path.join(pdf_dump, 'plot.pdf')
    pdf_file = PdfPages(pdf_path)

    grain = .001
    suffix = 'homogeneous'
    sex = 'male'


    population = (
        pd.read_csv(os.path.join(figures_directory, 'population_{}.csv'.format(suffix)), index_col = 0)
        .drop('sex', axis = 1)
        .query('age >= 65')
        .query('period == 2010')
        .copy()
        )
    pivot_table = population.groupby(['age', 'initial_state'])['population'].sum().unstack()


    pct_pivot_table = pivot_table.divide(pivot_table.sum(axis=1), axis=0)
    pct_pivot_table['population'] = pivot_table.sum(axis = 1)
    data = pct_pivot_table.reset_index()
    data['tick'] = data['population'] * grain  # bigger grain : able to smooth more if needed
    data['tick'] = data['tick'].astype(np.int)

    data_expanded = data.loc[np.repeat(data.index.values, data['tick'])].reset_index(drop=True)

    f = plt.figure()
    f.set_size_inches(24, 13.5)
    ax = f.add_subplot(1,1,1)

    shares = [data_expanded[col] for col in [0, 2, 3, 4]]

    # adjust to smooth
    smoothed_shares = [share.rolling(100, center = True,win_type='gaussian').mean(std = 15).fillna(method='bfill').fillna(method='ffill') for share in shares]
    y = np.row_stack(tuple(smoothed_shares))

    x = range(len(smoothed_shares[0]))

    ax.set_xticks([])
    ax.set_xlim(0, max(x))
    ax.set_ylim(0,1)
    ax.stackplot(x,y)

    x = np.cumsum([0] + list(data['population'].values))
    xplot = list((x[1:] + x[:-1]) / 2)
    ax2 = ax.twiny()
    ax2.set_xticks(xplot)
    ax2.set_xlim(0, sum(data['population']))
    ax2.set_xticklabels(list(data['age'].values))

    pdf_file.savefig()
    plt.close()
    pdf_file.close()

    return(None)
