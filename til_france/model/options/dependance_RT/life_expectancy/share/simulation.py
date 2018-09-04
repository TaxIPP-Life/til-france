#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division


import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import slugify
import sys


from til_core.config import Config

from til_france.model.options.dependance_RT.life_expectancy.share.transition_matrices import (
    assets_path,
    get_transitions_from_formula,
    get_transitions_from_file,
    )

from til_france.model.options.dependance_RT.life_expectancy.calibration import (
    get_insee_projected_mortality,
    get_insee_projected_population,
    plot_projected_target,
    regularize
    )

from til_france.model.options.dependance_RT.life_expectancy.share.calibration import (
    build_mortality_calibrated_target_from_formula,
    build_mortality_calibrated_target_from_transitions,
    correct_transitions_for_mortality
    )

from til_france.data.data.hsm_dependance_niveau import (
    create_dependance_initialisation_share,
    get_hsi_hsm_dependance_gir_mapping,
    )


from til_france.tests.base import ipp_colors
colors = [ipp_colors[cname] for cname in [
    'ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]


log = logging.getLogger(__name__)

life_table_path = os.path.join(
    assets_path,
    'lifetables_period.xlsx'
    )

config = Config()

figures_directory = config.get('dependance', 'figures_directory')


def add_lower_age_population(population = None, age_min = None):
    assert age_min is not None
    assert population is not None
    assert len(population.period.unique().tolist()) == 1, 'More than one period are present: {}'.format(
        population.period.unique().tolist())
    period = population.period.unique().tolist()[0]
    lower_age_population = (get_initial_population(age_min = age_min, rescale = True, period = period)
        .query('age in [@age_min, @age_min + 1]')
        )
    lower_age_population['period'] = period
    completed_population = pd.concat([lower_age_population, population]).sort_values(
        ['period', 'age', 'sex', 'initial_state'])

    assert completed_population.notnull().all().all(), completed_population.notnull().all()

    assert completed_population.notnull().all().all(), 'Missing values are present: {}'.format(
        completed_population.loc[completed_population.isnull()])
    return completed_population


def apply_transition_matrix(population = None, transition_matrix = None, age_min = None):
    assert age_min is not None
    death_state = 4
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
        .query('(initial_state == @death_state) & (age <= 120) & (age >= @age_min)')
        .groupby(['sex', 'age'])['population']
        .sum() / final_population
        .query('(age <= 120) & (age >= @age_min)')
        .groupby(['sex', 'age'])['population']
        .sum()
        ).reset_index()

    period = population.period.unique()[0]
    mortality = get_insee_projected_mortality().query('(year == @period) and (age >= @age_min)').reset_index().eval(
        'two_year_mortality = 1 - (1 - mortality) ** 2', inplace = False)

    log.debug(simulated_mortality.merge(mortality).query("sex == 'male'").head(50))
    log.debug(simulated_mortality.merge(mortality).query("sex == 'female'").head(50))

    final_population = (final_population
        .eval('age = age + 2', inplace = False)
        .eval('period = period + 2', inplace = False)
        .query('(initial_state != @death_state) & (age <= 120)')
        .copy()
        )
    assert final_population.age.max() <= 120
    assert final_population.age.min() >= age_min + 2, "population minimal age is {} instead of {}".format(
        final_population.age.min(), age_min + 2)
    return final_population


def build_suffix(survival_gain_cast = None, mu = None, vagues = None, survey = None):
    suffix = survival_gain_cast
    if mu is not None:
        suffix += '_mu_{}'.format(mu)
    if vagues is not None:
        suffix += slugify.slugify(str(vagues), separator = "_")
    if survey is not None:
        suffix += '_survey_{}'.format(mu)

    return suffix


def check_67_and_over(population, age_min):
    period = population.period.max()
    insee_population = get_insee_projected_population()
    log.info("period {}: insee = {} vs {} = til".format(
        period,
        insee_population.query('(age >= @age_min) and (year == @period)')['population'].sum(),
        population.query('(age >= @age_min) and (period == @period)')['population'].sum()
        ))


def correct_transitions(transitions, probability_name = 'calibrated_probability'):
    assert probability_name in transitions.columns, "Column {} not found in transitions columns {}".format(
        probability_name, transitions.columns)
    correction = False
    if correction:
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

    else:
        return transitions


def get_initial_population(age_min = None, period = None, rescale = False):
    assert age_min is not None
    if rescale:
        assert period is not None
    data_by_sex = dict()
    for sex in ['male', 'female']:
        sexe = 'homme' if sex == 'male' else 'femme'
        config = Config()
        filename = os.path.join(
            config.get('til', 'input_dir'),
            'dependance_initialisation_level_share_{}.csv'.format(sexe)
            )
        log.info('Loading initial population dependance states from {}'.format(filename))
        df = (pd.read_csv(filename, names = ['age', 0, 1, 2, 3], skiprows = 1)
            .query('(age >= @age_min)')
            )
        df['age'] = df['age'].astype('int')

        df = (
            pd.melt(
                df,
                id_vars = ['age'],
                value_vars = [0, 1, 2, 3],
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

    if rescale:
        insee_population = get_insee_projected_population()
        assert period in insee_population.reset_index().year.unique()
        rescaled_data = (
            data.groupby(['sex', 'age'])['population'].sum().reset_index()
            .merge(
                (insee_population.query("year == @period")
                    .reset_index()
                    .rename(columns = {"population": "insee_population"})
                    )
                    ,
                how = 'left',
                )
            .eval("calibration = insee_population / population", inplace = False)
            )

        data = (data
            .merge(rescaled_data[['sex', 'age', 'calibration']])
            .eval("population = calibration *population", inplace = False)
            )[['sex', 'age', 'initial_state', 'population']].fillna(0).copy()

    assert data.notnull().all().all(), data.notnull().all()

    return data


def get_population_by_gir(population):
    gir = (pd.concat([
            get_hsi_hsm_dependance_gir_mapping(sexe = sexe)
            .assign(sex = 'male' if sexe == 'homme' else 'female')
            for sexe in ['homme', 'femme']
            ])
        .reset_index()
        .set_index(['age', 'dependance_niveau', 'sex', 'gir'])
        .rename(columns = dict(poids = 'population_gir'))
        .unstack('gir').fillna(0).reset_index()
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
        value_vars = ['population_gir{}'.format(i) for i in range(1, 7)],
        var_name = 'gir',
        )
    population_by_gir['population'] = population_by_gir['population'] * population_by_gir['value']
    population_by_gir['gir'] = population_by_gir['gir'].str[-1:]
    return population_by_gir


def life_expectancy_diagnostic(uncalibrated_transitions = None, initial_period = 2010, age_min = None):
    assert age_min is not None
    initial_population = get_initial_population(age_min = age_min)
    initial_population['period'] = initial_period
    population = initial_population.copy()
    transitions = build_mortality_calibrated_target_from_transitions(
        transitions = uncalibrated_transitions,
        period = initial_period,
        dependance_initialisation = population,
        )
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
            population_sex = population.query('(sex == @sex) & (period - age == 2010 - @age_min)').copy()
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
#                .interpolate('cubic')population_{}
#                .sum() - .5
#                ))
#            for level in [2, 3, 4]:
#                population_sex['sans_incapacite'] = population_sex['initial_state'] < level
#                text_file.write("\nEVSI {} et plus: {}".format(
#                    level,
#                    (population_sex.query('sans_incapacite').groupby('age')['probability'].sum()
#                    .sum() - .5
#                    ))


def load_and_plot_dependance_niveau_by_period(survival_gain_cast = None, mu = None, periods = None, area = False,
          vagues = None):
    suffix = build_suffix(survival_gain_cast, mu, vagues)

    population = pd.read_csv(os.path.join(figures_directory, 'population_{}.csv'.format(suffix)))
    periods = range(2010, 2050, 2) if periods is None else periods
    for period in periods:
        plot_dependance_niveau_by_age(population, period, age_min = 50, age_max = 100, area = area)


def load_and_plot_gir_projections(survival_gain_cast = None, mu = None, age_min = 50, vagues = None):
    assert survival_gain_cast is not None
    suffix = build_suffix(survival_gain_cast, mu, vagues)
    if vagues is not None:
        vagues_suffix = slugify.slugify(str(vagues), separator = "_")

    reference_population = (
        pd.read_csv(
            os.path.join(figures_directory, 'population_{}{}.csv'.format('homogeneous', vagues_suffix)), index_col = 0
            )
        .query('age >= @age_min')
        .copy()
        )

    population = (
        pd.read_csv(os.path.join(figures_directory, 'population_{}.csv'.format(suffix)), index_col = 0)
        .query('age >= @age_min')
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


def load_and_plot_projected_target(survival_gain_cast = None, mu = None, age_min = 50, age_max = 100,
        initial_states = None, final_states = None, vagues = None):
    suffix = build_suffix(survival_gain_cast, mu, vagues)

    transitions = pd.read_csv(os.path.join(figures_directory, 'transitions_{}.csv'.format(suffix)))
    plot_projected_target(
        age_min = age_min,
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


def load_and_plot_scenarios_difference(survival_gain_cast = None, mu = None, age_min = None):
    # TODO adapt to vagues
    assert age_min is not None
    suffix = survival_gain_cast
    if mu is not None:
        suffix += '_mu_{}'.format(mu)

    population = (
        pd.read_csv(os.path.join(figures_directory, 'population_{}.csv'.format(suffix)), index_col = 0)
        .drop('sex', axis = 1)
        .query('age >= @age_min')
        .copy()
        )
    population_reference = (
        pd.read_csv(os.path.join(figures_directory, 'population_{}.csv'.format('homogeneous')), index_col = 0)
        .drop('sex', axis = 1)
        .query('age >= @age_min')
        .copy()
        )
    pivot_table = population.groupby(['period', 'initial_state'])['population'].sum().unstack()
    reference_pivot_table = population_reference.groupby(['period', 'initial_state'])['population'].sum().unstack()

    diff_pivot_table = pivot_table - reference_pivot_table
    diff_pivot_table.to_csv(os.path.join(figures_directory, 'share_proj_{}.csv'.format(
        suffix)))
    ax = diff_pivot_table.plot.line()
    figure = ax.get_figure()
    figure.savefig(os.path.join(figures_directory, 'share_diff_proj_{}.pdf'.format(
        suffix)), bbox_inches = 'tight')

    pct_pivot_table = pivot_table.divide(pivot_table.sum(axis=1), axis=0)
    pct_reference_pivot_table = reference_pivot_table.divide(reference_pivot_table.sum(axis=1), axis=0)
    diff_pct_pivot_table = pct_pivot_table - pct_reference_pivot_table
    ax = diff_pct_pivot_table.plot.line()
    figure = ax.get_figure()
    pct_pivot_table.to_csv(os.path.join(figures_directory, 'share_diff_proj_pct_{}.csv'.format(
        suffix)))
    figure.savefig(os.path.join(figures_directory, 'share_diff_proj_pct_{}.pdf'.format(suffix)), bbox_inches = 'tight')


def plot_dependance_niveau_by_age(population, period, sexe = None, area = False, pct = True, age_max = None,
          age_min = None, suffix = None):
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
        pivot_table = pivot_table.query('age <= @age_max').copy()

    if age_max:
        pivot_table = pivot_table.query('age >= @age_min').copy()

    xlim = [pivot_table.reset_index()['age'].min(), pivot_table.reset_index()['age'].max()]
    ylim = [0, 1]

    plot_kwargs = dict(
        color = colors,
        xlim = xlim,
        ylim = ylim,
        )
    pivot_table.index.name = u"Âge"
    # pivot_table.columns = ["0-1", "2", "3", "4"]

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


def run_scenario(uncalibrated_transitions = None, initial_population = None, initial_period = 2010, mu = None,
        survival_gain_cast = None, age_min = None):

    initial_population['period'] = initial_period
    population = initial_population.copy()

    uncalibrated_transitions = (uncalibrated_transitions
        .reset_index()
        .assign(period = initial_period)
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        )

    transitions = build_mortality_calibrated_target_from_transitions(
        transitions = uncalibrated_transitions,
        period = initial_period,
        dependance_initialisation = population,
        age_min = age_min,
        )

    period = initial_period

    transitions_by_period = dict()

    while period < 2058:
        print 'Running period {}'.format(period)
        period = population['period'].max()
        # plot_dependance_niveau_by_age(population, period)
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
#                plot_dependance_niveau_by_period(population, period)
#                plot_dependance_niveau_by_age(population, period)
#                plot_projected_target(age_min = 65, projected_target = transitions, years = [period],
#                    probability_name = 'calibrated_probability')
#                raw_input("Press Enter to continue...")
        # Iterate
        iterated_population = apply_transition_matrix(
            population = population.query('period == @period').copy(),
            transition_matrix = transitions,
            age_min = age_min,
            )
        check_67_and_over(iterated_population, age_min = age_min + 2)
        iterated_population = add_lower_age_population(population = iterated_population, age_min = age_min)
        population = pd.concat([population, iterated_population])

    return population, transitions_by_period


def save_data_and_graph(uncalibrated_transitions, mu = None, survival_gain_cast = None, vagues = None, age_min = None):
    assert age_min is not None
    log.info("Running with survival_gain_cast = {}".format(survival_gain_cast))
    initial_period = 2010
    initial_population = get_initial_population(age_min = age_min, rescale = True, period = initial_period)
    initial_population['period'] = initial_period
    population, transitions_by_period = run_scenario(
        uncalibrated_transitions = uncalibrated_transitions,
        initial_population = initial_population,
        mu = mu,
        survival_gain_cast = survival_gain_cast,
        age_min = age_min,
        )

    suffix = build_suffix(survival_gain_cast, mu, vagues)
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
    pivot_table.to_csv(os.path.join(figures_directory, 'share_proj_{}.csv'.format(
        suffix)))
    ax = pivot_table.plot.line()
    figure = ax.get_figure()
    figure.savefig(os.path.join(figures_directory, 'share_proj_{}.pdf'.format(
        suffix)), bbox_inches = 'tight')

    pct_pivot_table = pivot_table.divide(pivot_table.sum(axis = 1), axis = 0)
    ax = pct_pivot_table.plot.line()
    figure = ax.get_figure()
    pct_pivot_table.to_csv(os.path.join(figures_directory, 'share_proj_pct_{}.csv'.format(
        suffix)))
    figure.savefig(os.path.join(figures_directory, 'share_proj_pct_{}.pdf'.format(suffix)), bbox_inches = 'tight')


def graph_uncalibrated_transitions(initial_states = None, final_states = None):
    formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1) ** 2) + I(((age - 80) * 0.1) ** 3)'
    uncalibrated_transitions_12 = get_transitions_from_formula(formula = formula, vagues = [1, 2])
    uncalibrated_transitions_456 = get_transitions_from_formula(formula = formula, vagues = [4, 5, 6])

    vagues_periods = ["2004-2006", "2011-2013-2015"]
    uncalibrated_transitions_12['period'] = vagues_periods[0]
    uncalibrated_transitions_456['period'] = vagues_periods[1]
    uncalibrated_transitions = pd.concat([uncalibrated_transitions_12, uncalibrated_transitions_456])
    plot_projected_target(
        age_min = 50,
        age_max = 100,
        projected_target = uncalibrated_transitions,
        years = vagues_periods,
        probability_name = 'probability',
        french = True,
        save = True,
        title = 'Vagues SHARE',
        initial_states = initial_states,
        final_states = final_states,
        )


def comparison_share_early_late(age_min = 50):
    suffixes = ['homogeneous1_2', 'homogeneous4_5_6']
    population = pd.DataFrame()
    for suffix in suffixes:
        population = pd.concat([
            population,
            (pd.read_csv(os.path.join(figures_directory, 'population_{}.csv'.format(suffix)), index_col = 0)
                .drop('sex', axis = 1)
                .query('age >= @age_min')
                .query('(period > 2010) & (period <= 2060)')
                # .query('initial_state in @dependance_niveaux')
                .assign(scenario = suffix)
                .groupby(['period', 'initial_state', 'scenario'])['population'].sum()
                .reset_index())
                .rename(columns = {'period': u'Années'})
            ])

    fig, ax = plt.subplots()
    linestyle_by_scenario = {
        'homogeneous1_2': "--",
        'homogeneous4_5_6': "-",
        }

    for scenario, grp in sorted(population.groupby('scenario'), reverse = True):
        pivot_table = grp.groupby([u'Années', 'initial_state'])['population'].sum().unstack()
        if pivot_table.max().max() > 1e6:
            pivot_table = pivot_table / 1e6
            unite = 'Effectifs (millions)'
        else:
            pivot_table = pivot_table / 1e3
            unite = 'Effectifs (milliers)'

        lines = pivot_table.plot.line(
            label =str(scenario),
            ax=ax,
            linestyle = linestyle_by_scenario[scenario],
            color = ['b', 'g', 'y', 'r'],
            xlim = [2012, 2060],
            )
        ax.set_ylabel(unite)

    ax.legend(["0", "1", "2", "3"], frameon = True, edgecolor = 'k',
        framealpha = 1, title = u" 2004-06     --\n 2011-13-15 -")

    figure = ax.get_figure()
    figure_path_name = os.path.join(figures_directory, 'multi_scenario_vagues')
    # ax.set_title(u'Effectifs dépendants {}'.format(dependance_niveaux))
    figure.savefig(figure_path_name, bbox_inches = 'tight')
    figure.savefig(figure_path_name + ".pdf", bbox_inches = 'tight', format = 'pdf')


def comparison_share_paquid(age_min = 65):
    suffix = 'homogeneous4_5_6'
    share_population = (pd.read_csv(os.path.join(figures_directory, 'population_{}.csv'.format(suffix)), index_col = 0)
        .drop('sex', axis = 1)
        .query('age >= @age_min')
        .query('(period > 2010) & (period <= 2060)')
        .replace({'initial_state': {1: 0}})
        .assign(scenario = suffix)
        .groupby(['period', 'initial_state', 'scenario'])['population'].sum()
        .reset_index()
        .rename(columns = {'period': u'Années'})
        )

    suffix = 'homogeneous'
    paquid_population = (pd.read_csv(
            os.path.join(figures_directory, '..', 'population_{}.csv'.format(suffix)),
            index_col = 0
            )
        .drop('sex', axis = 1)
        .query('age >= @age_min')
        .query('(period > 2010) & (period <= 2060)')
        .replace({'initial_state': {4: 3}})
        .assign(scenario = suffix)
        .groupby(['period', 'initial_state', 'scenario'])['population'].sum()
        .reset_index()
        .rename(columns = {'period': u'Années'})
        )

    population = pd.concat([share_population, paquid_population])

    fig, ax = plt.subplots()
    linestyle_by_scenario = {
        'homogeneous': "--",
        'homogeneous4_5_6': "-",
        }

    for scenario, grp in sorted(population.groupby('scenario'), reverse = True):
        pivot_table = grp.groupby([u'Années', 'initial_state'])['population'].sum().unstack()
        if pivot_table.max().max() > 1e6:
            pivot_table = pivot_table / 1e6
            unite = 'Effectifs (millions)'
        else:
            pivot_table = pivot_table / 1e3
            unite = 'Effectifs (milliers)'

        lines = pivot_table.plot.line(
            label =str(scenario),
            ax=ax,
            linestyle = linestyle_by_scenario[scenario],
            color = ['b', 'y', 'r'],
            xlim = [2012, 2060],
            )
        ax.set_ylabel(unite)

    ax.legend(["0-1", "2", "3-4"], frameon = True, edgecolor = 'k',
        framealpha = 1, title = u" PAQUID --\n SHARE -")

    figure = ax.get_figure()
    figure_path_name = os.path.join(figures_directory, 'paquid_vs_share')
    # ax.set_title(u'Effectifs dépendants {}'.format(dependance_niveaux))
    figure.savefig(figure_path_name, bbox_inches = 'tight')
    figure.savefig(figure_path_name + ".pdf", bbox_inches = 'tight', format = 'pdf')


def plot_share_late(sex = None, age_min = 65):
    suffix = 'homogeneous4_5_6'
    figure_path_name = os.path.join(figures_directory, 'scenario_share_late')
    population = pd.read_csv(os.path.join(figures_directory, 'population_{}.csv'.format(suffix)), index_col = 0)

    if sex is not None:
        assert sex in ['male', 'female']
        population = population.query('sex == @sex')
        figure_path_name = figure_path_name + '_' + str(sex)

    population = (population
        .drop('sex', axis = 1)
        .query('age >= @age_min')
        .query('(period > 2010) & (period <= 2060)')
        # .query('initial_state in @dependance_niveaux')
        .assign(scenario = suffix)
        .groupby(['period', 'initial_state', 'scenario'])['population'].sum()
        .reset_index()
        .rename(columns = {'period': u'Années'})
        )

    fig, ax = plt.subplots()

    pivot_table = population.groupby([u'Années', 'initial_state'])['population'].sum().unstack()
    if pivot_table.max().max() > 1e6:
        pivot_table = pivot_table / 1e6
        unite = 'Effectifs (millions)'
    else:
        pivot_table = pivot_table / 1e3
        unite = 'Effectifs (milliers)'

    ax = pivot_table.plot.line(
        ax = ax,
        linestyle = '-',
        color = ['b', 'g', 'y', 'r'],
        xlim = [2012, 2060],
        ylim = [0, 6] if sex is not None else [0, 12],
        )
    ax.set_ylabel(unite)

    ax.legend(["0", "1", "2", "3"], frameon = True, edgecolor = 'k',
    framealpha = 1)

    figure = ax.get_figure()

    # ax.set_title(u'Effectifs dépendants {}'.format(dependance_niveaux))
    figure.savefig(figure_path_name, bbox_inches = 'tight')
    figure.savefig(figure_path_name + ".pdf", bbox_inches = 'tight', format = 'pdf')


def run(survival_gain_casts = None, uncalibrated_transitions = None, vagues = [4, 5, 6], age_min = None):
    assert age_min is not None
    assert uncalibrated_transitions is not None
    create_dependance_initialisation_share(smooth = True, survey = 'care', age_min = age_min)
    # life_expectancy_diagnostic(uncalibrated_transitions = uncalibrated_transitions, initial_period = 2010)

    for survival_gain_cast in survival_gain_casts:
        if survival_gain_cast in ['initial_vs_others', 'autonomy_vs_disability']:
            for mu in [0, 1]:
                save_data_and_graph(
                    uncalibrated_transitions = uncalibrated_transitions,
                    survival_gain_cast = survival_gain_cast,
                    mu = mu,
                    vagues = vagues,
                    age_min = age_min,
                    )
        else:
            save_data_and_graph(
                uncalibrated_transitions = uncalibrated_transitions,
                survival_gain_cast = survival_gain_cast,
                vagues = vagues,
                age_min = age_min,
                )


def graph_transitions_by_label(transitions_by_label, initial_states = None, final_states = None, title = None,
        age_min = None, age_max = None, title_template = None, legend_title = None):

    for label, transitions in transitions_by_label.iteritems():
        transitions['period'] = label

    concatenated_transitions= pd.concat(transitions_by_label.values())

    plot_projected_target(
        age_min = age_min,
        age_max = age_max,
        projected_target = concatenated_transitions,
        years = transitions_by_label.keys(),
        probability_name = 'probability',
        french = True,
        save = True,
        title = title,
        initial_states = initial_states,
        final_states = final_states,
        title_template = title_template,
        legend_title = legend_title,
        )



if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO, stream = sys.stdout)
    sns.set_style("whitegrid")

    STOP
#
#    print("til = {} vs insee = {}".format(
#        data_bis.groupby(['sex']).population.sum() / 1e6,
#        (insee_population
#            .query("year == @year")
#            .query("50 <= age")
#            .groupby(['sex']).population.sum() / 1e6,
#            )
#        ))
#
#
    transitions_by_label = {
#        "2004-2006": get_transitions_from_formula(formula = formula, vagues = [1, 2]),
#        "2011-2013-2015": get_transitions_from_formula(formula = formula, vagues = [4, 5, 6]),
        "Standard": get_transitions_from_file(),
        "Sans Alzheimer": get_transitions_from_file(alzheimer = 0),
        "Alzheimer": get_transitions_from_file(alzheimer = 1),
        }

    graph_transitions_by_label(transitions_by_label, age_min = 65, age_max = 90, title_template = "Effet Alzheimer - {}")

#
#        insee_pyramid_data = (insee_population.reset_index()
#           .query("year == @year")
#           .query("sex == @sex")
#            .query("50 <= age <= 100")
#            .sort_values('age')
#            )
#
#        bar_plot = sns.barplot(
#            x = "population",
#            y = "age",
#            color = "blue",
#            order = range(100, 49, -1),
#            data = pyramid_data,
#            orient = 'h'
#            )
#
#        insee_bar_plot = sns.barplot(
#            x = "population",
#            y = "age",
#            color = "blue",
#            order = range(100, 49, -1),
#            data = insee_pyramid_data,
#            orient = 'h'
#            )
#
#        insee_pyramid_data.set_index(['sex', 'age']) / pyramid_data.set_index(['sex', 'age'])
#
#        print("til = {} vs insee = {}".format(
#            pyramid_data.population.sum() / 1e6,
#            insee_pyramid_data.population.sum() / 1e6,
#            ))
#
#    BADABOUM


    # uncalibrated_transitions = get_transitions_from_file()
    vagues = [1, 2]
    formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1) ** 2) + I(((age - 80) * 0.1) ** 3)'
    uncalibrated_transitions = get_transitions_from_formula(formula = formula, vagues = vagues)

    survival_gain_casts = [
        'homogeneous',
        ]
    run(survival_gain_casts, uncalibrated_transitions = uncalibrated_transitions, vagues = vagues, age_min = 50)
    BOUM

    load_and_plot_gir_projections(survival_gain_cast = 'homogeneous', vagues = [4, 5, 6], age_min = 60)
    BAM

    comparison_share_early_late()
    BIM


#
    plot_share_late(sex = 'female')
#    BIM
    transitions = load_and_plot_projected_target(
        survival_gain_cast = 'homogeneous',
        age_min = 50,
        age_max = 120,
        vagues = [4, 5, 6],
        )
    BAM
    load_and_plot_dependance_niveau_by_period(survival_gain_cast = 'homogeneous', periods = None, area = True, vagues = [4, 5, 6])

    BIM
    transitions_by_label = {
        "Standard": get_transitions_from_file(),
        "Sans Alzheimer": get_transitions_from_file(alzheimer = 0),
        "Alzheimer": get_transitions_from_file(alzheimer = 1),
        }

    graph_transitions_by_label(
        transitions_by_label,
        age_min = 65,
        age_max = 90,
        title_template = "Effet Alzheimer - {}",
        legend_title = "",
        initial_states = [0, 1],
        final_states = [0, 1, 2, 4],
        )

    formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1) ** 2) + I(((age - 80) * 0.1) ** 3)'
    transitions_by_label = {
        "2004-2006": get_transitions_from_formula(formula = formula, vagues = [1, 2], age_min = 50, age_max = 100),
        "2011-2013-2015": get_transitions_from_formula(formula = formula, vagues = [4, 5, 6], age_min = 50, age_max = 100),
        }

    graph_transitions_by_label(
        transitions_by_label,
        age_min = 50,
        age_max = 95,
        title_template = "Effet vagues SHARE- {}",
        legend_title = "Vagues",
        initial_states = [0],
#        final_states = [0, 1, 2, 4],
        )


    transitions_by_label = {
        "Standard": get_transitions_from_file(),
        "Mémoire": get_transitions_from_file(memory = True),
        }

    graph_transitions_by_label(
        transitions_by_label,
        age_min = 50,
        age_max = 95,
        title_template = u"Effet mémoire - {}",
        legend_title = u"Spécifications",
#        initial_states = [0],
#        final_states = [0, 1, 2, 4],
        )

