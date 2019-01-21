# -*- coding: utf-8 -*-


from __future__ import division


from decimal import *
import logging
import math
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

from til_france.plot.population import get_insee_projection


from til_france.model.options.dependance_RT.life_expectancy.share.paths_prog import (
    til_france_path,
    )


log = logging.getLogger(__name__)

config = Config()
figures_directory = config.get('dependance', 'figures_directory')


eligible_survival_gain_casts = ['homogeneous', 'initial_vs_others', 'autonomy_vs_disability']


# Fonctions principales

def run(survival_gain_casts = None, mu = None, uncalibrated_transitions = None, vagues = [4, 5, 6], age_min = None,
        prevalence_survey = None, one_year_approximation = None, age_max_cale = None):
    """
        Run

        :param list vagues: share data waves used for transition estimation
        :param int age_min:
        :param DataFrame uncalibrated_transitions:
        :param str prevalence_survey: survey used to compute initial prevalence, should be 'care', 'hsm' or 'hsm_hsi'
        :param int age_max_cale:
        :param list survival_gain_casts: list of eligible survival_gain_casts, should be 'autonomy_vs_disability', 'homogeneous' or 'initial_vs_others',
    """

    assert vagues is not None
    assert age_min is not None
    assert uncalibrated_transitions is not None
    assert prevalence_survey in ['care', 'hsm', 'hsm_hsi']
    assert age_max_cale is not None
    assert set(survival_gain_casts) < set(eligible_survival_gain_casts)

    create_initial_prevalence(smooth = True, prevalence_survey = prevalence_survey, age_min = age_min)
    for survival_gain_cast in survival_gain_casts:
        save_data_and_graph(
            uncalibrated_transitions = uncalibrated_transitions,
            survival_gain_cast = survival_gain_cast,
            vagues = vagues,
            age_min = age_min,
            prevalence_survey = prevalence_survey,
            one_year_approximation = one_year_approximation,
            mu = mu,
            age_max_cale = age_max_cale
            )


def save_data_and_graph(uncalibrated_transitions, mu = None, survival_gain_cast = None, vagues = None, age_min = None, prevalence_survey = None, one_year_approximation = None, age_max_cale = None):
    if survival_gain_cast in ['initial_vs_others', 'autonomy_vs_disability']:
        assert mu is not None

    log.info("Running with survival_gain_cast = {}".format(survival_gain_cast))
    initial_period = 2010
    initial_population = get_initial_population(age_min = age_min, rescale = True, period = initial_period, prevalence_survey = prevalence_survey)
    initial_population['period'] = initial_period
    population, transitions_by_period = project_disability(
        uncalibrated_transitions = uncalibrated_transitions,
        initial_population = initial_population,
        mu = mu,
        survival_gain_cast = survival_gain_cast,
        age_min = age_min,
        prevalence_survey = prevalence_survey,
        one_year_approximation = one_year_approximation,
        age_max_cale = age_max_cale
        )

    # Save data
    suffix = build_suffix(survival_gain_cast, mu, vagues, prevalence_survey)
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

    # Plot data
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


def project_disability(uncalibrated_transitions = None, initial_population = None, initial_period = 2010, mu = None,
        survival_gain_cast = None, age_min = None, prevalence_survey = None, one_year_approximation = None, age_max_cale = None):
    """
        Project disabilyt levels by simulation according to a scenario specifying the survival gain casts

        :param DataFrame uncalibrated_transitions: transition matrix between different disability levels
        :param DataFrame initial_population: population at the initial period
        :param int initial_period: initial period of the simulation
        :param float mu: optional parameter for scenarios, should be in the [0, 1] interval.
        :param str survival_gain_cast: scenario for survival gain casting. Should be 'autonomy_vs_disability', 'homogeneous' or 'initial_vs_others'
        :param int age_min: minimal age for people to be disabled
        :param str prevalence_survey: survey used to compute initial prevalence, should be 'care', 'hsm' or 'hsm_hsi'
        :param one_year_approximation:
        :param int age_max_cale: minimal age at which the disability level transitions are freezed
    """
    assert prevalence_survey is not None
    assert age_max_cale is not None
    assert uncalibrated_transitions is not None

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
        one_year_approximation = one_year_approximation,
        survival_gain_cast = survival_gain_cast,
        mu = mu,
        age_max_cale = age_max_cale,
        uncalibrated_transitions = uncalibrated_transitions
        )
    period = initial_period
    transitions_by_period = dict()

    while period < 2058:
        log.info('Running period {}'.format(period))
        period = population['period'].max()
        if period > initial_period:
            dependance_initialisation = population.query('period == @period').copy()
            # Update the transitions matrix if necessary
            admissible_scenarios = ['homogeneous', 'initial_vs_others', 'autonomy_vs_disability']
            assert survival_gain_cast in admissible_scenarios, "survival_gain_cast should be int the following list:\n  {}".format(
                admissible_scenarios)
            log.info("Calibrate transitions for period = {} usning survival_gain_cast = {}".format(
                period, survival_gain_cast))
            delta = 1e-7
            transitions = regularize(
                transition_matrix_dataframe = transitions.rename(
                    columns = {'calibrated_probability': 'probability'}
                    ),
                by = ['period', 'sex', 'age', 'initial_state'],
                probability = 'probability',
                delta = delta,
                )
            transitions = build_mortality_calibrated_target_from_transitions(
                transitions = transitions,
                period = period,
                dependance_initialisation = dependance_initialisation,
                age_min = age_min,
                one_year_approximation = one_year_approximation,
                survival_gain_cast = survival_gain_cast,
                mu = mu,
                age_max_cale = age_max_cale,
                uncalibrated_transitions = uncalibrated_transitions
                )
            transitions_by_period[period] = transitions

        # Iterate
        iterated_population = apply_transition_matrix(
            population = population.query('period == @period').copy(),
            transition_matrix = transitions,
            age_min = age_min,
            age_max_cale = age_max_cale
            )
        check_67_and_over(iterated_population, age_min = age_min + 2)
        iterated_population = add_lower_age_population(population = iterated_population, age_min = age_min, prevalence_survey = prevalence_survey)
        population = pd.concat([population, iterated_population], sort = True)

    return population, transitions_by_period


def create_initial_prevalence(filename_prefix = None, smooth = False, window = 7, std = 2,
        prevalence_survey = None, age_min = None, scale = 4):
    """
        Create dependance_niveau variable initialisation file for use in til-france model (option dependance_RT)
    """
    assert scale in [4, 5], "scale should be equal to 4 or 5"
    assert age_min is not None
    config = Config()
    input_dir = config.get('til', 'input_dir')
    assert prevalence_survey in ['care', 'hsm', 'hsm_hsi']
    for sexe in ['homme', 'femme']:
        if prevalence_survey == 'hsm':
            pivot_table = get_hsm_prevalence_pivot_table(sexe = sexe, scale = 4)
        elif prevalence_survey == 'hsm_hsi':
            pivot_table = get_hsi_hsm_prevalence_pivot_table(sexe = sexe, scale = 4)
        elif prevalence_survey == 'care':
            pivot_table =  get_care_prevalence_pivot_table(sexe = sexe, scale = 4)
        #assert prevalence_survey is not None

        if filename_prefix is None:
            filename = os.path.join(input_dir, 'dependance_initialisation_level_{}_{}.csv'.format(prevalence_survey, sexe)) # dependance_initialisation_level_share_{}
        else:
            filename = os.path.join('{}_level_{}_{}.csv'.format(filename_prefix, prevalence_survey, sexe))
        level_pivot_table = (pivot_table.copy()
            .reset_index()
            .merge(pd.DataFrame({'age': range(0, 121)}), how = 'right')
            .sort_values('age')
            )
        level_pivot_table['age'] = level_pivot_table['age'].astype(int)
        level_pivot_table.fillna(0, inplace = True)

        if smooth:
            smoothed_pivot_table = smooth_pivot_table(pivot_table, window = window, std = std)
            # The windowing NaNifies some values on the edge age = age_min, we reuse the rough data for those ages
            smoothed_pivot_table.update(pivot_table, overwrite = False)
            pivot_table = smoothed_pivot_table.copy()
            del smoothed_pivot_table

        level_pivot_table.to_csv(filename, index = False)
        log.info('Saving {}'.format(filename))

        # Go from levels to pct
        pivot_table = pivot_table.divide(pivot_table.sum(axis=1), axis=0)
        if filename_prefix is None:
            filename = os.path.join(input_dir, 'dependance_initialisation_{}_{}.csv'.format(prevalence_survey,sexe))
        else:
            filename = os.path.join('{}_{}_{}.csv'.format(filename_prefix, prevalence_survey, sexe)) #prevalence_survey insere dans le nom du doc

        if filename is not None:
            pivot_table = (pivot_table
                .reset_index()
                .merge(pd.DataFrame({'age': range(0, 121)}), how = 'right')
                .sort_values('age')
                )
            pivot_table['age'] = pivot_table['age'].astype(int)

            pivot_table.fillna(0, inplace = True)
            pivot_table.loc[pivot_table.age < age_min, 0] = 1
            pivot_table.set_index('age', inplace = True)
            pivot_table.loc[pivot_table.sum(axis = 1) == 0, scale - 1] = 1
            pivot_table.to_csv(filename, header = False)
            assert ((pivot_table.sum(axis = 1) - 1).abs() < 1e-15).all(), \
                pivot_table.sum(axis = 1).loc[((pivot_table.sum(axis = 1) - 1).abs() > 1e-15)]

            # Add liam2 compatible header
            verbatim_header = '''age,initial_state,,,
,0,1,2,3,4
'''
            with open(filename, 'r') as original:
                data = original.read()
            with open(filename, 'w') as modified:
                modified.write(verbatim_header + data)
            log.info('Saving {}'.format(filename))


def get_care_prevalence_pivot_table(sexe = None, scale = None):
    config = Config()
    assert scale in [4, 5], "scale should be equal to 4 or 5"
    xls_path = os.path.join(
        config.get('raw_data', 'hsm_dependance_niveau'), 'care_extract.xlsx')
    data = (pd.read_excel(xls_path)
        .rename(columns = {
            'scale_v1': 'dependance_niveau',
            'femme': 'sexe',
            })
        )

    assert sexe in ['homme', 'femme']
    sexe = 1 if sexe == 'femme' else 0
    assert sexe in data.sexe.unique(), "sexe should be in {}".format(data.sexe.unique().tolist())
    pivot_table = (data[['dependance_niveau', 'poids_care', 'age', 'sexe']]
        .query('sexe == @sexe')
        .groupby(['dependance_niveau', 'age'])['poids_care'].sum().reset_index()
        .pivot('age', 'dependance_niveau', 'poids_care')
        .replace(0, np.nan)  # Next three lines to remove all 0 columns
        .dropna(how = 'all', axis = 1)
        .replace(np.nan, 0)
        )

    return pivot_table


## get_initial_population : Fonction dont l'output est la table data (variables : periode  | age      | initial_state | population    | sex)

def get_initial_population(age_min = None, period = None, rescale = True, prevalence_survey = None):
    """
        Produce intiial population with disabiliyt state

        :param int age_min: minimal age to retain
        :param bool rescale: rescale using INSEE population
        :param str prevalence_survey: survey used to compute initial prevalence, should be 'care', 'hsm' or 'hsm_hsi'


    """
    assert age_min is not None
    assert prevalence_survey is not None
    if rescale:
        assert period is not None
    data_by_sex = dict()
    for sex in ['male', 'female']:
        sexe = 'homme' if sex == 'male' else 'femme'
        config = Config()
        filename = os.path.join(
            config.get('til', 'input_dir'),
            'dependance_initialisation_level_{}_{}.csv'.format(prevalence_survey, sexe)
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


def add_lower_age_population(population = None, age_min = None, prevalence_survey = None):
    """
        Ajoute dans la base des personnes aux âges les plus bas
    """
    assert age_min is not None
    assert population is not None
    assert len(population.period.unique().tolist()) == 1, 'More than one period are present: {}'.format(
        population.period.unique().tolist())
    period = population.period.unique().tolist()[0]
    lower_age_population = (get_initial_population(age_min = age_min, rescale = True, period = period, prevalence_survey = prevalence_survey)
        .query('age in [@age_min, @age_min + 1]')
        )
    lower_age_population['period'] = period
    completed_population = pd.concat(
        [lower_age_population, population],
        sort = True,
        ).sort_values(
            ['period', 'age', 'sex', 'initial_state']
            )

    assert completed_population.notnull().all().all(), completed_population.notnull().all()

    assert completed_population.notnull().all().all(), 'Missing values are present: {}'.format(
        completed_population.loc[completed_population.isnull()])

    return completed_population


#  Mortalites calibrees
def build_mortality_calibrated_target_from_transitions(transitions = None, period = None, dependance_initialisation = None,
       age_min = None, one_year_approximation = None, survival_gain_cast = None, mu = None, age_max_cale = None, uncalibrated_transitions = None):
    assert age_min is not None
    assert period is not None
    assert transitions is not None
    assert age_max_cale is not None
    assert uncalibrated_transitions is not None

    mortality_calibrated_target = build_mortality_calibrated_target(
        transitions = transitions,
        period = period,
        dependance_initialisation = dependance_initialisation,
        age_min = age_min,
        one_year_approximation = one_year_approximation,
        survival_gain_cast = survival_gain_cast,
        mu = mu,
        age_max_cale = age_max_cale,
        uncalibrated_transitions = uncalibrated_transitions
        )
    assert_probabilities(
        dataframe = mortality_calibrated_target,
        by = ['sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )
    return mortality_calibrated_target



def build_mortality_calibrated_target(transitions = None, period = None, dependance_initialisation = None, scale = 4,
        age_min = None, one_year_approximation = None, survival_gain_cast = None, mu = None, age_max_cale = None, uncalibrated_transitions = None):
    """
    Compute the calibrated mortality by sex, age and disability state (initial_state) for a given period
    using data on the disability states distribution in the population at that period
    if dependance_initialisation = None
    TODO should be merged with build_mortality_calibrated_target_from_transitions
    """
    assert age_min is not None
    assert age_max_cale is not None
    assert uncalibrated_transitions is not None

    assert scale in [4, 5]
    if scale == 5:
        death_state = 5
    elif scale == 4:
        death_state = 4

    assert (transitions is not None) and (period is not None)
    calibrated_transitions = _get_calibrated_transitions(
        period = period,
        transitions = transitions,
        dependance_initialisation = dependance_initialisation,
        one_year_approximation = one_year_approximation,
        survival_gain_cast = survival_gain_cast,
        mu = mu,
        age_max_cale = age_max_cale,
        uncalibrated_transitions = uncalibrated_transitions
        )

    null_fill_by_year = calibrated_transitions.reset_index().query('age == @age_min').copy()
    null_fill_by_year['calibrated_probability'] = 0

    # Less than age_min years old -> no correction
    pre_age_min_null_fill = pd.concat([
        null_fill_by_year.assign(age = i).copy()
        for i in range(0, age_min)
        ]).reset_index(drop = True)

    pre_age_min_null_fill.loc[
        pre_age_min_null_fill.initial_state == pre_age_min_null_fill.final_state,
        'calibrated_probability'
        ] = 1

    assert_probabilities(
        dataframe = pre_age_min_null_fill,
        by = ['sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )

    # More than age_min years old
    age_max = calibrated_transitions.index.get_level_values('age').max()
    if age_max < 120:
        elder_null_fill = pd.concat([
            null_fill_by_year.assign(age = i).copy()
            for i in range(age_max + 1, 121)
            ]).reset_index(drop = True)

        elder_null_fill.loc[
            (elder_null_fill.age > age_max) & (elder_null_fill.final_state == death_state),
            'calibrated_probability'
            ] = 1
        assert_probabilities(
            dataframe = elder_null_fill,
            by = ['sex', 'age', 'initial_state'],
            probability = 'calibrated_probability',
            )
        age_full = pd.concat([
            pre_age_min_null_fill,
            calibrated_transitions.reset_index(),
            elder_null_fill
            ]).reset_index(drop = True)
    else:
        age_full = pd.concat([
            pre_age_min_null_fill,
            calibrated_transitions.reset_index(),
            ]).reset_index(drop = True)

    age_full['period'] = period
    mortality_calibrated_target = age_full[
        ['period', 'sex', 'age', 'initial_state', 'final_state', 'calibrated_probability']
        ].set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])

    assert_probabilities(
        dataframe = mortality_calibrated_target,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )

    return mortality_calibrated_target


def _get_calibrated_transitions(period = None, transitions = None, dependance_initialisation = None, one_year_approximation = None,
        survival_gain_cast = None, mu = None, age_max_cale = None, uncalibrated_transitions = None):
    """
    Calibrate transitions to match mortality from a specified period
    """
    death_state = 4
    assert (period is not None) and (transitions is not None)
    assert age_max_cale is not None
    assert uncalibrated_transitions is not None

    # Add calibration_coeffcients for mortality
    calibration = _compute_calibration_coefficient(
        period = period,
        transitions = transitions,
        dependance_initialisation = dependance_initialisation,
        one_year_approximation = one_year_approximation,
        )

    # Selectionne calibration jusque age_max
    calibration = calibration.loc[calibration['age'] < age_max_cale]

    assert not calibration.reset_index()[['sex', 'age']].duplicated().any(), \
        calibration.reset_index().loc[calibration.reset_index()[['sex', 'age']].duplicated()]

    # Calibrate mortality
    assert not transitions.reset_index()[['sex', 'age', 'initial_state', 'final_state']].duplicated().any(), \
        transitions.reset_index().loc[transitions.reset_index()[
            ['sex', 'age', 'initial_state', 'final_state']].duplicated()]

    mortality = (transitions
        .reset_index()
        .query('final_state == @death_state')
        .query('age < @age_max_cale') #a remplacer par age_max_cale
        .merge(
            calibration[['sex', 'age', 'cale_mortality_2_year']].copy(),
            on = ['sex', 'age'],
            )
        )
    mortality['calibrated_probability'] = np.minimum(
       mortality.probability * mortality.cale_mortality_2_year, 1)  # Avoid over corrections !

    # Dans df transitions on a encore les transitions vers le meme etat

    # Cree beta qui varie selon les SCENARIOS et applique beta

    ## CREATION DES BETA pour le scenario HOMOGENEOUS
    if survival_gain_cast == 'homogeneous':
        log.debug("Using homogeneous scenario")
        mortality.eval(
            'cale_other_transitions = (1 - calibrated_probability) / (1 - probability)',
            inplace = True,
            )
        assert not mortality[['sex', 'age', 'initial_state', 'final_state']].duplicated().any(), mortality.loc[mortality[['sex', 'age', 'initial_state', 'final_state']].duplicated()]

        # Calibrate other transitions
        cale_other_transitions = (mortality[['sex', 'age', 'initial_state', 'cale_other_transitions']]
            .copy()
            .reset_index()
            .set_index(['sex', 'age', 'initial_state'])
        )
        other_transitions = (transitions
            .reset_index()
            .set_index(['sex', 'age', 'initial_state'])
            .query('final_state != @death_state')
            .merge(
                cale_other_transitions,
                on = ['sex', 'age', 'initial_state'],
                )
            .eval('calibrated_probability = probability * cale_other_transitions', inplace = False)
            )

        log.debug("Utilise les cales pour calibrated_probability")
        assert other_transitions.calibrated_probability.notnull().all(), \
            other_transitions.loc[other_transitions.calibrated_probability.isnull()]

    ##CREATION DES BETA pour le scenario INITIAL VS OTHERS
    elif survival_gain_cast == "initial_vs_others":
        log.debug("Using initial_vs_others scenario")
        assert mu is not None
        mortality = (mortality
            .rename(columns = {'calibrated_probability': 'periodized_calibrated_probability'})
            )
        # Gain in survival probability feeds by a proportion of mu the initial_state and 1 - mu the other states
        other_transitions = initial_vs_others(
            # period = period,
            mortality = mortality.rename(
                columns = {'probability': 'calibrated_probability'}
                ),
            mu = mu,
            uncalibrated_probabilities = transitions.rename(
                columns = {'probability': 'calibrated_probability'}
                )
            )

        other_transitions = (other_transitions
            .set_index(['sex', 'age'])
            .sort_index()
            .fillna(method = 'ffill') #Remplit avec la valeur de l'age precedent pour eviter des missings aux ages eleves
            .reset_index()
            .set_index(['sex', 'age', 'initial_state', 'final_state'])
            )

        # Keep columns we want (en plus de l'index)
        other_transitions = other_transitions[['periodized_calibrated_probability']]

        # Calibrate other transitions
        other_transitions = (transitions
            .reset_index()
            .set_index(['sex', 'age', 'initial_state', 'final_state'])
            .query('final_state != @death_state')
            .merge(
                other_transitions,
                on = ['sex', 'age', 'initial_state', 'final_state'],
                )
            .rename(columns = {'periodized_calibrated_probability': 'calibrated_probability'})
            .reset_index()
            .set_index(['sex', 'age'])
            .sort_index()
            .fillna(method = 'ffill') # Remplit avec la valeur de l'age precedent pour eviter des missings aux ages eleves
            .reset_index()
            .set_index(['sex', 'age', 'initial_state', 'final_state'])
            #.rename(columns = {'periodized_calibrated_probability': 'cale_other_transition2'}) #Pour ne pas melanger ensuite
            )

     ##CREATION DES BETA pour le scenario AUTONOMY VS DISABILITY
    elif survival_gain_cast == 'autonomy_vs_disability':
            log.debug("Using autonomy_vs_disability scenario")
            assert mu is not None
            mortality = (mortality
            .rename(columns = {'calibrated_probability': 'periodized_calibrated_probability'})
            )
            other_transitions = autonomy_vs_disability(
                mortality = mortality.rename(
                    columns = {'probability': 'calibrated_probability'}),
                mu = mu,
                uncalibrated_probabilities = transitions.rename(columns = {'probability': 'calibrated_probability'})
                )

            #Keep columns we want (en plus de l'index)
            other_transitions = other_transitions[['periodized_calibrated_probability']]
            other_transitions = other_transitions.rename(columns = {'periodized_calibrated_probability': 'calibrated_probability'})

    else:
            raise NotImplementedError

    calibrated_transitions = pd.concat(
        [
            mortality.reset_index().set_index(['sex', 'age', 'initial_state', 'final_state']).sort_index(),
            other_transitions.reset_index().set_index(['sex', 'age', 'initial_state', 'final_state']).sort_index()
            ],
        sort = True
        ).sort_index()


    # Remplace les missings de calibrated proba par periodized calibrated proba si final_state=4 pour avoir une variable unique
    if (calibrated_transitions['calibrated_probability']).isnull().any():
        calibrated_transitions['calibrated_probability'] = calibrated_transitions['calibrated_probability'].fillna(value=calibrated_transitions['periodized_calibrated_probability'])


    #Correction

    # if (calibrated_transitions['calibrated_probability']).isnull().any():  # Deal with age > 100 with nobody in the population
    #     print("missing dans calibrated_probability")
    #     calibrated_transitions = (calibrated_transitions
    #         .reset_index()
    #         .loc[(calibrated_transitions.age >= 100) & (isnull(calibrated_transitions.calibrated_probability))], eval('calibrated_probability = 1 - periodized_calibrated_probability', inplace = False)
    #         .set_index(['sex', 'age', 'initial_state', 'final_state'])
    #         )

    #print("passe par l'imputation periodized_calibrated_probability recalibree")

    # Verification
    assert_probabilities(
        calibrated_transitions, by = ['sex', 'age', 'initial_state'], probability = 'calibrated_probability')

    #Impute les transitions aux ages eleves
    calibrated_transitions = impute_high_ages(
            data_to_complete = calibrated_transitions,
            uncalibrated_transitions = uncalibrated_transitions,
            period = period,
            age_max_cale = age_max_cale
        )
    return calibrated_transitions['calibrated_probability']


def impute_high_ages(data_to_complete = None, uncalibrated_transitions = None, period = None, age_max_cale = None):
    """
        Impute probability transition for high ages by forward filling values using a threshold age

        :param DataFrame data_to_complete: data to be complted by imputation
        :param DataFrame uncalibratd_transitions: disability transitions table
        :param int period: the period of the simulation
        :param int age_max_cale: minimal age at which the disability level transitions are freezed
    """

    assert age_max_cale is not None
    assert uncalibrated_transitions is not None
    transitions_data = (uncalibrated_transitions
        .reset_index()
        .query('age >= @age_max_cale')
        .assign(calibrated_probability = np.nan)
        .set_index(['sex', 'age', 'initial_state', 'final_state'])
        )

    complete_data = pd.concat([data_to_complete, transitions_data], sort = True)
    complete_data = (complete_data
        .reset_index()
        .set_index(['sex', 'initial_state', 'final_state', 'age'])
        .sort_index()
        .fillna(method = 'ffill')  # forward fill to avoid missings valeus at high ages
        .reset_index()
        .set_index(['sex', 'age', 'initial_state', 'final_state'])
        )

    return complete_data


def _compute_calibration_coefficient(age_min = 50, period = None, transitions = None, dependance_initialisation = None,
        one_year_approximation = None):
    """
        Calibrate mortality using the distribution of the disability states within population at a specific year
        for the given transition matrix and distribution of intiial_states

        Assuming the transition occur on a two-year period.

        From 2 years mortality to 1 year mortality by age, sex and intial_state if one_year_approximation is True
        2 years mortality by age, sex and intial_state if one_year_approximation is False

    """
    assert period is not None, "Mortality profile period is not set"
    assert transitions is not None
    predicted_mortality_table = get_predicted_mortality_table(transitions = transitions, one_year_approximation = one_year_approximation)
    # From 1 year mortality by age sex (use dependance_initialisation to sum over initial_state)
    mortality_after_imputation = (
        get_mortality_after_imputation(
            mortality_table = predicted_mortality_table,
            dependance_initialisation = dependance_initialisation,
            )
        .reset_index()
        .rename(columns = {'mortality_after_imputation': 'avg_mortality'})
        )

    assert (mortality_after_imputation.avg_mortality > 0).all(), \
        mortality_after_imputation.loc[~(mortality_after_imputation.avg_mortality > 0)]

    mortality_insee =(get_insee_projected_mortality()
        .query('year == @period')
        .rename(columns = {'year': 'period'})
        )

    assert mortality_insee.mortality_insee_next_period is not None

    model_to_target = (mortality_after_imputation
        .merge(
            mortality_insee.reset_index(),
            on = ['sex', 'age'],
            )
        )

    if not one_year_approximation: #transformations a 2 ans pour l'instant
        model_to_target = (model_to_target
            #.eval('avg_mortality_2_year = 1 - (1 - avg_mortality) ** 2', inplace = False) #ligne qui servait repasser des probabilites deja transformees
            # a modifier si transitions sur 1 an
            .eval('avg_mortality_2_year = avg_mortality', inplace = False)
            .eval('cale_mortality_2_year = mortalite_2_year_insee / avg_mortality_2_year', inplace = False)
            )

    return model_to_target


def get_mortality_after_imputation(mortality_table = None, dependance_initialisation = None, age_min = 50):
    """
        Compute total mortality from mortality by dependance initial state

        These states are given by the files dependance_initialisation_male/female.csv
        present in til/input_dir or from dependance_initialisation data_frame if not None
    """
    assert mortality_table is not None
    if dependance_initialisation is not None:
        data = dependance_initialisation.rename(columns = {'population': 'total'})
    else:
        raise NotImplementedError

    mortality_after_imputation = (data
        .merge(
            mortality_table.reset_index()[['sex', 'age', 'initial_state', 'mortality']],
            on = ['sex', 'age', 'initial_state'],
            how = 'inner',
            )
        .groupby(['sex', 'age'])[['total', 'mortality']].apply(lambda x: (
            (x.total * x.mortality).sum() / (x.total.sum() + (x.total.sum() == 0))  # Use last term to avoid 0 / 0
            ))
        )
    mortality_after_imputation.name = 'mortality_after_imputation'
    if (mortality_after_imputation == 0).any():  # Deal with age > 100 with nobody in the population
        log.info("missing dans mortality_after_imputation")
        mortality_after_imputation = mortality_after_imputation.reset_index()
        mortality_after_imputation.loc[
            (mortality_after_imputation.age >= 100) & (mortality_after_imputation.mortality_after_imputation == 0),
            'mortality_after_imputation'] = .84 # Mortalite à deux ans pour une mortalite à 1an de 0.6
        mortality_after_imputation.set_index(['sex', 'age'])

    return mortality_after_imputation


def correct_transitions_for_mortality(transitions, dependance_initialisation = None, mu = None, period = None,
        survival_gain_cast = None, previous_mortality = None):
    """
        Take a transition matrix = mortality_calibrated_target and correct transitions to match period's mortality target
        according to a scenario defined by survival_gain_cast and mu
    """
    log.debug("Entering correct_transitions_for_mortality")
    death_state = 4
    assert dependance_initialisation is not None
    admissible_survival_gain_cast = ["homogeneous", "initial_vs_others", 'autonomy_vs_disability']
    assert survival_gain_cast in admissible_survival_gain_cast, \
        "suvival_gain_cast should one of the following values {}".format(admissible_survival_gain_cast)
    assert period is not None
    delta = 1e-7
    regularize(
        transition_matrix_dataframe = transitions,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        delta = delta
        )

    projected_mortality = get_insee_projected_mortality()

    target_mortality = (projected_mortality
        .query('year == @period')
        #.rename(columns = {'mortality_insee': 'target_mortality'})
        .rename(columns = {'mortalite_2_year_insee': 'target_mortality_2_year'})
        .reset_index()
        .drop('year', axis = 1)
        )

    actual_mortality = (dependance_initialisation
            .merge(
                transitions.reset_index().drop('period', axis = 1),
                on = ['age', 'sex', 'initial_state'])
            .eval('new_population = population * calibrated_probability', inplace = False)
            .drop(['initial_state', 'calibrated_probability', 'population'], axis = 1)
            .rename(columns = {'new_population': 'population'})
            .groupby(['age', 'sex', 'period', 'final_state'])['population'].sum()
            .reset_index()
            .rename(columns = {'final_state': 'initial_state'})
            .copy()
            )

    actual_mortality['part'] = (
        (
            actual_mortality / actual_mortality.groupby(['age', 'sex']).transform(sum)
            )['population']
        .fillna(1)  # Kill all non dying
        )
    actual_mortality = (actual_mortality
        .query('initial_state == @death_state')[['age', 'sex', 'period', 'part']]
        .rename(columns = {'part': 'mortality', 'period': 'year'})
        )
    correction_coefficient = (target_mortality.reset_index()
        .merge(actual_mortality)
        .eval('correction_coefficient = target_mortality_2_year / mortality', inplace = False)
        .rename(columns = dict(year = 'period'))
        .drop(['mortality', 'target_mortality'], axis = 1)
        )

    assert not (correction_coefficient['correction_coefficient'].isnull().any())

    uncalibrated_probabilities = (transitions.reset_index()[
        ['sex', 'age', 'initial_state', 'final_state', 'calibrated_probability']
        ]
        .merge(correction_coefficient[['period', 'sex', 'age', 'correction_coefficient']])
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        .sort_index()
        )

    assert not (uncalibrated_probabilities['calibrated_probability'].isnull().any()), \
        "There are {} NaN(s) in uncalibrated_probabilities".format(
            uncalibrated_probabilities['calibrated_probability'].isnull().sum())

    return(uncalibrated_probabilities)

    assert (
        uncalibrated_probabilities.query('final_state == @death_state')['calibrated_probability'] < 1
        ).all(), "There are {} 1's in calibrated_probability".format(
            (uncalibrated_probabilities['calibrated_probability'] < 1).sum(),
            )

    assert_probabilities(
        dataframe = uncalibrated_probabilities,
        by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )

    mortality = (uncalibrated_probabilities
        .query('final_state == @death_state')
        ).copy()

    mortality['periodized_calibrated_probability'] = np.minimum(  # use minimum to avoid over corrections !
        mortality.calibrated_probability * mortality.correction_coefficient, 1 - delta)
    assert (
        (mortality['periodized_calibrated_probability'] < 1)
        ).all(), "There are {} 1's in periodized_calibrated_probability".format(
            (mortality['periodized_calibrated_probability'] < 1).sum(),
            )

    assert not (mortality['periodized_calibrated_probability'].isnull().any()), \
        "There are calibrated_probability NaNs in mortality"

    if survival_gain_cast == "homogeneous":
        log.debug("Using homogeneous scenario")
        # Gain in survival probability are dispatched to the other states respecting the orginal odds ratio
        assert (mortality.calibrated_probability < 1).all()
        mortality.eval(
            'cale_other_transitions = (1 - periodized_calibrated_probability) / (1 - calibrated_probability)',
            inplace = True,
            )
        assert not mortality.cale_other_transitions.isnull().any(), "Some calibration coeffecients are NaNs"
        assert (mortality.cale_other_transitions > 0).all(), "Some calibration coeffecients are negative"

        cale_other_transitions = mortality.reset_index()[
            ['period', 'sex', 'age', 'initial_state', 'cale_other_transitions']
            ].copy()
        other_transitions = (uncalibrated_probabilities
            .reset_index()
            .query('final_state != @death_state')
            .merge(cale_other_transitions)
            .eval(
                'periodized_calibrated_probability = calibrated_probability * cale_other_transitions',
                inplace = False,
                )
            )
        # Ensures that the mortality is the projected one assuming no variation in elderly disability distribution
        assert not (other_transitions['periodized_calibrated_probability'].isnull().any()), \
            "There are {} NaN(s) in other_transitions".format(
                other_transitions['periodized_calibrated_probability'].isnull().sum())
        assert (
            (other_transitions['periodized_calibrated_probability'] >= 0) &
            (other_transitions['periodized_calibrated_probability'] <= 1)
            ).all(), "Erroneous periodized_calibrated_probability"

    elif survival_gain_cast == "initial_vs_others":
        log.debug("Using initial_vs_others scenario")
        assert mu is not None
        # Gain in survival probability feeds by a proportion of mu the initial_state and 1 - mu the other states
        other_transitions = initial_vs_others(
            period = period,
            target_mortality = mortality,
            # previous_mortality = previous_mortality,
            mu = mu,
            uncalibrated_probabilities = uncalibrated_probabilities
            )

    elif survival_gain_cast == 'autonomy_vs_disability':
        log.debug("Using autonomy_vs_disability")
        assert mu is not None
        other_transitions = autonomy_vs_disability(
            mortality = mortality,
            mu = mu,
            uncalibrated_probabilities = uncalibrated_probabilities,
            )

    else:
        raise NotImplementedError

    mortality_recomputed = (other_transitions
        .query('final_state != @death_state')
        .groupby(
            ['period', 'sex', 'age', 'initial_state']
            )['periodized_calibrated_probability'].sum()
        .reset_index()
        )

    mortality_recomputed['periodized_calibrated_probability'] = 1 - mortality_recomputed['periodized_calibrated_probability']
    mortality_recomputed['final_state'] = death_state

    periodized_calibrated_transitions = (pd.concat([
        mortality_recomputed[
            ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
            ],
        other_transitions[
            ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
            ],
        ]))

    try:
        assert_probabilities(
            dataframe = periodized_calibrated_transitions,
            by = ['period', 'sex', 'age', 'initial_state'],
            probability = 'periodized_calibrated_probability',
            cut_off = delta,
            )
    except AssertionError as e:
        ValueError("Problem with probabilities for survival_gain_cast = {}:\n {}".format(
            survival_gain_cast, e))

    return (periodized_calibrated_transitions
        .rename(columns = {'periodized_calibrated_probability': 'calibrated_probability'})
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        .sort_index()
        )


def get_predicted_mortality_table(transitions = None, save = False, probability_name = 'probability', one_year_approximation = None):
    """
        Get mortality from simulation data
    """
    death_state = 4
    assert transitions is not None
    assert one_year_approximation is not None
    assert probability_name in transitions.columns, "{} not present in transitions columns: {}".format(
        probability_name,
        transitions.columns
        )

    if not one_year_approximation:
        # Transitions vers deces avec un pas de 2 ans
        mortality_table = (transitions
            .query('final_state == @death_state')
            .copy()
            .assign(mortality = lambda x: x[probability_name])
        )
    else:
        # one_year_approximation is true : transitions vers deces avec un pas de 1 an
        mortality_table = (transitions
            .query('final_state == @death_state')
            .copy()
            .assign(mortality = lambda x: (1 - np.sqrt(1 - x[probability_name])))
            )

    if save:
        mortality_table.to_csv('predicted_mortality_table.csv')

    mortality_table.loc[
        ~(mortality_table.mortality > 0),
        'mortality'
        ] = 1e-12  # Avoid null mortality
    assert (mortality_table.mortality > 0).all(), mortality_table.loc[~(mortality_table.mortality > 0)]

    return mortality_table


def get_insee_projected_mortality_interm():
    """
        Get mortality data from INSEE projections
    """
    data_path = os.path.join(til_france_path, 'param', 'demo')

    sheet_name_by_sex = dict(zip(
        ['male', 'female'],
        ['hyp_mortaliteH', 'hyp_mortaliteF']
        ))
    mortality_by_sex = dict(
        (
            sex,
            pd.read_excel(
                os.path.join(data_path, 'projpop0760_FECcentESPcentMIGcent.xls'),
                sheet_name = sheet_name, skiprows = 2, header = 2
                )[:121].set_index(
                    u"Âge atteint dans l'année", drop = True
                    ).reset_index()
            )
        for sex, sheet_name in sheet_name_by_sex.items()
        )

    for df in mortality_by_sex.values():
        del df[u"Âge atteint dans l'année"]
        df.index.name = 'age'

    mortality_insee = None
    for sex in ['male', 'female']:
        mortality_sex = ((mortality_by_sex[sex] / 1e4)
            .reset_index()
            )
        mortality_sex = pd.melt(
            mortality_sex,
            id_vars = 'age',
            var_name = 'annee',
            value_name = 'mortality_insee'
            )
        mortality_sex['sex'] = sex
        mortality_sex.rename(columns = dict(annee = 'year'), inplace = True)
        mortality_insee = pd.concat([mortality_insee, mortality_sex])

    return mortality_insee.set_index(['sex', 'age', 'year'])


def get_insee_projected_mortality():
    mortality_insee2= get_insee_projected_mortality_interm()
    mortality_insee_next_period = mortality_insee2
    mortality_insee_next_period.sort_values(by=['sex','year','age'])
    mortality_insee_next_period = mortality_insee_next_period.groupby(['sex','year']).shift(-1)
    mortality_insee_next_period.sort_values(by=['sex','age','year'])
    mortality_insee_next_period = mortality_insee_next_period.groupby(['sex','age']).shift(-1)
    mortality_insee2['mortality_insee_next_period'] =mortality_insee_next_period.mortality_insee
    mortality_insee = mortality_insee2

    mortality_insee['mortalite_2_year_insee'] = 0
    mortality_insee.loc[
        mortality_insee.mortality_insee_next_period.notnull(),
        'mortalite_2_year_insee'
        ] = 1 - (1 - mortality_insee.mortality_insee) * (1 - mortality_insee.mortality_insee_next_period)

    mortality_insee.loc[
        mortality_insee.mortality_insee_next_period.isnull(),
        'mortalite_2_year_insee'
        ] = 1 - (1 - mortality_insee.mortality_insee) ** 2

    return mortality_insee


def get_insee_projected_population():
    """
        Get population data from INSEE projections
    """
    population_by_sex = dict(
        (
            sex,
            get_insee_projection('population', sex)
            )
        for sex in ['male', 'female']
        )

    for df in population_by_sex.values():
        df.index.name = 'age'

    population = None
    for sex in ['male', 'female']:
        population_sex = ((population_by_sex[sex])
            .reset_index()
            )
        population_sex = pd.melt(
            population_sex,
            id_vars = 'age',
            var_name = 'annee',
            value_name = 'population'
            )
        population_sex['sex'] = sex
        population_sex.rename(columns = dict(annee = 'year'), inplace = True)
        population = pd.concat([population, population_sex])

    # Fix age values
    population.age = population.age.replace({'108 et +': 108}).astype(int)
    return population.set_index(['sex', 'age', 'year'])


##########
    #TOOLS
###########

def smooth_pivot_table(pivot_table, window = 7, std = 2):
    smoothed_pivot_table = pivot_table.copy()
    for dependance_niveau in smoothed_pivot_table.columns:
        smoothed_pivot_table[dependance_niveau] = (pivot_table[dependance_niveau]
            .rolling(win_type = 'gaussian', center = True, window = window, axis = 0)
            .mean(std = std)
            )

    return smoothed_pivot_table

def check_67_and_over(population, age_min):
    period = population.period.max()
    insee_population = get_insee_projected_population()
    pop_insee = insee_population.query('(age >= @age_min) and (year == @period)')['population'].sum()
    pop_sim = population.query('(age >= @age_min) and (period == @period)')['population'].sum()
    pop_insee_62 = insee_population.query('(age == @age_min) and (year == @period)')['population'].sum()
    pop_sim_62 = population.query('(age == @age_min) and (period == @period)')['population'].sum()
    pop_insee_80 = insee_population.query('(age == @age_min + 18) and (year == @period)')['population'].sum()
    pop_sim_80 = population.query('(age == @age_min + 18) and (period == @period)')['population'].sum()
    pop_insee_100 = insee_population.query('(age == @age_min + 38) and (year == @period)')['population'].sum()
    pop_sim_100 = population.query('(age == @age_min + 38) and (period == @period)')['population'].sum()
    pop_insee_60 = insee_population.query('(age == @age_min - 2) and (year == @period)')['population'].sum()
    pop_sim_60 = population.query('(age == @age_min - 2) and (period == @period)')['population'].sum()

    log.info("period {}: insee = {} vs {} = til".format(
        period,
        insee_population.query('(age >= @age_min) and (year == @period)')['population'].sum(),
        population.query('(age >= @age_min) and (period == @period)')['population'].sum()
        ))
    log.info("period {}: insee - til = {}".format(
        period,
        pop_insee - pop_sim
        ))
    log.info("period {}: insee_60:{} til_60:{} insee_60 - til_60 = {}".format(
       period, pop_insee_60, pop_sim_60,
       pop_insee_60 - pop_sim_60
       ))
    log.info("period {}: insee_62:{} til_62:{} insee_62 - til_62 = {}".format(
       period, pop_insee_62, pop_sim_62,
       pop_insee_62 - pop_sim_62
       ))
    log.info("period {}: insee_80:{} til_80:{} insee_80 - til_80 = {}".format(
       period, pop_insee_80, pop_sim_80,
       pop_insee_80 - pop_sim_80
       ))
    log.info("period {}: insee_100:{} til_100:{} insee_100 - til_100 = {}".format(
       period, pop_insee_100, pop_sim_100,
       pop_insee_100 - pop_sim_100
    ))


def regularize(transition_matrix_dataframe = None, by = None, probability = None, delta = None):
    assert transition_matrix_dataframe is not None
    assert by is not None
    assert probability is not None
    assert delta is not None
    final_state = 4
    assert_probabilities(dataframe = transition_matrix_dataframe, by = by, probability = probability)
    mortality_transitions = transition_matrix_dataframe.query('final_state == @final_state').copy()

    # by_without_initial_state = [by_value for by_value in by if by_value != 'initial_state']
    problematic_indices = (mortality_transitions[probability]
        .loc[mortality_transitions[probability] >= (1 - delta)]
        .reset_index()
        .drop(['final_state', probability], axis = 1)
        )
    count = (problematic_indices
        .merge(
            transition_matrix_dataframe.reset_index()[by + ['final_state']])
        .query('final_state != @final_state')
        .groupby(by)
        .count()
        )
    correction = delta / count.final_state.astype('float')
    correction.name = 'correction'

    corrected_transition_matrix = (transition_matrix_dataframe.reset_index()
        .merge(correction.reset_index())  # inner merge
        .fillna(0)
        )
    corrected_transition_matrix.loc[
        (corrected_transition_matrix.final_state == final_state),
        probability
        ] = (1 - delta)
    corrected_transition_matrix.loc[
        corrected_transition_matrix.final_state != final_state,
        probability
        ] = corrected_transition_matrix.correction

    corrected_transition_matrix.set_index(by + ['final_state'], inplace = True)
    assert (
        (corrected_transition_matrix[probability] > 0) &
        (corrected_transition_matrix[probability] < 1)
        ).all(), "There are {} 0's and {} 1's in corrected_transition_matrix[{}]".format(
            (corrected_transition_matrix[probability] > 0).sum(),
            (corrected_transition_matrix[probability] < 1).sum(),
            probability
            )
    transition_matrix_dataframe.update(corrected_transition_matrix)

    assert (
        transition_matrix_dataframe.query('final_state == @final_state')[probability] < 1
        ).all(), "There are {} 1's in transition_matrix_dataframe.{}".format(
            (transition_matrix_dataframe.query('final_state == @final_state')[probability] < 1).sum(),
            probability,
            )

    assert_probabilities(dataframe = transition_matrix_dataframe, by = by, probability = probability)
    return transition_matrix_dataframe


def apply_transition_matrix(population = None, transition_matrix = None, age_min = None, age_max_cale = None):
    assert age_min is not None
    death_state = 4
    assert population is not None and transition_matrix is not None
    assert len(population.period.unique()) == 1
    final_population = (population
        #.loc[population['age']<age_max_cale]
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

    mortality=get_insee_projected_mortality().query('(year == @period) and (age >= @age_min)').reset_index().eval('two_year_mortality=mortalite_2_year_insee')

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


def build_suffix(survival_gain_cast = None, mu = None, vagues = None, prevalence_survey = None):
    suffix = survival_gain_cast
    if mu is not None:
        suffix += '_mu_{}_'.format(mu)
    if vagues is not None:
        suffix += slugify.slugify(str(vagues), separator = "_")
    if prevalence_survey is not None:
        suffix += '_{prevalence_survey}_'.format(prevalence_survey = prevalence_survey)

    return suffix


def assert_probabilities(dataframe = None, by = ['period', 'sex', 'age', 'initial_state'],
        probability = 'calibrated_probability', cut_off = 1e-8):
    assert dataframe is not None
    assert not (dataframe[probability] < 0).any(), dataframe.loc[dataframe[probability] < 0]
    assert not (dataframe[probability] > 1).any(), dataframe.loc[dataframe[probability] > 1]
    diff = (
        dataframe.reset_index().groupby(by)[probability].sum() - 1)
    diff.name = 'error'
    assert (diff.abs().max() < cut_off).all(), "error is too big: {} > {}. Example: {}".format(
        diff.abs().max(), cut_off, (dataframe
            .reset_index()
            .set_index(by)
            .loc[diff.abs().argmax(), ['final_state', probability]]
            .reset_index()
            .set_index(by + ['final_state'])
            )
        )

###SCENARIOS

def initial_vs_others(mortality = None, mu = None, uncalibrated_probabilities = None):
    """
    Gain in survival probability feeds by a proportion of mu the initial_state and 1 - mu the other states
    """

    final_states_by_initial_state = {
        0: [0, 1, 4],
        1: [0, 1, 2, 4],
        2: [1, 2, 3, 4],
        3: [2, 3, 4],
        }

    mortality.eval(
        'delta_initial = - @mu * (periodized_calibrated_probability - calibrated_probability)',
        inplace = True,
        )

    mortality.eval(
        'delta_non_initial = - (1 - @mu) * (periodized_calibrated_probability - calibrated_probability)',
        inplace = True,
        )
    other_transitions = pd.DataFrame()
    for initial_state, final_states in final_states_by_initial_state.items():
        if 4 not in final_states:
            continue
        #
  #       if initial_state == 4:
  #           to_initial_transitions = (uncalibrated_probabilities
   #              .reset_index()
   #              .query('(initial_state == @initial_state) and (final_state == @initial_state) and (final_state != 5)')
   #              .merge(
   #                  mortality.reset_index()[
   #                      ['period', 'sex', 'age', 'initial_state', 'periodized_calibrated_probability']
   #                      ])
    #             )
    #         to_initial_transitions['periodized_calibrated_probability'] = (
    #             1 - to_initial_transitions['periodized_calibrated_probability']
    #             )
    #         to_non_initial_transitions = None
   #          other_transitions = pd.concat([
   #              other_transitions,
    #             to_initial_transitions[
     #                ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
     #                ],
    #             ])
        #
        else:
            delta_initial_transitions = mortality.reset_index()[
                #['period', 'sex', 'age', 'initial_state', 'delta_initial','calibrated_probability']
                ['period', 'sex', 'age', 'initial_state', 'delta_initial']
                ].copy()
            to_initial_transitions = (uncalibrated_probabilities
                .reset_index()
                .query('(initial_state == @initial_state) and (final_state == @initial_state) and (final_state != 4)')
                .merge(delta_initial_transitions)
                )

            to_initial_transitions['periodized_calibrated_probability'] = np.maximum(
                to_initial_transitions.calibrated_probability + to_initial_transitions.delta_initial, 0)

            delta_non_initial_transitions = mortality.reset_index()[
                ['period', 'sex', 'age', 'initial_state', 'delta_non_initial']
                ].copy()

            non_initial_transitions_aggregate_probability = (uncalibrated_probabilities
                .reset_index()
                .query('(initial_state == @initial_state) and (final_state != @initial_state) and (final_state != 4)')
                .groupby(['period', 'sex', 'age', 'initial_state'])['calibrated_probability'].sum()
                .reset_index()
                .rename(columns = dict(calibrated_probability = 'aggregate_calibrated_probability'))
                )

            to_non_initial_transitions = (uncalibrated_probabilities
                .reset_index()
                .query('(initial_state == @initial_state) and (final_state != @initial_state) and (final_state != 4)')
                .merge(non_initial_transitions_aggregate_probability.reset_index())
                .merge(delta_non_initial_transitions)
                .eval(
                    'periodized_calibrated_probability = calibrated_probability * (1 + delta_non_initial / aggregate_calibrated_probability)',
                    inplace = False,
                    )
                )

            other_transitions = pd.concat([
                other_transitions,
                to_initial_transitions[
                    ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
                    ],
                to_non_initial_transitions[
                    ['period', 'sex', 'age', 'initial_state', 'final_state', 'periodized_calibrated_probability']
                    ],
                ])


            assert not other_transitions[['period', 'sex', 'age', 'initial_state', 'final_state']].duplicated().any(), \
                'Duplicated transitions for initial_state = {}: {}'.format(
                    initial_state,
                    other_transitions.loc[
                        other_transitions[['period', 'sex', 'age', 'initial_state', 'final_state']].duplicated()
                        ]
                    )
    return other_transitions


def autonomy_vs_disability(mortality = None, mu = None, uncalibrated_probabilities = None):
    """
        Autonomy and disability states are treated differently

        Gain in survival probability feeds by a proportion of mu the autonomy initial_state and
        1 - mu the transition to the disability states.
        Other transition share proportionnally the survival probability gain

        :param DataFrame mortality: moratlity table
        :param mu mortality: Share of survival gain casted to the autonomy state
        :param DataFrame uncalibrated_probabilities: probability table to be updated
    """
    # Starting from autonomy
    # Staying autonomous
    assert mu is not None
    mortality.eval(
        'survival_gain = - (periodized_calibrated_probability - calibrated_probability)',  # It is postive gain
        inplace = True,
        )
    survival_gain = mortality.reset_index()[
        ['period', 'sex', 'age', 'initial_state', 'survival_gain']
        ].copy()

    stay_autonomous = (uncalibrated_probabilities
        .reset_index()
        .query('(initial_state == 0) & (final_state == 0)')
        .merge(survival_gain)
        .eval(
            'periodized_calibrated_probability = @mu * survival_gain + calibrated_probability',
            inplace = False,
            )
        )
    assert (stay_autonomous.periodized_calibrated_probability <= 1).all()

    # From autonomy to disablement
    disabling_transitions_aggregate_probability = (uncalibrated_probabilities
        .reset_index()
        .query('(initial_state == 0) & (final_state != 0) & (final_state != 4)')
        .groupby(['period', 'sex', 'age', 'initial_state'])['calibrated_probability'].sum()
        .reset_index()
        .rename(columns = dict(calibrated_probability = 'aggregate_calibrated_probability'))
        )
    become_disabled = (uncalibrated_probabilities
        .reset_index()
        .query('(initial_state == 0) & (final_state != 0) & (final_state != 4)')
        .merge(disabling_transitions_aggregate_probability)
        .merge(survival_gain)
        .eval(
            'periodized_calibrated_probability =  calibrated_probability * (1 + (1 - @mu) * survival_gain / aggregate_calibrated_probability)',
            inplace = False,
            )
        )

    # For initial_states >= 1
    mortality.eval(
        'cale_other_transitions = (1 - periodized_calibrated_probability) / (1 - calibrated_probability)',
        inplace = True,
        )
    cale_other_transitions = mortality.reset_index()[
        ['period', 'sex', 'age', 'initial_state', 'cale_other_transitions']
        ].copy()
    other_transitions = (uncalibrated_probabilities
        .reset_index()
        .query('(initial_state != 0) & (final_state != 4)')
        .merge(cale_other_transitions)
        .eval(
            'periodized_calibrated_probability = calibrated_probability * cale_other_transitions',
            inplace = False,
            )
        )

    all_transitions = pd.concat(
        [stay_autonomous, become_disabled, other_transitions,],
        sort = True,
        )

    all_transitions = (all_transitions
        .reset_index()
        .set_index(['period', 'sex', 'age', 'initial_state', 'final_state'])
        )

    return all_transitions
