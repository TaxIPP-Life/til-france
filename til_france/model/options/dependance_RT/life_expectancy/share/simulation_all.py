# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 14:59:09 2018

@author: a.rain
"""


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

# # Fonctions principales

def run(survival_gain_casts = None, uncalibrated_transitions = None, vagues = [4, 5, 6], age_min = None):
    assert age_min is not None
    assert uncalibrated_transitions is not None
    create_initial_prevalence(smooth = True, survey = 'care', age_min = age_min)

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


def save_data_and_graph(uncalibrated_transitions, mu = None, survival_gain_cast = None):
    log.info("Running with survival_gain_cast = {}".format(survival_gain_cast))
    initial_period = 2010
    initial_population = get_initial_population()
    initial_population['period'] = initial_period
    population, transitions_by_period = run_scenario(
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


def run_scenario(uncalibrated_transitions = None, initial_population = None, initial_period = 2010, mu = None,
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
    
# # Doc transition_matrices : fonctions get_transitions_from_formula, compute_prediction, build_estimation_sample, get_clean_share
# 
# get_transitions_from_formula : permet d'obtenir transitions, qu'on fixe tel que uncalibrated_transitions = transitions en argument de la fonction run()

###############################
# # compute_prediction : Calcule les probabilités de transition en appliquant un logit pour une spécification donnée, qui correspond à l'objet formula


# # get_clean_share : Get SHARE relevant data free of missing observations


########## Prevalence initiales : initialise les prevalences initiales dans les états de la population

# # create_initial_prevalence : renvoie la table pivot_table avec à chaque âge la répartition dans les états possibles
# anciennement Create_dependance_initialisation_share


def create_initial_prevalence(filename_prefix = None, smooth = False, window = 7, std = 2,
        survey = 'care', age_min = None, scale = 4):
    """
    Create dependance_niveau variable initialisation file for use in til-france model (option dependance_RT)
    """
    assert scale in [4, 5], "scale should be equal to 4 or 5"
    assert age_min is not None
    config = Config()
    input_dir = config.get('til', 'input_dir')
    assert survey in ['care', 'hsm', 'hsm_hsi']
    for sexe in ['homme', 'femme']:
        if survey == 'hsm':
            pivot_table = get_hsm_prevalence_pivot_table(sexe = sexe, scale = 4)
        elif survey == 'hsm_hsi':
            pivot_table = get_hsi_hsm_prevalence_pivot_table(sexe = sexe, scale = 4)
        elif survey == 'care':
            pivot_table =  get_care_prevalence_pivot_table(sexe = sexe, scale = 4)   

        if filename_prefix is None:
            filename = os.path.join(input_dir, 'dependance_initialisation_level_{}_{}.csv'.format(survey, sexe)) # dependance_initialisation_level_share_{} 
        else:
            filename = os.path.join('{}_level_{}_{}.csv'.format(filename_prefix, survey, sexe))
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
            filename = os.path.join(input_dir, 'dependance_initialisation_{}_{}.csv'.format(survey,sexe)) 
        else:
            filename = os.path.join('{}_{}_{}.csv'.format(filename_prefix, survey, sexe)) #survey insere dans le nom du doc

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
            with file(filename, 'r') as original:
                data = original.read()
            with file(filename, 'w') as modified:
                modified.write(verbatim_header + data)
            log.info('Saving {}'.format(filename))

    return pivot_table

def get_care_prevalence_pivot_table(sexe = None, scale = None):
    config = Config()
    assert scale in [4, 5], "scale should be equal to 4 or 5"
    xls_path = os.path.join(
            config.get('raw_data', 'hsm_dependance_niveau'),'CARe_scalev1v2.xls')
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

def get_initial_population(survey = 'care'):
        data_by_sex = dict()
        for sex in ['male', 'female']:
            sexe = 'homme' if sex == 'male' else 'femme'
            config = Config()
            input_dir = config.get('til', 'input_dir')
            filename = os.path.join(input_dir, 'dependance_initialisation_level_{}_{}.csv'.format(survey, sexe))
            log.info('Loading initial population dependance states from {}'.format(filename))
            df = (pd.read_csv(filename, names = ['age', 0, 1, 2, 3, 4], skiprows = 1)
                .query('(age >= 65)')
                )
            df['age'] = df['age'].astype('int')

            df = (
                pd.melt(
                    df,
                    id_vars = ['age'],
                    value_vars = [0, 1, 2, 3, 4],
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


#################  Mortalites calibrees

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


def build_mortality_calibrated_target_from_transitions(transitions = None, period = None, dependance_initialisation = None,
       age_min = None):
    assert age_min is not None
    assert period is not None
    assert transitions is not None
    mortality_calibrated_target = build_mortality_calibrated_target(
        transitions = transitions,
        period = period,
        dependance_initialisation = dependance_initialisation,
        age_min = age_min,
        )
    assert_probabilities(
        dataframe = mortality_calibrated_target,
        by = ['sex', 'age', 'initial_state'],
        probability = 'calibrated_probability',
        )
    return mortality_calibrated_target


def build_mortality_calibrated_target(transitions = None, period = None, dependance_initialisation = None, scale = 4,
        age_min = None):
    """
    Compute the calibrated mortality by sex, age and disability state (initial_state) for a given period
    using data on the disability states distribution in the population at that period
    if dependance_initialisation = None
    TODO should be merged with build_mortality_calibrated_target_from_transitions
    """
    assert age_min is not None
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
        )

    null_fill_by_year = calibrated_transitions.reset_index().query('age == @age_min').copy()
    null_fill_by_year['calibrated_probability'] = 0

    # Less than ageè_min years old -> no correction
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


def _get_calibrated_transitions(period = None, transitions = None, dependance_initialisation = None):
    """
    Calibrate transitions to match mortality from a specified period
    """
    death_state = 4
    assert (period is not None) and (transitions is not None)

    # Add calibration_coeffcients for mortality
    calibration = _compute_calibration_coefficient(
        period = period,
        transitions = transitions,
        dependance_initialisation = dependance_initialisation,
        )

    assert not calibration.reset_index()[['sex', 'age']].duplicated().any(), \
        calibration.reset_index().loc[calibration.reset_index()[['sex', 'age']].duplicated()]

    # Calibrate mortality
    assert not transitions.reset_index()[['sex', 'age', 'initial_state', 'final_state']].duplicated().any(), \
        transitions.reset_index().loc[transitions.reset_index()[
            ['sex', 'age', 'initial_state', 'final_state']].duplicated()]

    mortality = (transitions
        .reset_index()
        .query('final_state == @death_state')
        .merge(
            calibration[['sex', 'age', 'cale_mortality_2_year']].copy(),
            on = ['sex', 'age'],
            )
        )
    mortality['calibrated_probability'] = np.minimum(
        mortality.probability * mortality.cale_mortality_2_year, 1)  # Avoid over corrections !
    mortality.eval(
        'cale_other_transitions = (1 - calibrated_probability) / (1 - probability)',
        inplace = True,
        )
    assert not mortality[['sex', 'age', 'initial_state', 'final_state']].duplicated().any(), \
        mortality.loc[mortality[['sex', 'age', 'initial_state', 'final_state']].duplicated()]
    # Calibrate other transitions
    cale_other_transitions = mortality[['sex', 'age', 'initial_state', 'cale_other_transitions']].copy()
    other_transitions = (transitions
        .reset_index()
        .query('final_state != @death_state')
        .merge(
            cale_other_transitions,
            on = ['sex', 'age', 'initial_state'],
            )
        .eval('calibrated_probability = probability * cale_other_transitions', inplace = False)
        )
    assert other_transitions.calibrated_probability.notnull().all(), \
        other_transitions.loc[other_transitions.calibrated_probability.isnull()]

    calibrated_transitions = pd.concat(
        [
            mortality.set_index(['sex', 'age', 'initial_state', 'final_state']).sort_index(),
            other_transitions.set_index(['sex', 'age', 'initial_state', 'final_state']).sort_index()
            ]
        ).sort_index()

    # Verification
    assert_probabilities(
        calibrated_transitions, by = ['sex', 'age', 'initial_state'], probability = 'calibrated_probability')
    return calibrated_transitions['calibrated_probability']


def _compute_calibration_coefficient(age_min = 50, period = None, transitions = None, dependance_initialisation = None):
    """
    Calibrate mortality using the distribution of the disability states within population at a specific year
    for the given transition matrix and distribution of intiial_states
    Assuming the transition occur on a two-year period.
    """
    assert period is not None, "Mortality profile period is not set"
    assert transitions is not None
    # From 2yr mortality to 1yr mortality by age, sex and intial_state
    predicted_mortality_table = get_predicted_mortality_table(transitions = transitions)
    # From 1yr mrotality by age sex (use dependance_initialisation to sum over initial_state)
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

    projected_mortality = (get_insee_projected_mortality()  # brings in variable mortality
        .query('year == @period')
        .rename(columns = {'year': 'period'})
        )
    model_to_target = (mortality_after_imputation
        .merge(
            projected_mortality.reset_index(),
            on = ['sex', 'age'],
            )
        .eval('cale_mortality_1_year = mortality / avg_mortality', inplace = False)
        .eval('mortalite_2_year = 1 - (1 - mortality) ** 2', inplace = False)
        .eval('avg_mortality_2_year = 1 - (1 - avg_mortality) ** 2', inplace = False)
        .eval('cale_mortality_2_year = mortalite_2_year / avg_mortality_2_year', inplace = False)
        )
    return model_to_target


def get_mortality_after_imputation(mortality_table = None, dependance_initialisation = None, age_min = 50):
    """
    Compute total mortality from mortality by dependance initial state given dependance_initialisation_male/female.csv
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
        mortality_after_imputation = mortality_after_imputation.reset_index()
        mortality_after_imputation.loc[
            (mortality_after_imputation.age >= 100) & (mortality_after_imputation.mortality_after_imputation == 0),
            'mortality_after_imputation'] = .6
        mortality_after_imputation.set_index(['sex', 'age'])

    return mortality_after_imputation

## get_predicted_mortality_table et get_insee_projected_mortality : mortalités dans les données et INSEE

def get_predicted_mortality_table(transitions = None, save = True, probability_name = 'probability'):
    death_state = 4
    assert transitions is not None
    assert probability_name in transitions.columns, "{} not present in transitions colmns: {}".format(
        probability_name,
        transitions.columns
        )
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


def get_insee_projected_mortality():
    '''
    Get mortality data from INSEE projections
    '''
    data_path = os.path.join(til_france_path, 'param', 'demo')

    sheetname_by_sex = dict(zip(
        ['male', 'female'],
        ['hyp_mortaliteH', 'hyp_mortaliteF']
        ))
    mortality_by_sex = dict(
        (
            sex,
            pd.read_excel(
                os.path.join(data_path, 'projpop0760_FECcentESPcentMIGcent.xls'),
                sheetname = sheetname, skiprows = 2, header = 2
                )[:121].set_index(
                    u"Âge atteint dans l'année", drop = True
                    ).reset_index()
            )
        for sex, sheetname in sheetname_by_sex.iteritems()
        )

    for df in mortality_by_sex.values():
        del df[u"Âge atteint dans l'année"]
        df.index.name = 'age'

    mortality = None
    for sex in ['male', 'female']:
        mortality_sex = ((mortality_by_sex[sex] / 1e4)
            .reset_index()
            )
        mortality_sex = pd.melt(
            mortality_sex,
            id_vars = 'age',
            var_name = 'annee',
            value_name = 'mortality'
            )
        mortality_sex['sex'] = sex
        mortality_sex.rename(columns = dict(annee = 'year'), inplace = True)
        mortality = pd.concat([mortality, mortality_sex])

    return mortality.set_index(['sex', 'age', 'year'])