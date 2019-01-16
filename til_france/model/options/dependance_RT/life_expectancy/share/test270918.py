# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 09:37:22 2018

@author: a.rain
"""


from __future__ import division


import logging

import numpy as np
import os
import pandas as pd
import sys


from til_core.config import Config
from til_france.model.options.dependance_RT.life_expectancy.share.transition_matrices import (
    assets_path,
    final_states_by_initial_state,
    get_transitions_from_formula,
    til_france_path,
    )
from til_france.tests.base import ipp_colors
colors = [ipp_colors[cname] for cname in ['ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]


log = logging.getLogger(__name__)


life_table_path = os.path.join(
    assets_path,
    'lifetables_period.xlsx'
    )




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
        for sex, sheet_name in sheet_name_by_sex.iteritems()
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

####

vagues = [1, 2]
    formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1) ** 2) + I(((age - 80) * 0.1) ** 3)'
    uncalibrated_transitions = get_transitions_from_formula(formula = formula, vagues = vagues)

    survival_gain_casts = [
        'homogeneous',
        ]
    #survey = care
    run(survival_gain_casts, uncalibrated_transitions = uncalibrated_transitions, vagues = vagues, age_min = 60)



def run(survival_gain_casts = None, uncalibrated_transitions = None, vagues = [4, 5, 6], age_min = None):
    assert age_min is not None
    assert uncalibrated_transitions is not None
    create_initial_prevalence(smooth = True, survey = 'care', age_min = age_min)
