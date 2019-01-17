# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 14:55:35 2018

@author: a.rain
"""

import logging
import numpy as np
import os
import pandas as pd
import pkg_resources
import seaborn as sns
import sys

from til_core.config import Config
from til_france.data.data.hsm_dependance_niveau import smooth_pivot_table

log = logging.getLogger(__name__)


def get_care_prevalence_pivot_table(sexe = None, scale = None):
    config = Config()
    assert scale in [4, 5], "scale should be equal to 4 or 5"
#    if scale == 5:
#        xls_path = os.path.join(config.get('raw_data', 'hsm_dependance_niveau'), 'desc_dependance_scale5.xls')
 #       data = (pd.read_excel(xls_path)
 #           .rename(columns = {'disability_scale5': 'dependance_niveau', 'woman': 'sexe'})
 #           )
#    elif scale == 4:
    xls_path = os.path.join(
            config.get('raw_data', 'hsm_dependance_niveau'), 'CARe_scalev1v2.xls')
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

def create_dependance_initialisation_share_new(filename_prefix = None, smooth = False, window = 7, std = 2,
        survey = 'care', age_min = None, scale = 4):
    """
    Create dependance_niveau variable initialisation file for use in til-france model (option dependance_RT)
    """
    assert scale in [4, 5], "scale should be equal to 4 or 5"
    assert age_min is not None
    config = Config()
    input_dir = config.get('til', 'input_dir')
    assert survey in ['both', 'hsm', 'care']
    for sexe in ['homme', 'femme']:
        if survey == 'hsm':
            pivot_table = get_hsm_prevalence_pivot_table(sexe = sexe, scale = 4)
        elif survey == 'care':
            pivot_table = get_care_prevalence_pivot_table(sexe = sexe, scale = 4)    
        else:
            pivot_table = get_hsi_hsm_prevalence_pivot_table(sexe = sexe, scale = 4)


        if filename_prefix is None:
            filename = os.path.join(input_dir, 'dependance_initialisation_level_share2_{}.csv'.format(sexe))
        else:
            filename = os.path.join('{}_level_share_{}.csv'.format(filename_prefix, sexe))
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
            filename = os.path.join(input_dir, 'dependance_initialisation_share_{}.csv'.format(sexe))
        else:
            filename = os.path.join('{}_share_{}.csv'.format(filename_prefix, sexe))

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


df = create_dependance_initialisation_share_new(
        filename_prefix = None, smooth = True, window = 7, std = 2,
        survey = 'care', age_min = 50, scale = 4)

df2 = create_dependance_initialisation_share(
        filename_prefix = None, smooth = True, window = 7, std = 2,
        survey = 'care', age_min = 50, scale = 4)



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

df = get_predicted_mortality_table(transitions = transitions)



def load_and_plot_dependance_niveau_by_period(survival_gain_cast = None, mu = None, periods = None, area = False,
          vagues = None):
    suffix = build_suffix(survival_gain_cast, mu, vagues)

    population = pd.read_csv(os.path.join(figures_directory, 'population_{}.csv'.format(suffix)))
    periods = range(2010, 2050, 2) if periods is None else periods
    for period in periods:
        plot_dependance_niveau_by_age(population, period, age_min = 50, age_max = 100, area = area)
        
        ####
        vagues = [1, 2]
    formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1) ** 2) + I(((age - 80) * 0.1) ** 3)'

            uncalibrated_transitions = get_transitions_from_formula(formula = formula, vagues = vagues)


        save_data_and_graph(uncalibrated_transitions, survival_gain_cast = 'homogeneous')
        
        
        ####
        
            formula = 'final_state ~ I((age - 80) * 0.1) + I(((age - 80) * 0.1) ** 2) + I(((age - 80) * 0.1) ** 3)'
        vagues = [1,2]
        initial_states = [0]
         sex = ['male','female']

    df = build_estimation_sample(initial_state = initial_states, sex = sex, variables = ['age'], vagues = vagues)
    