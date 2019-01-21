# -*- coding: utf-8 -*-


import logging
import numpy as np
import os
import pandas as pd


from til_core.config import Config


log = logging.getLogger(__name__)


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


def smooth_pivot_table(pivot_table, window = 7, std = 2):
    smoothed_pivot_table = pivot_table.copy()
    for dependance_niveau in smoothed_pivot_table.columns:
        smoothed_pivot_table[dependance_niveau] = (pivot_table[dependance_niveau]
            .rolling(win_type = 'gaussian', center = True, window = window, axis = 0)
            .mean(std = std)
            )

    return smoothed_pivot_table
