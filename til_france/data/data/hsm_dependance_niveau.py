# -*- coding: utf-8 -*-

import logging
import numpy as np
import os
import pandas as pd
import pkg_resources
import seaborn as sns
import sys


from til_core.config import Config
from til_france.tests.base import ipp_colors
from til_france.model.options.dependance_RT.life_expectancy.transition_matrices import get_clean_paquid


colors = [ipp_colors[cname] for cname in ['ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]


log = logging.getLogger(__name__)


figures_directory = os.path.join(
    pkg_resources.get_distribution('til-france').location,
    'til_france',
    'figures'
    )


def create_dependance_initialisation(filename_prefix = None, smooth = False, window = 7, std = 2, survey = 'hsm'):
    """
    Create dependance_niveau variable initialisation file for use in til-france model (option dependance_RT)
    """
    config = Config()
    input_dir = config.get('til', 'input_dir')
    assert survey in ['both', 'hsm']
    for sexe in [0, 1]:
        if survey == 'hsm':
            pivot_table = get_hsm_prevalence_pivot_table(sexe = sexe)
        else:
            pivot_table = get_hsi_hsm_prevalence_pivot_table(sexe = sexe)
        if smooth:
            pivot_table = smooth_pivot_table(pivot_table, window = window, std = std)
        sexe_str = 'homme' if sexe == 0 else 'femme'

        if filename_prefix is None:
            filename = os.path.join(input_dir, 'dependance_initialisation_level_{}.csv'.format(sexe_str))
        else:
            filename = os.path.join('{}_level_{}.csv'.format(filename_prefix, sexe_str))
        level_pivot_table = (pivot_table.copy()
            .reset_index()
            .merge(pd.DataFrame({'age': range(0, 121)}), how = 'right')
            .sort_values('age')
            )
        level_pivot_table['age'] = level_pivot_table['age'].astype(int)
        level_pivot_table.fillna(0, inplace = True)
        level_pivot_table.to_csv(filename, index = False)
        log.info('Saving {}'.format(filename))

        # Go from levels to pct
        pivot_table = pivot_table.divide(pivot_table.sum(axis=1), axis=0)
        if filename_prefix is None:
            filename = os.path.join(input_dir, 'dependance_initialisation_{}.csv'.format(sexe_str))
        else:
            filename = os.path.join('{}_{}.csv'.format(filename_prefix, sexe_str))

        if filename is not None:
            pivot_table = (pivot_table
                .reset_index()
                .merge(pd.DataFrame({'age': range(0, 121)}), how = 'right')
                .sort_values('age')
                )
            pivot_table['age'] = pivot_table['age'].astype(int)

            pivot_table.fillna(0, inplace = True)
            pivot_table.loc[pivot_table.age < 60, 0] = 1
            pivot_table.set_index('age', inplace = True)
            pivot_table.loc[pivot_table.sum(axis = 1) == 0, 4] = 1
            pivot_table.to_csv(filename, header = False)
            assert ((pivot_table.sum(axis = 1) - 1).abs() < 1e-15).all(), \
                pivot_table.sum(axis = 1).loc[((pivot_table.sum(axis = 1) - 1).abs() > 1e-15)]

            # Add liam2 compatible header
            verbatim_header = '''age,initial_state,,,,
,0,1,2,3,4
'''
            with file(filename, 'r') as original:
                data = original.read()
            with file(filename, 'w') as modified:
                modified.write(verbatim_header + data)
            log.info('Saving {}'.format(filename))


def create_dependance_initialisation_merged_0_1(filename_prefix = None, smooth = False, window = 7, std = 2,
        survey = 'hsm'):
    """
    Create dependance_niveau variable initialisation file for use in til-france model (option dependance_RT)
    """
    config = Config()
    input_dir = config.get('til', 'input_dir')
    assert survey in ['both', 'hsm']
    for sexe in ['homme', 'femme']:
        if survey == 'hsm':
            pivot_table = get_hsm_prevalence_pivot_table(sexe = sexe)
        else:
            pivot_table = get_hsi_hsm_prevalence_pivot_table(sexe = sexe)

        pivot_table[0] = pivot_table[0] + pivot_table[1]
        del pivot_table[1]
        if smooth:
            pivot_table = smooth_pivot_table(pivot_table, window = window, std = std)

        if filename_prefix is None:
            filename = os.path.join(input_dir, 'dependance_initialisation_level_merged_0_1_{}.csv'.format(sexe))
        else:
            filename = os.path.join('{}_level_merged_0_1_{}.csv'.format(filename_prefix, sexe))
        level_pivot_table = (pivot_table.copy()
            .reset_index()
            .merge(pd.DataFrame({'age': range(0, 121)}), how = 'right')
            .sort_values('age')
            )
        level_pivot_table['age'] = level_pivot_table['age'].astype(int)
        level_pivot_table.fillna(0, inplace = True)
        level_pivot_table.to_csv(filename, index = False)
        log.info('Saving {}'.format(filename))

        # Go from levels to pct
        pivot_table = pivot_table.divide(pivot_table.sum(axis=1), axis=0)
        if filename_prefix is None:
            filename = os.path.join(input_dir, 'dependance_initialisation_merged_0_1_{}.csv'.format(sexe))
        else:
            filename = os.path.join('{}_merged_0_1_{}.csv'.format(filename_prefix, sexe))

        if filename is not None:
            pivot_table = (pivot_table
                .reset_index()
                .merge(pd.DataFrame({'age': range(0, 121)}), how = 'right')
                .sort_values('age')
                )
            pivot_table['age'] = pivot_table['age'].astype(int)

            pivot_table.fillna(0, inplace = True)
            pivot_table.loc[pivot_table.age < 60, 0] = 1
            pivot_table.set_index('age', inplace = True)
            pivot_table.loc[pivot_table.sum(axis = 1) == 0, 4] = 1
            pivot_table.to_csv(filename, header = False)
            assert ((pivot_table.sum(axis = 1) - 1).abs() < 1e-15).all(), \
                pivot_table.sum(axis = 1).loc[((pivot_table.sum(axis = 1) - 1).abs() > 1e-15)]

            # Add liam2 compatible header
            verbatim_header = '''age,initial_state,,,
,0,2,3,4
'''
            with file(filename, 'r') as original:
                data = original.read()
            with file(filename, 'w') as modified:
                modified.write(verbatim_header + data)
            log.info('Saving {}'.format(filename))


def create_dependance_initialisation_share(filename_prefix = None, smooth = False, window = 7, std = 2,
        survey = 'hsm', age_min = None, scale = 4):
    """
    Create dependance_niveau variable initialisation file for use in til-france model (option dependance_RT)
    """
    assert scale in [4, 5], "scale should be equal to 4 or 5"
    assert age_min is not None
    config = Config()
    input_dir = config.get('til', 'input_dir')
    assert survey in ['both', 'hsm']
    for sexe in ['homme', 'femme']:
        if survey == 'hsm':
            pivot_table = get_hsm_prevalence_pivot_table(sexe = sexe, scale = 4)
        else:
            pivot_table = get_hsi_hsm_prevalence_pivot_table(sexe = sexe, scale = 4)


        if filename_prefix is None:
            filename = os.path.join(input_dir, 'dependance_initialisation_level_share_{}.csv'.format(sexe))
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


def get_hsi_prevalence_pivot_table(sexe = None, scale = None):
    config = Config()
    assert scale in [4, 5], "scale should be equal to 4 or 5"
    if scale == 5:
        xls_path = os.path.join(config.get('raw_data', 'hsm_dependance_niveau'), 'hsi_desc_dependance_scale5_gir.xls')
        data = pd.read_excel(xls_path)
        log.debug('Droping NA values {}'.format(data.loc[data.age.isnull()]))
        data = (data
            .dropna()
            .rename(columns = {
                'scale5_c': 'dependance_niveau',
                #   est1_gir_s	est2_gir_s
            })
            )
    elif scale == 4:
        xls_path = os.path.join(
            config.get('raw_data', 'hsm_dependance_niveau'), 'desc_dependance_scale4_HSI_20180525.xls')
        data = pd.read_excel(xls_path)
        log.debug('Droping NA values {}'.format(data.loc[data.age.isnull()]))
        data = (data
            .dropna()
            .rename(columns = {
                'scale4': 'dependance_niveau',
                'femme': 'sexe'
                #   est1_gir_s	est2_gir_s
                })
            )
    assert data.sexe.isin([0, 1]).all()
    assert sexe in ['homme', 'femme']
    sexe = 1 if sexe == 'femme' else 0
    assert sexe in data.sexe.unique(), "sexe should be in {}".format(data.sexe.unique().tolist())
    pivot_table = (data[['dependance_niveau', 'poids_hsi', 'age', 'sexe']]
        .query('sexe == @sexe')
        .groupby(['age', 'dependance_niveau'])['poids_hsi'].sum().reset_index()
        .pivot('age', 'dependance_niveau', 'poids_hsi')
        .replace(0, np.nan)  # Next three lines to remove all 0 columns
        .dropna(how = 'all', axis = 1)
        .replace(np.nan, 0)
        )

    return pivot_table.copy()


def get_hsm_prevalence_pivot_table(sexe = None, scale = None):
    config = Config()
    assert scale in [4, 5], "scale should be equal to 4 or 5"
    if scale == 5:
        xls_path = os.path.join(config.get('raw_data', 'hsm_dependance_niveau'), 'desc_dependance_scale5.xls')
        data = (pd.read_excel(xls_path)
            .rename(columns = {'disability_scale5': 'dependance_niveau', 'woman': 'sexe'})
            )
    elif scale == 4:
        xls_path = os.path.join(
            config.get('raw_data', 'hsm_dependance_niveau'), 'desc_dependance_scale4_HSM_20180525.xls')
        data = (pd.read_excel(xls_path)
            .rename(columns = {
                'Scale4': 'dependance_niveau',
                'femme': 'sexe',
                'poids': 'poids_hsm',
                })
            )

    assert sexe in ['homme', 'femme']
    sexe = 1 if sexe == 'femme' else 0
    assert sexe in data.sexe.unique(), "sexe should be in {}".format(data.sexe.unique().tolist())
    pivot_table = (data[['dependance_niveau', 'poids_hsm', 'age', 'sexe']]
        .query('sexe == @sexe')
        .groupby(['dependance_niveau', 'age'])['poids_hsm'].sum().reset_index()
        .pivot('age', 'dependance_niveau', 'poids_hsm')
        .replace(0, np.nan)  # Next three lines to remove all 0 columns
        .dropna(how = 'all', axis = 1)
        .replace(np.nan, 0)
        )

    return pivot_table


def get_hsi_hsm_prevalence_pivot_table(sexe = None, scale = None):
    return (
        get_hsm_prevalence_pivot_table(sexe = sexe, scale = scale).add(
            get_hsi_prevalence_pivot_table(sexe = sexe, scale = scale), fill_value=0)
        )


def get_hsm_dependance_gir_mapping(sexe = None, scale = 4):
    config = Config()
    assert scale in [4, 5]
    if scale == 5:
        filename = 'desc_dependance_scale5_gir.xls'
    else:
        filename = 'desc_dependance_scale4_HSM_20180525.xls'

    xls_path = os.path.join(config.get('raw_data', 'hsm_dependance_niveau'), filename)
    data = pd.read_excel(xls_path)
    log.debug('Droping NA values {}'.format(data.loc[data.est1gir.isnull()]))
    data = (data
        .dropna()
        .astype({
#            'est1_gir_s': int,
#            'est2_gir_s': int,
            'est1gir': int,
            'est2gir': int,
            })
        .rename(columns = {
            'Age': 'age',
            'Sexe': 'sexe',
            'Scale5': 'dependance_niveau',
            'Scale4': 'dependance_niveau',
            'Poids': 'poids_hsm',
            'poids': 'poids_hsm',
            'est1gir': 'gir',
            'femme': 'sexe',
            #   est1_gir_s	est2_gir_s
        })
        )

    assert sexe in ['homme', 'femme']
    sexe = 1 if sexe == 'femme' else 0
    assert sexe in data.sexe.unique(), "sexe should be in {}".format(data.sexe.unique().tolist())
    pivot_table = (data[['dependance_niveau', 'gir', 'poids_hsm', 'age', 'sexe']]
        .query('sexe == @sexe')
        .groupby(['age', 'dependance_niveau', 'gir'])['poids_hsm'].sum().reset_index()
        .sort_values(['age', 'dependance_niveau', 'gir'])
        )

    return pivot_table.copy()


def get_hsi_dependance_gir_mapping(sexe = None, scale = 4):
    assert scale in [4, 5]
    config = Config()
    if scale == 5:
        filename = 'hsi_desc_dependance_scale5_gir.xls'
    else:
        filename = 'desc_dependance_scale4_HSI_20180525.xls'

    xls_path = os.path.join(config.get('raw_data', 'hsm_dependance_niveau'), filename)
    data = pd.read_excel(xls_path)
    log.debug('Droping NA values {}'.format(data.loc[data.age.isnull()]))
    print data.columns
    data = (data
        .dropna()
        .astype({
#            'est1_gir_s': int,
#            'est2_gir_s': int,
            'est1gir': int,
            'est2gir': int,

            })
        .rename(columns = {
            'scale5_c': 'dependance_niveau',
            'scale4': 'dependance_niveau',
            'est1gir': 'gir',
            'femme': 'sexe',
            #   est1_gir_s	est2_gir_s
        })
        )

    assert sexe in ['homme', 'femme']
    sexe = 1 if sexe == 'femme' else 0
    assert sexe in data.sexe.unique(), "sexe should be in {}".format(data.sexe.unique().tolist())
    pivot_table = (data[['dependance_niveau', 'gir', 'poids_hsi', 'age', 'sexe']]
        .query('sexe == @sexe')
        .groupby(['age', 'dependance_niveau', 'gir'])['poids_hsi'].sum().reset_index()
        .sort_values(['age', 'dependance_niveau', 'gir'])
        )

    return pivot_table.copy()


def get_hsi_hsm_dependance_gir_mapping(sexe = None):
    return (get_hsi_dependance_gir_mapping(sexe = sexe)
        .rename(columns = {'poids_hsi': 'poids'})
        .set_index(['age', 'dependance_niveau', 'gir'])
        .add(
            (get_hsm_dependance_gir_mapping(sexe = sexe)
                .rename(columns = {'poids_hsm': 'poids'})
                .set_index(['age', 'dependance_niveau', 'gir'])
                ),
            fill_value = 0,
            )
        )


def get_paquid_prevalence_pivot_table(sexe = None, year_max = None, year_min = None):
    assert sexe in ['homme', 'femme']
    sexe = 1 if sexe == 'homme' else 2
    data = get_clean_paquid()
    data.rename(columns = {'initial_state': 'dependance_niveau'}, inplace = True)
    year_max = year_max if year_max is not None else data.annee.max()
    year_min = year_min if year_min is not None else data.annee.min()
    data = data.query('(annee <= @year_max) and (annee >= @year_min)').copy()
    data['age'] = data['age'].round().astype(int)
    pivot_table = (data[['dependance_niveau', 'age', 'sexe']]
        .query('(sexe == @sexe) and (dependance_niveau != 5)')
        .groupby(['age', 'dependance_niveau']).size().reset_index(name = 'counts')
        .sort_values(['age', 'dependance_niveau'])
        .pivot('age', 'dependance_niveau', 'counts')
        .replace(0, np.nan)  # Next three lines to remove all 0 columns
        .dropna(how = 'all', axis = 1)
        .replace(np.nan, 0)
            )
    return pivot_table


def plot_prevalence(save_figure = False, smooth = False, window = 7, std = 2, age_min = 50, survey = 'hsm', scale = 4):
    assert survey in ['hsm', 'both']
    for sexe in ['homme', 'femme']:
        if survey == 'hsm':
            pivot_table = get_hsm_prevalence_pivot_table(sexe = sexe, scale = scale)
            title = u"HSM: Prévalence par âge des {}s".format(sexe)
            filename = 'dependance_niveau_hsm_{}.png'.format(sexe)
        else:
            pivot_table = get_hsi_hsm_prevalence_pivot_table(sexe = sexe, scale = scale)
            title = u"HSM/HSI: Prévalence par âge des {}s".format(sexe)
            filename = 'dependance_niveau_hsm_hsi_{}.png'.format(sexe)

        if smooth:
            pivot_table = smooth_pivot_table(pivot_table, window = window, std = std)
        # Go from levels to pct
        pivot_table = pivot_table.divide(pivot_table.sum(axis=1), axis=0).query('age >= @age_min')

        ax = pivot_table.plot.area(stacked = True, color = colors, title = title)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title = u'Niveau de \ndépendance')
        if save_figure:
            fig = ax.get_figure()
            fig.savefig(
                os.path.join(figures_directory, filename),
                bbox_inches = 'tight'
                )


def plot_prevalence_paquid(year_max = None, year_min = None, save_figure = False, smooth = False, window = 7, std = 2, age_min = 65):
    for sexe in ['homme', 'femme']:
        pivot_table = get_paquid_prevalence_pivot_table(sexe = sexe, year_max = year_max, year_min = year_min)

        if smooth:
            pivot_table = smooth_pivot_table(pivot_table, window = window, std = std)
        # Go from levels to pct
        pivot_table = pivot_table.divide(pivot_table.sum(axis = 1), axis = 0).query('age >= @age_min')

        ax = pivot_table.plot.area(stacked = True, color = colors, title = u"PAQUID: Prévalence par âge des {}s".format(sexe))
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title = u'Niveau de \ndépendance')
        if save_figure:
            fig = ax.get_figure()
            fig.savefig(
                os.path.join(figures_directory, 'dependance_niveau_paquid_{}_{}_{}.png'.format(
                    sexe, year_min, year_max)),
                bbox_inches = 'tight'
                )


def smooth_pivot_table(pivot_table, window = 7, std = 2):
    smoothed_pivot_table = pivot_table.copy()
    for dependance_niveau in smoothed_pivot_table.columns:
        smoothed_pivot_table[dependance_niveau] = (pivot_table[dependance_niveau]
            .rolling(win_type = 'gaussian', center = True, window = window, axis = 0)
            .mean(std = std)
            )

    return smoothed_pivot_table


def diagnostic_prevalence():
    pct = True
    import matplotlib.pyplot as plt
    sns.set_style("whitegrid")

    for year in [1988, 1998, 2008]:
        fig = plt.figure()
        i = 1
        for sexe in ['homme', 'femme']:
            paquid_pivot_table = get_paquid_prevalence_pivot_table(sexe = sexe, year_max = year, year_min = year)
            hsm_pivot_table = get_hsm_prevalence_pivot_table(sexe = sexe)
            if pct:
                hsm_pivot_table = hsm_pivot_table.divide(hsm_pivot_table.sum(axis = 1), axis = 0)
                paquid_pivot_table = paquid_pivot_table.divide(paquid_pivot_table.sum(axis = 1), axis = 0)
            fig.add_subplot(1, 2, i)
            (hsm_pivot_table - paquid_pivot_table).dropna().plot.line(ax = plt.gca(), sharey = True)
            i += 1
            fig.savefig(
                os.path.join(figures_directory, 'comparaison_paquid_hsm_{}.png'.format(
                    year)),
                bbox_inches = 'tight'
                )


if __name__ == "__main__":
    logging.basicConfig(level = logging.DEBUG, stream = sys.stdout)


    df = get_hsi_dependance_gir_mapping(sexe = 'homme', scale = 4)

    STOP
    pivot_table = create_dependance_initialisation_share(smooth = True, survey = 'both', age_min = 50)
    print pivot_table

    plot_prevalence(smooth = True, survey = 'both')

    boum
    plot_prevalence(save_figure = True, smooth = True, survey = 'both')

    BIM

    sexe = 'homme'
    df = get_hsm_prevalence_pivot_table(sexe = sexe)
    for year in [1988, 1998, 2008]:
        plot_prevalence_paquid(save_figure = True, year_min = year, year_max = year, smooth = False)

