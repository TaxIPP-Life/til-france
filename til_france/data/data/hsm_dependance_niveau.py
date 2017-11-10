# -*- coding: utf-8 -*-

import logging
import numpy as np
import os
import pandas as pd
import pkg_resources
import sys


from til_core.config import Config
from til_france.tests.base import ipp_colors

colors = [ipp_colors[cname] for cname in ['ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]


log = logging.getLogger(__name__)


figures_directory = os.path.join(
    pkg_resources.get_distribution('til-france').location,
    'til_france',
    'figures'
    )


def create_dependance_initialisation(filename_prefix = None, smooth = False, window = 7, std = 2):
    """
    Create dependance_niveau varaible initialisation file for use in til-france model (option dependance_RT)
    """
    for sexe in [0, 1]:
        pivot_table = get_hsm_prevalence_pivot_table(sexe = sexe)
        if smooth:
            pivot_table = smooth_pivot_table(pivot_table, window = window, std = std)
        # Go from levels to pct
        pivot_table = pivot_table.divide(pivot_table.sum(axis=1), axis=0)
        sexe_str = 'homme' if sexe == 0 else 'femme'
        if filename_prefix is None:
            config = Config()
            input_dir = config.get('til', 'input_dir')
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
            assert ((pivot_table.sum(axis = 1) - 1).abs() < 1e-15).all()

            # Add liam2 compatible header
            verbatim_header = '''age,initial_state,,,,
,0,1,2,3,4
'''
            with file(filename, 'r') as original:
                data = original.read()
            with file(filename, 'w') as modified:
                modified.write(verbatim_header + data)
            log.info('Saving {}'.format(filename))


def get_hsm_prevalence_pivot_table(sexe = None):
    config = Config()
    xls_path = os.path.join(config.get('raw_data', 'hsm_dependance_niveau'), 'desc_dependance.xls')
    data = (pd.read_excel(xls_path)
        .rename(columns = {'disability_scale3': 'dependance_niveau', 'woman': 'sexe'})
        )
    assert sexe is not None
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


def plot_prevalence(save_figure = False, smooth = False, window = 7, std = 2):
    for sexe in [0, 1]:
        pivot_table = get_hsm_prevalence_pivot_table(sexe = sexe)
        if smooth:
            pivot_table = smooth_pivot_table(pivot_table, window = window, std = std)
        # Go from levels to pct
        pivot_table = pivot_table.divide(pivot_table.sum(axis=1), axis=0)
        ax = pivot_table.plot.area(stacked = True, color = colors, title = "sexe = {}".format(int(sexe)))
        if save_figure:
            fig = ax.get_figure()
            fig.savefig(
                os.path.join(figures_directory, 'dependance_niveau_hsm_sexe_{}.png'.format(int(sexe))),
                bbox_inches = 'tight'
                )


def plot_prevalence_paquid(year_max = None, year_min = None, save_figure = False, smooth = False, window = 7, std = 2):
    from til_france.model.options.dependance_RT.life_expectancy.transition_matrices import get_clean_paquid
    data = get_clean_paquid()
    data.rename(columns = {'initial_state': 'dependance_niveau'}, inplace = True)
    year_max = year_max if year_max is not None else data.annee.max()
    year_min = year_min if year_min is not None else data.annee.min()
    data = data.query('(annee <= @year_max) and (annee >= @year_min)').copy()
    data['age'] = data['age'].round().astype(int)
    for sexe in data.sexe.unique():
        pivot_table = (data[['dependance_niveau', 'age', 'sexe']]
            .query('(sexe == @sexe) and (dependance_niveau != 5)')
            .groupby(['age', 'dependance_niveau']).size().reset_index(name = 'counts')
            .sort_values(['age', 'dependance_niveau'])
            .pivot('age', 'dependance_niveau', 'counts')
            .replace(0, np.nan)  # Next three lines to remove all 0 columns
            .dropna(how = 'all', axis = 1)
            .replace(np.nan, 0)
            )

        if smooth:
            pivot_table = smooth_pivot_table(pivot_table, window = window, std = std)

        pivot_table = pivot_table.divide(pivot_table.sum(axis=1), axis=0)
        ax = pivot_table.plot.area(stacked = True, color = colors, title = "sexe = {}".format(int(sexe)))

        if save_figure:
            fig = ax.get_figure()
            fig.savefig(
                os.path.join(figures_directory, 'dependance_niveau_paquid_sexe_{}.png'.format(int(sexe))),
                bbox_inches = 'tight'
                )


def smooth_pivot_table(pivot_table, window = 7, std = 2):
    smoothed_pivot_table = pivot_table.copy()
    for dependance_niveau in range(5):
        smoothed_pivot_table[dependance_niveau] = (pivot_table[dependance_niveau]
            .rolling(win_type = 'gaussian', center = True, window = window, axis = 0)
            .mean(std = std)
            )
    return smoothed_pivot_table


if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, stream = sys.stdout)
    plot_prevalence(save_figure = True, smooth = True)
    plot_prevalence_paquid(save_figure = True, year_min = 2008, year_max = 2008, smooth = True)
