# -*- coding:utf-8 -*-


# Copyright (C) 2011, 2012, 2013, 2014, 2015 OpenFisca Team
# https://github.com/openfisca
#
# This file is part of OpenFisca.
#
# OpenFisca is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# OpenFisca is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import pandas as pd
import seaborn as sns
from StringIO import StringIO


from til_france.tests.base import create_or_get_figures_directory, ipp_colors


def extract_dependance_csv(simulation):

    directory = os.path.dirname(simulation.data_sink.output_path)
    uniform_weight = simulation.uniform_weight
    file_path = os.path.join(directory, 'dependance.csv')

    input_file = open(file_path)
    txt = input_file.readlines()[1:]
    txt = [line for line in txt if not line.startswith('dependance')]
    date = None
    data = None
    data_by_date = dict()
    data = []
    is_period_line = False
    for line in txt:
        if is_period_line:
            if date and data:
                data_by_date.update({date: data})
                data = []
            date = int(line.strip())
        elif line.startswith('period'):
            pass
        else:
            data.append(line.strip())
        is_period_line = True if line.startswith('period') else False

    proto_panel = None
    for period, data in data_by_date.iteritems():
        csv_string = StringIO('\n'.join(data_by_date[period]))
        df = pd.read_csv(csv_string)
        df.set_index(df.columns[0], inplace = True)
        df.index.name = 'age'
        df.columns = df.iloc[0]
        df.drop(df.index[0], axis = 0, inplace = True)
        df.columns = ['male', 'female', 'total']
        df.columns.name = "dependants"
        df['period'] = period
        proto_panel = pd.concat([proto_panel, df]) if proto_panel is not None else df

    panel = proto_panel.loc['total'].set_index(['period'], append = True).to_panel().astype(int)
    return panel * uniform_weight


def extract_dependance_gir_csv(simulation):
    directory = os.path.dirname(simulation.data_sink.output_path)
    uniform_weight = simulation.uniform_weight
    file_path = os.path.join(directory, 'dependance_gir.csv')

    input_file = open(file_path)
    txt = input_file.readlines()[1:]
    txt = [line for line in txt if not(
        line.startswith('dependance') or line.startswith('period') or line.startswith(',,'))]

    df = pd.read_csv(
        StringIO('\n'.join(txt)),
        header = None,
        names = ['period', 'age', 'sexe', 'dependance_gir', 'total', 'total_global'],
        )
    df.drop('total_global', axis = 1, inplace = True)
    df.period = df.period.astype(int)
    df.total = df.total * uniform_weight
    return df


def extract_incidence_csv(simulation):
    directory = os.path.dirname(simulation.data_sink.output_path)
    uniform_weight = simulation.uniform_weight
    file_path = os.path.join(directory, 'dependance_incidence.csv')

    input_file = open(file_path)
    txt = input_file.readlines()[1:]
    txt = [line for line in txt if not(
        line.startswith('incidence') or
        line.startswith('period') or
        line.startswith(',,,') or
        line.startswith(',,total')
        )]

    df = pd.read_csv(
        StringIO('\n'.join(txt)),
        header = None,
        names = ['period', 'age', 'sexe', 'False', 'dependant', 'total'],
        )
    df.drop('False', axis = 1, inplace = True)
    df.period = df.period.astype(int)
    df.dependant = df.dependant * uniform_weight
    df.total = df.total * uniform_weight
    return df


def extract_deces_csv(simulation):
    directory = os.path.dirname(simulation.data_sink.output_path)
    uniform_weight = simulation.uniform_weight
    file_path = os.path.join(directory, 'dependance_deces.csv')

    input_file = open(file_path)
    txt = input_file.readlines()[1:]
    txt = [line for line in txt if not(
        line.startswith('period') or
        line.startswith(',,,') or
        line.startswith(',,total')
        )]

    df = pd.read_csv(
        StringIO('\n'.join(txt)),
        header = None,
        names = ['period', 'age', 'sexe', 'False', 'deces', 'total'],
        )
    df.drop('False', axis = 1, inplace = True)
    df.period = df.period.astype(int)
    df.deces = df.deces * uniform_weight
    df.total = df.total * uniform_weight
    return df


def plot_dependance_csv(simulation):
    figures_directory = create_or_get_figures_directory(simulation)
    panel_simulation = extract_dependance_csv(simulation)
    panel_simulation = panel_simulation.squeeze() / 1000
    panel_simulation.index.name = None
    panel_simulation.rename(
        columns = dict(
            male = "hommes",
            female = "femmes",
            ),
        inplace = True
        )
    ax = panel_simulation.plot(
        linewidth = 2,
        color = [ipp_colors[name] for name in ['ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']],
        xticks = [period for period in range(
            min(panel_simulation.index.astype(int)),
            max(panel_simulation.index.astype(int)),
            10
            )],
        # cmap = 'PuBu'
        )
    ax.set_ylabel(u"effectifs en milliers")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    fig = ax.get_figure()

    fig.savefig(os.path.join(figures_directory, 'dependance.pdf'), bbox_inches='tight')
    del ax, fig


def plot_dependance_gir_csv(simulation):
    figures_directory = create_or_get_figures_directory(simulation)
    data = (extract_dependance_gir_csv(simulation)
        .groupby(['period', 'dependance_gir'])['total'].sum()
        .unstack()
        .drop([0, -1], axis = 1) / 1000)
    ax = data.plot(
        linewidth = 2,
        # color = [ipp_colors[name] for name in ['ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']],
        xticks = [period for period in range(
            min(data.index.astype(int)),
            max(data.index.astype(int)),
            10
            )],
        )
    ax.set_ylabel(u"effectifs en milliers")
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    fig = ax.get_figure()

    fig.savefig(os.path.join(figures_directory, 'dependance_gir.pdf'), bbox_inches='tight')
    del ax, fig


def plot_dependance_prevalence_by_age(simulation, years = None, ax = None):
    assert years is not None
    figures_directory = create_or_get_figures_directory(simulation)

    data = extract_dependance_gir_csv(simulation)
    data = (data
        .groupby(['dependance_gir', 'period', 'age', 'sexe'])['total'].sum()
        .unstack(['dependance_gir'])
        )
    data[0] = data[-1] + data[0]
    data.drop(-1, axis = 1, inplace = True)
    data['prevalence'] = sum([data[i] for i in range(1, 5)])/ sum([data[i] for i in range(5)])
    data.drop(range(5), axis = 1, inplace = True)

    data = data.reset_index()
    data.age = data.age.astype(int)
    data['age_group'] = np.trunc((data.age - 60) / 5)
    data.age_group = data.age_group.astype(int)

    # data = data.query('age_group > 0').copy()

    data_plot = (data
        .query(
            '(age > 60) and (period in @years)'
            )
        )[['age', 'period', 'prevalence', 'sexe']]

    if ax == None:
        save_figure = False
        fig, ax = plt.subplots()

    colors = [ipp_colors[name] for name in ['ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]
    color_by_period = dict(zip(data_plot['period'].unique(), colors))
    legend_items = list()
    for grouped in data_plot.groupby(['period', 'sexe']):
        period, sexe = grouped[0]
        linestyle = '--' if sexe == 1 else "-"

        grouped[1].plot(
            x = 'age', y = 'prevalence', kind = 'line', ax = ax,
            linestyle = linestyle, color = color_by_period[period]
            )
        legend_items.append([sexe, period])

    plt.legend(legend_items, loc='best')

    if save_figure:
        fig = ax.get_figure()
        fig.savefig(os.path.join(figures_directory, 'prevalence.pdf'), bbox_inches='tight')
        del fig


def plot_dependance_incidence_by_age(simulation, years = None):
    assert years is not None
    figures_directory = create_or_get_figures_directory(simulation)

    data = extract_incidence_csv(simulation)
    data = (data
        .set_index(['period', 'age', 'sexe'])
        .eval('incidence = dependant / total', inplace = False)
        .reset_index()
        )
    data.age = data.age.astype(int)
    data['age_group'] = np.trunc((data.age - 60) / 5)
    data.age_group = data.age_group.astype(int)

    # data = data.query('age_group > 0').copy()

    data_plot = (data
        .query(
            '(age > 60) and (period in @years)'
            )
        )[['age', 'period', 'incidence', 'sexe']]

    fig, ax = plt.subplots()
    colors = [ipp_colors[name] for name in ['ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]
    color_by_period = dict(zip(data_plot['period'].unique(), colors))
    legend_items = list()
    for grouped in data_plot.groupby(['period', 'sexe']):
        print grouped[0]
        period, sexe = grouped[0]
        linestyle = '--' if sexe == 1 else "-"

        grouped[1].plot(
            x = 'age', y = 'incidence', kind = 'line', ax = ax,
            linestyle = linestyle, color = color_by_period[period]
            )
        legend_items.append([sexe, period])

    plt.legend(legend_items, loc='best')

    fig = ax.get_figure()
    fig.savefig(os.path.join(figures_directory, 'incidence.pdf'), bbox_inches='tight')
    del fig
    return ax

def plot_mortalite_by_age(simulation, years = None):
    assert years is not None
    figures_directory = create_or_get_figures_directory(simulation)

    data = extract_deces_csv(simulation)
    data = (data
        .set_index(['period', 'age', 'sexe'])
        .eval('mortalite = deces / total', inplace = False)
        .reset_index()
        )
    data.age = data.age.astype(int)

    data_plot = (data
        .query(
            '(age > 60) and (period in @years)'
            )
        )[['age', 'period', 'mortalite', 'sexe']]

    fig, ax = plt.subplots()
    colors = [ipp_colors[name] for name in ['ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]
    color_by_period = dict(zip(data_plot['period'].unique(), colors))
    legend_items = list()
    for grouped in data_plot.groupby(['period', 'sexe']):
        print grouped[0]
        period, sexe = grouped[0]
        linestyle = '--' if sexe == 1 else "-"

        grouped[1].plot(
            x = 'age', y = 'mortalite', kind = 'line', ax = ax,
            linestyle = linestyle, color = color_by_period[period]
            )
        legend_items.append([sexe, period])

    plt.legend(legend_items, loc='best')

    fig = ax.get_figure()
    fig.savefig(os.path.join(figures_directory, 'incidence.pdf'), bbox_inches='tight')
    del fig
    return ax