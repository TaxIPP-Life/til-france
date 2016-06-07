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
from StringIO import StringIO


from til_france.tests.base import create_or_get_figures_directory, ipp_colors, to_percent_round_formatter


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
    return _extract(
        simulation,
        filename = 'dependance_gir',
        removed_lines_prefixes = ['dependance', 'period', ',,'],
        columns = ['period', 'age', 'sexe', 'dependance_gir', 'total', 'total_global'],
        drop = ['total_global'],
        reweight = ['total']
        )


def extract_incidence_csv(simulation):
    return _extract(
        simulation,
        filename = 'dependance_incidence',
        removed_lines_prefixes = ['incidence', 'period', ',,,', ',,total'],
        columns = ['period', 'age', 'sexe', 'False', 'dependant', 'total'],
        drop = ['False'],
        reweight = ['dependant', 'total']
        )


def extract_deces_csv(simulation):
    return _extract(
        simulation,
        filename = 'dependance_deces',
        removed_lines_prefixes = ['dependance_deces', 'period', ',,,', ',,total'],
        columns = ['period', 'age', 'sexe', 'False', 'deces', 'total'],
        drop = ['False'],
        reweight = ['deces', 'total']
        )


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
    ax.set_ylabel("effectifs en milliers")
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
    ax.set_ylabel("effectifs en milliers")
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    fig = ax.get_figure()

    fig.savefig(os.path.join(figures_directory, 'dependance_gir.pdf'), bbox_inches='tight')
    del ax, fig


def plot_dependance_prevalence_by_age(simulation, years = None, ax = None, age_max = None):
    assert years is not None
    figures_directory = create_or_get_figures_directory(simulation)

    data = extract_dependance_gir_csv(simulation)
    data = (data
        .groupby(['dependance_gir', 'period', 'age', 'sexe'])['total'].sum()
        .unstack(['dependance_gir']))
    data[0] = data[-1] + data[0]
    data.drop(-1, axis = 1, inplace = True)
    data['prevalence'] = sum([data[i] for i in range(1, 5)]) / sum([data[i] for i in range(5)])
    data.drop(range(5), axis = 1, inplace = True)

    data = data.reset_index()
    data.age = data.age.astype(int)

    ylabel = u"taux de prévalence"
    return _plot_and_or_save(ax = ax, data = data, figures_directory = figures_directory,
        name = 'prevalence', pdf_name = None, years = years, age_max = age_max, ylabel = ylabel)


def plot_dependance_incidence_by_age(simulation, years = None, ax = None, age_max = None):
    assert years is not None
    figures_directory = create_or_get_figures_directory(simulation)

    data = extract_incidence_csv(simulation)
    data = (data
        .set_index(['period', 'age', 'sexe'])
        .eval('incidence = dependant / total', inplace = False)
        .reset_index())
    data.age = data.age.astype(int)
    ylabel = "taux d'incidence"
    return _plot_and_or_save(ax = ax, data = data, figures_directory = figures_directory,
                      name = 'incidence', pdf_name = None, years = years, age_max = age_max, ylabel = ylabel)


def plot_dependance_mortalite_by_age(simulation, years = None, ax = None, age_max = None):
    assert years is not None
    figures_directory = create_or_get_figures_directory(simulation)

    data = extract_deces_csv(simulation)
    data = (data
        .set_index(['period', 'age', 'sexe'])
        .eval('mortalite = deces / total', inplace = False)
        .reset_index())
    data.age = data.age.astype(int)
    ylabel = u"quotient de mortalité des personnes dépendantes"
    return _plot_and_or_save(ax = ax, data = data, figures_directory = figures_directory,
                      name = 'mortalite', pdf_name = None, years = years, age_max = age_max, ylabel = ylabel)


def plot_dependance_by_age(simulation, years = None, age_max = None, save = True):
    figures_directory = create_or_get_figures_directory(simulation)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    legend_handles, legend_labels = plot_dependance_prevalence_by_age(
        simulation, years = years, ax = ax1, age_max = age_max)
    plot_dependance_incidence_by_age(simulation, years = years, ax = ax2, age_max = age_max)
    plot_dependance_mortalite_by_age(simulation, years = years, ax = ax3, age_max = age_max)

    ax3.legend().set_visible(False)
    plt.draw()

    # fig.legend(tuple(legend_handles), tuple(legend_labels), loc='best')
    plt.legend(tuple(legend_handles), tuple(legend_labels), loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),
       bbox_transform = plt.gcf().transFigure, ncol = len(years) * 2)
    if save:
        fig.savefig(os.path.join(figures_directory, 'dependance_by_age.pdf'), bbox_inches='tight')


def plot_dependance_by_age_separate(simulation, years = None, age_max = None, save = True):
    plot_dependance_prevalence_by_age(simulation, years = years, age_max = age_max)
    plot_dependance_incidence_by_age(simulation, years = years, age_max = age_max, )
    plot_dependance_mortalite_by_age(
        simulation, years = years, age_max = age_max)


# Helpers

def _extract(simulation, filename, removed_lines_prefixes, columns, drop = None, reweight = None):

    directory = os.path.dirname(simulation.data_sink.output_path)
    uniform_weight = simulation.uniform_weight

    assert filename[-4:] != '.csv', 'filename should not contain extension'
    file_path = os.path.join(directory, '{}.csv'.format(filename))
    input_file = open(file_path)
    txt = input_file.readlines()[1:]

    for prefix in removed_lines_prefixes:
        txt = [line for line in txt if not(line.startswith(prefix))]
    df = pd.read_csv(
        StringIO('\n'.join(txt)),
        header = None,
        names = columns,
        )
    if drop:
        assert set(drop) < set(columns)
        df.drop(drop, axis = 1, inplace = True)
    if 'period' in columns:
        df.period = df.period.astype(int)

    for variable in reweight:
        assert variable in df.columns
        df[variable] = df[variable] * uniform_weight

    return df


def _plot_and_or_save(ax = None, data = None, figures_directory = None, name = None, pdf_name = None, years = None,
        age_max = None, ylabel = None):
    assert name is not None
    assert data is not None
    assert years is not None
    assert os.path.exists(figures_directory)
    data['age_group'] = np.trunc((data.age - 60) / 5)
    data.age_group = data.age_group.astype(int)

    if age_max:
        query = '(age >= 60) and (age <= @age_max) and (period in @years)'
    else:
        query = '(age >= 60) and (period in @years)'
    data_plot = data.query(query)[['age', 'period', name, 'sexe']]

    if ax is None:
        save_figure = True
        fig, ax = plt.subplots()
    else:
        save_figure = False

    colors = [ipp_colors[cname] for cname in ['ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]
    color_by_period = dict(zip(data_plot['period'].unique(), colors))
    legend_handles, legend_labels = list(), list()
    for grouped in data_plot.groupby(['period', 'sexe']):
        period, sexe = grouped[0]
        linestyle = '--' if sexe == 1 else "-"
        x = grouped[1]['age']
        y = grouped[1][name]
        line, = ax.plot(x, y, linestyle = linestyle, color = color_by_period[period], )
        legend_handles.append(line)
        sex_label = 'Hommes' if sexe == 0 else 'Femmes'
        legend_labels.append('{} - {}'.format(sex_label, period))

        ax.set_xlabel(u"âge")
        if ylabel is None:
            ax.set_ylabel(name)
        else:
            ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(to_percent_round_formatter)

    if save_figure:
        pdf_name = pdf_name if pdf_name else name
        ax.legend(legend_handles, legend_labels, loc = 'best')
        fig.savefig(os.path.join(figures_directory, '{}.pdf'.format(pdf_name)), bbox_inches='tight')
        del fig


    return legend_handles, legend_labels