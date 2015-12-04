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


from __future__ import division


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import pandas
from StringIO import StringIO


from til_base_model.tests.base import til_base_model_path, create_or_get_figures_directory, ipp_colors
from til_base_model.targets.population import  get_data_frame_insee


def extract_population_csv(simulation):
    directory = os.path.dirname(simulation.data_source.output_path)
    uniform_weight = simulation.uniform_weight
    file_path = os.path.join(directory, 'population2.csv')

    input_file = open(file_path)
    txt = input_file.readlines()[1:]
    txt = [line for line in txt if not line.startswith('population')]
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
        df = pandas.read_csv(csv_string)
        df.set_index(df.columns[0], inplace = True)
        df.index.name = 'age_group'
        df.columns = df.iloc[0]
        df.drop(df.index[0], axis = 0, inplace = True)
        df.columns = ['male', 'female', 'total']
        df.columns.name = "population"
        df['period'] = period // 100
        proto_panel = pandas.concat([proto_panel, df]) if proto_panel is not None else df

    panel = proto_panel.set_index(['period'], append = True).to_panel().fillna(0).astype(int)
    return panel * uniform_weight


def extract_population_by_age_csv(simulation):
    directory = os.path.dirname(simulation.data_source.output_path)
    uniform_weight = simulation.uniform_weight
    file_path = os.path.join(directory, 'population.csv')

    input_file = open(file_path)
    txt = input_file.readlines()[1:]
    txt = [line for line in txt if not line.startswith('population')]
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
        df = pandas.read_csv(csv_string)
        df.set_index(df.columns[0], inplace = True)
        df.index.name = 'age'
        df.columns = df.iloc[0]
        df.drop(df.index[0], axis = 0, inplace = True)
        df.columns = ['male', 'female', 'total']
        df.columns.name = "population"
        df['period'] = period // 100
        proto_panel = pandas.concat([proto_panel, df]) if proto_panel is not None else df

    panel = proto_panel.set_index(['period'], append = True).to_panel().fillna(0).astype(int)

    return panel * uniform_weight


def get_insee_projection(quantity, gender, function = None):

    assert quantity in ['births', 'deaths', 'migrants', 'population']
    assert gender in ['total', 'male', 'female']

    data_path = os.path.join(til_base_model_path, 'param/demo/projpop0760_FECcentESPcentMIGcent.xls')
    sheetname_by_gender_by_quantity = dict(
        births = 'nbre_naiss',
        deaths = dict(zip(
            ['total', 'male', 'female'],
            ['nbre_deces', 'nbre_decesH', 'nbre_decesF']
            )),
        migrants = dict(zip(
            ['male', 'female'],
            ['hyp_soldemigH', 'hyp_soldemigF']
            )),
        population = dict(zip(
            ['total', 'male', 'female'],
            ['populationTot', 'populationH', 'populationF']
            ))
        )
    age_label_by_quantity = dict(
        births = u"Âge au 1er janvier",
        deaths = u"Âge atteint dans l'année",
        migrants = u"Âge atteint dans l'année",
        population = u'Âge au 1er janvier'
        )

    age_label = age_label_by_quantity[quantity]
    if quantity == 'births':
        row_end = 36
    elif quantity == 'migrants':
        row_end = 111
    else:
        row_end = 109

    if quantity in ['deaths', 'population']:
        sheetname = sheetname_by_gender_by_quantity[quantity][gender]
    elif quantity == 'births':
        sheetname = sheetname_by_gender_by_quantity[quantity]
        age_label = u'Âge au 1er janvier'
    elif quantity == 'migrants' and gender in ['male', 'female']:
        sheetname = sheetname_by_gender_by_quantity[quantity][gender]
    elif quantity == 'migrants' and gender not in ['male', 'female']:
        return (
            get_insee_projection(quantity, 'male', function = function) +
            get_insee_projection(quantity, 'female', function = function)
            )

    data_frame = pandas.read_excel(
        data_path, sheetname = sheetname, skiprows = 2, header = 2)[:row_end].set_index(
            age_label)

    if function == 'sum':
        data_frame.reset_index(inplace = True)
        data_frame.drop(data_frame.columns[0], axis = 1, inplace = True)
        data_frame.index.name = 'age'
        return data_frame.sum()

    else:
        return data_frame


def plot_population2(simulation):
    figures_directory = create_or_get_figures_directory(simulation)

    panel_simulation = extract_population_csv(simulation)

    for gender in ['total', 'male', 'female']:
        data_frame_insee = get_data_frame_insee(gender)
        data_frame_simulation = panel_simulation[gender].drop('total')
        data_frame_simulation.index = data_frame_simulation.index.astype(int)
        data_frame_simulation.sort_index(inplace = True)

        absolute_diff = (
            data_frame_simulation - data_frame_insee[data_frame_simulation.columns]
            )

        relative_diff = (
            data_frame_simulation - data_frame_insee[data_frame_simulation.columns]
            ) / data_frame_insee[data_frame_simulation.columns]
        plt.figure()
        ax = relative_diff.T.plot(title = gender)
        fig = ax.get_figure()
        fig.savefig(os.path.join(figures_directory, 'population_rel_diff_{}.png'.format(gender)))
        del ax, fig

    for gender in ['total', 'male', 'female']:
        data_frame_insee_total = get_data_frame_insee(gender).sum()
        data_frame_simulation = panel_simulation[gender].drop('total')
        data_frame_simulation.index = data_frame_simulation.index.astype(int)
        data_frame_simulation.sort_index(inplace = True)
        data_frame_simulation_total = data_frame_simulation.sum()

        data_frame_insee_total = data_frame_insee_total.loc[data_frame_simulation_total.index].copy()

        plt.figure()
        ax = data_frame_insee_total.plot(label = 'insee', style = 'b-')
        data_frame_simulation_total.plot(label = 'til', style = 'r-', ax = ax)
        ax.legend()
        fig = ax.get_figure()
        fig.savefig(os.path.join(figures_directory, 'population_{}.png'.format(gender)))
        del ax, fig


def plot_ratio_demographique(simulation):
    figures_directory = create_or_get_figures_directory(simulation)

    panel_simulation = extract_population_by_age_csv(simulation)
    panel_simulation.drop('total', axis = 'major', inplace = True)

    def separate(age):
        age = int(age)
        if age <= 19:
            return 'kids'
        elif age <= 59:
            return 'active'
        elif age > 59:
            return 'retired'

    ratio_60 = panel_simulation['total'].groupby(separate, axis = 0).sum()
    ratio_60 = ratio_60.loc['retired', :]/ratio_60.loc['active', :]
    ratio_60.index.name = None
    plt.figure()
    ax = ratio_60.plot(
        color = ipp_colors['ipp_blue'],
        title = u'Ratio démographique',
        linewidth = 2,
        xticks = [period for period in range(
            min(ratio_60.index.astype(int)) + 1,
            max(ratio_60.index.astype(int)),
            10
            )],
        )
    ax.set_ylabel(u"(+ de 60 ans / 20-59 ans)")
    fig = ax.get_figure()
    fig.savefig(os.path.join(figures_directory, 'ratio_demographique.pdf'))
    del ax, fig


def population_diagnostic(simulation):
    figures_directory = create_or_get_figures_directory(simulation)
    directory = os.path.dirname(simulation.data_source.output_path)
    uniform_weight = simulation.uniform_weight

    population = None
    for csv_file in ['births', 'deaths', 'migrants']:
        simulation_data_frame = pandas.read_csv(os.path.join(directory, csv_file + '.csv'))
        simulation_data_frame.period = (simulation_data_frame['period'] / 100).round().astype(int)
        simulation_data_frame.set_index('period', inplace = True)

        insee_data_frame = pandas.DataFrame({
            '{}_insee'.format(csv_file): get_insee_projection(csv_file, 'total', function = 'sum')
            })
        insee_data_frame.index.name = 'period'

        data_frame = pandas.concat([simulation_data_frame * uniform_weight, insee_data_frame], axis = 1)

        if population is not None:
            population = pandas.concat([population, data_frame], axis = 1)
        else:
            population = data_frame

    population.dropna(inplace = True)
    population.index.name = None
    population = population / 1000
    population.rename(
        columns = dict(
            births = 'naissances',
            births_insee = 'naissances (INSEE)',
            deaths = u'décès',
            deaths_insee = u'décès (INSEE)',
            migrants = 'solde migratoire',
            migrants_insee = 'solde migratoire (INSEE)',
            ),
        inplace = True
        )
    ax = population.plot(
        color = ipp_colors.values(),
        linewidth = 2,
        title = u"Composantes de la croissance démographique",
        xticks = [period for period in range(
            min(population.index.astype(int)),
            max(population.index.astype(int)),
            10
            )],
        )
    ax.set_ylabel(u"effectifs en milliers")
    ax.legend(loc='center', bbox_to_anchor=(0.5, -.2), ncol=3)
    fig = ax.get_figure()
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    fig.savefig(os.path.join(figures_directory, 'population_growth_components.pdf'), bbox_inches='tight')
    del ax, fig
    return population
