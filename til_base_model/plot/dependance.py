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
import os
import pandas
from StringIO import StringIO


from til_base_model.tests.base import create_or_get_figures_directory, ipp_colors


def extract_dependance_csv(simulation):

    directory = os.path.dirname(simulation.data_source.output_path)
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
        df = pandas.read_csv(csv_string)
        df.set_index(df.columns[0], inplace = True)
        df.index.name = 'age'
        df.columns = df.iloc[0]
        df.drop(df.index[0], axis = 0, inplace = True)
        df.columns = ['male', 'female', 'total']
        df.columns.name = "dependants"
        df['period'] = period // 100
        proto_panel = pandas.concat([proto_panel, df]) if proto_panel is not None else df

    panel = proto_panel.loc['total'].set_index(['period'], append = True).to_panel().astype(int)
    return panel * uniform_weight


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
    plt.figure()
    ax = panel_simulation.plot(
        linewidth = 2,
        colors = [ipp_colors[name] for name in ['ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']],
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
