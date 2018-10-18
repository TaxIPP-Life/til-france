# -*- coding:utf-8 -*-


from __future__ import division


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import pandas as pd
from StringIO import StringIO


from til_france.tests.base import til_france_path, create_or_get_figures_directory, ipp_colors, get_data_directory
from til_france.targets.population import get_data_frame_insee


def extract(simulation, backup, csv_extract_function):
    directory = get_data_directory(simulation, backup = backup)
    uniform_weight = simulation.uniform_weight
    if os.path.exists(os.path.join(directory, 'replication_0')):
        multiple = True
        base_directory = directory

    if multiple:
        replication_number = 0
        result = list()
        while replication_number >= 0:
            directory = os.path.join(base_directory, 'replication_{}'.format(replication_number))
            if os.path.exists(directory):
                result.append(csv_extract_function(directory, weight = uniform_weight))
                replication_number += 1

            else:
                return result

    else:
        return csv_extract_function(directory, weight = uniform_weight)


def extract_population_csv(simulation, backup = None):

    directory = get_data_directory(simulation, backup = backup)
    uniform_weight = simulation.uniform_weight
    if os.path.exists(os.path.join(directory, 'replication_0')):
        multiple = True
        base_directory = directory
    else:
        multiple = False

    def _extract_population_csv(directory, weight = None):
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
            df = pd.read_csv(csv_string)
            df.set_index(df.columns[0], inplace = True)
            df.index.name = 'age_group'
            df.columns = df.iloc[0]
            df.drop(df.index[0], axis = 0, inplace = True)
            df.columns = ['male', 'female', 'total']
            df.columns.name = "population"
            df['period'] = period
            proto_panel = pd.concat([proto_panel, df]) if proto_panel is not None else df

        panel = proto_panel.reset_index()
        panel = panel.set_index(['period', 'age_group']).astype('int')

        return panel * weight

    if multiple:
        replication_number = 0
        result = list()
        while replication_number >= 0:
            directory = os.path.join(base_directory, 'replication_{}'.format(replication_number))
            if os.path.exists(directory):
                result.append(_extract_population_csv(directory, weight = uniform_weight))
                replication_number += 1

            else:
                return result

    else:
        return _extract_population_csv(directory, weight = uniform_weight)


def extract_population_by_age_csv(simulation, backup = None):
    directory = get_data_directory(simulation, backup = backup)
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
        df = pd.read_csv(csv_string)
        df.set_index(df.columns[0], inplace = True)
        df.index.name = 'age'
        df.columns = df.iloc[0]
        df.drop(df.index[0], axis = 0, inplace = True)
        df.columns = ['male', 'female', 'total']
        df.columns.name = "population"
        df['period'] = period
        proto_panel = pd.concat([proto_panel, df]) if proto_panel is not None else df

    panel = proto_panel.set_index(['period'], append = True).to_panel().fillna(0).astype(int)

    return panel * uniform_weight


def get_insee_projection(quantity, gender, function = None, scenario = central):

    assert quantity in ['naissances', 'deces', 'migrations', 'population']
    assert gender in ['total', 'male', 'female']
    assert scenario not None

    if scenario == 'central':

        data_path = os.path.join(til_france_path, 'param/demo/projpop0760_FECcentESPcentMIGcent.xls')
        sheetname_by_gender_by_quantity = dict(
            naissances = 'nbre_naiss',
            deces = dict(zip(
                ['total', 'male', 'female'],
                ['nbre_deces', 'nbre_decesH', 'nbre_decesF']
                )),
            migrations = dict(zip(
                ['male', 'female'],
                ['hyp_soldemigH', 'hyp_soldemigF']
                )),
            population = dict(zip(
                ['total', 'male', 'female'],
                ['populationTot', 'populationH', 'populationF']
                ))
            )
    age_label_by_quantity = dict(
        naissances = u"Âge au 1er janvier",
        deces = u"Âge atteint dans l'année",
        migrations = u"Âge atteint dans l'année",
        population = u'Âge au 1er janvier'
        )

    age_label = age_label_by_quantity[quantity]
    if quantity == 'naissances':
        row_end = 36
    elif quantity == 'migrations':
        row_end = 111
    else:
        row_end = 109

    if quantity in ['deces', 'population']:
        sheetname = sheetname_by_gender_by_quantity[quantity][gender]
    elif quantity == 'naissances':
        sheetname = sheetname_by_gender_by_quantity[quantity]
        age_label = u'Âge au 1er janvier'
    elif quantity == 'migrations' and gender in ['male', 'female']:
        sheetname = sheetname_by_gender_by_quantity[quantity][gender]
    elif quantity == 'migrations' and gender not in ['male', 'female']:
        return (
            get_insee_projection(quantity, 'male', function = function) +
            get_insee_projection(quantity, 'female', function = function)
            )

    data_frame = pd.read_excel(
        data_path, sheetname = sheetname, skiprows = 2, header = 2)[:row_end].set_index(
            age_label)

    if function == 'sum':
        data_frame.reset_index(inplace = True)
        data_frame.drop(data_frame.columns[0], axis = 1, inplace = True)
        data_frame.index.name = 'age'
        return data_frame.sum()

    else:
        return data_frame


def plot_population(simulation, backup = None):
    figures_directory = create_or_get_figures_directory(simulation, backup = backup)
    panel_simulation = extract_population_csv(simulation, backup = backup)

    if isinstance(panel_simulation, list):
        data_concat = pd.concat(panel_simulation)
        print(data_concat)
        by_row_index = data_concat.groupby(data_concat.index)
        panel_simulation = by_row_index.mean()
        multi_index = pd.MultiIndex.from_tuples(panel_simulation.index, names = ('period', 'age_group'))
        panel_simulation = panel_simulation.reindex(multi_index)

    for gender in ['total', 'male', 'female']:
        data_frame_insee = get_data_frame_insee(gender)
        data_frame_simulation = panel_simulation[gender].unstack('period').drop('total').fillna(0)
        data_frame_simulation.index = data_frame_simulation.index.astype(int)
        data_frame_simulation.sort_index(inplace = True)

        absolute_diff = (
            data_frame_simulation - data_frame_insee[data_frame_simulation.columns]
            )
        relative_diff = (
            data_frame_simulation - data_frame_insee[data_frame_simulation.columns]
            ) / data_frame_insee[data_frame_simulation.columns]
        ax = relative_diff.T.plot(title = gender)
        fig = ax.get_figure()
        fig.savefig(os.path.join(figures_directory, 'population_rel_diff_{}.png'.format(gender)))
        plt.draw()

    del ax, fig

    for gender in ['total', 'male', 'female']:
        data_frame_insee_total = get_data_frame_insee(gender).sum()
        data_frame_simulation = panel_simulation[gender].unstack('period').drop('total').fillna(0)
        data_frame_simulation.index = data_frame_simulation.index.astype(int)
        data_frame_simulation.sort_index(inplace = True)
        data_frame_simulation_total = data_frame_simulation.sum()

        data_frame_insee_total = data_frame_insee_total.loc[data_frame_simulation_total.index].copy()
        plt.figure()
        ax = data_frame_insee_total.plot(label = 'insee', style = 'b-', title = gender)
        ax2 = data_frame_simulation_total.plot(label = 'til', style = 'r-', ax = ax)
        ax.legend()
        fig = ax.get_figure()
        fig.savefig(os.path.join(figures_directory, 'population_{}.png'.format(gender)))
        plt.draw()

    del ax2, ax, fig


def plot_ratio_demographique(simulation, backup = None):
    figures_directory = create_or_get_figures_directory(simulation, backup = backup)

    panel_simulation = extract_population_by_age_csv(simulation, backup = backup)
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
    ratio_60 = ratio_60.loc['retired', :] / ratio_60.loc['active', :]
    ratio_60.index.name = None
    # plt.figure()
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


def population_diagnostic(simulation, backup = None):
    figures_directory = create_or_get_figures_directory(simulation, backup = backup)
    directory = os.path.dirname(simulation.data_sink.output_path)
    uniform_weight = simulation.uniform_weight

    population = None
    for csv_file in ['naissances', 'deces', 'migrations']:
        simulation_data_frame = pd.read_csv(os.path.join(directory, csv_file + '.csv'))
        simulation_data_frame.period = (simulation_data_frame['period']).round().astype(int)
        simulation_data_frame.set_index('period', inplace = True)

        insee_data_frame = pd.DataFrame({
            '{}_insee'.format(csv_file): get_insee_projection(csv_file, 'total', function = 'sum')
            })
        insee_data_frame.index.name = 'period'
        data_frame = pd.concat([simulation_data_frame * uniform_weight, insee_data_frame], axis = 1)

        if population is not None:
            population = pd.concat([population, data_frame], axis = 1)
        else:
            population = data_frame

    population.dropna(inplace = True)
    population.index.name = None
    population = population / 1000
    population.rename(
        columns = dict(
            naissances = 'naissances',
            naissances_insee = 'naissances (INSEE)',
            deces = u'décès',
            deces_insee = u'décès (INSEE)',
            migrations = 'solde migratoire',
            migrations_insee = 'solde migratoire (INSEE)',
            ),
        inplace = True
        )
    xtick_min, xtick_max = min(population.index.astype(int)), max(population.index.astype(int))
    xtick_interval = 10 if (xtick_max - xtick_min) > 20 else 1
    ax = population.plot(
        color = ipp_colors.values(),
        linewidth = 2,
        title = u"Composantes de la croissance démographique",
        xticks = [period for period in range(xtick_min, xtick_max + 1, xtick_interval)]
        )
    ax.set_ylabel(u"effectifs en milliers")
    ax.legend(loc='center', bbox_to_anchor = (0.5, -.2), ncol = 3)
    fig = ax.get_figure()
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    fig.savefig(os.path.join(figures_directory, 'population_growth_components.pdf'), bbox_inches='tight')
    plt.draw()
    del ax, fig
    return population


def plot_age_pyramid(age_min = 0, age_max = None, group_by = 1, year = 2010, backup = None):

    population = extract_population_by_age_csv(simulation, backup = backup)
    data = population.to_frame()[['male', 'female']]
    data = (data.stack()
        .unstack('period')
        .get(year)
        .unstack()
        .drop('total'))
    data.index = data.index.astype(int)
    data.sort_values(inplace = True)

    age_min = 60
    data = data.loc[age_min:age_max]

    data.group_by

    fig, axes = plt.subplots(ncols=2, sharey=True)
    axes[0].barh(data.index, data.male, align='center', color='gray', zorder=10)
    axes[0].set(title='Hommes')
    axes[1].barh(data.index, data.female, align='center', color='gray', zorder=10)
    axes[1].set(title='Femmes')

    axes[0].invert_xaxis()
    axes[0].set(yticks=data.index)
    axes[0].yaxis.tick_right()

    for ax in axes.flat:
        ax.margins(0.03)
        ax.grid(True)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.09)
    plt.show()
