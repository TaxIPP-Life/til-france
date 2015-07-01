# -*- coding:utf-8 -*-

import os
import pkg_resources


import pandas
import matplotlib
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter

from til_base_model.config import Config
from til.simulation import TilSimulation


matplotlib.style.use('ggplot')


path_model = os.path.join(
    pkg_resources.get_distribution('Til-BaseModel').location,
    'til_base_model',
    )


def create_til_simulation(capitalized_name = None, uniform_weight = None):
    assert capitalized_name is not None
    config = Config()
    name = capitalized_name.lower()

    input_dir = path_model
    input_file = '{}.h5'.format(capitalized_name)
    assert os.path.exists(os.path.join(input_dir, input_file))

    output_dir = os.path.join(config.get('til', 'output_dir'), name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    console_file = os.path.join(path_model, 'console.yml')
    simulation = TilSimulation.from_yaml(
        console_file,
        input_dir = input_dir,
        input_file = input_file,
        output_dir = output_dir,
        output_file = '{}_til_output.h5'.format(name),
        # tax_benefit_system = 'tax_benefit_system',  # Add the OpenFisca TaxBenfitSystem to use
        )
    if uniform_weight:
        simulation.uniform_weight = uniform_weight

    return simulation


def create_or_get_figures_directory(simulation):
    figures_directory = os.path.join(
        os.path.dirname(simulation.data_source.output_path),
        'figures',
        )
    if not os.path.exists(figures_directory):
        os.mkdir(figures_directory)
    return figures_directory


def plot_csv(simulation):
    figures_directory = create_or_get_figures_directory(simulation)

    directory = os.path.dirname(simulation.data_source.output_path)
    uniform_weight = simulation.uniform_weight

    for csv_file in ['civilstate', 'stat', 'workstate']:
        df = pandas.read_csv(os.path.join(directory, csv_file + '.csv'))
        df.period = (df['period'] / 100).round().astype(int)
        df.set_index('period', inplace = True)
        plt.figure()
        ax = (df * uniform_weight).plot(title = csv_file)
        ax.legend()
        fig = ax.get_figure()
        fig.savefig(os.path.join(figures_directory, '{}.png'.format(csv_file)))
        del ax, fig


def extract_population_csv(simulation):
    from StringIO import StringIO

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

    panel = proto_panel.set_index(['period'], append = True).to_panel().astype(int)
    return panel * uniform_weight


def get_data_frame_insee(gender, by = 'age_group'):
    data_path = '/home/benjello/openfisca/Til-BaseModel/til_base_model/param/demo/projpop0760_FECcentESPcentMIGcent.xls'
    sheetname_by_gender = dict(zip(
        ['total', 'male', 'female'],
        ['populationTot', 'populationH', 'populationF']
        ))
    population_insee = pandas.read_excel(
        data_path, sheetname = sheetname_by_gender[gender], skiprows = 2, header = 2)[:109].set_index(
            u'Âge au 1er janvier')

    population_insee.reset_index(inplace = True)
    population_insee.drop(population_insee.columns[0], axis = 1, inplace = True)
    population_insee.index.name = 'age'
    population_insee.columns = population_insee.columns + (-1)  # Passage en âge en fin d'année
    if by == 'age':
        return population_insee
    population_insee.reset_index(inplace = True)
    population_insee['age_group'] = population_insee.age // 10
    population_insee.drop('age', axis = 1, inplace = True)
    data_frame_insee = population_insee.groupby(['age_group']).sum()
    return data_frame_insee


def get_insee_projection(quantity, gender, function = None):

    assert quantity in ['births', 'deaths', 'migrants', 'population']
    assert gender in ['total', 'male', 'female']

    data_path = '/home/benjello/openfisca/Til-BaseModel/til_base_model/param/demo/projpop0760_FECcentESPcentMIGcent.xls'
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
    population.index.name = 'year'
    ax = population.plot(
        title = 'Components of population growth',
        style = ['k-', 'k--', 'g-', 'g--', 'b-', 'b--'],
        xticks = [period for period in list(population.index.astype(int))],
        )
    fig = ax.get_figure()
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    fig.savefig(os.path.join(figures_directory, 'population_growth_compoennts.png'))
    del ax, fig
    return population


def extract_dependance_csv(simulation):
    from StringIO import StringIO

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
    panel_simulation = extract_dependance_csv(simulation)
    panel_simulation = panel_simulation.squeeze()
    plt.figure()
    ax = panel_simulation.plot()
    ax.legend()
    fig = ax.get_figure()
    fig.savefig(os.path.join(figures_directory, 'dependance.png'))
    del ax, fig
