# -*- coding:utf-8 -*-

import os
import pkg_resources


import pandas


from til_base_model.config import Config
from til.simulation import TilSimulation


path_model = os.path.join(
    pkg_resources.get_distribution('Til-BaseModel').location,
    'til_base_model',
    )


def create_til_simulation(capitalized_name = None, uniform_weight = None):
    assert capitalized_name is not None
    config = Config()
    name = capitalized_name.lower()

    input_dir = path_model
    input_file = '{}.h5'.format(capitalized_name),
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


def plot_csv(simulation):
    directory = os.path.dirname(simulation.data_source.output_path)
    uniform_weight = simulation.uniform_weight
    files_path = [
        os.path.join(directory, csv_file + '.csv')
        for csv_file in 'civilstate', 'stat', 'workstate'
        ]
    for file_path in files_path:
        df = pandas.read_csv(file_path)
        df.period = (df['period'] / 100).round().astype(int)
        df.set_index('period', inplace = True)
        (df * uniform_weight).plot(title = file_path)


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
    print uniform_weight
    return panel * uniform_weight


def get_data_frame_insee(gender):
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
    population_insee.reset_index(inplace = True)
    population_insee['age_group'] = population_insee.age // 10
    population_insee.drop('age', axis = 1, inplace = True)
    data_frame_insee = population_insee.groupby(['age_group']).sum()
    return data_frame_insee


def plot_population2(simulation):
    figures_directory = os.path.join(
        os.path.dirname(simulation.data_source.output_path),
        'figures',
        )

    if not os.path.exists(figures_directory):
        os.mkdir(figures_directory)

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

        ax = relative_diff.T.plot(title = gender)
        fig = ax.get_figure()
        fig.savefig(os.path.join(figures_directory, 'population_{}.png'.format(gender)))