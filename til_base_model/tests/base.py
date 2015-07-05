# -*- coding:utf-8 -*-

import os
import pkg_resources


import pandas
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


from til_base_model.config import Config
from til.simulation import TilSimulation


# matplotlib.style.use('ggplot')


til_base_model_path = os.path.join(
    pkg_resources.get_distribution('Til-BaseModel').location,
    'til_base_model',
    )


def create_til_simulation(capitalized_name = None, uniform_weight = None):
    assert capitalized_name is not None
    config = Config()
    name = capitalized_name.lower()

    input_dir = til_base_model_path
    input_file = '{}.h5'.format(capitalized_name)
    assert os.path.exists(os.path.join(input_dir, input_file)), 'Input file path should be {}'.format(
        os.path.join(input_dir, input_file))

    output_dir = os.path.join(config.get('til', 'output_dir'), name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    console_file = os.path.join(til_base_model_path, 'console.yml')
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
    assert os.path.exists(figures_directory)
    return figures_directory


def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)


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
