# -*- coding:utf-8 -*-

import os
import pkg_resources

import pandas
from webcolors import rgb_to_hex

from til_core.config import Config
from til_core.simulation import TilSimulation


# RGB tuples
ipp_colors_not_normalized = dict(
    ipp_very_dark_blue = (0, 80, 101),
    ipp_dark_blue = (0, 135, 152),
    ipp_medium_blue = (146, 205, 220),
    ipp_light_blue = (183, 222, 232),
    ipp_very_light_blue = (218, 238, 243),
    ipp_blue = (75, 172, 197)
    )

ipp_colors = dict((name, rgb_to_hex(rgb)) for name, rgb in ipp_colors_not_normalized.iteritems())


til_france_path = os.path.join(
    pkg_resources.get_distribution('Til-France').location,
    'til_france',
    )


def create_til_simulation(input_name = None, option = None, output_name_suffix = 'output', uniform_weight = None):
    assert input_name is not None
    config = Config()
    name = input_name.lower()

    input_dir = config.get('til', 'input_dir')
    input_file = '{}_{}.h5'.format(input_name, uniform_weight)

    assertion_message = '''
Input file path is set to be {}.
Did you correctly create this file using DataTil ?
You should also check that the input path is correctly set in your config_local.ini'''.format(
        os.path.join(input_dir, input_file))

    assert os.path.exists(os.path.join(input_dir, input_file)), assertion_message

    output_dir = os.path.join(config.get('til', 'output_dir'), name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if option:
        assert isinstance(option, str)
        console_relative_path = os.path.join('options', 'console_{}.yml'.format(option))
    else:
        console_relative_path = os.path.join('standard', 'console.yml')

    console_file = os.path.join(til_france_path, 'model', console_relative_path)
    simulation = TilSimulation.from_yaml(
        console_file,
        input_dir = input_dir,
        input_file = input_file,
        output_dir = output_dir,
        output_file = '{}_{}.h5'.format(name, output_name_suffix),
        uniform_weight = uniform_weight,
        # tax_benefit_system = 'tax_benefit_system',  # Add the OpenFisca TaxBenfitSystem to use
        )

    return simulation


def create_or_get_figures_directory(simulation):
    figures_directory = os.path.join(
        os.path.dirname(simulation.data_sink.output_path),
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

    directory = os.path.dirname(simulation.data_sink.output_path)
    uniform_weight = simulation.uniform_weight

    for csv_file in ['civilstate', 'stat', 'workstate']:
        df = pandas.read_csv(os.path.join(directory, csv_file + '.csv'))
        df.set_index('period', inplace = True)
        ax = (df * uniform_weight).plot(title = csv_file)
        ax.legend()
        fig = ax.get_figure()
        fig.savefig(os.path.join(figures_directory, '{}.png'.format(csv_file)))
        del ax, fig
