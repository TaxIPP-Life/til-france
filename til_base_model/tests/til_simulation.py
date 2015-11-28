# -*- coding: utf-8 -*-


# OpenFisca -- A versatile microsimulation software
# By: OpenFisca Team <contact@openfisca.fr>
#
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import os
import pkg_resources


import pandas


from til_base_model.config import Config
from til.simulation import TilSimulation


til_base_model_path = os.path.join(
    pkg_resources.get_distribution('Til-BaseModel').location,
    'til_base_model',
    )


def get_simulation(capitalized_name = None, output_name_suffix = 'til_output', uniform_weight = None):

    assert capitalized_name is not None
    config = Config()
    name = capitalized_name.lower()

    input_dir = config.get('til', 'input_dir')
    input_file = '{}.h5'.format(capitalized_name)
    assertion_message = '''
Input file path should be {}.
You should run DataTil and check that the input path is correctly set in your config_local.ini'''.format(
        os.path.join(input_dir, input_file))
    assert os.path.exists(os.path.join(input_dir, input_file)), assertion_message

    output_dir = os.path.join(config.get('til', 'output_dir'), name)
    assert os.path.exists(output_dir)

    console_file = os.path.join(til_base_model_path, 'console.yml')
    simulation = TilSimulation.from_yaml(
        console_file,
        input_dir = input_dir,
        input_file = input_file,
        output_dir = output_dir,
        output_file = '{}_{}.h5'.format(name, output_name_suffix),
        # tax_benefit_system = 'tax_benefit_system',  # Add the OpenFisca TaxBenfitSystem to use
        )
    if uniform_weight:
        simulation.uniform_weight = uniform_weight

    return simulation


def test_panel():
    variable_name = 'dependance_level'
    simulation = get_simulation(
        capitalized_name = 'Patrimoine_next_metro_200',
        output_name_suffix = 'test_institutions',
        uniform_weight = 200
        )
    panel = simulation.get_variable([variable_name], fillna_value = 0)
    panel_mean = simulation.get_variable([variable_name], fillna_value = 0, function = "mean")
    panel_sum = simulation.get_variable([variable_name], fillna_value = 0, function = "sum")
    return panel, panel_mean, panel_sum


def test_csv():
    directory = os.path.abspath('/home/benjello/data/til/output/til/')
    files_path = [
        os.path.join(directory, csv_file + '.csv')
        for csv_file in 'civilstate', 'stat', 'workstate'
        ]
    for file_path in files_path:
        df = pandas.read_csv(file_path)
        df.period = (df['period'] / 100).round().astype(int)
        df.set_index('period', inplace = True)
        (df * 200).plot(title = file_path)





if __name__ == '__main__':

    # panel, panel_mean, panel_sum = test_panel()
    simulation = get_simulation(
        capitalized_name = 'Patrimoine_next_metro_200',
        output_name_suffix = 'test_institutions',
        uniform_weight = 200
        )
    dependance_level = simulation.get_variable(['dependance_level'])
    data_origin = simulation.get_variable(['data_origin'])