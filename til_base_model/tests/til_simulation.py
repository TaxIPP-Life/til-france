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


import matplotlib.pyplot as plt
import pandas


from til_base_model.config import Config
from til.simulation import TilSimulation
from til.simulation import weighted_sum
from til.simulation import weighted_count


path_model = os.path.join(
    pkg_resources.get_distribution('Til-BaseModel').location,
    'til_base_model',
    )


def get_simulation():
    config = Config()
    output_dir = config.get('til', 'output_dir')
    # output_dir = os.path.join(os.path.dirname(__file__), 'output'),
    console_file = os.path.join(path_model, 'console.yml')
    simulation = TilSimulation.from_yaml(
        console_file,
        input_dir = None,
        input_file = 'Patrimoine_next_200.h5',
        output_dir = output_dir,
        output_file = 'simul_long_2000.h5',
        )
    return simulation


def test_panel():
    variable_name = 'nb_children_af'
    simulation = get_simulation()
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

    TODO: this file must be cleaned
    panel, panel_mean, panel_sum = test_panel()
