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


import numpy
import os
import pandas

import tables


from til_base_model.tests.base import line_prepender, til_base_model_path


try:
    from liam2.importer import array_to_disk_array
except ImportError:
    from src.importer import array_to_disk_array


drees_excel_file_path = os.path.join(til_base_model_path, 'param', 'demo', 'drees', 'dss43_horizon_2060.xls')


def build_prevalence_2010():
    df = pandas.read_excel(drees_excel_file_path, sheetname ='Tab2', header = 3, parse_cols = 'B:O', skip_footer = 4)
    for column in df.columns:
        if column.startswith('Unnamed') or column.startswith('Ensemble'):
            del df[column]
    df.index = [index.year for index in df.index]
    df.columns = range(1, 7)
    csv_file_path = os.path.join(til_base_model_path, 'param', 'demo', 'dependance_prevalence_2010.csv')
    data = pandas.DataFrame(df.xs(2010)).T
    data = (data / 200).apply(numpy.round)   # TODO fix this
    data.astype(int).to_csv(csv_file_path, index = False)
    line_prepender(csv_file_path, 'age_category')


def build_prevalence_all_years():
    df = pandas.read_excel(drees_excel_file_path, sheetname ='Tab6A', header = 3, parse_cols = 'B:E', skip_footer = 3)
    # "Au 1er janvier"
    df.columns = ['year', 'dependants_optimiste', 'DEPENDANTS', 'dependants_pessimiste']
    df.set_index('year', inplace = True)
    data = df.reindex(index = range(2010, 2061)).interpolate(method='polynomial', order = 6)
    # On passe en année pleine
    data.index = [int(str(year - 1) + "01") for year in data.index]
    data.index.name = "PERIOD"

    simulation_file_path = os.path.join(til_base_model_path, 'Patrimoine_next_metro_200.h5')
    h5file = tables.open_file(simulation_file_path, mode="a")
    array_to_disk_array(h5file, '/globals', 'dependance_prevalence_all_years', data.DEPENDANTS.values)
    h5file.close()

    csv_file_path = os.path.join(til_base_model_path, 'param', 'demo', 'dependance_prevalence_all_years.csv')
    data = data.reset_index()[['PERIOD', 'DEPENDANTS']]
    data.astype(int).to_csv(csv_file_path, index = False)

