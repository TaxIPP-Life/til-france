# -*- coding:utf-8 -*-

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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import division


import os
import pandas


from til_france.tests.base import til_france_path


def get_data_frame_insee(gender, by = 'age_group'):

    data_path = os.path.join(til_france_path, 'param/demo/projpop0760_FECcentESPcentMIGcent.xls')
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
