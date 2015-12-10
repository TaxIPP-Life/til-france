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


from __future__ import division


import logging
import os
import pandas
import pkg_resources
import tables

from liam2.importer import array_to_disk_array


log = logging.getLogger(__name__)


def add_mortality_rates(simulation_file_name = None):
    assert simulation_file_name is not None
    path_model = os.path.join(
        pkg_resources.get_distribution('Til-France').location,
        'til_france',
        )
    # Data from INSEE projections
    data_path = os.path.join(path_model, 'param', 'demo')

    sheetname_by_gender = dict(zip(
        ['male', 'female'],
        ['hyp_mortaliteH', 'hyp_mortaliteF']
        ))
    mortality_by_gender = dict(
        (
            gender,
            pandas.read_excel(
                os.path.join(data_path, 'projpop0760_FECcentESPcentMIGcent.xls'),
                sheetname = sheetname, skiprows = 2, header = 2
                )[:121].set_index(
                    u"Âge atteint dans l'année", drop = True
                    ).reset_index()
            )
        for gender, sheetname in sheetname_by_gender.iteritems()
        )

    for df in mortality_by_gender.values():
        del df[u"Âge atteint dans l'année"]

    mortality_by_gender['male'].columns = [
        "period_{}".format(column) for column in mortality_by_gender['male'].columns
        ]
    mortality_by_gender['female'].columns = [
        "period_{}".format(column) for column in mortality_by_gender['female'].columns
        ]

    male_array = mortality_by_gender['male'].values / 1e4
    female_array = mortality_by_gender['female'].values / 1e4

    male_1997 = pandas.read_csv(os.path.join(data_path, 'mortality_rate_male_1997.csv'))
    female_1997 = pandas.read_csv(os.path.join(data_path, 'mortality_rate_female_1997.csv'))

    male_1997_array = male_1997['mortality_rate_male_1997'].values
    female_1997_array = female_1997['mortality_rate_female_1997'].values

    h5file = tables.open_file(os.path.join(path_model, simulation_file_name), mode="r+")
    try:
        globals_node = h5file.create_group("/", 'globals')
        log.info("Creating node globals")
    except:
        log.info("Reinitializing node globals")
        h5file.remove_node("/globals", recursive= True)
        globals_node = h5file.create_group("/", 'globals')

    array_to_disk_array(globals_node, 'mortality_rate_male', male_array)
    array_to_disk_array(globals_node, 'mortality_rate_female', female_array)
    array_to_disk_array(globals_node, 'mortality_rate_male_1997', male_1997_array)
    array_to_disk_array(globals_node, 'mortality_rate_female_1997', female_1997_array)
    h5file.close()
