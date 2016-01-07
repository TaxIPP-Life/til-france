# -*- coding: utf-8 -*-


from __future__ import division


import logging
import os
import pandas
import pkg_resources


from liam2.importer import array_to_disk_array
from til_france.targets.population import build_mortality_rates

log = logging.getLogger(__name__)


def add_mortality_rates(globals_node):
    path_model = os.path.join(
        pkg_resources.get_distribution('Til-France').location,
        'til_france',
        )
    data_path = os.path.join(path_model, 'param', 'demo')

    mortality_by_gender = build_mortality_rates(to_csv = False)
    array_by_gender, df_1997_by_gender, array_1997_by_gender = (dict(), ) * 3
    for gender in ['male', 'female']:
        mortality_by_gender[gender].columns = [
            "period_{}".format(column) for column in mortality_by_gender[gender].columns
            ]
        array_by_gender[gender] = mortality_by_gender[gender].values
        df_1997_by_gender[gender] = pandas.read_csv(
            os.path.join(data_path, 'mortality_rate_{}_1997.csv'.format(gender)))
        array_1997_by_gender[gender] = df_1997_by_gender[gender]['mortality_rate_{}_1997'.format(gender)].values

        array_to_disk_array(globals_node, 'mortality_rate_{}'.format(gender), array_by_gender[gender])
        array_to_disk_array(globals_node, 'mortality_rate_{}_1997'.format(gender), array_1997_by_gender[gender])


if __name__ == "__main__":
    log.setLevel(logging.INFO)
    build_mortality_rates(to_csv = True)
