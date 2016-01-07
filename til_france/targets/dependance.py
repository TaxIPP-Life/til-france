# -*- coding:utf-8 -*-


from __future__ import division


import logging
import numpy
import os
import pandas

from liam2.importer import array_to_disk_array
from til_core.config import Config
from til_france.tests.base import line_prepender, til_france_path


log = logging.getLogger(__name__)


drees_excel_file_path = os.path.join(til_france_path, 'param', 'demo', 'drees', 'dss43_horizon_2060.xls')


def build_prevalence_2010(input_dir = None, uniform_weight = None):
    assert uniform_weight is not None
    if input_dir is None:
        config = Config()
        input_dir = config.get('til', 'input_dir')

    df = pandas.read_excel(drees_excel_file_path, sheetname ='Tab2', header = 3, parse_cols = 'B:O', skip_footer = 4)
    for column in df.columns:
        if column.startswith('Unnamed') or column.startswith('Ensemble'):
            del df[column]
    df.index = [index.year for index in df.index]
    df.columns = range(1, 7)
    check_dependance_directory_existence(input_dir)
    csv_file_path = os.path.join(input_dir, 'parameters', 'dependance', 'dependance_prevalence_2010.csv')
    data = pandas.DataFrame(df.xs(2010)).T
    data = (data / uniform_weight).apply(numpy.round)
    data.astype(int).to_csv(csv_file_path, index = False)
    line_prepender(csv_file_path, 'age_category')


def build_prevalence_all_years(globals_node = None, input_dir = None, to_csv = None):
    assert globals_node or to_csv
    if to_csv:
        if input_dir is None:
            config = Config()
            input_dir = config.get('til', 'input_dir')

    df = pandas.read_excel(drees_excel_file_path, sheetname ='Tab6A', header = 3, parse_cols = 'B:E', skip_footer = 3)
    # "Au 1er janvier"
    df.columns = ['year', 'dependants_optimiste', 'DEPENDANTS', 'dependants_pessimiste']
    df.set_index('year', inplace = True)
    data = df.reindex(index = range(2010, 2061)).interpolate(method='polynomial', order = 6)
    data.index = [int(str(year - 1)) for year in data.index]
    data.index.name = "PERIOD"

    if globals_node:
        array_to_disk_array(globals_node, 'dependance_prevalence_all_years', data.DEPENDANTS.values)
    elif to_csv:
        check_dependance_directory_existence(input_dir)
        csv_file_path = os.path.join(input_dir, 'parameters', 'dependance', 'dependance_prevalence_all_years.csv')
        data = data.reset_index()[['PERIOD', 'DEPENDANTS']]
        data.astype(int).to_csv(csv_file_path, index = False)


def check_dependance_directory_existence(input_dir):
    dependance_directory_path = os.path.join(os.path.join(input_dir, 'parameters', 'dependance'))
    if not os.path.exists(dependance_directory_path):
        log.info('Creating directory {}'.format(dependance_directory_path))
        os.makedirs(dependance_directory_path)
