# -*- coding: utf-8 -*-


import pandas as pd


def separate_cuts_from_variables(parameters_value_by_name):
    cuts = {
        cut: parameters_value_by_name.get(cut, None)
        for cut in parameters_value_by_name.keys()
        if cut.startswith('cut')
        }
    parameters = {
        cut: parameters_value_by_name.get(cut, None)
        for cut in parameters_value_by_name.keys()
        if not cut.startswith('cut')
        }
    return cuts, parameters


def build_function_str(function_name, cuts, variables):
    header = function_name
    value_expr = " + ".join([
        "{value} * {key}".
        format(key = key, value = value)
        for key, value in variables.iteritems()
        ])
    value_expr = """
    - value: {}
""".format(value_expr)
    cut_expr = """
    - niveau: if(
        value > {cut_4},
        5,
        if(
          value > {cut_3},
          4,
          if(
            value > {cut_2},
            3,
            if(
              value > {cut_1},
              2,
              1
              )
            )
          )
        )
""".format(**cuts)

file_path = "/home/benjello/Dropbox/Projet Dépendance - IPP (Dropbox)/Cadrage HSM/Tables/TIL_prevalence coef_RT.xls"

sheetnames_by_function = {
    'compute_dependance_niveau_homme_sup_75': 'Men_Above75',
    'compute_dependance_niveau_homme_inf_75': 'Men_Under75',
    'compute_dependance_niveau_femme_sup_80': 'Women_Above80',
    'compute_dependance_niveau_femme_inf_80': 'Women_Under80',
    }

vars = []
for function_name, sheetname in sheetnames_by_function.iteritems():
    parameters_value_by_name = pd.read_excel(file_path, sheetname = sheetname).transpose().to_dict()[0]
    cuts, variables = separate_cuts_from_variables(parameters_value_by_name)
    print variables
    vars += variables.keys()
    build_function_str(function_name, cuts, variables)

print list(set(vars))