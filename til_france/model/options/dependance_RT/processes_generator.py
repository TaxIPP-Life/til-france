# -*- coding: utf-8 -*-


import os
import pandas as pd
import pkg_resources
import yaml


dependance_functions_yml_path = os.path.join(
    pkg_resources.get_distribution('til-france').location,
    'til_france',
    'model',
    'options',
    'dependance_RT',
    'dependance_initialisation_functions.yml'
    )


dependance_transition_yml_path = os.path.join(
    pkg_resources.get_distribution('til-france').location,
    'til_france',
    'model',
    'options',
    'dependance_RT',
    'dependance_transition_functions.yml'
    )

prevalence_coef_path = os.path.join(
    pkg_resources.get_distribution('til-france').location,
    'til_france',
    'model',
    'options',
    'dependance_RT',
    'assets',
    'TIL_prevalence coef_RT.xls'
    )

sheetnames_by_function = {
    'compute_dependance_niveau_homme_sup_75': 'Men_Above75',
    'compute_dependance_niveau_homme_inf_75': 'Men_Under75',
    'compute_dependance_niveau_femme_sup_80': 'Women_Above80',
    'compute_dependance_niveau_femme_inf_80': 'Women_Under80',
    }


def build_function_str(function_name, cuts, variables):
    # header = function_name
    value_formula = " + ".join([
        "{value} * {key}".
        format(key = key, value = value)
        for key, value in variables.iteritems()
        ])
    value_expr = {'value': value_formula}
    cut_expr = {
        'dependance_niveau': "if(value > {cut_4}, 5, if(value > {cut_3}, 4, if(value > {cut_2}, 3, if(value > {cut_1}, 2, 1))))".format(**cuts)
        }
    return [value_expr, cut_expr, 'return dependance_niveau']


def rename_variables(variables):
    new_by_old_variable = {
        'lqm_2008': 'lq',
        'nb_children': 'nb_enfants',
        'partner': 'INCOUPLE',
        'child_0': '(nb_enfants <= 0)',
        'child_3_more': '(nb_enfants >= 3)',
        'educ_1': '(education_niveau == 1)',
        'educ_2': '((education_niveau >= 2) and (education_niveau <= 3))',
        'educ_3': '(education_niveau >= 4)',
        'age_80': '(age >= 80)',
        'age_90': '(age >= 90)',
        'age_more90': '(age >= 99)',
        'seul': '(not INCOUPLE)',
        'femme': 'ISFEMALE',
        }
    for old_key, new_key in new_by_old_variable.iteritems():
        if old_key in variables:
            variables[new_key] = variables.pop(old_key)
    return variables


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


def create_initialisation():
    # renamed_vars = []
    # vars = []
    process_by_initial_state = dict()
    processes = dict(processes = process_by_initial_state)
    individus = dict(individus = processes)
    main = dict(entities = individus)
    for function_name, sheetname in sheetnames_by_function.iteritems():
        parameters_value_by_name = pd.read_excel(prevalence_coef_path, sheetname = sheetname).transpose().to_dict()[0]
        cuts, variables = separate_cuts_from_variables(parameters_value_by_name)
        # vars += variables.keys()
        variables = rename_variables(variables)
        # renamed_vars += variables.keys()
        process_by_initial_state[function_name + '(lq, nb_enfants)'] = build_function_str(function_name, cuts, variables)

    with open(dependance_functions_yml_path, 'w') as outfile:
        yaml.dump(main, outfile, default_flow_style = False, width = 1000)


process_by_initial_state = dict()
processes = dict(processes = process_by_initial_state)
individus = dict(individus = processes)
main = dict(entities = individus)
file_path_by_state = dict(
    [
        (
            state,
            os.path.join(
                pkg_resources.get_distribution('til-france').location,
                'til_france',
                'model',
                'options',
                'dependance_RT',
                'assets',
                'etat_initial_{}.xlsx'.format(state)
                )
            ) for state in range(5)
    ])

for initial_state, file_path in file_path_by_state.iteritems():
    variables_by_final_state = pd.read_excel(file_path).to_dict()
    initial_state_actions = list()
    process_by_initial_state['etat_{}()'.format(initial_state)] = initial_state_actions
    for final_state, variables in variables_by_final_state.iteritems():
        variables = rename_variables(variables)
        value_formula = " + ".join([
            "{value} * {key}".format(key = key, value = value)
            for key, value in variables.iteritems()
            if (key != 'cons') & (value != 0)
            ])
        if 'cons' in variables and variables['cons'] != 0:
            value_formula = '{} + {}'.format(variables['cons'], value_formula)
        if value_formula == '':
            value_formula = 0
        initial_state_actions.append({
            "prob_" + str(final_state): "exp({})".format(value_formula)
            })

    initial_state_actions.append({
        "z": " + ".join([
            "prob_" + str(final_state)
            for final_state in variables_by_final_state.keys()
            ])
        })

    for final_state in variables_by_final_state.keys():
        initial_state_actions.append({
            "prob_" + str(final_state): "prob_{} / z".format(final_state)
            })

    prob_final_states = ["prob_" + final_state for final_state in variables_by_final_state.keys()]
    probabilities_list_str = "[" + ", ".join(prob_final_states) + "]"
    final_states_index_list_str = [
        int(final_state[-1:]) for final_state in prob_final_states
        ]
    initial_state_actions.append(
        "return choice({}, {})".format(
            final_states_index_list_str,
            probabilities_list_str
            )
        )

with open(dependance_transition_yml_path, 'w') as outfile:
    yaml.dump(main, outfile, default_flow_style = False, width = 1000)
