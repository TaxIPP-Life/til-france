# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import pkg_resources


assets_path = config_files_directory = os.path.join(
    pkg_resources.get_distribution('til-france').location,
    'til_france',
    'model',
    'options',
    'dependance_RT',
    'assets',
    )


def rename_variables(variables):
    new_by_old_variable = {
        'age': '(age - 80)',
        'age2': '(age - 80)**2',
        'age3': '(age - 80)**3',
        }
    for old_key, new_key in new_by_old_variable.iteritems():
        if old_key in variables:
            variables[new_key] = variables.pop(old_key)
    return variables


def create_transition(cohort = 'paquid', sexe = None):
    assert cohort in ['paquid', '3c']
    if sexe is None:
        template = 'etat_initial_{}_corr.xlsx'
    elif sexe is not None:
        assert sexe in ['homme', 'femme']
        template = 'etat_initial_{}_corr' + '_{sexe}.xlsx'.format(sexe = sexe)
    else:
        raise
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
                    cohort,
                    template.format(state)
                    )
                ) for state in range(5)
        ])

    process_by_initial_state = dict()
    proba_by_initial_state = dict()
    for initial_state, file_path in file_path_by_state.iteritems():
        print initial_state
        variables_by_final_state = pd.read_excel(file_path).to_dict()
        initial_state_actions = list()
        process_by_initial_state['etat_{}()'.format(initial_state)] = initial_state_actions
        for final_state, variables in variables_by_final_state.iteritems():
            variables = rename_variables(variables)
            value_formula = " + ".join([
                "{value} * {key}".format(key = key, value = value)
                for key, value in variables.iteritems()
                if (key != 'constante') & (value != 0)
                ])
            if 'constante' in variables and variables['constante'] != 0:
                value_formula = '{} + {}'.format(variables['constante'], value_formula)
            if value_formula == '':
                value_formula = 0
            initial_state_actions.append({
                "prob_" + str(final_state): "exp({})".format(value_formula)
                })

        print(initial_state_actions)
        df = pd.DataFrame({'age': range(65, 120)})
        for action in initial_state_actions:
            assert len(action) == 1
            df[action.keys()[0]] = df.eval(action.values()[0])
        df.set_index('age', inplace = True)
        df = df.div(df.sum(axis = 1), axis = 0)  # Normalization
        assert (abs(df.sum(axis=1) - 1) < .000001).all()
        proba_by_initial_state[initial_state] = df
    return proba_by_initial_state

proba_by_initial_state = create_transition()