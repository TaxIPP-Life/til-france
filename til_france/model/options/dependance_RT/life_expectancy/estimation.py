# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import pkg_resources



moratliteH_path = os.path.join(
    pkg_resources.get_distribution('til-france').location,
    'til_france',
    'param',
    'demo',
    'hyp_mortaliteH.csv',
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
print proba_by_initial_state
bim


def get_transition(age, proba_by_initial_state):
    assert age in range(65, 120)
    probabilities = dict()
    for etat, dataframe in proba_by_initial_state.iteritems():
        extraction = dataframe.query('age == {}'.format(age)).reset_index(drop = True).squeeze()
        extraction.name = etat
        probabilities[etat] = extraction

    transition = pd.concat(probabilities, axis = 1)
    # Introduce absorbant dead state
    transition[5] = 0
    transition.loc['prob_etat_5', 5] = 1
    transition.fillna(0, inplace = True)
    rename_func = lambda x: x[-1:]
    transition.rename(index = rename_func, inplace = True)
    return transition


initial_age = 65

transition_by_age = dict()
for age in range(65, 120):
    transition_by_age[age] = get_transition(age = age, proba_by_initial_state = proba_by_initial_state)


transition = transition_by_age[65]
initial_population = np.array([1, 0, 0, 0, 0, 0])
assert initial_population.sum() == 1
assert (transition.dot(initial_population).sum() - 1) < 1e-10


def get_population_2_year(transition_by_age):
    initial_population = np.array([1, 0, 0, 0, 0, 0])
    assert initial_population.sum() == 1

    population_1 = pd.DataFrame({initial_age: initial_population}) / 2
    population_2 = pd.DataFrame({initial_age: initial_population}) / 2
    population = population_1 + population_2
    for age in range(initial_age + 1, 120 + 1):
        transition = transition_by_age[age - 1]

        if age % 2 == 0:
            population_1[age] = transition.dot(population_1[age - 1]).values
            population_2[age] = population_2[age - 1]
        else:
            population_2[age] = transition.dot(population_2[age - 1]).values
            population_1[age] = population_1[age - 1]

        population[age] = population_1[age] + population_2[age]
        assert abs(population[age].sum() - 1.0) < 1e-10, 'age = {}: sum = {}'.format(age, population[age].sum())
    return population


def get_population_1_year(transition_by_age):
    initial_population = np.array([1, 0, 0, 0, 0, 0])
    assert initial_population.sum() == 1
    population = pd.DataFrame({initial_age: initial_population})
    for age in range(initial_age + 1, 120 + 1):
        transition = transition_by_age[age - 1]
        population[age] = transition.dot(population[age - 1]).values
        assert abs(population[age].sum() - 1.0) < 1e-10, 'age = {}: sum = {}'.format(age, population[age].sum())
    return population


population = get_population_1_year(transition_by_age)
alive = population.loc[0:4].sum()
alive.sum()
autonomous = population.loc[0].sum()
autonomous.sum()
dependant = population.loc[1:4].sum()
dependant.sum()
dead = population.loc[5].sum()
dead.sum()

from til_france.targets.population import build_mortality_rates

mortalite = build_mortality_rates()['female'][2007]
mortalite[65] = 0
alive2 = (1-mortalite[65:]).cumprod()
pd.concat([alive, alive2], axis = 1).plot()


mortality = (alive.shift() - alive) / alive.shift()
mortality[65] = 0

pd.concat([mortalite[66:], mortality], axis = 1).plot()


