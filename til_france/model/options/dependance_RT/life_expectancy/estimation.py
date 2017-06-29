# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import pkg_resources


from til_france.targets.population import build_mortality_rates

from til_france.model.options.dependance_RT.life_expectancy.transition_matrices import (
    build_tansition_matrix_from_proba_by_initial_state
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


def create_transition_matrix(cohort = 'paquid', sexe = None):
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
                "proba_" + str(final_state): "exp({})".format(value_formula)
                })

        df = pd.DataFrame({'age': range(65, 120)})
        for action in initial_state_actions:
            assert len(action) == 1
            df[action.keys()[0]] = df.eval(action.values()[0])
        df.set_index('age', inplace = True)
        df = df.div(df.sum(axis = 1), axis = 0)  # Normalization
        assert (abs(df.sum(axis=1) - 1) < .000001).all()
        proba_by_initial_state[initial_state] = df.reset_index()

    return build_tansition_matrix_from_proba_by_initial_state(proba_by_initial_state)




def get_transition_by_age(transition_matrix, age_range = range(65, 120)):
    transition_by_age = dict()
    for age in range(65, 120):
        transition = (transition_matrix
            .query('age == @age')
            .loc[age]
            .unstack('initial_state')
            .squeeze()
            .xs('probability', axis=1, drop_level=True)
            )
        # Introduce absorbant dead state
        transition[ 5] = 0
        transition.loc[5, 5] = 1
        transition.fillna(0, inplace = True)
        transition_by_age[age] = transition.copy()
    return transition_by_age


def get_population_2_year(transition_by_age):
    initial_age = min(transition_by_age.keys())
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
    initial_age = min(transition_by_age.keys())
    initial_population = np.array([1, 0, 0, 0, 0, 0])
    assert initial_population.sum() == 1
    population = pd.DataFrame({initial_age: initial_population})
    for age in range(initial_age + 1, 120 + 1):
        transition = transition_by_age[age - 1]
        population[age] = transition.dot(population[age - 1]).values
        assert abs(population[age].sum() - 1.0) < 1e-10, 'age = {}: sum = {}'.format(age, population[age].sum())
    return population


def diagnostic(cohort = None , sexe = None, initial_population = np.array([1, 0, 0, 0, 0, 0])):
    transition_matrix = create_transition_matrix(cohort = cohort, sexe = sexe)
    transition_by_age = get_transition_by_age(transition_matrix)
    initial_population
    assert initial_population.sum() == 1
    assert (transition_by_age[65].dot(initial_population).sum() - 1) < 1e-10
    population = get_population_1_year(transition_by_age)

    alive = population.loc[0:4].sum()
    life_expectancy = 64 + alive.sum()
    print('life_expectancy at 65: {}'.format(life_expectancy))
    remaining_years = alive.sum()
    autonomous_years = population.loc[0].sum()
    dependant_years = population.loc[1:4].sum().sum()

    print('{} = {} + {}'.format(
        remaining_years, autonomous_years, dependant_years
        ))

    mortalite = build_mortality_rates()['female'][2007]
    mortalite[65] = 0

    # survival
    alive2 = (1 - mortalite[65:]).cumprod()
    pd.concat([alive, alive2], axis = 1).plot()

    # mortality
    mortality = (alive.shift() - alive) / alive.shift()
    mortality[65] = 0
    pd.concat([mortalite[66:], mortality], axis = 1).plot()


if __name__ == "__main__":
    diagnostic(cohort = 'paquid')
    # TODO check origin of year/age
    # male/female
    # cohort and data

