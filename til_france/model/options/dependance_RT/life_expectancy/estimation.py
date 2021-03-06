# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import pkg_resources


from til_france.model.options.dependance_RT.life_expectancy.transition_matrices import (
    build_tansition_matrix_from_proba_by_initial_state
    )

from til_france.model.options.dependance_RT.life_expectancy.calibration import (
    get_historical_mortality,
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


def get_transitions_from_cohort(cohort = 'paquid', sex = None):
    assert cohort in ['paquid', '3c']
    assert (sex in ['male', 'female)']) or (sex is None)
    if sex is None:
        template = 'etat_initial_{}_corr.xlsx'
        sex = 'all'
    elif sex is not None:
        if sex == 'male':
            sexe = 'homme'
        elif sex == 'female':
            sexe = 'femme'
        template = 'etat_initial_{}_corr' + '_{sexe}.xlsx'.format(sexe = sexe)

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

    return build_tansition_matrix_from_proba_by_initial_state(proba_by_initial_state, sex = sex)


def get_transition_by_age(transition_matrix = None, age_range = range(65, 120), sex = None):
    assert transition_matrix is not None
    transition_by_age = dict()
    if sex is None:
        sex = 'all'
    assert sex in ['all', 'female', 'male']
    for age in range(65, 120):
        transition = (transition_matrix
            .query('(age == @age) & (sex == @sex)')
            .loc[sex, age]
            .unstack('initial_state')
            .squeeze()
            .xs('probability', axis=1, drop_level=True)
            )
        # Introduce absorbant dead state
        transition[5] = 0
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


def diagnostic(cohort = None, sex = None, initial_population = np.array([1, 0, 0, 0, 0, 0]),
        mortality_year = 2007, upper_age_limit = 100):
    transition_matrix = get_transitions_from_cohort(cohort = cohort, sex = sex)
    transition_by_age = get_transition_by_age(transition_matrix, sex = sex)
    initial_population
    assert initial_population.sum() == 1
    assert (transition_by_age[65].dot(initial_population).sum() - 1) < 1e-10
    population = get_population_1_year(transition_by_age)

    alive = population.loc[0:4].sum()
    mortality = (alive.shift() - alive) / alive.shift()
    mortality[65] = 0

    life_expectancy = 64 + alive.sum()
    print('life_expectancy at 65: {}'.format(life_expectancy))
    remaining_years = alive.sum()
    autonomous_years = population.loc[0].sum()
    dependant_years = population.loc[1:4].sum().sum()

    print('{} = {} + {}'.format(
        remaining_years, autonomous_years, dependant_years
        ))

    alive.index.name = 'age'
    mortality.index.name = 'age'
    if sex is None:
        sex = 'all'
    alive.name = 'survival_rate_{}_{}'.format(sex, cohort)
    mortality.name = 'mortality_{}_{}'.format(sex, cohort)

    survivors = [alive]
    mortalities = [mortality]

    if sex is None:
        sexes = ['male', 'female']
    else:
        sexes = [sex]

    for sexe in sexes:
        reference_mortality = (get_historical_mortality()
            .query("(annee == @mortality_year) and (sex == @sexe)")[['mortalite', 'age']]
            .set_index('age')
            .squeeze()
            )
        reference_mortality[65] = 0
        reference_mortality.name = 'mortality_{}_{}'.format(sexe, mortality_year)
        alive_reference = (1 - reference_mortality[65:]).cumprod()
        alive_reference.name = 'survival_rate_{}_{}'.format(sexe, mortality_year)

        survivors.append(alive_reference)
        mortalities.append(reference_mortality[65:].copy())

    xlim = [65, upper_age_limit]
    ylim = [0, max([mortality_.loc[65:upper_age_limit].max() for mortality_ in mortalities])]
    # survival
    pd.concat(survivors, axis = 1).plot(title = "Survival rate", xlim = xlim)
    # mortality
    pd.concat(mortalities, axis = 1).plot(title = "Mortality", xlim = xlim, ylim = ylim)


if __name__ == "__main__":
    diagnostic(cohort = 'paquid')
    # TODO check origin of year/age
    # male/female
    # cohort and data
