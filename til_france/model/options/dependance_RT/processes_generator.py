# -*- coding: utf-8 -*-


import os
import pandas as pd
import pkg_resources
import yaml


from til_france.model.options.dependance_RT.life_expectancy.transition import final_states_by_initial_state

til_france_path = os.path.join(
    pkg_resources.get_distribution('Til-France').location,
    'til_france',
    )

dependance_functions_yml_path = os.path.join(
    til_france_path,
    'model',
    'options',
    'dependance_RT',
    'dependance_initialisation_functions.yml'
    )

prevalence_coef_path = os.path.join(
    til_france_path,
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
    probabilities = [
        {'probabilite_0': "Phi({cut_1} - value)".format(**cuts)},
        {'probabilite_1': "Phi({cut_2} - value) - probabilite_0".format(**cuts)},
        {'probabilite_2': "Phi({cut_3} - value) - (probabilite_0 + probabilite_1)".format(**cuts)},
        {'probabilite_3': "Phi({cut_4} - value) - (probabilite_0 + probabilite_1 + probabilite_2)".format(**cuts)},
        {'probabilite_4': "1 - (probabilite_0 + probabilite_1 + probabilite_2 + probabilite_3)".format(**cuts)},
        ]
    dependance_expr = {
        "dependance_niveau":
            "choice([0, 1, 2, 3, 4], [probabilite_0, probabilite_1, probabilite_2, probabilite_3, probabilite_4])"
        }
    return [value_expr] + probabilities + [dependance_expr, 'return dependance_niveau']


def rename_variables(variables):
    # PAQUID +65 ans
    # gen age_70 = 0
    # replace age_70 = 1 if age < 70
    # gen age_80 = 0
    # replace age_80 = 1 if age>= 70 & age < 80
    # gen age_90 = 0
    # replace age_90 = 1 if age>= 80 & age < 90
    # gen age_more90 = 0
    # replace age_more90 = 1 if age >=90
    new_by_old_variable = {
        'lqm_2008': 'lq',
        'nb_children': 'nb_enfants',
        'partner': 'INCOUPLE',
        'child_0': '(nb_enfants <= 0)',
        'child_3_more': '(nb_enfants >= 3)',
        'educ_1': '(education_niveau == 1)',
        'educ_2': '((education_niveau >= 2) and (education_niveau <= 3))',
        'educ_3': '(education_niveau >= 4)',
        'age_80': '((age >= 70) and (age < 80))',
        'age_90': '((age >= 80) and (age < 90))',
        'age_more90': '(age >= 90)',
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
    process_by_initial_state = dict()
    processes = dict(processes = process_by_initial_state)
    individus = dict(individus = processes)
    process_by_initial_state["Phi(x)"] = [
        {"x_adjusted": "x / 1.41421356237"},
        "return (1.0 + 0.5 * erf(x_adjusted))",
        ]
    main = dict(entities = individus)
    for function_name, sheetname in sheetnames_by_function.iteritems():
        parameters_value_by_name = pd.read_excel(prevalence_coef_path, sheetname = sheetname).transpose().to_dict()[0]
        cuts, variables = separate_cuts_from_variables(parameters_value_by_name)
        # vars += variables.keys()
        variables = rename_variables(variables)
        # renamed_vars += variables.keys()
        process_by_initial_state[function_name + '(lq, nb_enfants)'] = build_function_str(
            function_name, cuts, variables)

    with open(dependance_functions_yml_path, 'w') as outfile:
        yaml.dump(main, outfile, default_flow_style = False, width = 1000)


def create_transition_functions(cohort = None):
    assert cohort in ['paquid', '3c']
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
                    cohort,
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

        prob_final_states = sorted(["prob_" + final_state for final_state in variables_by_final_state.keys()])
        probabilities_list_str = "[" + ", ".join(prob_final_states[:-1])
        probabilities_list_str = probabilities_list_str + ", 1 - (" + '+'.join(prob_final_states[:-1]) + ")]"
        final_states_index_list_str = [
            int(final_state[-1:]) for final_state in prob_final_states
            ]
        initial_state_actions.append(
            "return choice({}, {})".format(
                final_states_index_list_str,
                probabilities_list_str
                )
            )

    dependance_transition_yml_path = os.path.join(
        pkg_resources.get_distribution('til-france').location,
        'til_france',
        'model',
        'options',
        'dependance_RT',
        'dependance_{}_transition_functions.yml'.format(cohort)
        )

    with open(dependance_transition_yml_path, 'w') as outfile:
        yaml.dump(main, outfile, default_flow_style = False, width = 1000)


def create_scaled_transition_functions(cohort = 'paquid'):
    dependance_RT_path = os.path.join(
        pkg_resources.get_distribution('til-france').location,
        'til_france',
        'model',
        'options',
        'dependance_RT',
        )

    # individus entry base structure
    dependance_transition_yml_path = os.path.join(
        dependance_RT_path,
        'dependance_scaled_{}_transition_functions.yml'.format(cohort)
        )
    process_by_initial_state = dict()
    processes = dict(processes = process_by_initial_state)
    individus = dict(individus = processes)
    main = dict(entities = individus)

    # generation entry base structure
    dependance_generation_transition_yml_path = os.path.join(
        dependance_RT_path,
        'dependance_generation_transition_calibration_variables.yml'.format(cohort)
        )
    generation_fields = list()
    mise_a_jour = ['education()', 'mortality_rates_update()', 'dependance_transition_mise_a_jour()']
    options_initialisation = ['mortality_rates_initialisation()', 'dependance_transition_mise_a_jour()']
    dependance_transition_mise_a_jour = list()
    generation_processes = dict(
        mise_a_jour = mise_a_jour,
        options_initialisation = options_initialisation,
        dependance_transition_mise_a_jour = dependance_transition_mise_a_jour,
        )
    generations = dict(generation = dict(fields = generation_fields, processes = generation_processes))
    generations_main = dict(entities = generations)

    for initial_state, final_states in final_states_by_initial_state.iteritems():
        initial_state_actions = list()
        process_by_initial_state['etat_{}()'.format(initial_state)] = initial_state_actions
        # probability computation lines
        for final_state in final_states:
            male_link = 'dependance_transition_homme_{}_{}'.format(initial_state, final_state)
            female_link = 'dependance_transition_femme_{}_{}'.format(initial_state, final_state)
            initial_state_actions.append({
                "prob_{}".format(final_state): "if(ISMALE, individu2generation.{}, individu2generation.{})".format(
                    male_link, female_link)
                })
            # generation_fields.append({
            #     male_link: dict(type = 'float', initialdata = 'False')
            #     })
            # generation_fields.append({
            #     str(female_link): dict(type = 'float', initialdata = 'False')
            #     })
            generation_fields.append({
                male_link: '{type: float, initialdata: False}'
                })
            generation_fields.append({
                female_link: '{type: float, initialdata: False}'
                })
            dependance_transition_mise_a_jour.append({
                male_link: 'dependance_transition_homme[period - 2010, 0:121, {}, {}]'.format(
                    initial_state, final_state),
                # First argument of transition is period
                })
            dependance_transition_mise_a_jour.append({
                female_link: 'dependance_transition_femme[period - 2010, 0:121, {}, {}]'.format(
                    initial_state, final_state),
                })
        # return line for individus
        prob_final_states = sorted(["prob_" + str(final_state) for final_state in final_states])
        probabilities_list_str = "[" + ", ".join(prob_final_states[:-1])
        probabilities_list_str = probabilities_list_str + ", 1 - (" + ' + '.join(prob_final_states[:-1]) + ")]"
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

    with open(dependance_generation_transition_yml_path, 'w') as outfile:
        yaml.dump(generations_main, outfile, default_flow_style = False, width = 1000)

    # Read in the file
    with open(dependance_generation_transition_yml_path, 'r') as outfile:
        filedata = outfile.read()
    # Replace the target string
    filedata = filedata.replace("'{type: float, initialdata: False}'", '{type: float, initialdata: False}')
    # Write the file out again
    with open(dependance_generation_transition_yml_path, 'w') as outfile:
        outfile.write(filedata)


if __name__ == "__main__":
    # create_initialisation()
    # create_transition_functions(cohort = "paquid")
    # create_transition_functions(cohort = "3c")
    create_scaled_transition_functions(cohort = "paquid")
