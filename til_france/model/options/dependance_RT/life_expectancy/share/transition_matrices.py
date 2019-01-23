#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division


import sys


import logging
import os
import pkg_resources
import statsmodels.formula.api as smf


import pandas as pd
import patsy

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


from til_core.config import Config

log = logging.getLogger(__name__)


# Paths

til_france_path = os.path.join(
    pkg_resources.get_distribution('Til-France').location,
    'til_france',
    )

assets_path = os.path.join(
    til_france_path,
    'model',
    'options',
    'dependance_RT',
    'assets',
    )


config = Config()
data_path = os.path.join(
    config.get('raw_data', 'share'),
    'share_data_for_microsimulation_newvar.csv', ##A modifier si on veut modifier la base transition
    )

assert os.path.exists(data_path)


# Transition matrix structure
final_states_by_initial_state = {
    0: [0, 1, 4],
    1: [0, 1, 2, 4],
    2: [1, 2, 3, 4],
    3: [2, 3, 4],
    }

replace_by_initial_state = {
    0: {
        2: 1,
        3: 1,
        },
    1: {
        3: 2
        },
    2: {
        0: 1,
        },
    3: {
        0: 2,
        1: 2,
        },
    }


def get_clean_share(extra_variables = None):
    """
        Get SHARE relevant data free of missing observations
    """
    death_state = 4
    if extra_variables is None:
        extra_variables = list()
    df = pd.read_csv(data_path)

    log.debug("Share data contains the following variables: {}".format(df.columns))

    renaming = {
        'scale': 'initial_state',
        'mergeid': 'id',
        'year_int': 'year',
        'age_int': 'age',
        }
    assert set(renaming.keys()) < set(df.columns)
    df = (df
        .rename(columns = renaming)
        .copy()
        )
    df['sexe'] = df.male + 2 * (df.male == 0)
    del df['male']
    variables = ['id', 'initial_state', 'sexe', 'year', 'age', 'vague']

    if extra_variables:
        assert isinstance(extra_variables, list)
        variables = list(set(['id', 'initial_state', 'sexe', 'year', 'age', 'vague']).union(set(extra_variables)))

    df = df[variables].copy().dropna()  # TODO Remove the dropna
    assert df.notnull().all().all(), \
        "Some columns contains NaNs:\n {}".format(df.isnull().sum())

    for extra_variable in extra_variables:
        assert df[extra_variable].notnull().all(), "Variable {} contains null values. {}".format(
            extra_variable, df[extra_variable].value_counts(dropna = False))

    assert df.sexe.isin([1, 2]).all()

    filtered = df.dropna()
    log.debug("There are {} missing observation of initial_state out of {}".format(df.initial_state.isnull().sum(), len(df)))

    log.debug("There are {} valid observations out of {} for the following variables {}".format(
        len(filtered), len(df), df.columns))

    assert (filtered.isnull().sum() == 0).all()

    filtered["sexe"] = filtered["sexe"].astype('int').astype('category')
    filtered["initial_state"] = filtered["initial_state"].astype('int').astype('category')

    return filtered


def build_estimation_sample(initial_state, sex = None, variables = None, vagues = None):
    """
        Build estimation sample from share data
    """
    final_states = final_states_by_initial_state[initial_state]
    assert (sex in ['male', 'female']) or (sex is None)
    extra_variables = None
    if variables is not None:
        extra_variables = [variable for variable in variables if variable not in ['final_state']]

    clean_share = (get_clean_share(extra_variables = extra_variables)
        .sort_values( by = ['id', 'vague'])
        )
    assert clean_share.notnull().all().all()
    assert initial_state in final_states
    no_transition = (clean_share
        .groupby('id')['initial_state']
        .count() == 1
        )
    no_transition_with_specific_initial_state = clean_share.loc[
        clean_share.id.isin(no_transition.index[no_transition].tolist())
        ].query('initial_state == {}'.format(initial_state))
    log.info(
        "There are {} individuals out of {} with only one observation (no transition) with intiial state = {}".format(
            len(no_transition_with_specific_initial_state), no_transition.sum(), initial_state))

    clean_share['final_state'] = clean_share.groupby('id')['initial_state'].shift(-1).copy()
    log.info("There are {} individuals with intiial state = {} with no subsequent transition".format(
        len(
            clean_share.loc[clean_share.final_state.isnull()]
            .query('(initial_state == {})'.format(initial_state))
            ),
        initial_state,
        ))
    log.debug("Transifions from initial_state = {}:\n {}".format(
        initial_state,
        (
            clean_share
            .query('(initial_state == {})'.format(initial_state))['final_state']
            .value_counts(dropna = False)
            .sort_index()
            )
        ))
    wrong_transition = (clean_share.loc[clean_share.final_state.notnull()]
        .query('(initial_state == {}) & (final_state not in {})'.format(
            initial_state, final_states)
            )
        .groupby('final_state')['initial_state']
        .count()
        )
    log.info("There are {} individuals with intiial state = {} transiting to forbidden states: \n {}".format(
        wrong_transition.sum(), initial_state, wrong_transition[wrong_transition > 0]))

    log.info("Final states:\n {}".format(clean_share.final_state.value_counts(dropna = False)))

    if wrong_transition.sum() > 0:
        if initial_state in replace_by_initial_state.keys():
            log.info("Using the following replacement rule: {}".format(
                replace_by_initial_state[initial_state]
                ))
            log.debug("Sample size before cleaning bad final_states:\n {}".format(
                len(clean_share
                    .query('(initial_state == {})'.format(
                        initial_state,
                        ))
                    .dropna()
                    )
                ))
            log.info(clean_share.query('(initial_state == {})'.format(
                initial_state,
                )).final_state.value_counts(dropna = False))
            sample = (clean_share
                .query('initial_state == {}'.format(initial_state))
                .dropna()
                .replace({
                    'final_state': replace_by_initial_state[initial_state]
                    })
                .copy()
                )
            log.debug("Sample size after cleaning bad final_states:\n {}".format(len(sample)))
            log.info(sample.final_state.value_counts())
    else:
        sample = (clean_share
            .query('(initial_state == {})'.format(
                initial_state,
                ))
            .dropna()
            .copy()
            )

    log.debug("Sample size after eventually cleaning bad final_states {}".format(len(sample)))

    if sex:
        if sex == 'male':
            sample = sample.query('sexe == 1').copy()
        elif sex == 'female':
            sample = sample.query('sexe == 2').copy()
    sample["final_state"] = sample["final_state"].astype('int').astype('category')

    if sex:
        log.info("Keeping sample of size {} for sex = {}".format(len(sample), sex))
    else:
        log.info("Keeping sample of size {}".format(len(sample)))
        del sample['sexe']

    if vagues:
        sample = sample.query('vague in @vagues').copy()

    assert set(sample.final_state.value_counts().index.tolist()) == set(final_states), 'Final states in data are {} and differs from {}'.format(
        set(sample.final_state.value_counts().index.tolist()), set(final_states)
        )
    return sample.reset_index(drop = True).copy()


def build_tansition_matrix_from_proba_by_initial_state(proba_by_initial_state, sex = None):
    assert sex in ['male', 'female', 'all']
    transition_matrices = list()
    for initial_state, proba_dataframe in proba_by_initial_state.items():
        transition_matrices.append(
            pd.melt(
                proba_dataframe,
                id_vars = ['age'],
                var_name = 'final_state',
                value_name = 'probability',
                )
            .replace({'final_state': dict([('proba_etat_{}'.format(index), index) for index in range(6)])})
            .assign(initial_state = initial_state)
            .assign(sex = sex)
            [['sex', 'age', 'initial_state', 'final_state', 'probability']]
            )
    return pd.concat(
        transition_matrices,
        ignore_index = True
        ).set_index(
            ['sex', 'age', 'initial_state', 'final_state']
            )


def estimate_model(initial_state, formula, sex = None, variables = ['age', 'final_state']):
    assert (sex in ['male', 'female']) or (sex is None)
    final_states = final_states_by_initial_state[initial_state]
    sample = build_estimation_sample(initial_state, sex = sex, variables = variables)
    result = smf.mnlogit(
        formula = formula,
        data = sample[variables],
        ).fit()
    log.debug(result.summary())
    formatted_params = result.params.copy()
    formatted_params.columns = sorted(set(final_states))[1:]

    def rename_index_func(index):
        index = index.lower()
        if index.startswith('i('):
            index = index[1:]
        elif index.startswith('intercept'):
            index = '(age > 0)'  # Hack to deal with https://github.com/pandas-dev/pandas/issues/16363
        return index

    formatted_params.rename(index = rename_index_func, inplace = True)
    formatted_params[sorted(set(final_states))[0]] = 0

    return result, formatted_params


def direct_compute_predicition(initial_state, formula, formatted_params, sex = None):
    assert (sex in ['male', 'female']) or (sex is None)
    computed_prediction = build_estimation_sample(initial_state, sex = sex)
    for final_state, column in formatted_params.items():
        proba_str = "exp({})".format(
            " + ".join([index + " * " + str(value) for index, value in zip(column.index, column.values)])
            )
        computed_prediction['proba_etat_{}'.format(final_state)] = computed_prediction.eval(proba_str)

    computed_prediction['z'] = computed_prediction[[
        col for col in computed_prediction.columns if col.startswith('proba')
        ]].sum(axis = 1)

    for col in computed_prediction.columns:
        if col.startswith('proba'):
            computed_prediction[col] = computed_prediction[col] / computed_prediction['z']

    return computed_prediction


def compute_prediction(initial_state = None, formula = None, variables = ['age'], exog = None, sex = None, vagues = None):
    """
        Compute prediction on exogneous if given or on sample
    """
    assert initial_state is not None
    assert (sex in ['male', 'female']) or (sex is None)
    sample = build_estimation_sample(initial_state, sex = sex, variables = variables, vagues = vagues)
    if exog is None:
        exog = sample[variables]

    result = smf.mnlogit(
        formula = formula,
        data = sample,
        ).fit()
    expurged_formula = formula.split('~', 1)[-1]
    x = patsy.dmatrix(expurged_formula, data= exog)  # exog is data for prediction
    prediction = result.predict(x, transform=False)
    (abs(prediction.sum(axis = 1) - 1) < .00001).all()
    prediction = pd.DataFrame(prediction)
    final_states = final_states_by_initial_state[initial_state]
    prediction.columns = ['proba_etat_{}'.format(state) for state in sorted(final_states)]
    return prediction.reset_index(drop = True)


def get_transitions_from_formula(formula = None, age_min = 50, age_max = 120, vagues = None):
    transitions = None
    for sex in ['male', 'female']:
        assert formula is not None
        proba_by_initial_state = dict()
        exog = pd.DataFrame(dict(age = range(age_min, age_max + 1)))
        for initial_state, final_states in final_states_by_initial_state.items():
            proba_by_initial_state[initial_state] = pd.concat(
                [
                    exog,
                    compute_prediction(initial_state, formula, sex = sex, exog = exog, vagues = vagues)
                    ],
                axis = 1,
                )
        transitions = pd.concat([
            transitions,
            build_tansition_matrix_from_proba_by_initial_state(proba_by_initial_state, sex = sex)
            ])

    transitions.reset_index().set_index(['sex', 'age', 'initial_state', 'final_state'])
    return transitions


def test(formula = None, initial_state = None, sex = None):
    assert formula is not None
    assert initial_state is not None
    assert (sex is None) or (sex in ['male', 'female'])
    result, formatted_params = estimate_model(initial_state, formula, sex = sex)
    computed_prediction = direct_compute_predicition(initial_state, formula, formatted_params, sex = sex)
    prediction = compute_prediction(initial_state, formula, sex = sex)
    diff = computed_prediction[prediction.columns] - prediction
    log.debug("Max of absolute error = {}".format(diff.abs().max().max()))
    assert (diff.abs().max() < 1e-5).all(), "error is too big: {} > 1e-5".format(diff.abs().max())


def get_formatted_params_by_initial_state(formula = None, variables = None):
    formatted_params_by_initial_state = dict([
        (
            initial_state,
            estimate_model(
                initial_state = initial_state, formula = formula, sex = None, variables = variables)[1]
            ) for initial_state in range(5)
        ])
    return formatted_params_by_initial_state


def get_transitions_from_file(alzheimer = None, memory = False):

    directory = os.path.join(
        config.get('raw_data', 'share'),
        '..',
        'Sorties',
        'Final'
        )
    if alzheimer is not None:
        assert alzheimer in [0, 1]
        filename = 'predict_alzheimer.csv'
    elif memory:
        filename = 'predict_specif2.csv'
    else:
        filename = 'predict_benchmark.csv'

    df = pd.read_csv(os.path.join(directory, filename), sep = ";", decimal = ',')
    if alzheimer is not None:
        assert set(df.alzheimer.unique()) == set([0, 1])
        df = df.query('alzheimer == @alzheimer').drop('alzheimer', axis = 1)

    age_max = df.age[df.age.str.isnumeric()].unique().max()
    df.replace(
        {'age': {age_max + '+': int(age_max) + 1}},
        inplace = True)
    df['age'] = df.age.astype(int)
    df = df.dropna()
    df['sex'] = 'male'
    # 'male, sexe == 1'
    # 'female, sexe == 2'
    df.loc[df.sexe == 2, 'sex'] = 'female'
    del df['sexe']
    df.rename(
        columns = {
            'etat_initial': 'initial_state',
            'etat_final': 'final_state',
            },
        inplace = True,
        )
    edge_age = int(age_max) + 1
    large_age_extension = pd.concat([
        df.query('age == @edge_age').assign(age = i).copy()
        for i in range(edge_age, 121)
        ])

    df = pd.concat([df, large_age_extension])
    df = df[['sex', 'age', 'initial_state', 'final_state', 'probability']].sort_values(
            ['sex', 'age', 'initial_state', 'final_state']
            )
    return df.set_index(['sex', 'age', 'initial_state', 'final_state'])
