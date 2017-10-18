#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division


import logging
import os
import pandas as pd
import patsy
import pkg_resources
import statsmodels.formula.api as smf
import sys


log = logging.getLogger(__name__)


# Paths

til_france_path = os.path.join(
    pkg_resources.get_distribution('Til-France').location,
    'til_france',
    )

assets_path = config_files_directory = os.path.join(
    til_france_path,
    'model',
    'options',
    'dependance_RT',
    'assets',
    )


paquid_path = u'/home/benjello/data/dependance/paquid_panel_3.csv'
paquid_dta_path = u'/home/benjello/data/dependance/paquid_panel_3_utf8.dta'


# Transition matrix structure
final_states_by_initial_state = {
    0: [0, 1, 4, 5],
    1: [0, 1, 2, 4, 5],
    2: [1, 2, 3, 4, 5],
    3: [2, 3, 4, 5],
    4: [4, 5],
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
    4: {
        0: 4,
        1: 4,
        2: 4,
        3: 4,
        },
    }


def get_clean_paquid():
    """
    Get PAQUID relevant data free of missing observations
    """
    df = pd.read_stata(paquid_dta_path)
    df = df[['numero', 'annee', 'age', 'scale5', 'sexe']].copy()
    assert df[['numero', 'annee', 'age', 'sexe']].notnull().all().all(), df.notnull().all()
    filtered = (df
        .dropna()
        .rename(columns = {'scale5': 'initial_state'})
        )
    log.info("There are {} valid observations out of {}".format(
        len(filtered), len(df)))

    assert (filtered.isnull().sum() == 0).all()

    filtered["sexe"] = filtered["sexe"].astype('int').astype('category')
    filtered["initial_state"] = filtered["initial_state"].astype('int').astype('category')
    return filtered


def build_estimation_sample(initial_state, sex = None):
    """Build estimation sample from paquid data
    """
    final_states = final_states_by_initial_state[initial_state]
    assert (sex in ['male', 'female']) or (sex is None)
    clean_paquid = get_clean_paquid()
    assert clean_paquid.notnull().all().all()
    assert initial_state in final_states
    no_transition = (clean_paquid
        .groupby('numero')['initial_state']
        .count() == 1
        )
    no_transition_with_specific_initial_state = clean_paquid.loc[
        clean_paquid.numero.isin(no_transition.index[no_transition].tolist())
        ].query('initial_state == {}'.format(initial_state))
    log.info("There are {} individuals out of {} with only one observation (and thus no transition) with intiial state = {}".format(
        len(no_transition_with_specific_initial_state), initial_state, no_transition.sum()))
    clean_paquid['final_state'] = clean_paquid.groupby('numero')['initial_state'].shift(-1).copy()
    log.info("There are {} individuals with intiial state = {} with no subsequent transition".format(
        len(
            clean_paquid.loc[clean_paquid.final_state.isnull()]
            .query('(initial_state == {})'.format(initial_state))
            ),
        initial_state,
        ))
    log.debug(clean_paquid
        .query('(initial_state == {})'.format(initial_state))
        .count()
        )
    log.debug('\n' + str(clean_paquid
        .query('(initial_state == {})'.format(initial_state))
        ['final_state'].value_counts(dropna = False).sort_index()
        ))
    wrong_transition = (clean_paquid.loc[clean_paquid.final_state.notnull()]
        .query('(initial_state == {}) & (final_state not in {})'.format(
            initial_state, final_states)
            )
        .groupby('final_state')['initial_state']
        .count()
        )
    log.info("There are {} individuals with intiial state = {} transiting to forbidden states: \n {}".format(
        wrong_transition.sum(), initial_state, wrong_transition[wrong_transition > 0]))

    log.info("Using the following replacement rule: {}".format(
        replace_by_initial_state[initial_state]
        ))
    log.debug("Sample size before cleaning bad final_states {}".format(
        len(clean_paquid
            .query('(initial_state == {})'.format(
                initial_state,
                final_states,
                ))
            .dropna()
            )
        ))
    sample = (clean_paquid
        .query('initial_state == {}'.format(initial_state))
        .replace({
            'final_state': replace_by_initial_state[initial_state]
            })
        .dropna()
        .copy()
        )
    log.debug("Sample size after cleaning bad final_states {}".format(len(sample)))

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
    assert set(sample.final_state.value_counts().index.tolist()) == set(final_states)
    return sample.reset_index(drop = True).copy()


def build_tansition_matrix_from_proba_by_initial_state(proba_by_initial_state, sex = None):
    assert sex in ['male', 'female', 'all']
    transition_matrices = list()
    for initial_state, proba_dataframe in proba_by_initial_state.iteritems():
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
    sample = build_estimation_sample(initial_state, sex = sex)
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
    for final_state, column in formatted_params.iteritems():
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


def compute_prediction(initial_state, formula = None, variables = ['age'], exog = None, sex = None):
    assert (sex in ['male', 'female']) or (sex is None)
    sample = build_estimation_sample(initial_state, sex = sex)
    if exog is None:
        exog = sample[variables]

    result = smf.mnlogit(
        formula = formula,
        data = sample,
        ).fit()
    expurged_formula = formula.split('~', 1)[-1]
    x = patsy.dmatrix(expurged_formula, data= exog)  # df is data for prediction
    prediction = result.predict(x, transform=False)
    (abs(prediction.sum(axis = 1) - 1) < .00001).all()
    prediction = pd.DataFrame(prediction)
    final_states = final_states_by_initial_state[initial_state]
    prediction.columns = ['proba_etat_{}'.format(state) for state in sorted(final_states)]
    return prediction.reset_index(drop = True)


def get_transitions_from_formula(formula = None):
    transitions = None
    for sex in ['male', 'female']:
        assert formula is not None
        proba_by_initial_state = dict()
        exog = pd.DataFrame(dict(age = range(65, 120)))
        for initial_state, final_states in final_states_by_initial_state.iteritems():
            proba_by_initial_state[initial_state] = pd.concat(
                [
                    exog,
                    compute_prediction(initial_state, formula, exog = exog, sex = sex)
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


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG, stream = sys.stdout)
    sex = None
    formula = 'final_state ~ I((age - 80)) + I(((age - 80))**2) + I(((age - 80))**3)'
    for initial_state in range(5):
        test(initial_state = initial_state, formula = formula, sex = sex)
