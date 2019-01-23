# -*- coding: utf-8 -*-


import logging


from til_france.model.options.dependance_RT.life_expectancy.share.transition_matrices import (
    build_estimation_sample,
    compute_prediction,
    direct_compute_predicition,
    estimate_model,
    get_clean_share,
    get_transitions_from_file,
    )


log = logging.getLogger(__name__)


def test_alzeihmer():
    df = get_transitions_from_file(alzheimer = 0)


def test_build_estimation_sample_share():
    df = get_clean_share()
    sex = 'male'
    initial_state = 1
    sample = build_estimation_sample(initial_state, sex = sex)


def test_estimation_share():
    sex = None
    formula = 'final_state ~ I((age - 80)) + I(((age - 80))**2) + I(((age - 80))**3)'

    for initial_state in range(3):
        test(initial_state = initial_state, formula = formula, sex = sex)


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


if __name__ == "__main__":
    test_alzeihmer()
    test_build_estimation_sample_share()
    test_estimation_share()
