# -*- coding:utf-8 -*-

import os
import pkg_resources


import pandas


from til_base_model.config import Config
from liam2.simulation import Simulation


path_model = os.path.join(
    pkg_resources.get_distribution('Til-BaseModel').location,
    'til_base_model',
    )

import openfisca_france_data
TaxBenefitSystem = openfisca_france_data.init_country()


def dump(simulation, file_path = None, overwrite = False):
    assert os.path.isabs(file_path)
    assert file_path.endswith('.h5')
    if not overwrite:
        assert not os.path.exists(file_path), "Cannot overwrite on already existing file {}".format(
            file_path)
    store = pandas.HDFStore(file_path)

    tax_benefit_system = simulation.tax_benefit_system
    for key_plural, entity_class in tax_benefit_system.entity_class_by_key_plural.iteritems():
        for variable in entity_class.column_by_name:
            try:
                simulation.calculate(variable)
            except:
                simulation.calculate_add(variable)
        data_frame = pandas.DataFrame(dict([
            (variable, simulation.get_or_new_holder(variable).get_array(period = simulation.period))
            for variable in entity_class.column_by_name
            ]))
        data_frame['id'] = data_frame.index.values.copy()
        data_frame['period'] = 200901
        data_frame.sort_index(axis = 1, inplace = True)
        store.put('entities/{}'.format(entity_class.key_singular), data_frame)

    store.close()


def test(regenerate_hdf_file = False, test_case = False):
    if regenerate_hdf_file:
        tax_benefit_system = TaxBenefitSystem()
        scenario = tax_benefit_system.new_scenario().init_single_entity(
            period = 2014,
            parent1 = dict(
                age = 25,
                salaire_imposable = 12000,
                ),
            )
        simulation = scenario.new_simulation()
        dump(
            simulation,
            file_path = os.path.join(path_model, 'toto.h5'),
            overwrite = True,
            )

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    console_file = os.path.join(path_model, 'test_case' if test_case else '', 'console.yml')
    simulation = Simulation.from_yaml(
        console_file,
        input_dir = None,
        input_file = 'toto.h5',
        output_dir = output_dir,
        output_file = None,
        )
    # simulation.run(False)
    return simulation

    # import cProfile
    # command = """simulation.run(False)"""
    # cProfile.runctx( command, globals(), locals(), filename="OpenGLContext.profile1")


def test_variables():
    patrimoine_store = pandas.HDFStore(os.path.join(path_model, 'Patrimoine_1500.h5'))
    openfisca_store = pandas.HDFStore(os.path.join(path_model, 'toto.h5'))

    entities = [('person', 'individu'), ('declar', 'foyer_fiscal'), ('menage', 'menage')]
    for patrimoine_entity, openfisca_entity in entities:
        print patrimoine_entity, openfisca_entity
        patrimoine_columns = patrimoine_store['/entities/{}'.format(patrimoine_entity)]
        openfisca_columns = openfisca_store['/entities/{}'.format(openfisca_entity)]
        print set(patrimoine_columns).difference(set(openfisca_columns))

if __name__ == '__main__':
    simulation = test()
