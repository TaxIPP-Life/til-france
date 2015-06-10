# -*- coding:utf-8 -*-

import numpy as np
import os
import pkg_resources



from til_base_model.config import Config
from til.simulation import TilSimulation


path_model = os.path.join(
    pkg_resources.get_distribution('Til-BaseModel').location,
    'til_base_model',
    )


def test():
    config = Config()
    output_dir = config.get('til', 'output_dir')
    # output_dir = os.path.join(os.path.dirname(__file__), 'output'),
    console_file = os.path.join(path_model, 'console.yml')
    simulation = TilSimulation.from_yaml(
        console_file,
        input_dir = None,
        input_file = 'Patrimoine_next_200.h5',
        output_dir = output_dir,
        output_file = 'simul_short_erase.h5',
        tax_benefit_system = 'tax_benefit_system',
        )
    simulation.run(False)

    # import cProfile
    # command = """simulation.run(False)"""
    # cProfile.runctx( command, globals(), locals(), filename="OpenGLContext.profile1")



if __name__ == '__main__':
    test()
