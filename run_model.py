# -*- coding:utf-8 -*-

path_til = 'C:\\til\\'
path_pension = 'C:\\Til-Pension\\'
path_liam = 'C:\\liam2\\'

import sys
sys.path.append(path_liam)
from src.simulation import Simulation

path_model = 'C:\\Til-BaseModel\\console.yml'

simulation = Simulation.from_yaml(path_model,
                    input_dir = None,
                    input_file = None,
                    output_dir = path_til + 'output',                    
                    output_file = None)
simulation.run(False)
