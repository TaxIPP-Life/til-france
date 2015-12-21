# -*- coding:utf-8 -*-

from til_france.tests.base import create_til_simulation, plot_csv
from til_france.plot.dependance import plot_dependance_csv
from til_france.plot.population import plot_population2, population_diagnostic, plot_ratio_demographique

simulation = create_til_simulation(
    input_name = 'patrimoine',
    option = 'dependance',
    output_name_suffix = 'test_institutions',
    uniform_weight = 200
    )
simulation.run()

plot_population2(simulation)
plot_csv(simulation)
#
population_diagnostic_data_frame = population_diagnostic(simulation)
plot_ratio_demographique(simulation)
plot_dependance_csv(simulation)
