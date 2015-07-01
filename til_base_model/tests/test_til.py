# -*- coding:utf-8 -*-

from til_base_model.tests.base import create_til_simulation, plot_population2, population_diagnostic, plot_csv

simulation = create_til_simulation(capitalized_name = 'Patrimoine_next_metro_200', uniform_weight = 200)
simulation.run()

plot_population2(simulation)
plot_csv(simulation)


population_diagnostic_data_frame = population_diagnostic(simulation)
