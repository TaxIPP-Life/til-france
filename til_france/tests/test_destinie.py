# -*- coding:utf-8 -*-


from til_france.tests.base import create_til_simulation


simulation = create_til_simulation(capitalized_name = 'Destinie', uniform_weight = 1922.721)

simulation.run()

# population_diagnostic_data_frame = population_diagnostic(simulation)


# plot_population2(simulation)
# plot_csv(simulation)
