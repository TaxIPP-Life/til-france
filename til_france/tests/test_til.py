# -*- coding:utf-8 -*-

import os
import shutil

import seaborn as sns
sns.set_style("whitegrid")


from til_france.tests.base import create_til_simulation, plot_csv
from til_france.plot.dependance import (
    plot_dependance_csv, plot_dependance_gir_csv, plot_dependance_prevalence_by_age,
    plot_dependance_incidence_by_age, plot_dependance_mortalite_by_age, plot_dependance_by_age,
    plot_dependance_by_age_separate, multi_extract_dependance_csv,
    plot_multi_prevalence_csv, plot_multi_dependance_csv
    )
from til_france.plot.population import (
    plot_population, population_diagnostic, plot_ratio_demographique,
    )


#for option in ['dependance', 'dependance_aligned', 'dependance_pessimistic', 'dependance_medium']:
#    simulation = create_til_simulation(
#        input_name = 'patrimoine',
#        option = option,
#        output_name_suffix = 'test_institutions',
#        uniform_weight = 200,
#        )
#    simulation.run_multiple(seeds = 30)
#    simulation.backup(option, erase = True)

# simulation.run(log = True)

# simulation.load_backup('aligned_reference')

# plot_population2(simulation)
# plot_csv(simulation)
#
# population_diagnostic_data_frame = population_diagnostic(simulation)
#

# plot_ratio_demographique(simulation)
option = 'dependance'
simulation = create_til_simulation(
    input_name = 'patrimoine',
    option = option,
    output_name_suffix = 'test_institutions',
    uniform_weight = 200,
    )

#plot_population(simulation, backup = option)
#
#bim
# plot_dependance_csv(simulation, backup = option)
#options = ['dependance', 'dependance_aligned']
#for option in options:
#    plot_dependance_by_age_separate(
#        simulation, backup = option, years = [2010, 2025, 2040], save = True, age_max = 95)
#
#
#plot_dependance_gir_csv(simulation)
#
#plot_dependance_by_age(simulation, years = [2010, 2020, 2030], save = True, age_max = 100)
#plot_dependance_prevalence_by_age(simulation, years = [2010, 2025, 2040], age_max = 100)
#
#plot_dependance_by_age_separate(simulation, years = [2010, 2025, 2040], save = True, age_max = 95)

# avec alignement 403
# sans alignement



options = ['dependance', 'dependance_aligned', 'dependance_pessimistic', 'dependance_medium']

age_max = 95
plot_multi_dependance_csv(simulation, options = options, save_figure = True)

bdadou
#plot_multi_prevalence_csv(
#    simulation, options = options, save_figure = True, years = years, age_max = age_max)



def export_to_slides():
    options = ['dependance', 'dependance_aligned', 'dependance_pessimistic', 'dependance_medium']
    graphs = ['prevalence', 'incidence', 'mortalite']
    destination_dir = "/home/benjello/Graphiques".decode('utf-8')
    for graph in graphs:
        for option in options:
            source_path = '/home/benjello/data/til/output/{}/figures/{}.pdf'.format(
                option, graph)
            destination_path = os.path.join(destination_dir, "{}_{}.pdf".format(
                graph, option[11:]
                ))
            shutil.copyfile(source_path, destination_path)

export_to_slides()