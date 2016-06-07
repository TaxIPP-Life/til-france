# -*- coding:utf-8 -*-

import seaborn as sns
sns.set_style("whitegrid")


from til_france.tests.base import create_til_simulation, plot_csv
from til_france.plot.dependance import (
    plot_dependance_csv, plot_dependance_gir_csv, plot_dependance_prevalence_by_age,
    plot_dependance_incidence_by_age, plot_dependance_mortalite_by_age, plot_dependance_by_age,
    plot_dependance_by_age_separate
    )
from til_france.plot.population import (
    plot_population2, population_diagnostic, plot_ratio_demographique,
    )


simulation = create_til_simulation(
    input_name = 'patrimoine',
    option = 'dependance',
    output_name_suffix = 'test_institutions',
    uniform_weight = 200,
    )

simulation.run(log = True)

# simulation.backup('unaligned_anciennete')
# simulation.load_backup('aligned_reference')

# plot_population2(simulation)
# plot_csv(simulation)
#
population_diagnostic_data_frame = population_diagnostic(simulation)
#
#plot_ratio_demographique(simulation)
plot_dependance_csv(simulation)
# plot_dependance_gir_csv(simulation)


# plot_dependance_by_age(simulation, years = [2010, 2020, 2030], save = True, age_max = 100)
plot_dependance_by_age_separate(simulation, years = [2010, 2025, 2040], save = True, age_max = 85)

# avec alignement 403
# sans alignement


