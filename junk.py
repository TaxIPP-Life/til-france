# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:23:07 2018

@author: a.rain
"""

from til_france.data.data.hsm_dependance_niveau import create_dependance_initialisation_share 

from til_france.model.options.dependance_RT.life_expectancy.calibration import (
    get_insee_projected_mortality,
    get_insee_projected_population,
    )

df = create_dependance_initialisation_share(
        filename_prefix = None, smooth = False, window = 7, std = 2,
        survey = 'hsm', age_min = 50, scale = 4)

print(df.loc[90:120,:])

df2 = create_dependance_initialisation_share(
        filename_prefix = None, smooth = True, window = 7, std = 2,
        survey = 'hsm', age_min = 50, scale = 4)

print(df2.loc[90:120,:])

#FONCTION QUI IMPORTE LES PREVALENCES DEPUIS CARE

def get_care_prevalence_pivot_table(sexe = None, scale = None):
    config = Config()
    assert scale in [4, 5], "scale should be equal to 4 or 5"
#    if scale == 5:
#        xls_path = os.path.join(config.get('raw_data', 'hsm_dependance_niveau'), 'desc_dependance_scale5.xls')
 #       data = (pd.read_excel(xls_path)
 #           .rename(columns = {'disability_scale5': 'dependance_niveau', 'woman': 'sexe'})
 #           )
#    elif scale == 4:
    xls_path = os.path.join(
            config.get('raw_data', 'hsm_dependance_niveau'), 'CARe_scalev1v2.xls')
    data = (pd.read_excel(xls_path)
            .rename(columns = {
                'scale_v1': 'dependance_niveau',
                'femme': 'sexe',
                })
            )

    assert sexe in ['homme', 'femme']
    sexe = 1 if sexe == 'femme' else 0
    assert sexe in data.sexe.unique(), "sexe should be in {}".format(data.sexe.unique().tolist())
    pivot_table = (data[['dependance_niveau', 'poids_hsm', 'age', 'sexe']]
        .query('sexe == @sexe')
        .groupby(['dependance_niveau', 'age'])['poids_hsm'].sum().reset_index()
        .pivot('age', 'dependance_niveau', 'poids_hsm')
        .replace(0, np.nan)  # Next three lines to remove all 0 columns
        .dropna(how = 'all', axis = 1)
        .replace(np.nan, 0)
        )

    return pivot_table

df = create_dependance_initialisation_share(
        filename_prefix = None, smooth = False, window = 7, std = 2,
        survey = 'hsm', age_min = 50, scale = 4)


def get_initial_population(age_min = None, period = None, rescale = False):
    assert age_min is not None
    if rescale:
        assert period is not None
    data_by_sex = dict()
    for sex in ['male', 'female']:
        sexe = 'homme' if sex == 'male' else 'femme'
        config = Config()
        filename = os.path.join(
            config.get('til', 'input_dir'),
            'dependance_initialisation_level_share_{}.csv'.format(sexe)
            )
        log.info('Loading initial population dependance states from {}'.format(filename))
        df = (pd.read_csv(filename, names = ['age', 0, 1, 2, 3], skiprows = 1)
            .query('(age >= @age_min)')
            )
        df['age'] = df['age'].astype('int')

        df = (
            pd.melt(
                df,
                id_vars = ['age'],
                value_vars = [0, 1, 2, 3],
                var_name = 'initial_state',
                value_name = 'population'
                )
            .sort_values(['age', 'initial_state'])
            .set_index(['age', 'initial_state'])
            )
        assert (df.query('initial_state == -1')['population'] == 0).all()
        data_by_sex[sex] = (df
            .assign(sex = sex)
            )
    data = pd.concat(data_by_sex.values()).reset_index()

    if rescale:
        insee_population = get_insee_projected_population()
        assert period in insee_population.reset_index().year.unique()
        rescaled_data = (
            data.groupby(['sex', 'age'])['population'].sum().reset_index()
            .merge(
                (insee_population.query("year == @period")
                    .reset_index()
                    .rename(columns = {"population": "insee_population"})
                    )
                    ,
                how = 'left',
                )
            .eval("calibration = insee_population / population", inplace = False)
            )

        data = (data
            .merge(rescaled_data[['sex', 'age', 'calibration']])
            .eval("population = calibration *population", inplace = False)
            )[['sex', 'age', 'initial_state', 'population']].fillna(0).copy()

    assert data.notnull().all().all(), data.notnull().all()

    return data

df = get_initial_population(age_min = 60, period = None, rescale = False)


def check_67_and_over(population, age_min):
    period = population.period.max()
    insee_population = get_insee_projected_population()
    log.info("period {}: insee = {} vs {} = til".format(
        period,
        insee_population.query('(age >= @age_min) and (year == @period)')['population'].sum(),
        population.query('(age >= @age_min) and (period == @period)')['population'].sum()
        ))