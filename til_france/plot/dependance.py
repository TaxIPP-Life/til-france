# -*- coding:utf-8 -*-


import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import pandas as pd
from StringIO import StringIO


from til_france.tests.base import (
    create_or_get_figures_directory, ipp_colors, to_percent_round_formatter, get_data_directory
    )


log = logging.getLogger(__name__)


def extract_dependance_csv(simulation, backup = None):
    directory = get_data_directory(simulation, backup = backup)
    uniform_weight = simulation.uniform_weight

    if os.path.exists(os.path.join(directory, 'replication_0')):
        multiple = True
        base_directory = directory
    else:
        multiple = False

    def _extract_dependance_csv(directory):
        file_path = os.path.join(directory, 'dependance.csv')
        input_file = open(file_path)
        txt = input_file.readlines()[1:]
        txt = [line for line in txt if not line.startswith('dependance')]
        date = None
        data = None
        data_by_date = dict()
        data = []
        is_period_line = False
        for line in txt:
            if is_period_line:
                if date and data:
                    data_by_date.update({date: data})
                    data = []
                date = int(line.strip())
            elif line.startswith('period'):
                pass
            else:
                data.append(line.strip())
            is_period_line = True if line.startswith('period') else False

        proto_panel = None
        for period, data in data_by_date.iteritems():
            csv_string = StringIO('\n'.join(data_by_date[period]))
            df = pd.read_csv(csv_string)
            df.set_index(df.columns[0], inplace = True)
            df.index.name = 'age'
            df.columns = df.iloc[0]
            df.drop(df.index[0], axis = 0, inplace = True)
            df.columns = ['male', 'female', 'total']
            df.columns.name = "dependants"
            df['period'] = period
            proto_panel = pd.concat([proto_panel, df]) if proto_panel is not None else df

        panel = proto_panel.loc['total'].reset_index(drop = True)
        panel = panel.set_index(['period']).astype('int')

        return panel * uniform_weight

    if multiple:
        replication_number = 0
        result = list()
        while replication_number >= 0:
            directory = os.path.join(base_directory, 'replication_{}'.format(replication_number))
            if os.path.exists(directory):
                result.append(_extract_dependance_csv(directory))
                replication_number += 1

            else:
                return result

    else:
        return _extract_dependance_csv(directory)


def extract_dependance_gir_csv(simulation, backup = None):
    return _extract(
        simulation,
        filename = 'dependance_gir',
        removed_lines_prefixes = ['dependance', 'period', ',,'],
        columns = ['period', 'age', 'sexe', 'dependance_gir', 'total', 'total_global'],
        drop = ['total_global'],
        reweight = ['total'],
        backup = backup,
        )


def extract_dependance_niveau_csv(simulation, backup = None):
    return _extract(
        simulation,
        filename = 'dependance_niveau',
        removed_lines_prefixes = ['dependance', 'period', ',,'],
        columns = ['period', 'age', 'sexe', 'dependance_niveau', 'total', 'total_global'],
        drop = ['total_global'],
        reweight = ['total'],
        backup = backup,
        )


def extract_incidence_csv(simulation, backup = None):
    return _extract(
        simulation,
        filename = 'dependance_incidence',
        removed_lines_prefixes = ['incidence', 'period', ',,,', ',,total'],
        columns = ['period', 'age', 'sexe', 'False', 'dependant', 'total'],
        drop = ['False'],
        reweight = ['dependant', 'total'],
        backup = backup,
        )


def extract_deces_csv(simulation, backup = None):
    return _extract(
        simulation,
        filename = 'dependance_deces',
        removed_lines_prefixes = ['dependance_deces', 'period', ',,,', ',,total'],
        columns = ['period', 'age', 'sexe', 'False', 'deces', 'total'],
        drop = ['False'],
        reweight = ['deces', 'total'],
        backup = backup,
        )


def clean_dependance_csv(panel_simulation):
    def _clean_panel(panel):
        panel = panel / 1000
        panel.index.name = None
        panel.rename(
            columns = dict(
                male = "hommes",
                female = "femmes",
                ),
            inplace = True
            )
        return panel

    clean_panels = list()
    if isinstance(panel_simulation, list):
        for panel in panel_simulation:
            clean_panels.append(_clean_panel(panel))
    else:
        clean_panels.append(_clean_panel(panel_simulation))

    panel_simulation_concat = pd.concat(clean_panels)
    by_row_index = panel_simulation_concat.groupby(panel_simulation_concat.index)
    panel_simulation = by_row_index.mean()

    return panel_simulation


def plot_dependance_csv(simulation, backup = None, year_min = None, year_max = None):
    figures_directory = create_or_get_figures_directory(simulation, backup = backup)
    panel_simulation = extract_dependance_csv(simulation, backup = backup)
    panel_simulation = clean_dependance_csv(panel_simulation)

    if year_min:
        panel_simulation = panel_simulation.query('index >= @year_min')
    if year_max:
        panel_simulation = panel_simulation.query('index <= @year_max')

    ax = panel_simulation.plot(
        linewidth = 2,
        color = [ipp_colors[name] for name in ['ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']],
        xticks = [period for period in range(
            min(panel_simulation.index.astype(int)),
            max(panel_simulation.index.astype(int)),
            10
            )],
        )
    ax.set_ylabel("effectifs en milliers")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    fig = ax.get_figure()

    fig.savefig(os.path.join(figures_directory, 'dependance.pdf'), bbox_inches='tight')
    del ax, fig


def plot_dependance_gir_csv(simulation, backup = None):
    figures_directory = create_or_get_figures_directory(simulation, backup = backup)
    data = extract_dependance_gir_csv(simulation, backup = backup)

    def _clean_data(df):
        return (df
            .groupby(['period', 'dependance_gir'])['total'].sum()
            .unstack()
            .drop([0, -1], axis = 1) / 1000)

    clean_data = list()
    if isinstance(data, list):
        for df in data:
            clean_data.append(_clean_data(df))
    else:
        clean_data.append(_clean_data(df))

    data_concat = pd.concat(clean_data)
    by_row_index = data_concat.groupby(data_concat.index)
    data = by_row_index.mean()

    ax = data.plot(
        linewidth = 2,
        # color = [ipp_colors[name] for name in ['ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']],
        xticks = [period for period in range(
            min(data.index.astype(int)),
            max(data.index.astype(int)),
            10
            )],
        )
    ax.set_ylabel("effectifs en milliers")
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    fig = ax.get_figure()

    fig.savefig(os.path.join(figures_directory, 'dependance_gir.pdf'), bbox_inches='tight')
    del ax, fig


def clean_prevalence_csv(data):

    columns = data[0].columns if isinstance(data, list) else data.columns

    if 'dependance_gir' in columns:
        dependance_level_variable = 'dependance_gir'
        dependance_levels_number = 4
        log.debug("Cleaning girs")

    elif 'dependance_niveau' in columns:
        dependance_level_variable = 'dependance_niveau'
        dependance_levels_number = 4
        log.debug("Cleaning dependance_niveau")

    else:
        raise

    def _clean_data(df):
        df = (df
            .groupby([dependance_level_variable, 'period', 'age', 'sexe'])['total'].sum()
            .unstack([dependance_level_variable])
            .fillna(0)
            )
        df[0] = df[-1] + df[0]
        df.drop(-1, axis = 1, inplace = True)
        df['prevalence'] = sum(
            [df[i] for i in range(1, dependance_levels_number + 1)]
            ) / sum(
                [df[i] for i in range(dependance_levels_number + 1)]
                )
        df.drop(range(dependance_levels_number + 1), axis = 1, inplace = True)

        df = df.reset_index()
        df.age = df.age.astype(int)
        df.set_index(['period', 'age', 'sexe'], inplace = True)
        return df

    data = _apply_cleaning(_clean_data, data)
    return data


def clean_prevalence_all_levels_csv(data):

    columns = data[0].columns if isinstance(data, list) else data.columns

    if 'dependance_niveau' in columns:
        dependance_level_variable = 'dependance_niveau'
        log.debug("Cleaning dependance_niveau")
    else:
        raise

    def _clean_data(df):
        df = (df
            .groupby([dependance_level_variable, 'period', 'age', 'sexe'])['total'].sum()
            .unstack([dependance_level_variable])
            .fillna(0)
            )
        df[0] = df[-1] + df[0]
        df.drop(-1, axis = 1, inplace = True)
        prevalence_cols = df.columns.tolist()
        for column in df.columns:
            df['prevalence_{}'.format(column)] = df[column] / sum(
                [df[i] for i in prevalence_cols ]
                )

        df = df.reset_index()
        df.age = df.age.astype(int)
        df.set_index(['period', 'age', 'sexe'], inplace = True)

        return df[[col for col in df if str(col).startswith('prevalence')]].copy()

    data = _apply_cleaning(_clean_data, data)
    return data


def plot_dependance_prevalence_by_age(simulation, years = None, ax = None, age_max = None, age_min = None,
        backup = None):
    assert years is not None
    figures_directory = create_or_get_figures_directory(simulation, backup = backup)

    try:
        df = extract_dependance_gir_csv(simulation, backup = backup)
        log.debug("Extracting girs")
    except IOError:
        log.debug("Abrot extracting girs because irrelevant")
        df = extract_dependance_niveau_csv(simulation, backup = backup)
        log.debug("Extracting dependance_niveau")

    data = clean_prevalence_csv(df)

    ylabel = u"taux de prévalence"
    # ylabel = "prevalence rate"
    return _plot_and_or_save(ax = ax, data = data, figures_directory = figures_directory,
        name = 'prevalence', pdf_name = None, years = years, age_max = age_max, age_min = age_min, ylabel = ylabel)


def plot_dependance_prevalence_all_levels_by_age(simulation, years = None, ax = None, age_max = None, age_min = None,
        backup = None):
    figures_directory = create_or_get_figures_directory(simulation, backup = backup)

    try:
        df = extract_dependance_gir_csv(simulation, backup = backup)
        log.debug("Extracting girs")
    except IOError:
        log.debug("Abort extracting girs because irrelevant")
        df = extract_dependance_niveau_csv(simulation, backup = backup)
        log.debug("Extracting dependance_niveau")

    return df
    # data = clean_prevalence_all_levels_csv(df)
    # ylabel = u"taux de prévalence"  # ylabel = "prevalence rate"
    # return data
#    return _plot_and_or_save(ax = ax, data = data, figures_directory = figures_directory,
#         name = 'prevalence', pdf_name = None, years = years, age_max = age_max, age_min = age_min, ylabel = ylabel)


def plot_dependance_incidence_by_age(simulation, years = None, ax = None, age_max = None, age_min = None,
        backup = None):
    assert years is not None
    figures_directory = create_or_get_figures_directory(simulation, backup = backup)
    data = extract_incidence_csv(simulation, backup = backup)

    def _clean_data(df):
        return (df
            .set_index(['period', 'age', 'sexe'])
            .eval('incidence = dependant / total', inplace = False))

    data = _apply_cleaning(_clean_data, data)

    ylabel = "taux d'incidence"
    # ylabel = "incidence rate"
    return _plot_and_or_save(ax = ax, data = data, figures_directory = figures_directory,
        name = 'incidence', pdf_name = None, years = years, age_max = age_max, age_min = age_min, ylabel = ylabel)


def plot_dependance_mortalite_by_age(simulation, years = None, ax = None, age_max = None, age_min = None,
        backup = None):
    assert years is not None
    figures_directory = create_or_get_figures_directory(simulation, backup = backup)
    data = extract_deces_csv(simulation, backup = backup)

    def _clean_data(df):
        return (df
            .set_index(['period', 'age', 'sexe'])
            .eval('mortalite = deces / total', inplace = False))

    data = _apply_cleaning(_clean_data, data)
    data.fillna(0, inplace = True)
    print(data)
    ylabel = u"quotient de mortalité des personnes dépendantes"
    return _plot_and_or_save(ax = ax, data = data, figures_directory = figures_directory,
      name = 'mortalite', pdf_name = None, years = years, age_max = age_max, age_min = age_min, ylabel = ylabel)


def plot_dependance_by_age(simulation, years = None, age_max = None, age_min = None, save = True, backup = None):
    figures_directory = create_or_get_figures_directory(simulation, backup = backup)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    legend_handles, legend_labels = plot_dependance_prevalence_by_age(
        simulation, years = years, ax = ax1, age_max = age_max)
    plot_dependance_incidence_by_age(simulation, years = years, ax = ax2, age_max = age_max, age_min = age_min)
    plot_dependance_mortalite_by_age(simulation, years = years, ax = ax3, age_max = age_max, age_min = age_min)

    ax3.legend().set_visible(False)
    plt.draw()

    # fig.legend(tuple(legend_handles), tuple(legend_labels), loc='best')
    plt.legend(tuple(legend_handles), tuple(legend_labels), loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),
       bbox_transform = plt.gcf().transFigure, ncol = len(years) * 2)
    if save:
        fig.savefig(os.path.join(figures_directory, 'dependance_by_age.pdf'), bbox_inches='tight')


def plot_dependance_by_age_separate(simulation, years = None, age_max = None, age_min = None, save = True,
        backup = None):
    plot_dependance_prevalence_by_age(simulation, years = years, age_max = age_max, age_min = age_min, backup = backup)
    plot_dependance_incidence_by_age(simulation, years = years, age_max = age_max, age_min = age_min, backup = backup)
    plot_dependance_mortalite_by_age(
        simulation, years = years, age_max = age_max, age_min = age_min, backup = backup)


# Helpers


def _apply_cleaning(_clean_data, data):
    clean_data = list()
    if isinstance(data, list):
        for df in data:
            clean_data.append(_clean_data(df))
    else:
        df = data
        clean_data.append(_clean_data(df))

    data_concat = pd.concat(clean_data)
    by_row_index = data_concat.groupby(data_concat.index)
    data = by_row_index.mean()
    data.index = pd.MultiIndex.from_tuples(data.index, names = ['period', 'age', 'sexe'])
    data.reset_index(inplace = True)
    return data


def _extract(simulation, filename, removed_lines_prefixes, columns, drop = None, reweight = None, backup = None):
    directory = get_data_directory(simulation, backup = backup)
    uniform_weight = simulation.uniform_weight

    if os.path.exists(os.path.join(directory, 'replication_0')):
        multiple = True
        base_directory = directory
        del directory
    else:
        multiple = False

    def _single_extract(directory, filename):
        assert filename[-4:] != '.csv', 'filename should not contain extension'
        file_path = os.path.join(directory, '{}.csv'.format(filename))
        log.debug("Extracting data from file {}".format(file_path))
        input_file = open(file_path)
        txt = input_file.readlines()[1:]
        for prefix in removed_lines_prefixes:
            txt = [line for line in txt if not(line.startswith(prefix))]
        df = pd.read_csv(
            StringIO('\n'.join(txt)),
            header = None,
            names = columns,
            )
        if drop:
            assert set(drop) < set(columns)
            df.drop(drop, axis = 1, inplace = True)
        if 'period' in columns:
            df.period = df.period.astype(int)
        for variable in reweight:
            assert variable in df.columns
            df[variable] = df[variable] * uniform_weight
        return df

    if multiple:
        replication_number = 0
        result = list()
        while replication_number >= 0:
            directory = os.path.join(base_directory, 'replication_{}'.format(replication_number))
            if os.path.exists(directory):
                result.append(_single_extract(directory, filename))
                replication_number += 1

            else:
                return result

    else:
        return _single_extract(directory, filename)


def _plot_and_or_save(ax = None, data = None, figures_directory = None, name = None, pdf_name = None, years = None,
        age_max = None, age_min = None, ylabel = None, english = False):
    assert name is not None
    assert data is not None
    assert years is not None
    assert os.path.exists(figures_directory)

    data['age_group'] = np.trunc((data.age - 60) / 5)
    data.age_group = data.age_group.astype(int)

    if not age_min:
        age_min = 60
    if age_max:
        query = '(age >= @age_min) and (age <= @age_max) and (period in @years)'
    else:
        query = '(age >= @age_min) and (period in @years)'

    data_plot = data.query(query)[['age', 'period', name, 'sexe']]

    if ax is None:
        save_figure = True
        fig, ax = plt.subplots()
    else:
        save_figure = False

    colors = [ipp_colors[cname] for cname in [
        'ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]
    color_by_period = dict(zip(data_plot['period'].unique(), colors))
    legend_handles, legend_labels = list(), list()
    for grouped in data_plot.groupby(['period', 'sexe']):
        period, sexe = grouped[0]
        linestyle = '--' if sexe == 1 else "-"
        x = grouped[1]['age']
        y = grouped[1][name]
        line, = ax.plot(x, y, linestyle = linestyle, color = color_by_period[period], )
        legend_handles.append(line)
        if english:
            sex_label = 'Men' if sexe == 0 else 'Women'
        else:
            sex_label = 'Hommes' if sexe == 0 else 'Femmes'
        legend_labels.append('{} - {}'.format(sex_label, period))

        if english:
            ax.set_xlabel("age")
        else:
            ax.set_xlabel(u"âge")
        if ylabel is None:
            ax.set_ylabel(name)
        else:
            ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(to_percent_round_formatter)

    if save_figure:
        pdf_name = pdf_name if pdf_name else name
        ax.legend(legend_handles, legend_labels, loc = 'best')
        fig.savefig(os.path.join(figures_directory, '{}.pdf'.format(pdf_name)), bbox_inches='tight')
        del fig

    return legend_handles, legend_labels


def multi_extract_dependance_csv(simulation, backups = None):
    dataframes = list()
    for backup in backups:
        df = clean_dependance_csv(
            extract_dependance_csv(simulation, backup = backup)
            )
        df.reset_index(inplace = True)
        df['option'] = backup
        dataframes.append(df)

    return pd.concat(dataframes)


def plot_multi_dependance_csv(simulation, options, save_figure = True):
    figures_directory = create_or_get_figures_directory(simulation)
    df = multi_extract_dependance_csv(simulation, backups = options)
    df.index = df.index.astype(int)
    df2 = df.set_index(['index', 'option'])
    df2 = df2['total'].unstack()
    df2['dependance_aligned'] = df2.dependance_aligned.shift()

#    df2.rename(
#        columns = {
#            'dependance': 'optimistic ($\mu$ = 1)',
#            'dependance_aligned': 'aligned',
#            'dependance_pessimistic': 'pessimistic ($\mu$ = 0)',
#            'dependance_medium': 'median ($\mu$ = 0)'
#            },
#        inplace = True,
#        )

    df2.rename(
        columns = {
            'dependance': r'sc. optimiste ($\mu$ = 1)',
            'dependance_aligned': r"sc. align$\acute{e}$",
            'dependance_pessimistic': r'sc. pessimiste ($\mu$ = 0)',
            'dependance_medium': r"m$\acute{e}$dian ($\mu$ = .5)"
            },
        inplace = True,
        )


    # plt.rc('text', usetex=True)
    df2.columns.name = None
    colors = [ipp_colors[cname] for cname in [
        'ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]
    ax = df2.loc[df2.index >= 2010].plot(colors = colors)
    #    ax.set_xlabel(u"year")  # TODO french
    #    ax.set_ylabel("elderly disabled (thousands)")
#    ax.set_xlabel(u"année")
#    ax.set_ylabel(u"personnes dépendantes (milliers)")

    if save_figure:
        pdf_name = 'multi_dependance'
        ax.figure.savefig(os.path.join(figures_directory, '{}.pdf'.format(pdf_name)), bbox_inches='tight')


def multi_extract_prevalence_csv(simulation, backups = None):
    dataframes = list()
    for backup in backups:
        df = clean_prevalence_csv(
            extract_dependance_gir_csv(simulation, backup = backup)
            )
        df.reset_index(inplace = True)
        df['option'] = backup
        dataframes.append(df)
    return pd.concat(dataframes)


def plot_multi_prevalence_csv(simulation, options = None, save_figure = True, years = None, age_max = None,
                              pdf_name = None, ylabel = None, ):
    figures_directory = create_or_get_figures_directory(simulation)
    data = multi_extract_prevalence_csv(simulation, backups = options)
    name = 'prevalence'
    assert options is not None
    if age_max:
        query = '(age >= 60) and (age <= @age_max) and (period in @years) and (option in @options)'
    else:
        query = '(age >= 60) and (period in @years) and (option in @options)'
    data_plot = data.query(query)[['age', 'period', name, 'sexe', 'option']]

    save_figure = True
    fig, ax = plt.subplots()

    colors = [ipp_colors[cname] for cname in [
        'ipp_very_dark_blue', 'ipp_dark_blue', 'ipp_medium_blue', 'ipp_light_blue']]
    color_by_period = dict(zip(data_plot['period'].unique(), colors))
    color_by_option = dict(zip(data_plot['option'].unique(), colors))
    name_by_option = dict(dependance_aligned = 'aligned', dependance = 'optimistic')
    legend_handles, legend_labels = list(), list()
    for grouped in data_plot.groupby(['period', 'sexe', 'option']):
        period, sexe, option = grouped[0]
        linestyle = '--' if sexe == 1 else "-"
        x = grouped[1]['age']
        y = grouped[1][name]
        line, = ax.plot(x, y, linestyle = linestyle, color = color_by_option[option], )
        legend_handles.append(line)
        sex_label = 'Hommes' if sexe == 0 else 'Femmes'
        legend_labels.append('{} - {} - {}'.format(sex_label, period, name_by_option[option]))

        ax.set_xlabel(u"âge")
        if ylabel is None:
            ax.set_ylabel(name)
        else:
            ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(to_percent_round_formatter)

    if save_figure:
        pdf_name = pdf_name if pdf_name else name
        ax.legend(legend_handles, legend_labels, loc = 'best')
        fig.savefig(os.path.join(figures_directory, 'multi_{}.pdf'.format(pdf_name)), bbox_inches='tight')
        del fig

    return legend_handles, legend_labels