#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys

from glob import glob
from shutil import copyfile
from subprocess import call

import openfisca_survey_manager
import til_core
import til_france

from til_france.data.data.Patrimoine import test_build as patrimoine
from til_france.data.data.handicap_sante_institutions import create_hsi_data as hsi
from openfisca_survey_manager.scripts import build_collection

from build_demographic_model_data import main as build_demographic_model_data_
from population_and_dependance_builder import main as build_parameters_

app_name = os.path.splitext(os.path.basename(__file__))[0]
log = logging.getLogger(app_name)


# utils
def read_config_example(option):
    """
    str -> str
    Reads the file containing pre-filled configuration for the option
    """

    filename = 'config_example_{}.txt'.format(option)
    fichier = os.path.join(
        os.path.dirname(til_france.__file__),
        'scripts',
        'til_init',
        filename
        )

    assert os.path.exists(fichier), \
        "Could not find {} config example file".format(option)

    with open(fichier) as fichier:
        text = fichier.read() + '\n'

    return(text)


def copy_init_file_to_config(option):
    """
    str * dict[str->str] -> None
    Reads the file containing pre-filled configuration and writes a file with
    the proper name to ~/.config
    """

    config_dir = get_config_dir(option)

    if not os.path.exists(config_dir):
        log.info('Creating a .config folder here: {}'.format(config_dir))
        os.mkdir(config_dir)

    assert os.path.exists(config_dir), "Could not locate your config folder"

    destination = os.path.join(
        config_dir,
        get_config_filename(option)
        )

    with open(destination, 'w') as f:
        text = read_config_example(option)
        f.write(text)

    return


def get_config_dir(option):
    """
    str -> str
    Returns the abs path to ~/.config/[option directory]
    """
    home = os.path.expanduser("~")
    name = 'til-core' if option == 'til' else 'openfisca-survey-manager'
    config_dir = os.path.join(
        home,
        ".config/" + name
        )
    return(config_dir)


def get_config_filename(option):
    """
    str -> str
    Returns the name fo the config file for the option
    """
    assert option in ['til', 'osm', 'hsi']
    filename = 'raw_data.ini' if option == 'hsi' else 'config.ini'
    return(filename)


# main functions
def configuration(option):
    """
    str -> None
    Given an option, prompts the user to fill the appropriate config files from
    a template
    """
    assert option in ['til', 'osm', 'hsi'], "Unrecognized option"

    attributes_by_option = {
        'til': {
            'raw_input': (
                "You now will be taken to your text editor.\n" +
                "You must give appropriate directories for where raw data should be stored, " +
                "as well as appropriate directories for Til Core. \n" +
                "Your config file should contain at least:\n\n"),
            'module': til_core,
            'name': 'Til Core',
            },
        'osm': {
            'raw_input': (
                "You now will be taken to your text editor.\n" +
                "HSI data is first processed by openfisca survey manager (openfisca_survey_manager), " +
                "then by til_france. You must give directories where to store openfisca_survey_manager files\n" +
                "Your config file should contain at least:\n\n"),
            'module': openfisca_survey_manager,
            'name': 'Openfisca Survey Manager',
            },
        'hsi': {
            'raw_input': ("You now will be taken to your text editor (for the last time).\n" +
                "You must give a directory where to get HSI raw data. " +
                "Your config file should contain at least:\n\n"),
            'module': openfisca_survey_manager,
            'name': 'HSI raw data',
            }
        }

    attributes = attributes_by_option[option]

    name = attributes['name']
    raw_input_text = attributes['raw_input']
    config_filename = get_config_filename(option)
    config_example = read_config_example(option)
    config_dir = get_config_dir(option)

    log.info('Starting {} configuration ...'.format(name))

    config_file = os.path.join(
        config_dir,
        config_filename
        )

    if not os.path.isfile(config_file):
        log.info('Creating a {} configuration template...'.format(name))
        init_file_exit_value = copy_init_file_to_config(option)
        assert init_file_exit_value is None, "Unable to copy {} init template to your .config folder".format(name)

    assert os.path.isfile(config_file), "Config template was not created"

    raw_input(
        raw_input_text + config_example
        )

    editor = os.environ.get('EDITOR', 'vim')
    call([editor, config_file], shell = True)
    # from now on, assume we have a correct config file

    return


def build_insee_data():
    """
    None -> None
    Build INSEE data
    """
    log.info(u"Starting basic insee data processing ...")
    build_demographic_model_data_()
    log.info('... done')
    return


def build_patrimoine_data():
    """
    None -> None
    Build Patrimoine data /// NB: should not work! Patrimoine script is broken
    """
    log.info(u"Starting Patrimoine data processing ...")
    patrimoine()
    log.info('... done')
    return


def build_hsi_data():
    """
    None -> None
    Build HSI data
    """
    log.info(u"Starting HSI data processing ...")
    sys.argv.extend(['-chsi', '-d', '-m'])
    build_collection.main()
    hsi()
    log.info('... done')
    return


def build_parameters():
    """
    None -> None
    Build Parameters file
    """

    # no idea what this is for
    sys.argv.extend(['-d'])

    build_parameters_()
    return


def data_prompt():
    """
    None -> None
    Prompt the user to check if the data is where it should be
    """
    raw_input((
        "Make sure the input folders you chose actually contain the input data " +
        "(Patrimoine and HSI) and press ENTER to proceed"
        ))
    return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-v',
        '--verbose',
        action = 'store_true',
        default = False,
        help = "increase output verbosity"
        )

    args = parser.parse_args()

    logging.basicConfig(
        level = logging.DEBUG if args.verbose else logging.INFO,
        stream = sys.stdout)

    to_do = [
        lambda: configuration('til'),
        lambda: configuration('osm'),
        lambda: configuration('hsi'),
        data_prompt,
        build_insee_data,
        build_patrimoine_data,
        build_hsi_data,
        build_parameters
        ]

    for task in to_do:
        exit_value = task()
        assert exit_value is None, "{} failed".format(task.__name__)

    return


if __name__ == "__main__":
    sys.exit(main())
