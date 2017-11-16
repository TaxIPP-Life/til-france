#! /usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import logging
import os
import sys


from glob import glob
from shutil import copyfile
from subprocess import call


import openfisca_survey_manager as osm
import til_core


from til_france.data.data.Patrimoine import test_build as patrimoine
from til_france.data.data.handicap_sante_institutions import create_hsi_data as hsi
from openfisca_survey_manager.scripts import build_collection


from build_demographic_model_data import main as bdmd
from population_and_dependance_builder import main as build_parameters


app_name = os.path.splitext(os.path.basename(__file__))[0]
log = logging.getLogger(app_name)


# utils
def read_config_example(option):
    filename = 'config_example_{}.txt'.format(option)
    fichier = os.path.abspath(filename)

    assert os.path.exists(fichier), "Could not find config example file"

    with open(fichier) as fichier:
        text = fichier.read() + '\n'

    return(text)


def copy_init_file_to_config(attributes):
    name = attributes['name']
    config_dir = attributes['config_dir']
    config_filename = attributes['config_filename']
    module = attributes['module']

    log.info('Creating a {} configuration template...'.format(name))

    if not os.path.exists(config_dir):
        log.info('Creating a .config folder here: {}'.format(config_dir))
        os.mkdir(config_dir)

    assert os.path.exists(config_dir), "Could not locate your config folder"

    # locating the template config in module installation folder
    module_path = os.path.dirname(module.__file__)
    relative_path_of_init = '../' + config_filename

    init_template = os.path.join(
        module_path,
        relative_path_of_init
        )

    destination = os.path.join(
        config_dir,
        remove_templateini(config_filename)
        )

    copyfile(init_template, destination)

    return(0)


def remove_templateini(s):
    if s.endswith('_template.ini'):
        remove = len('_template.ini')
        s = s[:-remove] + '.ini'
    return(s)


def get_config_dir(option):
    home = os.path.expanduser("~")
    name = 'til-core' if option == 'til' else 'openfisca-survey-manager'
    config_dir = os.path.join(
        home,
        ".config/" + name
        )
    return(config_dir)


def get_config_filename(option):
    assert option in ['til', 'osm']

    module = til_core if option == 'til' else osm
    module_path = os.path.dirname(module.__file__)

    search = os.path.join(
        module_path,
        '../*.ini'
        )

    candidates = glob(search)
    assert len(candidates) == 1, "Not clear what config template should be used"

    filename = os.path.basename(candidates[0])

    return(filename)


# main functions
def configuration(option):
    assert option in ['til', 'osm', 'hsi'], "Unrecognized option"

    last_words = (
        "NB: If your editor is not well set, this will start vim as a default editor.\n" +
        "If you are lost, you can exit it by pressing escape then :q! then enter.\n" +
        "Set your editor properly and re-run this script.\n" +
        "Press ENTER to continue\n"
        )

    attributes_by_option = {
        'til': {'raw_input': ["You now will be taken to your text editor.\n" +
                              "You must give appropriate directories for where raw data should be stored, " +
                              "as well as appropriate directories for Til Core. \n" +
                              "Your config file should contain at least:\n\n",
                              last_words],
                'module': til_core,
                'name': 'Til Core',
                'config_dir': get_config_dir('til'),
                'config_filename': get_config_filename('til'),
                'config_example': read_config_example('til'),
                },
        'osm': {'raw_input': ["You now will be taken to your text editor.\n" +
                              "HSI data is first processed by openfisca survey manager (osm), " +
                              "then by til_france. You must give directories where to store osm files\n" +
                              "Your config file should contain at least:\n\n",
                              "If it is already the case, just close your editor.\n" +
                              last_words],
                'module': osm,
                'name': 'Openfisca Survey Manager',
                'config_dir': get_config_dir('osm'),
                'config_filename': 'config_template.ini',
                'config_example': read_config_example('osm')
                },
        'hsi': {'raw_input': ["You now will be taken to your text editor (for the last time).\n" +
                              "You must give a directory where to get HSI raw data. " +
                              "Your config file should contain at least:\n\n",
                              "If it is already the case, just close your editor.\n" +
                              last_words],
                'module': osm,
                'name': 'HSI raw data',
                'config_dir': get_config_dir('osm'),
                'config_filename': 'raw_data_template.ini',
                'config_example': read_config_example('hsi')
                }
        }

    attributes = attributes_by_option[option]

    name = attributes['name']
    raw_input_1, raw_input_2 = attributes['raw_input']
    config_dir = attributes['config_dir']
    config_filename = attributes['config_filename']
    config_example = attributes['config_example']

    log.info('Starting {} configuration ...'.format(name))

    fichier_config = os.path.join(
        config_dir,
        remove_templateini(config_filename)
        )

    if not os.path.isfile(fichier_config):
        init_file_exit_value = copy_init_file_to_config(attributes)
        assert init_file_exit_value == 0, "Unable to copy {} init template to your .config folder".format(name)

    assert os.path.isfile(fichier_config), "Config template was not created"

    raw_input((
        raw_input_1 +
        config_example + '\n' +
        raw_input_2
        ))

    editor = os.environ.get('EDITOR', 'vim')
    call([editor, fichier_config], shell = True)
    # from now on, assume we have a correct config file

    return(0)


def build_insee_data():
    log.info(u"Starting basic insee data processing ...")
    bdmd()
    log.info('... done')
    return(0)


def build_patrimoine_data():
    log.info(u"Starting Patrimoine data processing ...")
    patrimoine()
    log.info('... done')
    return(0)


def build_hsi_data():
    log.info(u"Starting HSI data processing ...")
    sys.argv.extend(['-chsi', '-d', '-m'])
    build_collection.main()
    hsi()
    log.info('... done')
    return(0)


def build_parameters():
    sys.argv.extend(['-d'])
    build_parameters()
    return(0)


def data_prompt():
    raw_input(("Make sure the input folders you chose actually contain the input data " +
               "(Patrimoine and HSI) and press ENTER to proceed" 
             ))
    return(0)

def main():
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

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
        assert exit_value == 0, "{} failed".format(task.__name__)

    return(0)

if __name__ == "__main__":
    sys.exit(main())
