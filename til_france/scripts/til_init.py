#! /usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
from glob import glob
import logging
import os
import pip
from shutil import copy
from subprocess import call
import sys

from til_france.data.data.Patrimoine import test_build as patrimoine_main


app_name = os.path.splitext(os.path.basename(__file__))[0]
log = logging.getLogger(app_name)


def til_configuration():
    log.info('Starting Til Core configuration ...')

    til_core_install_exit_value = install_til_core()
    assert til_core_install_exit_value in [0, 1], "Could not install Til Core"

    # now til-core is installed or the script broke before
    global til_core
    import til_core

    if til_core_install_exit_value:
        log.info('Creating a Til Core configuration template...')
        init_file_exit_value = copy_init_file_to_config()
        assert init_file_exit_value == 0, "Unable to copy Til Core init template to your .config folder"

    # now we have at worst a config template in the right place, at best a config file with all we want
    config_dir = get_config_dir()
    config_filename = get_config_filename()

    fichier_config = os.path.join(
        config_dir,
        config_filename
        )

    assert os.path.isfile(fichier_config), "Config template was not created"

    config_example = read_config_example()

    raw_input((
        "You now will be taken to your text editor.\n" +
        "You must give appropriate directories for where raw data should be stored, " +
        "as well as appropriate directories for Til Core. \n" +
        "Your config file should contain at least:\n" +
        config_example + '\n' +
        "If it is already the case, just close your editor.\n" +
        "NB: If your editor is not well set, this will start vim as a default editor.\n" +
        "If you are lost, you can exit it by pressing escape then :q! then enter.\n" +
        "Set your editor properly and re-run this script.\n" +
        "Press any key to continue\n"
        ))

    editor = os.environ.get('EDITOR', 'vim')
    call([editor, fichier_config], shell = True)
    # from now on, assume we have a correct config file

    return(0)


def read_config_example():
    filename = 'config_example.txt'
    fichier = os.path.abspath(filename)
    assert os.path.exists(fichier), "Could not find config example file"

    with open(fichier) as fichier:
        text = fichier.read() + '\n'

    return(text)


def copy_init_file_to_config():
    config_dir = get_config_dir()

    if not os.path.exists(config_dir):
        log.info('Creating a .config folder here: {}'.format(config_dir))
        os.mkdir(config_dir)

    assert os.path.exists(config_dir), "Could not locate your config folder"

    # locating the template config in til-core installation folder
    config_filename = get_config_filename()
    til_core_path = os.path.dirname(til_core.__file__)
    relative_path_of_init = '../' + config_filename

    init_template = os.path.join(
        til_core_path,
        relative_path_of_init
        )

    copy(init_template, config_dir)

    return(0)


def get_config_dir():
    home = os.path.expanduser("~")
    config_dir = os.path.join(
        home,
        ".config/til-core"
        )
    return(config_dir)


def get_config_filename():
    assert 'til_core' in sys.modules, "Til_Core should have been imported at this stage"

    til_core_path = os.path.dirname(til_core.__file__)

    search = os.path.join(
        til_core_path,
        '../*.ini'
        )

    candidates = glob(search)
    assert len(candidates) == 1, "Not clear what config template should be used"

    filename = os.path.basename(candidates[0])

    return(filename)


def install_til_core():
    installed_packages = pip.get_installed_distributions()
    installed_packages_names = [package.project_name for package in installed_packages]
    if 'Til-Core' not in installed_packages_names:
        answer = raw_input((
            "Til-Core is not installed but is required. " +
            "Do you want to download and install Til Core? " +
            "y/n \n"
            )).lower()

        if answer in ['y', 'yes']:
            log.info('Installing Til Core...')
            pip.main(['install', 'Til-Core'])
            installed_packages = pip.get_installed_distributions()
            installed_packages_names = [package.project_name for package in installed_packages]
            assert 'Til-Core' in installed_packages_names, "Failed to install Til-Core with pip"
            return(1)
        elif answer in ['n', 'no']:
            log.info('Til Core is required. Aborting...')
            return(-2)
        else:
            log.info('Could not understand answer. Aborting...')
            return(-1)

    return(0)


def build_insee_data():
    return(0)


def build_patrimoine_data():
    log.info(u"Starting Patrimoine data processing ...")
    # patrimoine_main()
    log.info('... done')
    return(0)


def build_hsi_data():

    return(0)


def build_parameters():

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
        til_configuration,
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
