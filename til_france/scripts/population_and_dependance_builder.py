# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys

import ipp_macro_series_parser.demographie.build_parameters as constructeur


def main():
    parser = argparse.ArgumentParser()
   
    parser.add_argument(
        '-d',
        '--download',
        action = 'store_true',
        help = "download all input files from their web sources"
        )
    
    parser.add_argument(
        '-v',
        '--verbose',
        action = 'store_true',
        default = False,
        help = "increase output verbosity"
        )
      
    parser.add_argument(
        '-o',
        '--output',
        type = str,
        default = None,
        help = "output directory"
        )

    parser.add_argument(
        '-p',
        '--pop_input',
        type = str,
        default = None,
        help = "input directory for population files"
        )

    parser.add_argument(
        '-w',
        '--weight',
        default = 200,
        help = "weight used for TIL-France"
        )  # TODO remove weight from here

    parser.add_argument(
        '-t',
        '--til_input',
        default = None,
        help = "input directory for til-specific files (dependance)"
        )


    args = parser.parse_args()
    
    if not args.download and not (args.til_input and args.pop_input):
        print("Error: no source given for input files ")
        print("You must:")
        print(" - give directories containing the input files using *both* -t and -p")
        print(" - or download them from the web with -d")
        sys.exit(-1)
  
    if args.output is None:
        default_output = "../param/demo2"
        print("No output directory given. Default output directory used: " + default_output)
        sys.argv.append("-o" + default_output)

    constructeur.main()

if __name__ == '__main__':
   sys.exit(main())