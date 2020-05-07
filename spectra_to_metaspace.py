#!/usr/bin/env python

"""
To write!
"""


import pandas as pd
import numpy as np
import argparse
import glob
import os
import json
import mona
from string import digits
import re
import shlex
from subprocess import Popen, PIPE
from threading import Timer

# RDKit
from rdkit import Chem
from molmass import Formula


__author__ = "Christopher M Baxter Rath"
__copyright__ = "Copyright 2020"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"


def main():
    # Main captures input variables when called as command line script.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--input_expt",
                        default=None,
                        type=str,
                        help="Merged experimental df path")
    parser.add_argument("--input_theo",
                        default=None,
                        type=str,
                        help="Merged theoretical df path")
    parser.add_argument("--input_spectra_db",
                        default=None,
                        type=str,
                        help="Folder containing parsed spectra!")
    parser.add_argument("--output", default=None, type=str, help="Path to output df with spectra")
    parser.add_argument("--polarity",
                        default=None,
                        type=str,
                        help="Try: positive, negative, or both!")


    args = parser.parse_args()

    if glob.glob(args.input_expt) == []:
        print('Experimental input df does not exist!')
        exit(1)
    elif glob.glob(args.input_theo) == []:
        print('Theoretical input df does not exist!')
        exit(1)
    else:
        continue

    if args.polarity == 'both' or args.polarity == 'positive':
        function
        print()
    elif args.polarity == 'both' or args.polarity == 'negative':
        function
        print()
    else:
        print('Polarity was not recognized, try: positive, negative, or both!')
        exit(1)

    print('Spectra annotation with Sirius and export to METASPACE format complete!')
    return 1

if __name__ == "__main__":
    main()