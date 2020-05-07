#!/usr/bin/env python

"""
Get HMDB exptl neg and HMDB theo neg df...
"""

import pandas as pd
import glob
import re
import os
import numpy as np
import argparse


__author__ = "Christopher M Baxter Rath"
__copyright__ = "Copyright 2020"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"


def tag_finder(head, tail, fh):
    query = head + '(.*)' + tail
    result = re.search(query, fh)
    if result == None:
        return 'nd'
    else:
        return result.group(1)


def get_att(fh, head, tail, directory):
    root_path = directory + '/' + fh
    with open (root_path, 'r') as file:
        data = file.read()
        out = tag_finder(head, tail, data)
        return out


def parse_hmdb(directory, polarity, theo_flag=False):
    # Filters dataset for high mass accuracy instruments and <20 ppm mass error
    file_paths = os.listdir(directory)
    df = pd.DataFrame()
    df['file_paths'] = list(sorted(file_paths))
    df['id'] = df.file_paths.apply(lambda x: x.split('_')[0])

    if theo_flag is False:
        df['instrument'] = df.file_paths.apply(lambda x: get_att(x,
                                                                 '<instrument-type>',
                                                                 '</instrument-type>',
                                                                 directory
                                                                 )
                                               )
    elif theo_flag is True:
        df['instrument'] = 'predicted'

    good_hmdb_intruments = ['LC-ESI-qTof',
                            'LC-ESI-ITFT (LTQ Orbitrap XL, Thermo Scientfic)',
                            'LC-ESI-QTOF (UPLC Q-Tof Premier, Waters)',
                            'DI-ESI-qTof',
                            'CE-ESI-TOF (CE-system connected to 6210 Time-of-Flight MS, Agilent)',
                            'LC-ESI-ITTOF (LCMS-IT-TOF)',
                            'LC-ESI-qTOF',
                            'MALDI-TOF (Voyager DE-PRO, Applied Biosystems)',
                            'LC-ESI-qToF',
                            'LC-ESI-Hybrid FT',
                            'LC-ESI-QTOF (ACQUITY UPLC System, Waters)',
                            'DI-ESI-Q-Exactive Plus',
                            'LC-ESI-ITTOF (Shimadzu LC20A-IT-TOFMS)',
                            'LC-ESI-ITFT (LTQ Orbitrap XL Thermo Scientific)',
                            'predicted'
                            ]

    df = df[df.instrument.isin(good_hmdb_intruments)]

    # Filters dataset for appropriate polarity
    df['polarity'] = df.file_paths.apply(lambda x: get_att(x,
                                                           '<ionization-mode>',
                                                           '</ionization-mode>',
                                                           directory
                                                           )
                                         )

    if polarity == 'positive':
        df = df[df.polarity == 'Positive']
        df['adduct'] = 'M+H'
    elif polarity == 'negative':
        df = df[df.polarity == 'Negative']
        df['adduct'] = 'M-H'
    return df


def hmdb_parse_loop(input, polarity):
    # Pareses experimental and predicted spectra sepearately
    hmdb_pos_exptl_df = parse_hmdb(input + 'hmdb_experimental_msms_spectra',
                                   polarity,
                                   theo_flag=False,
                                   )
    hmdb_pos_theo_df = parse_hmdb(input + 'hmdb_predicted_msms_spectra',
                                  polarity,
                                  theo_flag=True,
                                  )
    return([hmdb_pos_exptl_df,
            hmdb_pos_theo_df
    ])


def main():
    # Main captures input variables when called as command line script.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--input", default=None,
                        type=str,
                        help="Path to input db, e.g. core_metabolome_V3"
                        )
    parser.add_argument("--output",
                        default=None,
                        type=str,
                        help="Path to output df pickle for MONA"
                        )
    parser.add_argument("--polarity",
                        default='both',
                        type=str,
                        help="polarities to run: both, positive, negative"
                        )

    args = parser.parse_args()

    # Check input values are okay!

    if glob.glob(args.input) == []:
        print('Input path does not exist!')
        exit(1)

    good_polarities = ['both', 'positive', 'negative']
    if args.polarity not in good_polarities:
        print('Value for polarity is not recognized!\ntry: ', good_polarities)
        exit(1)

    if args.polarity == 'both' or args.polarity == 'positive':
        hmdb_dfs = hmdb_parse_loop(args.input,
                                   'positive'
                                   )
        hmdb_dfs[0].to_pickle(args.output + 'hmdb_exp_positive.pickle')
        hmdb_dfs[1].to_pickle(args.output + 'hmdb_theo_positive.pickle')

    if args.polarity == 'both' or args.polarity == 'negative':
        hmdb_dfs = hmdb_parse_loop(args.input,
                                   'negative'
                                   )
        hmdb_dfs[0].to_pickle(args.output + 'hmdb_exp_negative.pickle')
        hmdb_dfs[1].to_pickle(args.output + 'hmdb_theo_negative.pickle')

    print('HMDB parsing complete!')
    return 1


if __name__ == "__main__":
    main()