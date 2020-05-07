#!/usr/bin/env python

"""

"""

import pandas as pd
import numpy as np
import mona
import argparse
import glob

__author__ = "Christopher M Baxter Rath"
__copyright__ = "Copyright 2020"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"


def pd_tidy(in_str, df):
    t_dict = dict(zip(df['name'], df['value']))
    return t_dict[in_str]


def search_mona(ref_db, polarity):
    mona_df = pd.DataFrame()
    inchi_ref_list = list(ref_db.inchi.unique())
    counter = 0

    for inchi_x in inchi_ref_list:
        counter += 1
        print(inchi_x)
        print(counter)
        print()
        # 1834

        failure_list = []

        try:
            spectral_hits = mona.mona_main(inchi_x, 'MS2', polarity)
        except:
            failure_list.append(counter)

        counter2 = 0
        for spectra in spectral_hits:
            counter2 -= 1
            print(counter2)
            temp_dict = {}
            temp_dict['name'] = spectra['compound'][0]['names'][0]['name']
            temp_dict['id'] = spectra['id']
            temp_dict['spectrum'] = spectra['spectrum']
            try:
                temp_dict['source'] = spectra['library']['library']
            except:
                temp_dict['source'] = 'unknown'

            try:
                temp_dict['link'] = spectra['library']['link']
            except:
                temp_dict['link'] = 'unknown'

            df = pd.DataFrame(spectra['compound'][0]['metaData'])
            temp_dict['formula'] = pd_tidy('molecular formula', df)
            temp_dict['total_exact'] = pd_tidy('total exact mass', df)
            temp_dict['inchi'] = pd_tidy('InChI', df)

            df = pd.DataFrame(spectra['metaData'])

            try:
                temp_dict['instrument'] = pd_tidy('instrument', df)
            except:
                try:
                    temp_dict['instrument'] = pd_tidy('instrument type', df)
                except:
                    temp_dict['instrument'] = 'unknown'

            temp_dict['polarity'] = pd_tidy('ionization mode', df)
            temp_dict['MSn'] = pd_tidy('ms level', df)

            try:
                temp_dict['adduct'] = pd_tidy('precursor type', df)
            except:
                temp_dict['adduct'] = 'unknown'

            try:
                temp_dict['precursor'] = pd_tidy('precursor m/z', df)
            except:
                temp_dict['precursor'] = np.nan

            try:
                temp_dict['exact'] = pd_tidy('exact mass', df)
            except:
                temp_dict['exact'] = np.nan

            try:
                temp_dict['dppm'] = pd_tidy('mass accuracy', df)
            except:
                temp_dict['dppm'] = 'unknown'
            mona_df = mona_df.append(temp_dict, ignore_index=True)

    print(failure_list)
    return mona_df


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
        mona_df = search_mona(pd.read_pickle(args.input),
                              'positive'
                              )
        mona_df.to_pickle(args.output + 'mona_positive.pickle')

    if args.polarity == 'both' or args.polarity == 'negative':
        print('here')
        mona_df = search_mona(pd.read_pickle(args.input),
                              'negative'
                              )
        mona_df.to_pickle(args.output + 'mona_negative.pickle')

    print('MONA search complete!')
    return 1


if __name__ == "__main__":
    main()
