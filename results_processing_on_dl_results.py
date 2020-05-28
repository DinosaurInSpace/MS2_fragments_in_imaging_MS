#!/usr/bin/env python

"""
To write!



"""


import pandas as pd
import numpy as np
import argparse
import pathlib
import pickle
import glob
import re
from collections import Counter

from scipy.ndimage import median_filter
from sklearn.metrics.pairwise import cosine_similarity




__author__ = "Christopher M Baxter Rath"
__copyright__ = "Copyright 2020"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"


def load_arr_pickle(image_path):
    # Loads pickled np.array
    with open(image_path, 'rb') as f:
        return pickle.load(f)


def median_thresholded_cosine_v2(a, b):
    # Modified function emulating METASPACE with 2D arr input and score out
    def preprocess(img):
       img = img.copy()
       img[img < np.quantile(img, 0.5)] = 0
       return median_filter(img, (3, 3)).reshape(1, -1)
    return cosine_similarity(preprocess(a), preprocess(b))[0, 0]


def cos_df(x, path_to_folder, ds_id):
    # Annotates cosine similarity using df formulas and np.arrays of images
    path_prefix = path_to_folder + '/' + ds_id + '/by_id/' + x.id_x + '/'
    a = path_prefix + x.par_formula + '_P.pickle'
    b = path_prefix + x.formula + '_' + x.par_or_frag + '.pickle'
    return median_thresholded_cosine_v2(load_arr_pickle(a), load_arr_pickle(b))


def parent_or_fragment(input_str):
    # Annotates METASPACE result as parent or fragment ion
    if input_str.find('p') != -1:
        return 'P'
    else:
        return 'F'


def annotate_cos_parent_fragment(path_to_folder, unique_ds_id, out_file):
    # Will generate dataframe with pairwise cosine similarity for parent and fragments
    master_df = pd.DataFrame()
    counter = 0
    error_counter = 1
    for ds_id in unique_ds_id:
        counter += 1
        print(counter, ' ', ds_id)
        pf_df_path = glob.glob(path_to_folder + '/' + ds_id + '/*.pickle')[0]
        pf_df = pd.read_pickle(pf_df_path)
        pf_df['par_or_frag'] = pf_df.par_frag.apply(lambda x: parent_or_fragment(x))
        p_df = pf_df[pf_df['par_or_frag'] == 'P'][['id_x', 'formula']].copy(deep=True)
        p_df = p_df.rename(columns={'formula':'par_formula'})
        pf_df = pd.merge(pf_df, p_df, on='id_x', how='left')
        # Where, when is it failing?
        try:
            pf_df['cos'] = pf_df.apply((lambda x: cos_df(x,
                                                         path_to_folder,                
                                                         ds_id)), axis = 1)
        except:
            pf_df['cos'] = 'error!'
            error_counter += 1
            print('error:', error_counter)

        master_df = pd.concat([master_df, pf_df], sort=True)
    master_df.to_pickle(out_file)
    return master_df


def parse_formula(formula):
    # Parses chemical formula to counter, thanks Vitally!
    FORMULA_REGEXP = re.compile(r'([A-Z][a-z]*)([0-9]*)')

    regexp_matches = [(elem, int(n or '1')) for (elem, n) in FORMULA_REGEXP.findall(formula)]
    return Counter(dict(regexp_matches))


def delta_formula(parent_formula, fragment_formula):
    # Finds delta formula between parent and fragment
    out_list = []
    delta = parse_formula(parent_formula) - parse_formula(fragment_formula)
    for key, value in delta.items():
        out_list.append(key)
        out_list.append(str(value))
    return "".join(out_list)


def find_3_ppm(mz, col, target, df):
    # Returns id's within +/- 3 ppm range from target, includes self
    ppm3 = (mz * 3)/1000000
    df = df[(df[col] >= mz - ppm3) & (df[col] <= mz + ppm3)]
    isobars = list(df[target])
    return isobars


def find_3_ppm_overlap_in_ds(df, group='ds_id', col='ion_mass',
                       target='id', expt=True, out_col='isobars'):
    # Finds matching ions with 3 ppm for each dataset in a set.
    df_list = []
    big_df = df.sort_values(by=[col])
    if expt == True:
        ds_counter = 0
        for ds_id in list(df.ds_id.unique()):
            print(ds_counter, ds_id)
            ds_counter += 1
            df = big_df[big_df.ds_id == ds_id]
            df[out_col] = df[col].apply(lambda x: find_3_ppm(x, col, target, df))
            df_list.append(df)
    else:
        df[out_col] = df[col].apply(lambda x: find_3_ppm(x, col, target, df))
        df_list.append(df)
    return pd.concat(df_list, sort=True)


def n_isobars(df, new_col_name, input_col_name):
    # Counts number of isobars
    df[new_col_name] = df[input_col_name].apply(lambda x: len(x) - 1)
    return df


def processing_loop():
    pass
    return


def main():
    # Main captures input variables when called as command line script.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--limit_list",
                        default=[],
                        type=list,
                        help="List of ids to limit results to, e.g. MS1 hits")
    parser.add_argument("--out_name",
                        default=None,
                        type=str,
                        help="Name of output METASPACE db")
    args = parser.parse_args()

    processing_loop(args.limit_list,
                 args.out_name,
                 )

    print('Sirius spectra with formulas were converted to METASPACE MS1 and MS2 db')
    return 1

if __name__ == "__main__":
    main()