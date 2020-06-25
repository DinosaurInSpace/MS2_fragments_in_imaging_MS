#!/usr/bin/env python

"""
To write!



"""


import pandas as pd
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


def filter_me(df, f):
    # Filters input df based on inputs
    if f.analyzer == 'all':
        inst = ['FTICR', 'Orbitrap']
    else:
        inst = [f.analyzer]

    groups = None
    if f.group != 'all':
        if type(f.group) == str:
            groups = [f.group]
        else:
            groups = f.group

    # expt_type
    if f.expt_type == 'all':
        if f.group == 'all':
            return df[(df.polarity == f.polarity) &
                      (df.analyzer.isin(inst))]
        elif f.group != 'all':
            return df[(df.polarity == f.polarity) &
                      (df.analyzer.isin(inst)) &
                      (df.group.isin(groups))]
    else:
        if f.group == 'all':
            return df[(df.polarity == f.polarity) &
                      (df.analyzer.isin(inst)) &
                      (df.expt_type == f.expt_type)]
        elif f.group != 'all':
            return df[(df.polarity == f.polarity) &
                      (df.analyzer.isin(inst)) &
                      (df.group.isin(groups)) &
                      (df.expt_type == f.expt_type)]


def frag_per_par(df, result_type):
    # Calculates fragments per parent
    df = df[['id_x', 'ds_id', 'par_formula', 'cos']].copy(deep=True)
    df['cos'] = 1
    df = df.groupby(['id_x', 'ds_id', 'par_formula']).sum().reset_index()
    if result_type == 'avg':
        return df.cos.mean()
    else:
        return df.cos.std()


def find_top_10_frags(df):
    # Calculates top-10 most abundant fragments present at least once
    df = df[['id_x', 'par_formula', 'formula', 'par_frag']].copy(deep=True)
    df['n'] = 1
    df = df.groupby(['id_x', 'par_formula', 'formula', 'par_frag']).sum().reset_index()
    df.sort_values(by=['n'], inplace=True, ascending=False)
    df = df[df.n > 1].copy(deep=True)
    if df.shape[0] > 10:
        df = df.iloc[0:10, :].copy(deep=True)
    else:
        df = df.iloc[0:df.shape[0], :].copy(deep=True)
    df['out'] = df['id_x'] + "_" + df['par_frag'] + "_" + df['formula'] + '_' + df['n'].astype(str)

    return list(df.out)


def score_filtered(f, df):
    # Scores results on a per filter level, and exports dict.
    df1 = df[['id_x', 'ds_id', 'ion_mass', 'cos', 'db_n_isobar',
              'ds_n_isobar', 'par_frag', 'par_or_frag', 'formula',
              'par_formula']].copy(deep=True)
    df = df1.copy(deep=True)
    n_ds_id = float(len(df.ds_id.unique()))
    f['n_ds_id'] = n_ds_id

    # Divide by zero issues below with empty df
    if n_ds_id == 0:
        return f

    # Parent results
    f['n_par'] = df[df.par_or_frag == 'P'].shape[0] / n_ds_id

    # Fragment results
    df = df[df.par_or_frag == 'F']
    f['cos_avg'] = df.cos.mean()
    f['cos_std'] = df.cos.std()
    f['n_frag_00'] = df[df.cos >= 0.00].shape[0] / n_ds_id
    f['n_frag_50'] = df[df.cos >= 0.50].shape[0] / n_ds_id
    f['n_frag_75'] = df[df.cos >= 0.75].shape[0] / n_ds_id
    f['n_frag_90'] = df[df.cos >= 0.90].shape[0] / n_ds_id
    f['f_per_p_avg_00'] = frag_per_par(df[df.cos >= 0.00], 'avg')
    f['f_per_p_std_00'] = frag_per_par(df[df.cos >= 0.00], 'stdev')
    f['f_per_p_avg_50'] = frag_per_par(df[df.cos >= 0.50], 'avg')
    f['f_per_p_std_50'] = frag_per_par(df[df.cos >= 0.50], 'stdev')
    f['f_per_p_avg_75'] = frag_per_par(df[df.cos >= 0.75], 'avg')
    f['f_per_p_std_75'] = frag_per_par(df[df.cos >= 0.75], 'stdev')
    f['f_per_p_avg_90'] = frag_per_par(df[df.cos >= 0.90], 'avg')
    f['f_per_p_std_90'] = frag_per_par(df[df.cos >= 0.90], 'stdev')

    # Unique and 1 isobar results
    df = df1[df1.par_or_frag == 'F'].copy(deep=True)
    df = df[df.db_n_isobar == 0]
    f['n_u_db_frag_00'] = df[df.cos >= 0.00].shape[0] / n_ds_id
    f['n_u_db_frag_50'] = df[df.cos >= 0.50].shape[0] / n_ds_id
    f['n_u_db_frag_75'] = df[df.cos >= 0.75].shape[0] / n_ds_id
    f['n_u_db_frag_90'] = df[df.cos >= 0.90].shape[0] / n_ds_id

    df = df1[df1.par_or_frag == 'F'].copy(deep=True)
    df = df[df.db_n_isobar == 1]
    f['n_1_db_frag_00'] = df[df.cos >= 0.00].shape[0] / n_ds_id
    f['n_1_db_frag_50'] = df[df.cos >= 0.50].shape[0] / n_ds_id
    f['n_1_db_frag_75'] = df[df.cos >= 0.75].shape[0] / n_ds_id
    f['n_1_db_frag_90'] = df[df.cos >= 0.90].shape[0] / n_ds_id

    df = df1[df1.par_or_frag == 'F'].copy(deep=True)
    df = df[df.ds_n_isobar == 0]
    f['n_u_ds_frag_00'] = df[df.cos >= 0.00].shape[0] / n_ds_id
    f['n_u_ds_frag_50'] = df[df.cos >= 0.50].shape[0] / n_ds_id
    f['n_u_ds_frag_75'] = df[df.cos >= 0.75].shape[0] / n_ds_id
    f['n_u_ds_frag_90'] = df[df.cos >= 0.90].shape[0] / n_ds_id

    df = df1[df1.par_or_frag == 'F'].copy(deep=True)
    df = df[df.ds_n_isobar == 1]
    f['n_1_ds_frag_00'] = df[df.cos >= 0.00].shape[0] / n_ds_id
    f['n_1_ds_frag_50'] = df[df.cos >= 0.50].shape[0] / n_ds_id
    f['n_1_ds_frag_75'] = df[df.cos >= 0.75].shape[0] / n_ds_id
    f['n_1_ds_frag_90'] = df[df.cos >= 0.90].shape[0] / n_ds_id

    # Finds top-10 fragments by count present at least once
    df = df1[df1.par_or_frag == 'F'].copy(deep=True)
    f['top_10'] = find_top_10_frags(df)

    return f


def generate_vector(df, counter):
    # Generates a vector for analyzing result per Theo's request.
    vector = ['cos', 'db_n_isobar_par', 'db_n_isobar_frag',
              'ds_n_isobar_frag']
    metadata = ['ds_id', 'id_x', 'formula', 'par_frag', 'polarity',
                'analyzer', 'group', 'expt_type', 'filter']

    cols = vector + metadata

    p_df = df[df.par_or_frag == 'P'].copy(deep=True)
    p_df['db_n_isobar_par'] = df['db_n_isobar']
    p_df = p_df[['id_x', 'db_n_isobar_par']]

    f_df = df[df.par_or_frag == 'F'].copy(deep=True)
    f_df['db_n_isobar_frag'] = f_df['db_n_isobar']
    f_df['ds_n_isobar_frag'] = f_df['ds_n_isobar']

    df = pd.merge(f_df, p_df, how='left', on='id_x')
    df['filter'] = counter

    return df[cols].copy(deep=True)


def generate_results(filter_df, scored_df):
    # Loops through filters to analyze results
    m = filter_df.shape[0]
    counter = 0
    out_list = []
    vect_list = []
    while counter < m:
        print(counter)
        # Filter level results
        f = filter_df.iloc[counter, :]
        df = filter_me(scored_df, f)
        f_dict = score_filtered(dict(f), df)
        out_list.append(f_dict)

        # Annotated vectors
        vect = generate_vector(df, counter)
        vect_list.append(vect)
        counter += 1

    return out_list, vect_list


def main():
    # Main captures input variables when called as command line script.
    # Not yet updated!
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

    generate_results(args.limit_list,
                 args.out_name,
                 )

    print('Sirius spectra with formulas were converted to METASPACE MS1 and MS2 db')
    return 1

if __name__ == "__main__":
    main()