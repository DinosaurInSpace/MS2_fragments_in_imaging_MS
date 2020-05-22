#!/usr/bin/env python

"""
To write!
"""


import pandas as pd
import numpy as np
import argparse
from string import digits
from molmass import Formula


__author__ = "Christopher M Baxter Rath"
__copyright__ = "Copyright 2020"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"


def ms_pd_reader(ms_path, db_index):
    # Reads Sirius output obtaining m/z and formula
    df = pd.read_csv(ms_path, sep='\t', header=0)
    df = df[['exactmass', 'explanation']]
    df['db_index'] = db_index
    return df


def df_merge(input_df):
    # Merges MS2 data with metadata and output to
    # pre-METASPACE df
    out_df = pd.DataFrame()

    # Loops over dataframe
    index_list = list(input_df.index)
    ctr = 0
    for idx in index_list:
        ctr += 1
        if ctr % 1000 == 0:
            print('df_merge: ', ctr)
        else:
            pass
        ser = input_df.loc[idx]
        df = ms_pd_reader(ser.exists, ser.db_index)
        out_df = pd.concat([out_df, df])
    out_df = pd.merge(out_df, input_df, how='left', on='db_index')
    return out_df


def ex(formula):
    if type(formula) is str:
        return Formula(formula).isotope.mass
    else:
        print('Bad formula?', formula)
        return formula


def add_A(A, formula, n):
    # Safely add/subtract elements such as H to molecular formulas
    # E.g. A = 'H', formula = 'C6H12O6', n = 1
    x = formula
    if n == 0:
        return x

    elif A not in x and n >= 0:
        prefix = x
        if n == 1:
            final_suffix = A
        else:
            final_suffix = A + str(n)

    elif A not in x and n < 0:
        print('Error! No ', A, ' to increment by ', n, ' in X!')
        return np.nan

    else:
        x = x.split(A)
        prefix = x[0]
        N_suffix = x[1]
        if N_suffix == '':
            if n == -1:
                final_suffix = ''
            else:
                final_suffix = A + str(n + 1)
        else:
            ln = len(N_suffix)
            suffix = N_suffix.lstrip(digits)
            ls = len(suffix)
            # print(ln, " ", ls)

            if ln - ls == 0:
                N = 1
            elif ln - ls == 1:
                N = N_suffix[0:1]
            elif ln - ls == 2:
                N = N_suffix[0:2]
            elif ln - ls == 3:
                N = N_suffix[0:3]
            else:
                print('Bad formula!')
                return
            # print(N)
            if int(N) + n == 0:
                final_suffix = suffix
            else:
                final_suffix = A + str(int(N) + n) + suffix
    return prefix + final_suffix


def ion_form(formula, dmass):
    # Generates ion formula from formula and dmass
    # Will need to be updated for new adducts!
    # print(formula, '\t', type(formula), '\t', dmass)
    if dmass == 0:
        formula = formula
    elif dmass == -1:
        formula = add_A('H', formula, -1)
    elif dmass == 1:
        formula = add_A('H', formula, 1)
    elif dmass == 2:
        formula = add_A('H', formula, 2)
    elif dmass == 22:
        # formula = add_A('H', formula, -1)
        formula = add_A('Na', formula, 1)
    elif dmass == 23:
        formula = add_A('Na', formula, 1)
    elif dmass == 38:
        # formula = add_A('H', formula, -1)
        formula = add_A('K', formula, 1)
    elif dmass == 39:
        formula = add_A('K', formula, 1)
    else:
        print('dmass not known!')
        formula = np.nan
    return formula


def ionmasspos(ionformula):
    return ex(ionformula) - 0.00055


def ionmassneg(ionformula):
    return ex(ionformula) + 0.00055


def mass_check(em, im):
    if em - im <= 0.001:
        return True
    else:
        return False


def results_clean_up(has_ms2_df, sirius_output_df, polarity):
    # Empty!  columns = [exits, source, polarity]
    print(2, 'sirius_output_df', sirius_output_df.shape, sirius_output_df)

    # Merges Sirius results and metadata
    ms2_meta_df = pd.merge(has_ms2_df,
                           sirius_output_df,
                           how='inner',
                           left_on='db_index',
                           right_index=True)

    # Joins MS2 spectra to metadata
    print('10', ms2_meta_df.shape)
    df = df_merge(ms2_meta_df)
    df = df.dropna()
    df['expl_ex'] = df.explanation.apply(lambda x: ex(x))
    df['dmass'] = df['exactmass'] - df['expl_ex']
    df = df.astype({'dmass': int})
    df[['formula', 'explanation', 'exactmass', 'expl_ex', 'dmass']]
    df['H_check'] = df.explanation.str.contains('H')

    print('2', df.shape, df.dmass.astype(int).value_counts())

    if polarity == 'positive':
        df = df[df.dmass >= 0]
    elif polarity == 'negative':
        df = df[df.dmass <= 0]
    else:
        print('Bad polarity!')
        exit(1)

    print('3', df.shape)

    print('Observed ions: \n', df['dmass'].value_counts())
    # Drops artifact masses?
    df = df[~((df.dmass == 38) & (df.H_check == False))]
    df = df[~((df.dmass == 22) & (df.H_check == False))]

    print('4', df.shape)

    df['ion_formula'] = df.apply(lambda x: ion_form(x.explanation, x.dmass), axis=1)

    # Drops inherently charged formulas, rare.
    df['bad_if'] = df.ion_formula.str.contains('-')
    df = df[df.bad_if == False]
    df = df.copy(deep=True)

    # ionmasneg if negative mode!
    if polarity == 'positive':
        df['ion_mass'] = df.ion_formula.apply(lambda x: ionmasspos(x))
    elif polarity == 'negative':
        df['ion_mass'] = df.ion_formula.apply(lambda x: ionmassneg(x))

    df['good_mass_calc'] = df.apply(lambda x: mass_check(x.exactmass,
                                                         x.ion_mass),
                                    axis=1)
    print('Correct ionformulas: \n', df.good_mass_calc.value_counts())
    return df


def f_or_p_ion(d, p):
    if d == p:
        return 'p'
    else:
        return 'f'


def output_metaspace(pre_metaspace_df):
    # Filters columns, renames, and prepares for export to METASPACE
    df = pre_metaspace_df[['explanation', 'formula', 'id', 'name',
                           'ion_formula', 'inchi']].copy(deep=True)
    df['f_num'] = df.groupby(['name']).cumcount() + 1
    df['f_or_p'] = df.apply(lambda x: f_or_p_ion(x.explanation,
                                                 x.formula), axis=1)
    df['out_id'] = df.id + '_' + df.f_num.astype(str) + df.f_or_p
    df['out_name'] = df.out_id + '_' + df.name
    df_prefilter = df[['out_id', 'out_name', 'ion_formula', 'inchi', 'id']]
    df_prefilter = df_prefilter.rename(columns={'out_id': 'id',
                                                'out_name': 'name',
                                                'ion_formula': 'formula',
                                                'id': 'old_id'
                                                })
    return df_prefilter


def primary_loop(limit_list,
                 out_name,
                 out_path,
                 polarity,
                 expt_pos,
                 theo_pos,
                 expt_neg,
                 theo_neg,
                 ref_expt,
                 ref_theo
                 ):
    # Label and concat 4x input dataframes:
    exp_negative = pd.read_pickle(expt_neg)
    exp_negative['source'] = 'experimental'
    exp_negative['polarity'] = 'negative'
    exp_positive = pd.read_pickle(expt_pos)
    exp_positive['source'] = 'experimental'
    exp_positive['polarity'] = 'positive'
    theo_negative = pd.read_pickle(theo_neg)
    theo_negative['source'] = 'theoretical'
    theo_negative['polarity'] = 'negative'
    theo_positive = pd.read_pickle(theo_pos)
    theo_positive['source'] = 'theoretical'
    theo_positive['polarity'] = 'positive'

    out_df = pd.concat([exp_negative,
                        exp_positive,
                        theo_negative,
                        theo_positive], sort=True
                       )
    if out_df.empty:
        print('DataFrame is empty!')

    out_df = out_df[['exists', 'source', 'polarity']].copy(deep=True)
    out_df = out_df[out_df.polarity == polarity]

    # Clean-up and merge output with MS2 spectra
    ref_db = pd.concat([pd.read_pickle(ref_expt),
                        pd.read_pickle(ref_theo)], sort=True
                       )
    joined_out = results_clean_up(ref_db, out_df, polarity)

    # Clean-up database for METASPACE
    metaspace_db = output_metaspace(joined_out)
    if limit_list != []:
        metaspace_db = metaspace_db[metaspace_db.old_id.isin(limit_list)]

    # Export entire db
    if metaspace_db.empty:
        print('DataFrame is empty!')

    else:
        metaspace_db.iloc[:, 0:4].to_csv(out_path + out_name + '.csv',sep='\t')
        print(out_path + out_name + '.csv')
        return 1


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
    parser.add_argument("--out_path",
                        default=None,
                        type=str,
                        help="Path to output folder")
    parser.add_argument("--polarity",
                        default=None,
                        type=str,
                        help="Try: positive, or negative")
    parser.add_argument("--expt_pos",
                        default=None,
                        type=str,
                        help="Experimental positive spectra")
    parser.add_argument("--theo_pos",
                        default=None,
                        type=str,
                        help="Theoretical positive spectra")
    parser.add_argument("--expt_neg",
                        default=None,
                        type=str,
                        help="Experimental negative spectra")
    parser.add_argument("--theo_neg",
                        default=None,
                        type=str,
                        help="Theoretical negative spectra")
    parser.add_argument("--ref_expt",
                        default=None,
                        type=str,
                        help="Reference experimental db")
    parser.add_argument("--ref_theo",
                        default=None,
                        type=str,
                        help="Reference theoretical db")
    args = parser.parse_args()

    primary_loop(args.limit_list,
                 args.out_name,
                 args.out_path,
                 args.polarity,
                 args.expt_pos,
                 args.theo_pos,
                 args.expt_neg,
                 args.theo_neg,
                 args.ref_expt,
                 args.ref_theo
                 )

    print('Sirius spectra with formulas were converted to METASPACE MS1 and MS2 db')
    print('Outname: ', args.out_path + args.out_name)
    return 1

if __name__ == "__main__":
    main()