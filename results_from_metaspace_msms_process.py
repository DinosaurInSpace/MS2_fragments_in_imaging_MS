#!/usr/bin/env python

"""
To write!

To-do:
1) Make sure it works.
2) Check that the download results works.
3) Replace many static variables.
4) Check downloading of images files works.
5) Any other df stuff parsing needed?
6) Implement proposed scoring function


"""


import pandas as pd
import numpy as np
import argparse
import ast
from glob import glob
from matplotlib import pyplot as plt
import pathlib
from shutil import copyfile

from results_local import results as results2
from metaspace.sm_annotation_utils import SMInstance
from metaspace.sm_annotation_utils import GraphQLClient
del GraphQLClient.DEFAULT_ANNOTATION_FILTER['hasHiddenAdduct']
sm = SMInstance(host='https://beta.metaspace2020.eu')
sm

import types
# Assign external function as method to object
sm.results2 = types.MethodType(results2, sm)


__author__ = "Christopher M Baxter Rath"
__copyright__ = "Copyright 2020"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"


def logon_metaspace(path_to_pw='/Users/dis/.metaspace.json'):
    # Logs onto METASPACE server secretly
    f = open('/Users/dis/.metaspace.json', "r")
    secret = (f.read())
    secret = secret.replace('\n', '')
    secret = ast.literal_eval(secret)
    f.close()

    sm.login(secret['email'], secret['password'])
    return


def split_data_frame_list(df, target_column):
    # Accepts a column with multiple types and splits list variables to
    # several rows.

    row_accumulator = []

    def split_list_to_rows(row):
        split_row = row[target_column]

        if isinstance(split_row, list):

          for s in split_row:
              new_row = row.to_dict()
              new_row[target_column] = s
              row_accumulator.append(new_row)

          if split_row == []:
              new_row = row.to_dict()
              new_row[target_column] = None
              row_accumulator.append(new_row)

        else:
          new_row = row.to_dict()
          new_row[target_column] = split_row
          row_accumulator.append(new_row)

    df.apply(split_list_to_rows, axis=1)
    new_df = pd.DataFrame(row_accumulator)

    return new_df


def a_label(x, a):
    # Finds string a in string x
    if a in x:
        return 1
    else:
        return 0


def extract_results_METASPACE(ms_out):
    # Read METASPACE output
    ms_out = pd.read_csv(ms_out, header=2)

    ms_out = ms_out[(ms_out.adduct == 'M[M]+')|(ms_out.adduct == 'M[M]-')]
    ms_out['n_ids_formula'] = ms_out.moleculeIds.apply(lambda x: len(x.split(',')))
    gc = ['datasetId', 'formula', 'adduct', 'mz', 'fdr',
          'moleculeNames']
    ms_out = ms_out[gc]
    ms_out['moleculeNames'] = ms_out['moleculeNames'].apply(lambda x: x.split(', '))
    ms_out = split_data_frame_list(ms_out, 'moleculeNames')
    ms_out['moleculeNames'] = ms_out['moleculeNames'].apply(lambda x: x.split('_', 2))
    df = pd.DataFrame(ms_out['moleculeNames'].tolist(), columns=['id', 'par_frag', 'name'])
    ms_out = pd.concat([df, ms_out, ], axis=1)
    ms_out = ms_out.sort_values(by=['name'])

    # Label with number of parents and fragments for each id
    ms_out['parent'] = ms_out['par_frag'].apply(lambda x: a_label(x, 'p'))
    ms_out['n_frag'] = ms_out['par_frag'].apply(lambda x: a_label(x, 'f'))
    df = ms_out[['id', 'parent', 'n_frag']]
    df = df.groupby('id').sum()
    ms_out = ms_out.merge(df, on='id', how='left')
    gc = ['id', 'par_frag', 'name', 'datasetId', 'formula',
          'adduct', 'mz', 'fdr', 'moleculeNames',
          'parent_y', 'n_frag_y']
    ms_out = ms_out[gc].copy(deep=True)

    # Label with number of degenerate formulas at each mass
    df = ms_out[['id', 'formula']]
    df = df.groupby('formula').nunique()
    df = df.iloc[:, 0:1]
    ms_out = ms_out.merge(df, on='formula', how='left')

    return ms_out



def process_manual_metaspace_report_dl(beta_id, prod_id, outpath):
    print(beta_id)
    target = 'any_results/metaspace_report/' + beta_id + '.csv'
    fh = glob.glob(target)
    print(fh)
    ms_out = pd.read_csv(fh[0], header=2)
    print(ms_out.shape)
    ms_out2 = extract_results_METASPACE(ms_out)
    print(ms_out2.shape)
    outpath = outpath + prod_id + '_msms_report.csv'
    ms_out2.to_csv(outpath)
    print(outpath)
    return outpath


def filter_report_parent_wfragment(report, outpath):
    # Filters METASAPCE MSMS output for 1) parent observed and 2) 1+ fragment obs.
    df = pd.read_csv(report)
    df = df[(df.parent_y == 1) & (df.n_frag_y > 0)]
    ds_name = report.split('/')[-1]
    ds_name = ds_name.split('.')[0]
    outpath = outpath + ds_name + '.csv'
    df.to_csv(outpath)
    print(outpath)
    return outpath


def metaspace_hotspot_removal(img):
    awkwardness = np.max(img) / 255 # METASPACE holdover from 8-bit
    hot_thresold = np.percentile(img[img >= awkwardness], 99) or 1
    return np.clip(img, 0, hot_thresold) / hot_thresold


def dl_img(main_id, beta_id, db, fdr_max, save_img):
    # Load and save dataset as image or arrays (saved as df)
    ds = sm.dataset(id=beta_id)
    print(ds)

    # Generate and save images
    x = ds.all_annotation_images(fdr=fdr_max,
                                 database=db,
                                 only_first_isotope=True,
                                 scale_intensity=False,
                                 hasHiddenAdduct=True)
    if x == []:
        return 'Error, empty annotations!'
    else:
        for n in x:
            if save_img == True:
                image = metaspace_hotspot_removal(n._images[0])
                plt.imshow(image)
                pathlib.Path('formula/' + main_id + '/').mkdir(parents=True, exist_ok=True)
                img_name = 'formula/' + main_id + '/' + n.formula + '.png'
                plt.imsave(img_name, image)
            else:
                # For saving arrays as df per Theo's request
                df = pd.DataFrame(data=metaspace_hotspot_removal(n._images[0]))
                pathlib.Path('any_results/formula_arr/' + main_id + '/').mkdir(parents=True, exist_ok=True)
                arr_name = 'any_results/formula_arr/' + main_id + '/' + n.formula + '.txt'
                df.to_pickle(arr_name)
    return 1


def copy_by_parent(new_id, report, img_true):
    ms_out = pd.read_csv(report)
    print(new_id)

    # iterate through ms_out, find img by formula, save in folder by id
    max_rows = ms_out.shape[0]
    counter = 0
    while counter < max_rows:
        ser = ms_out.iloc[counter, :]
        form = ser.formula
        par_id = ser.id_x
        name = ser.moleculeNames.join('_')
        ion_type = ser.par_frag

        if ion_type.find('p') != -1:
            out_ion = '_P'
        elif ion_type.find('f') != -1:
            out_ion = '_F'
        else:
            print('unknown ion type!')

        # Man edit line below for out path!
        outpath = 'any_results/by_id2_arr/' + new_id + '/' + par_id + '/'
        pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)

        if img_true == True:
            infile = glob.glob('any_results/formula/' +
                               new_id + '/' + form + '.png')[0]
            outfile = outpath + form + out_ion + '.png'

        else:
            infile = glob.glob('any_results/formula_arr/' +
                               new_id + '/' + form + '.txt')[0]
            outfile = outpath + form + out_ion + '.txt'

        # print(infile, '\n', outfile)
        copyfile(infile, outfile)
        counter += 1
    print(new_id, ' Counter: ', counter)
    return 1


def reporting_loop(input_ds_id, output_ds_id):
    # Call all the fun stuff above!

    # Loop to download images from METASPACE for datasets
    # Last arguement False == arrays, True = images
    dl_img(input_ds_id,
           output_ds_id,
           'any_ds_db_msms_2020_Apr_28',
           0.5,
           False)

    # Loop to group ion images or arrays by formula into by parent id
    msms_reports = glob.glob('any_results/msms_theo_man_report/*.csv')
    # msms_reports = glob.glob('any_results/msms_report/*.csv')
    for report in msms_reports:
        new_id = report.split('/')[-1]
        new_id = new_id.split('_msms_')[0]
        copy_by_parent(new_id, report, False)
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

    reporting_loop(args.limit_list,
                 args.out_name,
                 )

    print('Sirius spectra with formulas were converted to METASPACE MS1 and MS2 db')
    return 1

if __name__ == "__main__":
    main()