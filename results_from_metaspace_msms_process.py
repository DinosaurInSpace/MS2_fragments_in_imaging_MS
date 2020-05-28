#!/usr/bin/env python

"""
To-do:
1) Replace many static variables.
2) Fix main function at end

"""


import pandas as pd
import numpy as np
import argparse
import ast
from glob import glob
from matplotlib import pyplot as plt
import pathlib
from shutil import copyfile
import pickle

from metaspace.sm_annotation_utils import SMInstance
from metaspace.sm_annotation_utils import GraphQLClient



__author__ = "Christopher M Baxter Rath"
__copyright__ = "Copyright 2020"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"


def logon_metaspace(sm, path_to_pw='/Users/dis/.metaspace.json'):
    # Logs onto METASPACE server secretly
    f = open(path_to_pw, "r")
    secret = (f.read()).replace('\n', '')
    secret = ast.literal_eval(secret)
    f.close()

    sm.login(secret['email'], secret['password'])
    return sm


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


def extract_results_metaspace(ds_id, ms_out):
    # Read METASPACE output from API

    ms_out = ms_out[(ms_out.adduct == '[M]+')|(ms_out.adduct == '[M]-')]
    ms_out['n_ids_formula'] = ms_out.moleculeIds.apply(lambda x: len(x))
    gc = ['formula', 'adduct', 'mz', 'fdr', 'moleculeNames']
    ms_out = ms_out[gc]
    ms_out['ds_id'] = ds_id
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
    gc = ['id', 'par_frag', 'name', 'ds_id', 'formula',
          'adduct', 'mz', 'fdr',
          'parent_y', 'n_frag_y']
    ms_out = ms_out[gc].copy(deep=True)

    # Label with number of degenerate formulas at each mass
    df = ms_out[['id', 'formula']]
    df = df.groupby('formula').nunique()
    df = df.iloc[:, 0:1]
    ms_out = ms_out.merge(df, on='formula', how='left')
    ms_out = ms_out.iloc[:,0:10]

    return ms_out


def metaspace_hotspot_removal(img):
    awkwardness = np.max(img) / 255 # METASPACE holdover from 8-bit
    hot_thresold = np.percentile(img[img >= awkwardness], 99) or 1
    return np.clip(img, 0, hot_thresold) / hot_thresold


def dl_img(ds,
           primary_id,
           db,
           fdr_max,
           out_path,
           save_img=True):
    # Generate and save images from METASPACE to local by formula
    x = ds.all_annotation_images(fdr=fdr_max,
                                 database=db,
                                 only_first_isotope=True,
                                 scale_intensity=False,
                                 hasHiddenAdduct=True)
    if x == []:
        return 'Error, empty annotations!'
    else:
        imgs = {}
        for n in x:
            pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
            image = metaspace_hotspot_removal(n._images[0])
            if save_img == True:
                out_name = out_path + n.formula + '.png'
                plt.imsave(out_name, image)
            else:
                # For saving arrays as df per Theo's request
                out_name = out_path + n.formula + '.pickle'
                with open(out_name, 'wb') as f:
                    pickle.dump(image, f)
            imgs[n.formula] = out_name
    return imgs


def copy_by_parent(img_dict, new_id, report, out_path, save_image=True):
    # Organizes images by formula into folder by parent id

    ms_out = pd.read_pickle(report)
    if save_image == True:
        extension = '.png'
    else:
        extension = '.pickle'

    # iterate through ms_out, find img by formula, save in folder by id
    max_rows = ms_out.shape[0]
    counter = 0
    while counter < max_rows:
        ser = ms_out.iloc[counter, :]

        if ser.par_frag.find('p') != -1:
            out_ion = '_P'
        elif ser.par_frag.find('f') != -1:
            out_ion = '_F'
        else:
            print('unknown ion type!')

        out_by_id = out_path + new_id + '/by_id/' + ser.id_x + '/'
        pathlib.Path(out_by_id).mkdir(parents=True, exist_ok=True)

        infile = img_dict[report][ser.formula]
        outfile = out_by_id + ser.formula + out_ion + extension
        # print(infile, '\n', outfile)
        copyfile(infile, outfile)
        counter += 1
    print(new_id, ' Counter: ', counter)
    return 1


def reporting_loop(ori_ds_id,
                   db_id,
                   msms_ds_id,
                   out_path,
                   parent_and_fragment_req=True,
                   fdr_max=0.5,
                   save_image=True):
    # Access server and logon!
    sm = SMInstance(host='https://beta.metaspace2020.eu')
    sm = logon_metaspace(sm)

    # Accesses results with target db and parses
    ds = sm.dataset(id=msms_ds_id)
    results_df = ds.results(database=db_id).reset_index()
    results_df = extract_results_metaspace(msms_ds_id, results_df)


    if parent_and_fragment_req == True:
        results_df = results_df[(results_df.parent_y == 1) &
                                (results_df.n_frag_y > 0)]

    pathlib.Path(out_path + msms_ds_id + '/').mkdir(parents=True, exist_ok=True)
    out_df = out_path + msms_ds_id + '/' + "ms2_" + msms_ds_id + "_db_" + db_id + "_ms1_" + ori_ds_id + '.pickle'
    results_df.to_pickle(out_df)

    # Loop to download images from METASPACE for datasets
    img_dict = {}
    img_dict[out_df] = dl_img(ds,
                              msms_ds_id,
                              db_id,
                              fdr_max,
                              out_path + msms_ds_id + '/by_formula/',
                              save_image)

    # Loop to group ion images or arrays by formula into by parent id
    copy_by_parent(img_dict, msms_ds_id, out_df, out_path, save_image)

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