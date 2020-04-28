#!/usr/bin/env python

"""

METASPACE is a powerful online engine for processing imaging mass spectrometry data.
https://www.biorxiv.org/content/10.1101/539478v1
https://metaspace2020.eu/

A challenge for any analysis of imaging mass spectrometry experiment is assigning ions
with confidence.  Here we assume that at least some ions in the dataset are due to in-
source fragmentation or neutral losses.  This then let's us assign these ions, and
ideally increase the confidence for which the parent ion can be assigned.  Fragments
are identified from authentic standard data in reference MS/MS datasets, and then
assigned formulas with the tool "Sirius".
https://www.nature.com/articles/s41592-019-0344-8

Databases of MS/MS fragments are then exported to METASPACE for searching.

The workflow will consist of the following steps:
1. Load database of ions such as core_metabolome_v3.
2. Search for and collect authentic standard spectra in GNPS (https://gnps.ucsd.edu/).
3. Search for and download authentic standard spectra in MONA (https://mona.fiehnlab.ucdavis.edu/).
4. Submit standard spectra to Sirius via command line.
5. Collect and parse .json Sirius tree outputs to MS/MS spectra.
6. Filter spectral database for observed ions in standard METASPACE search.
7. Output dataset specific database to METASPACE for MS/MS ion searching.

# Identify other missed peaks (...) ?

Future:
1. Score colocalization
2. Extract Mona negative hits
3. Implement more adducts in sirius input.  Currently [M+, M+H, M+Na, M+K] next: M-H, M-
4. Are smiles search and inchi search close enough?
5. Search by name
6. Rescue GNPS in silico lipids from PNNL:
    the library membership is
    PNNL-LIPIDS-POSITIVE
    PNNL-LIPIDS-NEGATIVE
    --> Need structures somehow for this!  "Few weeks" Ming

Usage:

1. Steps 1-5 need only be run when a database is changed.
2. Steps 6-7 should be run with each experimental database.
3. The workflow can be run interactively from the following Jupyter notebook:
    http://localhost:8888/notebooks/PycharmProjects/word2vec/database_expt_msms_to_METASPACE.ipynb

4. The workflow can be run from the command line as follows:

python main_functions <bunch of arguements>

Output:

Path to plots, number of FBMN features, number of FBMN features within mz/rt tolerance.

"""

import pandas as pd
import numpy as np
import argparse
import glob
import os
import json
import mona
# import subprocess
# from subprocess import call
from string import digits
import re
from shutil import copyfile
import shlex
from subprocess import Popen, PIPE
from threading import Timer

# RDKit
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from molmass import Formula


__author__ = "Christopher M Baxter Rath"
__copyright__ = "Copyright 2020"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"


def load_molecule_database(path_to_db):
    # Should be pickle
    ref_db = pd.read_pickle(path_to_db)
    if list(ref_db) != ['id', 'name', 'formula', 'inchi']:
        return "Check that df columns are: ['id', 'name', 'formula', 'inchi']"
    else:
        ref_db = ref_db.reset_index(drop=True)
        return ref_db


def lipid_name(d):
    x = d[0]['Compound_Name'].split(' ')[2]
    return x


def formula_writer(df):
    # Writes formula for PNNL-LIPIDS database from incorrect field
    df = df.copy(deep=True)
    df['Formula_smiles'] = df.annotation_history.apply(lambda x: lipid_name(x))
    return df


def parse_GNPS_json(GNPS_json):
    # Parses experimental data and filters for acceptable Sirius input (high-res)
    GNPS_json = '/Users/dis/PycharmProjects/word2vec/ALL_GNPS.json'
    with open(GNPS_json, "r") as read_file:
        GNPS_data = json.load(read_file)

    GNPS_df = pd.DataFrame(GNPS_data)

    high_res_insts = ['ESI-QFT', 'HCD; Velos', 'qTof', 'LC-ESI-ITFT',
                      'Maxis II HD Q-TOF Bruker', 'LC-ESI-QFT',
                      'LC-ESI-QTOF', 'Orbitrap', 'ESI-QTOF',
                      'Q-TOF', 'LC-Q-TOF/MS', ' impact HD', 'CID; Lumos',
                      'HCD; Lumos', 'Q-Exactive Plus', 'qToF',
                      'Hybrid FT', 'Q-Exactive Plus Orbitrap Res 70k',
                      'LC-ESI-TOF', 'Maxis HD qTOF', 'LTQ-FT-ICR',
                      'IT-FT/ion trap with FTMS', 'qTOF',
                      'Q-Exactive Plus Orbitrap Res 14k',
                      'Orbitrap Elite', 'APCI-ITFT', 'ESI-ITFT',
                      'Hybrid Ft', 'LC-ESI-ITTOF', 'LTQ-FTICR',
                      'TOF', 'QIT-TOF', 'FT', 'Maxis', 'ESI-FTICR',
                      'HPLC-ESI-TOF', 'UPLC-ESI-QTOF',
                      'QIT-FT', 'Q-Exactive']

    # Fix METASPACE syntax
    good_adducts = ['M+H', '[M+H]+', '[M-H]-', 'M-H', 'M+NH4', '[M+Na]+',
                    '[M+K]+', '[M+NH4]+', '[M+H]', 'M+Na', 'M-H2O+H',
                    '[M-H]', 'M+H-H2O', '[M+Na]', 'M+K', '[M]+',
                    'M+', 'M+Cl', '[M+NH4]', 'M-H2O-H', 'M',
                    'M+H-NH3', 'M-', '[M]-', '[M+Cl]', '[M+K]',
                    'M+Na+H2O', 'M-H2O', '[M+H-H2O]+', '[M-1]-',
                    'M-H-H2O', 'M-H-', 'M+H+H2O',
                    ' M+H', 'M+H-H20']

    good_pol = ['Positive', 'Negative', 'positive', 'negative',
                'Positive-20eV', 'Negative-20eV', 'Positive-10eV',
                'Positive-40eV', 'Negative-40eV', 'Positive-0eV',
                ' Negative', ' Positive']

    good_cols = ['Compound_Name', 'spectrum_id', 'peaks_json',
                 'Adduct', 'Smiles', 'INCHI', 'Ion_Mode',
                 'library_membership']

    rescue_libaries = ['GNPS-EMBL-MCF', 'PNNL-LIPIDS-POSITIVE']

    good_libs = ['GNPS-LIBRARY', 'GNPS-EMBL-MCF',
             'MMV_POSITIVE', 'MMV_NEGATIVE',
             'LDB_POSITIVE', 'LDB_NEGATIVE',
             'GNPS-NIST14-MATCHES', 'GNPS-COLLECTIONS-MISC',
             'GNPS-MSMLS', 'BILELIB19',
             'PNNL-LIPIDS-POSITIVE', 'PNNL-LIPIDS-NEGATIVE',
             'MIADB', 'MASSBANK', 'MASSBANKEU', 'MONA',
             'RESPECT', 'HMDB', 'CASMI', 'SUMNER']

    GNPS_df.Smiles = GNPS_df.Smiles.replace(['N/A', ' ', 'NA', 'c', 'n/a'], 'x')
    GNPS_df.INCHI = GNPS_df.INCHI.replace(['N/A', ' ', ''], 'x')

    GNPS_df['temp'] = GNPS_df.Smiles + GNPS_df.INCHI
    GNPS_df1 = GNPS_df[GNPS_df.temp != 'xx']
    GNPS_df2 = GNPS_df[GNPS_df.library_membership == 'PNNL-LIPIDS-POSITIVE']
    GNPS_df2 = formula_writer(GNPS_df2)
    GNPS_df = pd.concat([GNPS_df1, GNPS_df2])

    # Turn the next line off to include PNNL_lipids without structures.
    GNPS_df = GNPS_df[GNPS_df.temp != 'xx']

    GNPS_df = GNPS_df[(GNPS_df.ms_level == '2')]
    GNPS_df = GNPS_df[(GNPS_df.Instrument.isin(high_res_insts))]
    GNPS_df = GNPS_df[GNPS_df.Adduct.isin(good_adducts)]
    GNPS_df = GNPS_df[GNPS_df.Ion_Mode.isin(good_pol)]

    GNPS_df['Ion_Mode'] = GNPS_df.Ion_Mode.replace(['Positive', 'Positive-20eV',
                                                 'Positive-10eV', 'Positive-40eV',
                                                 'Positive-0eV', ' Positive'], 'positive')

    GNPS_df.Ion_Mode = GNPS_df.Ion_Mode.replace(['Negative', 'Negative-20eV',
                                                 'Negative-40eV', ' Negative'], 'negative')

    adduct_dict = {'M+H': ['[M+H]+', ' M+H', '[M+H]'],
                   'M-H': ['[M-H]-', '[M-H]', 'M-H-', '[M-1]-'],
                   'M+Na': ['[M+Na]+', '[M+Na]'],
                   'M+K': ['[M+K]+', '[M+K]'],
                   'M+NH4+': ['[M+NH4]+', '[M+NH4]'],
                   'M+': ['[M]+', 'M'],
                   'M+Cl': ['[M+Cl]', ],
                   'M-': ['[M]-'],
                   'M-H2O+H': ['M+H-H20', 'M+H-H2O', '[M+H-H2O]+'],
                   'M+H2O+H': ['M+H+H2O'],
                   'M-H2O+': ['M-H2O'],
                   'M-H2O-H': ['M-H-H2O'],
                   'M-NH3+H': ['M+H-NH3'],
                   'M+H2O+Na': ['M+Na+H2O']}

    for k, v in adduct_dict.items():
        GNPS_df.Adduct = GNPS_df.Adduct.replace(v, k)

    GNPS_df = GNPS_df[good_cols].copy(deep=True)
    return GNPS_df


def can_no_stereo_smiles_from_mol(molecule):
    # Removes sterochemistry for matching MS/MS spectra and returns SMILES
    Chem.RemoveStereochemistry(molecule)
    out_smiles = Chem.MolToSmiles(molecule)
    return out_smiles


def molecule_from_gnps(s):
    # Generates canonical smiles without stereochem from smiles or inchi for GNPS_df
    if s.Smiles == 'x' and s.INCHI == 'x':
        print('Some GNPS entires have neither INCHI nor Smiles structures')
        exit(1)
    elif s.Smiles != 'x':
        mol = Chem.MolFromSmiles(s.Smiles)
    else:
        mol = Chem.inchi.MolFromInchi(s.INCHI)

    if mol == None:
        return None
    else:
        can_smiles = can_no_stereo_smiles_from_mol(mol)
        return can_smiles


def search_GNPS_targets(ref_db, GNPS_df):
    # Returns subset of GNPS df with matching canonical smiles to ref_db
    GNPS_df['can_smiles'] = GNPS_df.apply(lambda x: molecule_from_gnps(x), axis=1)
    GNPS_df = GNPS_df[GNPS_df.can_smiles != None]
    ref_db['can_smiles'] = ref_db['inchi'].apply(lambda x: can_no_stereo_smiles_from_mol(Chem.inchi.MolFromInchi(x)))
    db_can_smiles = list(ref_db['can_smiles'].unique())
    GNPS_hits_df = GNPS_df[GNPS_df.can_smiles.isin(db_can_smiles)]
    return GNPS_hits_df


def pd_tidy(in_str, df):
    t_dict = dict(zip(df['name'], df['value']))
    return t_dict[in_str]


def search_MONA(ref_db):
    mona_df = pd.DataFrame()
    inchi_ref_list = list(ref_db.inchi.unique())
    counter = 0

    for inchi_x in inchi_ref_list:
        counter += 1
        print(inchi_x)
        print(counter)
        print()

        spectral_hits = mona.mona_main(inchi_x, 'MS2', 'positive')

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
    return mona_df


def parse_MONA_out(Mona_hits_df):
    # Filters dataset for high mass accuracy instruments and <20 ppm mass error
    # Filters and renames adducts
    good_instruments = ['Q Exactive Plus Orbitrap Thermo Scientific',
                        'Thermo Q Exactive HF', 'Agilent QTOF 6550', 'UPLC Q-Tof Premier, Waters',
                        'Q-Tof Premier, Waters', 'Orbitrap',
                        'maXis plus UHR-ToF-MS, Bruker Daltonics', 'Q-Exactive Plus',
                        'SCIEX TripleTOF 6600', 'Agilent 6530 Q-TOF',
                        'LTQ Orbitrap XL, Thermo Scientfic',
                        'Thermo Fisher Scientific Q Exactive GC Orbitrap GC-MS/MS',
                        '6530 QTOF LC/MS Agilent', 'LC-ESI-Q-Orbitrap',
                        'LTQ Orbitrap Velos Thermo Scientific', 'Bruker maXis Impact',
                        'LTQ Orbitrap XL Thermo Scientific', 'Agilent 6550 iFunnel',
                        'maXis (Bruker Daltonics)', 'QTOF Premier',
                        '6550 Q-TOF (Agilent Technologies)',
                        'LTQ Orbitrap XL, Thermo Scientfic; HP-1100 HPLC, Agilent',
                        'LC, Waters Acquity UPLC System; MS, Waters Xevo G2 Q-Tof',
                        'API QSTAR Pulsar i', 'Q Exactive Orbitrap Thermo Scientific',
                        'micrOTOF-Q', 'Micromass Q-TOF II', 'Sciex Triple TOF 6600', 'qTof',
                        'Q-TOF', 'qToF', 'Q Exactive HF', 'Agilent 1200 RRLC; Agilent 6520 QTOF',
                        'Waters Xevo G2 Q-Tof', 'Maxis II HD Q-TOF Bruker', 'qTOF', 'LTQ-Orbitrap XL',
                        'JMS-S3000', 'Xevo G2-S QtOF, Waters (USA) coupled to ACQUITY UPLC, Waters (USA).',
                        'LCMS-IT-TOF', 'Hybrid FT']
    Mona_hits_df = Mona_hits_df[Mona_hits_df.instrument.isin(good_instruments)]
    Mona_hits_df = Mona_hits_df[Mona_hits_df.dppm != 'unknown']
    Mona_hits_df = Mona_hits_df[Mona_hits_df.dppm <= 20]

    adduct_dict = {'M+H': ['[M+H]+', ' M+H', '[M+H]'],
                   'M-H': ['[M-H]-', '[M-H]', 'M-H-', '[M-1]-'],
                   'M+Na': ['[M+Na]+', '[M+Na]'],
                   'M+K': ['[M+K]+', '[M+K]'],
                   'M+NH4+': ['[M+NH4]+', '[M+NH4]'],
                   'M+': ['[M]+', 'M'],
                   'M+Cl': ['[M+Cl]', ],
                   'M-': ['[M]-'],
                   'M-H2O+H': ['M+H-H20', 'M+H-H2O', '[M+H-H2O]+', '[M-H2O+H]+'],
                   'M+H2O+H': ['M+H+H2O'],
                   'M-H2O+': ['M-H2O'],
                   'M-H2O-H': ['M-H-H2O'],
                   'M-NH3+H': ['M+H-NH3'],
                   'M+H2O+Na': ['M+Na+H2O']}

    for k, v in adduct_dict.items():
        Mona_hits_df.adduct = Mona_hits_df.adduct.replace(v, k)

    good_adducts = list(adduct_dict.keys())
    Mona_hits_df = Mona_hits_df[Mona_hits_df.adduct.isin(good_adducts)]

    return Mona_hits_df

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


def parse_hmdb(directory, theo_flag=False):
    # Filters dataset for high mass accuracy instruments and <20 ppm mass error
    file_paths = os.listdir(directory)
    hpe_df = pd.DataFrame()
    hpe_df['file_paths'] = list(sorted(file_paths))
    hpe_df['id'] = hpe_df.file_paths.apply(lambda x: x.split('_')[0])

    if theo_flag is False:
        hpe_df['instrument'] = hpe_df.file_paths.apply(lambda x: get_att(x,
                                                                     '<instrument-type>',
                                                                     '</instrument-type>',
                                                                     directory))
    if theo_flag is True:
        hpe_df['instrument'] = 'predicted'

    hpe_df['polarity'] = hpe_df.file_paths.apply(lambda x: get_att(x,
                                                                   '<ionization-mode>',
                                                                   '</ionization-mode>',
                                                                   directory))

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

    hpe_df = hpe_df[hpe_df.polarity == 'Positive']
    hpe_df = hpe_df[hpe_df.instrument.isin(good_hmdb_intruments)]
    hpe_df['adduct'] = 'M+H'
    return hpe_df


def hmdb_theo_finder(ref_df, expt_df):
    # Finds theoretical MS/MS spectra only for files which do not have experimental
    # To delete, not used.
    pass
    return


def preparser_Sirius(ref_db, GNPS_hits_df, Mona_hits_df, HMDB_ex, HMDB_theo):
    # Cleans up ref db for only entries with experimental MS/MS spectra
    df = ref_db[['id', 'name', 'formula', 'inchi', 'can_smiles']].copy(deep=True)
    df['temp'] = list(df.index)
    df['db_index'] = df.temp.astype(str) + '_' + df.id + '_' + df.name
    df['db_index'] = df['db_index'].str.replace(' ', '_')
    df['db_index'] = df['db_index'].str.replace('[^a-zA-Z0-9_]', '')
    m = list(Mona_hits_df.inchi)
    g = list(GNPS_hits_df.can_smiles)
    he = list(HMDB_ex.id)
    ht = list(HMDB_theo.id)
    df1 = df[(df.inchi.isin(m) |
              df.can_smiles.isin(g) |
              df.id.isin(he))].copy(deep=True)

    # Only load predicted MS/MS spectra for samples without experimentaldata
    expt_ids = list(df1.id)
    df2 = df[~df.id.isin(expt_ids)]
    df2 = df2[df2.id.isin(ht)]
    expt_df = df1
    theo_df = df2
    return [df1, df2]


def hmdb_spectra_copy(df, path, spectral_paths, ex_flag):
    # parses HMDB spectra file from xml
    if ex_flag is True:
        from_path = '/Users/dis/Desktop/hmdb_LCMS/hmdb_experimental_msms_spectra'
    else:
        from_path = '/Users/dis/Desktop/hmdb_LCMS/hmdb_predicted_msms_spectra'

    for file in df.file_paths:
        f = from_path + '/' + file
        #print(f)
        mz_list = []
        int_list = []
        with open(f, 'r') as input_file:
            lines = input_file.readlines()
            for line in lines:
                if '<mass-charge>' in line:
                    mz = line.split('>')[1]
                    mz = mz.split('<')[0]
                    mz_list.append(float(mz))
                elif '<intensity>' in line:
                    i = line.split('>')[1]
                    i = i.split('<')[0]
                    int_list.append(float(i))
                else:
                    continue

        out_file = path + file.split('.')[0] + '.txt'
        spectral_paths.append(out_file)
        mz_i_list = zip(mz_list, int_list)
        with open(out_file, 'w+') as f:
            for mz_i in mz_i_list:
                to_write = str(mz_i[0]) + ' ' + str(mz_i[1]) + '\n'
                f.write(to_write)

    return spectral_paths


def ms_format_writer(m_df, g_df, e_df, t_df, db_index, add):
    # Writes the GNPS json and the Mona text spectra to file, returning paths
    # Copies HMDB spectra files to directory
    out_path = '/Users/dis/PycharmProjects/word2vec/spectra/'
    spectral_paths = []
    counter = 0
    path = out_path + db_index + '/'

    # Errors if directory already exists
    try:
        os.mkdir(path)
    except:
        pass

    if not m_df.empty:
        for s in list(m_df.spectrum):
            counter += 1
            s = s.replace(' ', '\n')
            s = s.replace(':', ' ')
            o = out_path + db_index + '/' + add + '_' + str(counter) + '.txt'
            o = o.replace('+', 'p')
            o = o.replace('-', 'n')
            spectral_paths.append(o)
            with open(o, 'w+') as out_file:
                out_file.write(s)
    else:
        pass

    if not g_df.empty:
        for s in list(g_df.peaks_json):
            counter += 1
            s = s.replace('],[', '\n')
            s = s.replace('], [', '\n')
            s = s.replace(',', ' ')
            s = s.replace('[[', '')
            s = s.replace(']]', '')
            o = out_path + db_index + '/' + add + '_' + str(counter) + '.txt'
            o = o.replace('+', 'p')
            o = o.replace('-', 'n')
            spectral_paths.append(o)
            spectral_paths.append(o)
            with open(o, 'w+') as out_file:
                out_file.write(s)
    else:
        pass

    if not e_df.empty:
        spectral_paths = hmdb_spectra_copy(e_df,
                                           path,
                                           spectral_paths,
                                           True)
    else:
        pass

    if not t_df.empty:
        spectral_paths = hmdb_spectra_copy(t_df,
                                           path,
                                           spectral_paths,
                                           False)
    else:
        pass

    return spectral_paths


def adduct_translate(add):
    # Translates adduct from METASPACE vocab to Sirius equivalent.
    m_s_dict = {'M+': '[M]+', 'M+H': '[M+H]+', 'M+Na': '[M+Na]+', 'M+K': '[M+K]+',
                'M-H': '[M-H]-', 'M-': '[M]-'}
    return m_s_dict[add]


def exists(ms_path):
    if ms_path is None:
        return 0
    else:
        target = glob.glob(ms_path)
        # print('target, 578:', target)
        if target != []:
            return glob.glob(ms_path)[0]
        else:
            return 0


def run(cmd, timeout_sec):
    # Timeout test
    # https://stackoverflow.com/questions/1191374/using-module-subprocess-with-timeout

    proc = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    timer = Timer(timeout_sec, proc.kill)
    try:
        timer.start()
        stdout, stderr = proc.communicate()

    finally:
        timer.cancel()

    if stdout.decode("utf-8") == '':
        return ('failed')

    else:
        return stdout.decode("utf-8")


def loop_Sirius(df,
                Mona_hits_df,
                GNPS_hits_df,
                hmdb_pos_exptl_df,
                hmdb_pos_theo_df):
    # Main loop for running Sirius
    # Have to delete old spectra and trees before rerunning

    sirius_output_dict = {}
    mona_df = Mona_hits_df[['inchi', 'adduct', 'spectrum']].copy(deep=True)
    gnps_df = GNPS_hits_df[['can_smiles', 'Adduct', 'peaks_json']].copy(deep=True)
    hex_df = hmdb_pos_exptl_df[['id', 'adduct', 'file_paths']].copy(deep=True)
    hth_df = hmdb_pos_theo_df[['id', 'adduct', 'file_paths']].copy(deep=True)

    # If positive mode
    current_adducts = ['M+', 'M+H', 'M+Na', 'M+K']
    mona_df = mona_df[mona_df.adduct.isin(current_adducts)]
    gnps_df = gnps_df[gnps_df.Adduct.isin(current_adducts)]
    hex_df = hex_df[hex_df.adduct.isin(current_adducts)]
    hth_df = hth_df[hth_df.adduct.isin(current_adducts)]


    # Loops over dataframe
    index_list = list(df.index)
    for idx in index_list:
        ser = df.loc[idx]
        print('\n', 'series', idx, ser.db_index)
        formula = ser.formula
        db_index = ser.db_index

        mo_df = mona_df[mona_df.inchi == ser.inchi]
        gn_df = gnps_df[gnps_df.can_smiles == ser.can_smiles]
        he_df = hex_df[hex_df.id == ser.id]
        ht_df = hth_df[hth_df.id == ser.id]

        unique_adducts = list(set(list(mo_df.adduct)
                                  + list(gn_df.Adduct)
                                  + list(he_df.adduct)
                                  + list(ht_df.adduct)))

        add_counter = 0
        for add in unique_adducts:
            print(idx, ' ', add)
            add_counter += 1
            output_dir = '/Users/dis/PycharmProjects/word2vec/trees/' + db_index

            m_df = mo_df[mo_df.adduct == add]
            g_df = gn_df[gn_df.Adduct == add]
            e_df = he_df[he_df.adduct == add]
            t_df = ht_df[ht_df.adduct == add]

            print('m_df:', len(list(m_df.spectrum)),
                  'g_df', len(list(g_df.peaks_json)),
                  'e_df', len(list(e_df.file_paths)),
                  't_df', len(list(t_df.file_paths)))

            # Make copy function!
            spectra_list = ms_format_writer(m_df,
                                            g_df,
                                            e_df,
                                            t_df,
                                            db_index,
                                            add)

            t_add = adduct_translate(add)
            sirius_input = runner_Sirius(formula,
                                         t_add,
                                         spectra_list,
                                         output_dir,
                                         db_index)

            # Run with timeout as Sirius chokes >120 mins in some large compounds
            #sirius_output = subprocess.check_output(sirius_input)
            sirius_input = " ".join(sirius_input)
            sirius_output = run(sirius_input, 180)

            #sirius_output = sirius_output.decode('utf-8')
            # print('sirius_output:', '\n', sirius_output, '\n',)
            sirius_output_dict[db_index] = output_Sirius_parser(sirius_output,
                                                                output_dir,
                                                                db_index,
                                                                add_counter)
    print('sirius_output_dict:', sirius_output_dict)
    sirius_output_df = pd.DataFrame.from_dict(sirius_output_dict,
                                              orient='index',
                                              columns=['file'])
    sirius_output_df['exists'] = sirius_output_df['file'].apply(lambda x: exists(x))
    # print('Sirius success: ', sirius_output_df.exists.value_counts())
    sirius_output_df = sirius_output_df[sirius_output_df.exists != 0]

    return sirius_output_df


def runner_Sirius(formula, ion, spectra_list, output_dir, db_index):
    # Generates query for Sirius and runs on command line.
    '''
    binary = '/Users/dis/PycharmProjects/word2vec/sirius/bin/sirius'
    s = ' '
    spectra_query = ' -2 ' + s.join(spectra_list)
    query = binary + ' -f ' + formula + ' -i ' + ion + spectra_query + ' -o ' + output_dir + '/' + db_index
    print('query: ', query, '\n', type(query))
    return query
    '''

    query_list = ['/Users/dis/PycharmProjects/word2vec/sirius/bin/sirius',
                  '-f',
                  formula,
                  '-i',
                  ion,
                  '-2']
    for s in spectra_list:
        query_list.append(s)

    query_list.append('-o')
    query_list.append(output_dir + '/' + db_index)
    #print(query_list)
    return(query_list)


def output_Sirius_parser(sirius_output, output_dir, db_index, n):
    # Parses Sirius output returning paths to ms file with formulas
    # print('sirius_output:', sirius_output, '\n', type(sirius_output))

    try:
        x = sirius_output.split("Sirius results for")[1]
        x = x.split(".txt")[0]
        x = x.split(": '")[1]
        x = str(x)
        # print('x', x)

    except:
        return None


    search_string = output_dir + '/' + db_index + '/' + str(n) + '_' + x + '_/spectra/*.ms'
    # print('search_string:', search_string)
    return search_string


def ms_pd_reader(ms_path, db_index):
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
        if ctr % 500 == 0:
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
        print(formula)
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
        print('Error!')
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
    # Will need to be updated for new adducts and - mode
    # print(formula, '\t', type(formula), '\t', dmass)

    if dmass == 0:
        formula = formula
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


def mass_check(em, im):
    if em - im <= 0.001:
        return True
    else:
        return False


def results_clean_up(has_MS2_df, sirius_output_df):
    # Merges Sirius results and metadata
    MS2_meta_df = pd.merge(has_MS2_df,
                           sirius_output_df,
                           how='inner',
                           left_on='db_index',
                           right_index=True)

    # Joins MS2 spectra to metadata
    pre_METASPACE_df = df_merge(MS2_meta_df)
    pre_METASPACE_df = pre_METASPACE_df.dropna()
    pre_METASPACE_df['expl_ex'] = pre_METASPACE_df.explanation.apply(lambda x:
                                                                     ex(x))
    pre_METASPACE_df['dmass'] = pre_METASPACE_df['exactmass'] - pre_METASPACE_df['expl_ex']
    pre_METASPACE_df = pre_METASPACE_df.astype({'dmass': int})
    pre_METASPACE_df = pre_METASPACE_df[pre_METASPACE_df.dmass >= 0]
    pre_METASPACE_df[['formula', 'explanation', 'exactmass', 'expl_ex', 'dmass']]

    print('Observed ions: \n', pre_METASPACE_df['dmass'].value_counts())

    pre_METASPACE_df = pre_METASPACE_df.dropna()
    pre_METASPACE_df['H_check'] = pre_METASPACE_df.explanation.str.contains('H')
    df = pre_METASPACE_df
    df = df[~((df.dmass == 38) & (df.H_check == False))]
    df = df[~((df.dmass == 22) & (df.H_check == False))]
    pre_METASPACE_df = df

    pre_METASPACE_df['ion_formula'] = pre_METASPACE_df.apply(lambda x:
                                                             ion_form(x.explanation,
                                                                      x.dmass), axis=1)

    pre_METASPACE_df['bad_if'] = pre_METASPACE_df.ion_formula.str.contains('-')
    pre_METASPACE_df = pre_METASPACE_df[pre_METASPACE_df.bad_if == False]

    pre_METASPACE_df = pre_METASPACE_df.copy(deep=True)
    pre_METASPACE_df['ion_mass'] = pre_METASPACE_df.ion_formula.apply(lambda x: ionmasspos(x))

    df = pre_METASPACE_df
    df['good_mass_calc'] = df.apply(lambda x: mass_check(x.exactmass,
                                                         x.ion_mass),
                                    axis=1)
    pre_METASPACE_df = df
    print('Correct ionformulas: \n', pre_METASPACE_df.good_mass_calc.value_counts())
    return pre_METASPACE_df


def f_or_p_ion(d, p):
    if d == p:
        return 'p'
    else:
        return 'f'


def output_METASPACE(pre_METASPACE_df):
    # Filters columns, renames, and prepares for export to METASPACE

    df = pre_METASPACE_df[['explanation', 'formula', 'id', 'name',
                           'ion_formula', 'inchi']].copy(deep=True)
    df['f_num'] = df.groupby(['name']).cumcount() + 1
    df['f_or_p'] = df.apply(lambda x: f_or_p_ion(x.explanation,
                                                 x.formula), axis=1)
    df['out_id'] = df.id + '_' + df.f_num.astype(str) + df.f_or_p
    df['out_name'] = df.out_id + '_' + df.name
    df_prefilter = df[['out_id', 'out_name', 'ion_formula', 'inchi', 'id']]
    df_prefilter = df_prefilter.rename(columns={'out_id': 'id', 'out_name': 'name',
                                                'ion_formula': 'formula', 'id': 'old_id'})
    return df_prefilter


def api_upload_db_METASPACE(path, database_msms):
    # Waiting on development from Vitally
    pass
    return


def split_data_frame_list(df, target_column):
    # Accepts a column with multiple types and splits list variables to several rows.

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
    # Generated at 4/17/2020 12:08:16 PM. For help see https://bit.ly/2HO2uz4
    # URL: https://metaspace2020.eu/annotations?db=whole_body_MSMS_test_v3&prj=a493c7b8-e27f-11e8-9d75-3bb2859d3748&ds=2017-05-17_19h49m04s&fdr=0.5&sort=-mz&hideopt=1&sections=3&page=7

    ms_out = pd.read_csv('metaspace_annotations.csv', header=2)
    ms_out = ms_out[ms_out.adduct == 'M[M]+']
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

    return


def main():
    ### Main ###
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--path", default='networks/', type=str, help="Directory with GNPS output")
    parser.add_argument("--mztol", default=20, type=int, help="Mass error in ppm")
    parser.add_argument("--rttol", default=20, type=int, help="RT tolerance in seconds")
    parser.add_argument("--output_path", default='reports/', type=str, help="output directory")
    args = parser.parse_args()

    input_filenames = glob.glob(os.path.join(args.path, "*"))
    if len(input_filenames):
        print('Incorrect input directory or empty directory!')
        exit(1)

    for input_filename in input_filenames:
        output_filename = os.path.join(args.output_path, os.path.basename(input_filename) + ".png")
        fbmn_evaluate(input_filename, output_filename)

    # Run all the steps above...


if __name__ == "__main__":
    main()