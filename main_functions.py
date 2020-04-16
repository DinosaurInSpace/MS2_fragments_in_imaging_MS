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
1. Future: HMDB spectra, lipid spectra (lipidblast?) simulated from somewhere
2. Score colocalization
3. Extract Mona negative hits
4. Implement more adducts in sirius input.  Currently [M+, M+H, M+Na, M+K] next: M-H, M-
5. Are smiles search and inchi search close enough?
6. Search by name
8. Rescue GNPS in silico lipids from PNNL:
    the library membership is
    PNNL-LIPIDS-POSITIVE
    PNNL-LIPIDS-NEGATIVE

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
import subprocess

# RDKit
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem


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
                 'Adduct', 'Smiles', 'INCHI', 'Ion_Mode']

    GNPS_df.Smiles = GNPS_df.Smiles.replace(['N/A', ' ', 'NA', 'c', 'n/a'], 'x')
    GNPS_df.INCHI = GNPS_df.INCHI.replace(['N/A', ' ', ''], 'x')

    GNPS_df['temp'] = GNPS_df.Smiles + GNPS_df.INCHI
    GNPS_df = GNPS_df[GNPS_df.temp != 'xx']

    GNPS_df = GNPS_df[(GNPS_df.ms_level == '2') &
                      (GNPS_df.Instrument.isin(high_res_insts)) &
                      GNPS_df.Adduct.isin(good_adducts) &
                      GNPS_df.Ion_Mode.isin(good_pol)
                      ].copy(deep=True)

    GNPS_df.Ion_Mode = GNPS_df.Ion_Mode.replace(['Positive', 'Positive-20eV',
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


def preparser_Sirius(ref_db, GNPS_hits_df, Mona_hits_df):
    # Cleans up ref db for only entries with experimental MS/MS spectra
    df = ref_db[['id', 'name', 'formula', 'inchi', 'can_smiles']].copy(deep=True)
    df['temp'] = list(df.index)
    df['db_index'] = df.temp.astype(str) + '_' + df.id + '_' + df.name
    df['db_index'] = df['db_index'].str.replace(' ', '_')
    df['db_index'] = df['db_index'].str.replace('[^a-zA-Z0-9_]', '')
    m = list(Mona_hits_df.inchi)
    g = list(GNPS_hits_df.can_smiles)
    df = df[(df.inchi.isin(m) | df.can_smiles.isin(g))].copy(deep=True)
    return df


def ms_format_writer(m_df, g_df, db_index, add):
    # Writes the GNPS json and the Mona text spectra to file, returning paths
    out_path = '/Users/dis/PycharmProjects/word2vec/spectra/'
    spectral_paths = []
    counter = 0
    path = out_path + db_index + '/'

    # Errors if directroy already exists
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

    return spectral_paths


def adduct_translate(add):
    # Translates adduct from METASPACE vocab to Sirius equivalent.
    m_s_dict = {'M+': '[M]+', 'M+H': '[M+H]+', 'M+Na': '[M+Na]+', 'M+K': '[M+K]+',
                'M-H': '[M-H]-', 'M-': '[M]-'}
    return m_s_dict[add]


def exists(ms_path):
    if os.path.isfile(ms_path):
        return 1
    else:
        return 0


def loop_Sirius(df, Mona_hits_df, GNPS_hits_df):
    # Main loop for running Sirius
    # Have to delete old spectra and trees before rerunning
    sirius_output_dict = {}
    mona_df = Mona_hits_df[['inchi', 'adduct', 'spectrum']].copy(deep=True)
    gnps_df = GNPS_hits_df[['can_smiles', 'Adduct', 'peaks_json']].copy(deep=True)

    # If positive mode
    current_adducts = ['M+', 'M+H', 'M+Na', 'M+K']
    mona_df = mona_df[mona_df.adduct.isin(current_adducts)]
    gnps_df = gnps_df[gnps_df.Adduct.isin(current_adducts)]

    # Loops over dataframe
    index_list = list(df.index)
    for idx in index_list:
        print('\n', idx)
        ser = df.loc[idx]
        formula = ser.formula
        inchi = ser.inchi
        can_smiles = ser.can_smiles
        db_index = ser.db_index
        mo_df = mona_df[mona_df.inchi == inchi]
        gn_df = gnps_df[gnps_df.can_smiles == can_smiles]
        unique_adducts = list(set(list(mo_df.adduct) + list(gn_df.Adduct)))
        print(unique_adducts)
        add_counter = 0
        for add in unique_adducts:
            print(idx, ' ', add)
            add_counter += 1
            output_dir = '/Users/dis/PycharmProjects/word2vec/trees/' + db_index
            m_df = mo_df[mo_df.adduct == add]
            g_df = gn_df[gn_df.Adduct == add]
            spectra_list = ms_format_writer(m_df, g_df, db_index, add)
            t_add = adduct_translate(add)
            sirius_input = runner_Sirius(formula, t_add, spectra_list, output_dir, db_index)
            # In Jupyter it was: "sirius_output = !{sirius_input}"
            sirius_output = subprocess.check_output([sirius_input])
            sirius_output_dict[db_index] = output_Sirius_parser(sirius_output, output_dir,
                                                                db_index, add_counter)

    sirius_output_df = pd.DataFrame.from_dict(sirius_output_dict, orient='index', columns=['file'])
    sirius_output_df['exists'] = sirius_output_df['file'].apply(lambda x: exists(x))
    print('Sirius success: ', sirius_output_df.exists.value_counts())
    sirius_output_df = sirius_output_df[sirius_output_df.exists == 1]

    return sirius_output_df


def runner_Sirius(formula, ion, spectra_list, output_dir, db_index):
    # Generates query for Sirius and runs on command line.
    binary = '/Users/dis/PycharmProjects/word2vec/sirius/bin/sirius'
    s = ' '
    spectra_query = ' -2 ' + s.join(spectra_list)
    query = binary + ' -f ' + formula + ' -i ' + ion + spectra_query + ' -o ' + output_dir + '/' + db_index
    return query


def output_Sirius_parser(sirius_output, output_dir, db_index, n):
    # Parses Sirius output returning paths to ms file with formulas
    idx = None
    for line in sirius_output:
        if 'Sirius results for' in line:
            idx = sirius_output.index(line)

    if idx is None:
        # Catch errors?
        print('No sirius output')
        #print(sirius_output)
        return None

    else:
        print('ran sirius')
        #print(sirius_output[idx:-1])

        short = sirius_output[idx]
        short = short.split('.')[0]
        short = short.split(" '")[1]
        idx += 1
        first_hit = sirius_output[idx]
        first_hit = first_hit.split('\tscore')[0]
        first_hit = first_hit.replace('.) ', '_')
        first_hit = first_hit.replace('\t', '_')
        first_hit = first_hit.replace('[', '')
        first_hit = first_hit.replace(']', '')
        first_hit = first_hit.replace(' ', '')
        first_hit = first_hit + '.ms'
        search_string = output_dir + '/' + db_index +'/' + str(n) + '_' + short + '_/spectra/' + first_hit
        #print(search_string)
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
    for idx in index_list:
        ser = input_df.loc[idx]
        df = ms_pd_reader(ser.file, ser.db_index)
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
    MS2_meta_df = pd.merge(has_MS2_df, sirius_output_df, how='inner',
                           left_on='db_index', right_index=True)

    # Joins MS2 spectra to metadata
    pre_METASPACE_df = df_merge(MS2_meta_df)
    pre_METASPACE_df = pre_METASPACE_df.dropna()
    pre_METASPACE_df['expl_ex'] = pre_METASPACE_df.explanation.apply(lambda x:
                                                                     ex(x))
    pre_METASPACE_df['dmass'] = pre_METASPACE_df['exactmass'] - pre_METASPACE_df['expl_ex']
    pre_METASPACE_df = pre_METASPACE_df.astype({'dmass': int})
    pre_METASPACE_df = pre_METASPACE_df[pre_METASPACE_df.dmass >= 0]
    pre_METASPACE_df[['formula', 'explanation', 'exactmass', 'expl_ex', 'dmass']]

    print('Observed ions: ', pre_METASPACE_df['dmass'].value_counts())

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
    print('Correct ionformulas: ', pre_METASPACE_df.good_mass_calc.value_counts())
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
    pass
    return


def extract_results_METASPACE(path, database_msms):
    pass
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