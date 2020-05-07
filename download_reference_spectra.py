#!/usr/bin/env python

"""
To execute after obtaining Mona neg!
"""

import pandas as pd
import argparse
import glob
import json

# RDKit
from rdkit import Chem

__author__ = "Christopher M Baxter Rath"
__copyright__ = "Copyright 2020"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"


def load_molecule_database(path_to_db):
    # Expected database format is pd.Dataframe() as pickle with columns below.
    ref_db = pd.read_pickle(path_to_db)
    if list(ref_db) != ['id', 'name', 'formula', 'inchi']:
        return "Check that df columns are: ['id', 'name', 'formula', 'inchi']"
    else:
        ref_db = ref_db.reset_index(drop=True)
        return ref_db


def parse_gnps_json(gnps_json):
    # Parses experimental data and filters for acceptable Sirius input (high-res)
    with open(gnps_json, "r") as read_file:
        gnps_df = pd.DataFrame(json.load(read_file))

    # Drops entries with no machine readable structures!
    gnps_df.Smiles = gnps_df.Smiles.replace(['N/A', ' ', 'NA', 'c', 'n/a'], 'x')
    gnps_df.INCHI = gnps_df.INCHI.replace(['N/A', ' ', ''], 'x')
    gnps_df['temp'] = gnps_df.Smiles + gnps_df.INCHI
    gnps_df = gnps_df[gnps_df.temp != 'xx']

    # Filter diverse syntax users input in GNPS
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
                      'QIT-FT', 'Q-Exactive'
                      ]

    good_adducts = ['M+H', '[M+H]+', '[M-H]-', 'M-H', 'M+NH4', '[M+Na]+',
                    '[M+K]+', '[M+NH4]+', '[M+H]', 'M+Na', 'M-H2O+H',
                    '[M-H]', 'M+H-H2O', '[M+Na]', 'M+K', '[M]+',
                    'M+', 'M+Cl', '[M+NH4]', 'M-H2O-H', 'M',
                    'M+H-NH3', 'M-', '[M]-', '[M+Cl]', '[M+K]',
                    'M+Na+H2O', 'M-H2O', '[M+H-H2O]+', '[M-1]-',
                    'M-H-H2O', 'M-H-', 'M+H+H2O',
                    ' M+H', 'M+H-H20'
                    ]

    good_pol = ['Positive', 'Negative', 'positive', 'negative',
                'Positive-20eV', 'Negative-20eV', 'Positive-10eV',
                'Positive-40eV', 'Negative-40eV', 'Positive-0eV',
                ' Negative', ' Positive'
                ]

    gnps_df = gnps_df[(gnps_df.ms_level == '2')]
    gnps_df = gnps_df[(gnps_df.Instrument.isin(high_res_insts))]
    gnps_df = gnps_df[gnps_df.Adduct.isin(good_adducts)]
    gnps_df = gnps_df[gnps_df.Ion_Mode.isin(good_pol)]

    # Replace various names with common name:
    gnps_df['Ion_Mode'] = gnps_df.Ion_Mode.replace(['Positive', 'Positive-20eV',
                                                    'Positive-10eV', 'Positive-40eV',
                                                    'Positive-0eV', ' Positive'], 'positive'
                                                   )

    gnps_df['Ion_Mode'] = gnps_df.Ion_Mode.replace(['Negative', 'Negative-20eV',
                                                    'Negative-40eV', ' Negative'], 'negative'
                                                   )

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
        gnps_df.Adduct = gnps_df.Adduct.replace(v, k)

    # Remove columns unexpected by downstream steps.
    good_cols = ['Compound_Name', 'spectrum_id', 'peaks_json',
                 'Adduct', 'Smiles', 'INCHI', 'Ion_Mode',
                 'library_membership'
                 ]

    gnps_df = gnps_df[good_cols].copy(deep=True)
    return gnps_df


def can_no_stereo_smiles_from_mol(molecule):
    # Removes sterochemistry for matching MS/MS spectra and returns SMILES
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule)


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
    GNPS_df['can_smiles'] = GNPS_df.apply(lambda x: molecule_from_gnps(x),
                                          axis=1)
    GNPS_df = GNPS_df[GNPS_df.can_smiles != None]
    ref_db['can_smiles'] = ref_db['inchi'].apply(lambda x: can_no_stereo_smiles_from_mol(Chem.inchi.MolFromInchi(x)))
    db_can_smiles = list(ref_db['can_smiles'].unique())
    GNPS_hits_df = GNPS_df[GNPS_df.can_smiles.isin(db_can_smiles)]
    return GNPS_hits_df


def parse_mona_out(df):
    # Filters MONA output for high mass accuracy instruments and <20 ppm mass error
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
    df = df[df.instrument.isin(good_instruments)]
    df = df[df.dppm != 'unknown']
    df = df[df.dppm <= 20]

    # Filters and renames adducts
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
        df.adduct = df.adduct.replace(v, k)
    df = df[df.adduct.isin(list(adduct_dict.keys()))]

    return df


def preparser_Sirius(ref_db, GNPS_df, Mona_df, HMDB_ex, HMDB_theo):
    # Makes new aggregated name descriptor
    df = ref_db[['id', 'name', 'formula', 'inchi', 'can_smiles']].copy(deep=True)
    df['temp'] = list(df.index)
    df['db_index'] = df.temp.astype(str) + '_' + df.id + '_' + df.name
    df['db_index'] = df['db_index'].str.replace(' ', '_')
    df['db_index'] = df['db_index'].str.replace('[^a-zA-Z0-9_]', '')

    # Cleans up ref db for only entries with experimental MS/MS spectra
    df1 = df[(df.inchi.isin(list(Mona_df.inchi)) |
              df.can_smiles.isin(list(GNPS_df.can_smiles)) |
              df.id.isin(list(HMDB_ex.id)))].copy(deep=True)

    # Only load predicted MS/MS spectra for samples without experimental data
    df2 = df[~df.id.isin(list(df1.id))]
    df2 = df2[df2.id.isin(list(HMDB_theo.id))]

    # Return experimental and theoretical ids
    return [df1, df2]


def output_loop(input_db,
                output_path,
                gnps_path,
                hmdb_pos_expt,
                hmdb_pos_theo,
                hmdb_neg_expt,
                hmdb_neg_theo,
                mona_pos_path,
                mona_neg_path):
    # Executes main loop to parse input database and pull spectra from local dumps.
    ref_db = load_molecule_database(input_db)
    gnps_df = parse_gnps_json(gnps_path)
    gnps_hits_df = search_GNPS_targets(ref_db, gnps_df)

    hmdb_exptl_df = pd.concat([pd.read_pickle(hmdb_pos_expt),
                               pd.read_pickle(hmdb_neg_expt)])
    hmdb_theo_df = pd.concat([pd.read_pickle(hmdb_pos_theo),
                              pd.read_pickle(hmdb_neg_theo)])

    mona_df = pd.concat([pd.read_pickle(mona_pos_path),
                         pd.read_pickle(mona_neg_path)])

    mona_hits_df = parse_mona_out(mona_df)

    ref_dfs = preparser_Sirius(ref_db,
                               gnps_hits_df,
                               mona_hits_df,
                               hmdb_exptl_df,
                               hmdb_theo_df
                               )

    ref_dfs[0].to_pickle(output_path + 'expt_df.pickle')
    ref_dfs[1].to_pickle(output_path + 'theo_df.pickle')
    print("Parse_reference_spectra.py executed successfully!")
    print(output_path + 'expt_df.pickle')
    print(output_path + 'theo_df.pickle')
    return 1


def main():
    # Main captures input variables when called as command line script.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--input", default=None, type=str, help="Path to input db, e.g. core_metabolome_V3")
    parser.add_argument("--output", default=None, type=str, help="Path to output df with spectra")
    parser.add_argument("--gnps", default=None, type=str, help="Path to GNPS json dump")
    parser.add_argument("--hmdb_p_exp", default=None, type=str, help="Path to HMDB pos experimental dump")
    parser.add_argument("--hmdb_n_exp", default=None, type=str, help="Path to HMDB neg experimental dump")
    parser.add_argument("--hmdb_p_theo", default=None, type=str, help="Path to HMDB pos theoretical")
    parser.add_argument("--hmdb_n_theo", default=None, type=str, help="Path to HMDB neg theoretical")
    parser.add_argument("--mona_p", default=None, type=str, help="Path to MONA pos dump")
    parser.add_argument("--mona_n", default=None, type=str, help="Path to MONA neg dump")

    args = parser.parse_args()

    for a in args:
        if len(glob.glob(a)) == []:
            print('Input path does not exist!')
            exit(1)
        else:
            continue

    output_loop(args.input,
                args.output,
                args.gnps,
                args.hmdb_p_exp,
                args.hmdb_n_exp,
                args.hmdb_p_theo,
                args.hmdb_n_theo,
                args.mona_p,
                args.mona_n
                )

    print('Download reference spectra complete!')
    return 1

if __name__ == "__main__":
    main()