#!/usr/bin/env python


"""
To write!
"""


import pandas as pd
import argparse
import glob
import os
import shlex
from subprocess import Popen, PIPE
from threading import Timer


__author__ = "Christopher M Baxter Rath"
__copyright__ = "Copyright 2020"
__credits__ = ["Christopher M Baxter Rath"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Christopher M Baxter Rath"
__email__ = "chrisrath@gmail.com"
__status__ = "Development"


# Paths to local files, setup when running:
sirius_executable_path = '/Users/dis/PycharmProjects/word2vec/sirius/bin/sirius'
hmdb_experimental_msms_path = '/Users/dis/Desktop/hmdb_LCMS/hmdb_experimental_msms_spectra'
hmdb_theoretical_msms_path = '/Users/dis/Desktop/hmdb_LCMS/hmdb_predicted_msms_spectra'


def hmdb_spectra_copy(df, path, spectral_paths, ex_flag):
    # parses HMDB spectra file from xml
    if ex_flag is True:
        from_path = hmdb_experimental_msms_path
    else:
        from_path = hmdb_theoretical_msms_path

    for file in df.file_paths:
        f = from_path + '/' + file
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


def ms_format_writer(m_df, g_df, e_df, t_df, db_index, add, polarity):
    # Writes the GNPS json and the Mona text spectra to file, returning paths
    # Copies HMDB spectra files to directory
    spectral_paths = []
    counter = 0

    # Changed to support polarity
    path = 'spectra_' + polarity + '/' + db_index

    try:
        os.mkdir(path)
    except:
        pass

    path = path + '/' + add + '_'

    if not m_df.empty:
        for s in list(m_df.spectrum):
            counter += 1
            s = s.replace(' ', '\n')
            s = s.replace(':', ' ')
            # Changed to support polarity
            o = path + str(counter) + '.txt'
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
            # Changed to support polarity
            o = path + str(counter) + '.txt'
            o = o.replace('+', 'p')
            o = o.replace('-', 'n')
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

    query_list = [sirius_executable_path,
                  '-f',
                  formula,
                  '-i',
                  ion,
                  '-2']
    for s in spectra_list:
        query_list.append(s)

    query_list.append('-o')
    query_list.append(output_dir + '/' + db_index)
    return(query_list)


def output_Sirius_parser(sirius_output, output_dir, db_index, n, polarity):
    # Parses Sirius output returning paths to ms file with formulas
    # print('sirius_output:', sirius_output, '\n', type(sirius_output))

    try:
        x = sirius_output.split("Sirius results for")[1]
        x = x.split(".txt")[0]
        x = x.split(": '")[1]
        x = str(x)
    except:
        return None

    search_string = output_dir + '/' + db_index + '/' + str(n) + '_' + x \
                    + '_/spectra/*.ms'
    # print('search_string:', search_string)
    return search_string


def loop_Sirius(df,
                mona_df,
                gnps_df,
                hex_df,
                hth_df,
                polarity):
    # Main loop for running Sirius
    # Have to delete old spectra and trees before rerunning
    sirius_output_dict = {}
    mona_df = mona_df[['inchi', 'adduct', 'spectrum']].copy(deep=True)
    gnps_df = gnps_df[['can_smiles', 'Adduct', 'peaks_json']].copy(deep=True)
    gnps_df = gnps_df.rename(columns={'Adduct':'adduct'})
    hex_df = hex_df[['id', 'adduct', 'file_paths']].copy(deep=True)
    hth_df = hth_df[['id', 'adduct', 'file_paths']].copy(deep=True)

    db_df_dict = {'mona_df': mona_df,
                  'gnps_df': gnps_df,
                  'hex_df': hex_df,
                  'hth_df':hth_df
                  }

    if polarity == 'positive':
        current_adducts = ['M+', 'M+H', 'M+Na', 'M+K']
    elif polarity == 'negative':
        current_adducts = ['M-', 'M-H']

    for name, db_df in db_df_dict.items():
        db_df_dict[name] = db_df[db_df.adduct.isin(current_adducts)]

    mona_df = db_df_dict['mona_df']
    gnps_df = db_df_dict['gnps_df']
    hex_df = db_df_dict['hex_df']
    hth_df = db_df_dict['hth_df']

    # Loops over each compound in reference dataframe
    for idx in list(df.index):
        ser = df.loc[idx]
        print('\n', 'series', idx, ser.db_index)

        mo_df = mona_df[mona_df.inchi == ser.inchi]
        gn_df = gnps_df[gnps_df.can_smiles == ser.can_smiles]
        he_df = hex_df[hex_df.id == ser.id]
        ht_df = hth_df[hth_df.id == ser.id]

        unique_adducts = list(set(list(mo_df.adduct)
                                  + list(gn_df.adduct)
                                  + list(he_df.adduct)
                                  + list(ht_df.adduct)))

        # Loops over each adduct for each compound
        add_counter = 0
        for add in unique_adducts:
            print(idx, ' ', add)
            add_counter += 1
            # Changed to support polarity
            output_dir = 'trees/' + polarity + '_' + ser.db_index

            m_df = mo_df[mo_df.adduct == add]
            g_df = gn_df[gn_df.adduct == add]
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
                                            ser.db_index,
                                            add,
                                            polarity)

            t_add = adduct_translate(add)
            sirius_input = runner_Sirius(ser.formula,
                                         t_add,
                                         spectra_list,
                                         output_dir,
                                         ser.db_index)

            # Run with timeout as Sirius chokes >120 mins in some large compounds
            #sirius_output = subprocess.check_output(sirius_input)
            sirius_input = " ".join(sirius_input)
            sirius_output = run(sirius_input, 180)

            #sirius_output = sirius_output.decode('utf-8')
            # print('sirius_output:', '\n', sirius_output, '\n',)
            sirius_output_dict[ser.db_index] = output_Sirius_parser(sirius_output,
                                                                    output_dir,
                                                                    ser.db_index,
                                                                    add_counter,
                                                                    polarity
                                                                    )
    print('sirius_output_dict:', sirius_output_dict)
    sirius_output_df = pd.DataFrame.from_dict(sirius_output_dict,
                                              orient='index',
                                              columns=['file'])
    sirius_output_df['exists'] = sirius_output_df['file'].apply(lambda x: exists(x))
    # print('Sirius success: ', sirius_output_df.exists.value_counts())
    sirius_output_df = sirius_output_df[sirius_output_df.exists != 0]

    return sirius_output_df


def master_loop(ref,
                output,
                gnps,
                hmdb_p_exp,
                hmdb_n_exp,
                hmdb_p_theo,
                hmdb_n_theo,
                mona_p,
                mona_n,
                polarity,
                ref_type):
    # Loop captures execution of Sirius command line script for
    # theoretical/experimental and positive/negative

    # Read all the paths as dfs
    ref = pd.read_pickle(ref)
    gnps = pd.read_pickle(gnps)
    hmdb_p_exp = pd.read_pickle(hmdb_p_exp)
    hmdb_n_exp = pd.read_pickle(hmdb_n_exp)
    hmdb_p_theo = pd.read_pickle(hmdb_p_theo)
    hmdb_n_theo = pd.read_pickle(hmdb_n_theo)
    mona_p = pd.read_pickle(mona_p)
    mona_n = pd.read_pickle(mona_n)


    if ref_type == 'exp':
        if polarity == 'positive':
            sirius_df = loop_Sirius(ref,
                                    mona_p,
                                    gnps,
                                    hmdb_p_exp,
                                    pd.DataFrame(columns=['id', 'adduct', 'file_paths']),
                                    polarity
                                    )

        elif polarity == 'negative':
            sirius_df = loop_Sirius(ref,
                                    mona_n,
                                    gnps,
                                    hmdb_n_exp,
                                    pd.DataFrame(columns=['id', 'adduct', 'file_paths']),
                                    polarity
                                    )

    elif ref_type == 'theo':
        if polarity == 'positive':
            sirius_df = loop_Sirius(ref,
                                    pd.DataFrame(columns=['spectrum', 'adduct', 'inchi']),
                                    pd.DataFrame(columns=['can_smiles', 'Adduct', 'peaks_json']),
                                    pd.DataFrame(columns=['id', 'adduct', 'file_paths']),
                                    hmdb_p_theo,
                                    polarity
                                    )
        elif polarity == 'negative':
            sirius_df = loop_Sirius(ref,
                                    pd.DataFrame(columns=['spectrum', 'adduct', 'inchi']),
                                    pd.DataFrame(columns=['can_smiles', 'Adduct', 'peaks_json']),
                                    pd.DataFrame(columns=['id', 'adduct', 'file_paths']),
                                    hmdb_n_theo,
                                    polarity
                                    )
    try:
        os.mkdir(output)
    except:
        pass

    outname = output + ref_type + '_' + polarity + '.pickle'
    sirius_df.to_pickle(outname)
    print('Sirius out: ', outname, sirius_df.shape)
    return 1


def main():
    # Main captures input variables when called as command line script.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--input_exp", default=None, type=str, help="experimental spectra")
    parser.add_argument("--input_theo", default=None, type=str, help="theoretical spectra")
    parser.add_argument("--output", default=None, type=str, help="Path to output df with spectra")
    parser.add_argument("--gnps", default=None, type=str, help="Path to GNPS json dump")
    parser.add_argument("--hmdb_p_exp", default=None, type=str, help="Path to HMDB pos experimental dump")
    parser.add_argument("--hmdb_n_exp", default=None, type=str, help="Path to HMDB neg experimental dump")
    parser.add_argument("--hmdb_p_theo", default=None, type=str, help="Path to HMDB pos theoretical")
    parser.add_argument("--hmdb_n_theo", default=None, type=str, help="Path to HMDB neg theoretical")
    parser.add_argument("--mona_p", default=None, type=str, help="Path to MONA pos dump")
    parser.add_argument("--mona_n", default=None, type=str, help="Path to MONA neg dump")
    parser.add_argument("--polarity",
                        default=None,
                        type=str,
                        help="Try: positive, negative, or both!")

    args = parser.parse_args()

    # Potentially loops for both Theo/Exp x Pos/Neg:
    for ref_type, ref_path in {'exp': args.input_exp,
                        'theo': args.input_theo}.items():

        if args.polarity == 'both' or args.polarity == 'positive':
            print('Running positive mode data!')
            master_loop(ref_path,
                        args.output,
                        args.gnps,
                        args.hmdb_p_exp,
                        args.hmdb_n_exp,
                        args.hmdb_p_theo,
                        args.hmdb_n_theo,
                        args.mona_p,
                        args.mona_n,
                        'positive',
                        ref_type
                        )
        elif args.polarity == 'both' or args.polarity == 'negative':
            print('Running positive mode data!')
            master_loop(ref_path,
                        args.output,
                        args.gnps,
                        args.hmdb_p_exp,
                        args.hmdb_n_exp,
                        args.hmdb_p_theo,
                        args.hmdb_n_theo,
                        args.mona_p,
                        args.mona_n,
                        'positive',
                        ref_type
                        )
        else:
            print('Polarity was not recognized, try: positive, negative, or both!')
            exit(1)

    print('Spectra annotation with Sirius and export to METASPACE format complete!')
    return 1

if __name__ == "__main__":
    main()