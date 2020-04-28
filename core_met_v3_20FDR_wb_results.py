'''
This file contains links and names to the result files from searching the whole-body
datasets against core_metabolome_v3 on METASPACE.

As well, this file contains the the local address and names of the downloaded datasets.

Finally, this file contains the mapping between the two.

'''

import pandas as pd

cmv3_wb_result_paths = ['/Users/dis/Desktop/whole_body_word_2_vec/core_metabolome_v3_wb_METASPACE_results/2016-09-21_16h07m45s_Technologie_ServierTES-WBrat-vehicle.csv',
                        '/Users/dis/Desktop/whole_body_word_2_vec/core_metabolome_v3_wb_METASPACE_results/2017-05-17_19h49m04s_whole body xenograft (1) [RMS norm].csv',
                        '/Users/dis/Desktop/whole_body_word_2_vec/core_metabolome_v3_wb_METASPACE_results/2017-05-17_19h50m07s_wb xenograft trp pathway dosed- rms_corrected.csv',
                        '/Users/dis/Desktop/whole_body_word_2_vec/core_metabolome_v3_wb_METASPACE_results/2017-05-29_07h28m52s_servier_TT_mouse_wb_fmpts_derivatization_CHCA.csv',
                        '/Users/dis/Desktop/whole_body_word_2_vec/core_metabolome_v3_wb_METASPACE_results/2017-07-24_19h42m31s_Servier_Ctrl_mouse_wb_lateral_plane_9aa.csv',
                        '/Users/dis/Desktop/whole_body_word_2_vec/core_metabolome_v3_wb_METASPACE_results/2017-07-26_18h25m14s_Servier_Ctrl_mouse_wb_median_plane_9aa.csv',
                        '/Users/dis/Desktop/whole_body_word_2_vec/core_metabolome_v3_wb_METASPACE_results/2017-08-03_15h09m06s_Servier_Ctrl_mouse_wb_median_plane_chca.csv',
                        '/Users/dis/Desktop/whole_body_word_2_vec/core_metabolome_v3_wb_METASPACE_results/2017-08-03_15h09m51s_Servier_Ctrl_mouse_wb_lateral_plane_chca.csv',
                        '/Users/dis/Desktop/whole_body_word_2_vec/core_metabolome_v3_wb_METASPACE_results/2017-08-11_07h59m58s_Servier_Ctrl_mouse_wb_lateral_plane_DHB.csv',
                        '/Users/dis/Desktop/whole_body_word_2_vec/core_metabolome_v3_wb_METASPACE_results/2017-08-11_08h01m02s_Servier_Ctrl_mouse_wb_median_plane_DHB.csv'
                        ]

cmv3_wb_result_names = ['2016-09-21_16h07m45s_Technologie_ServierTES-WBrat-vehicle.csv',
                        '2017-05-17_19h49m04s_whole body xenograft (1) [RMS norm].csv',
                        '2017-05-17_19h50m07s_wb xenograft trp pathway dosed- rms_corrected.csv',
                        '2017-05-29_07h28m52s_servier_TT_mouse_wb_fmpts_derivatization_CHCA.csv',
                        '2017-07-24_19h42m31s_Servier_Ctrl_mouse_wb_lateral_plane_9aa.csv',
                        '2017-07-26_18h25m14s_Servier_Ctrl_mouse_wb_median_plane_9aa.csv',
                        '2017-08-03_15h09m06s_Servier_Ctrl_mouse_wb_median_plane_chca.csv',
                        '2017-08-03_15h09m51s_Servier_Ctrl_mouse_wb_lateral_plane_chca.csv',
                        '2017-08-11_07h59m58s_Servier_Ctrl_mouse_wb_lateral_plane_DHB.csv',
                        '2017-08-11_08h01m02s_Servier_Ctrl_mouse_wb_median_plane_DHB.csv'
                        ]

cmv3_wb_ds_name_dict = {'2016-09-21_16h07m45s': 'Technologie_ServierTES-WBrat-vehicle',
                        '2017-05-17_19h49m04s': 'whole body xenograft (1) [RMS norm]',
                        '2017-05-17_19h50m07s': 'wb xenograft trp pathway dosed- rms_corrected',
                        '2017-05-29_07h28m52s': 'servier_TT_mouse_wb_fmpts_derivatization_CHCA',
                        '2017-07-24_19h42m31s': 'Servier_Ctrl_mouse_wb_lateral_plane_9aa',
                        '2017-07-26_18h25m14s': 'Servier_Ctrl_mouse_wb_median_plane_9aa',
                        '2017-08-03_15h09m06s': 'Servier_Ctrl_mouse_wb_median_plane_chca',
                        '2017-08-03_15h09m51s': 'Servier_Ctrl_mouse_wb_lateral_plane_chca',
                        '2017-08-11_07h59m58s': 'Servier_Ctrl_mouse_wb_lateral_plane_DHB',
                        '2017-08-11_08h01m02s': 'Servier_Ctrl_mouse_wb_median_plane_DHB'
                        }

cm3_wb_ds_prod_beta_dict = {'2016-09-21_16h07m45s': '2016-09-21_16h07m45s',
                        '2017-05-17_19h49m04s': '2017-05-17_19h49m04s',
                        '2017-05-17_19h50m07s': '2017-05-17_19h50m07s',
                        '2017-05-29_07h28m52s': '2017-05-29_07h28m52s',
                        '2017-07-24_19h42m31s': '2017-07-24_19h42m31s',
                        '2017-07-26_18h25m14s': '2017-07-26_18h25m14s',
                        '2017-08-03_15h09m06s': '2017-08-03_15h09m06s',
                        '2017-08-03_15h09m51s': '2017-08-03_15h09m51s',
                        '2017-08-11_07h59m58s': '2017-08-11_07h59m58s',
                        '2017-08-11_08h01m02s': '2017-08-11_08h01m02s'
                        }

# .imzML, .ibid
cm3_wb_name_name_fn_dict = {'Technologie_ServierTES-WBrat-vehicle': '',
                        'whole body xenograft (1) [RMS norm]': 'wb xenograft in situ metabolomics test - rms_corrected',
                        'wb xenograft trp pathway dosed- rms_corrected': '15-0879 wb xenograft trp pathway dosed- rms_corrected',
                        'servier_TT_mouse_wb_fmpts_derivatization_CHCA': 'servier_TT_mouse_wb_fmpts_derivatization_CHCA',
                        'Servier_Ctrl_mouse_wb_lateral_plane_9aa': 'Servier_Ctrl_mouse_wb_9aa',
                        'Servier_Ctrl_mouse_wb_median_plane_9aa': 'Servier_Ctrl_mouse_wb2_9aa',
                        'Servier_Ctrl_mouse_wb_median_plane_chca': 'Servier_Ctrl_mouse_wb_median_plane_chca',
                        'Servier_Ctrl_mouse_wb_lateral_plane_chca': 'Servier_Ctrl_mouse_wb_lateral_plane_chca',
                        'Servier_Ctrl_mouse_wb_lateral_plane_DHB': 'Servier_Ctrl_mouse_wb_lateral_plane_DHB',
                        'Servier_Ctrl_mouse_wb_median_plane_DHB': 'Servier_Ctrl_mouse_wb_median_plane_DHB'
                        }

cmv3_wb_datasets_to_rename = ['2016-09-21_16h07m45s_Technologie_ServierTES-WBrat-vehicle.csv',
                        ]

paths = ['/Users/dis/Desktop/whole_body_word_2_vec/core_metabolome_v3_wb_METASPACE_results',
         '/Users/dis/Desktop/whole_body_word_2_vec/datasets'
         ]

