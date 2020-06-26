import pandas as pd

whole_body_ds_ids = [
    '2020-05-26_17h57m50s',
    '2020-05-26_17h57m53s',
    '2020-05-26_17h57m57s',
    '2020-05-26_17h58m00s',
    '2020-05-26_17h58m04s',
    '2020-05-26_17h58m08s',
    '2020-05-26_17h58m11s',
    '2020-05-26_17h58m15s',
    '2020-05-26_17h58m19s',
    '2020-05-26_17h58m22s',
]
high_quality_ds_ids = [
    '2020-05-18_17h08m38s',
    '2020-05-18_17h08m40s',
    '2020-05-18_17h08m42s',
    '2020-05-18_17h08m44s',
    '2020-05-18_17h08m47s',
    '2020-05-18_17h08m49s',
    '2020-05-18_17h08m51s',
    '2020-05-18_17h08m54s',
    '2020-05-18_17h08m56s',
    '2020-05-18_17h08m58s',
    '2020-05-18_17h09m01s',
    '2020-05-18_17h09m03s',
    '2020-05-18_17h09m05s',
    '2020-05-18_17h09m07s',
    '2020-05-18_17h09m10s',
]
spotting_only_spotted_mols_ds_ids = [
    '2020-05-14_16h32m01s',
    '2020-05-14_16h32m04s',
    '2020-05-14_16h32m07s',
    '2020-05-14_16h32m10s',
    '2020-05-14_16h32m14s',
    '2020-05-14_16h32m16s',
    '2020-05-14_16h32m19s',
    '2020-05-14_16h32m22s',
    '2020-05-14_16h32m26s',
]
spotting_ds_ids = [
    # '2020-06-19_16h38m19s',  # Lipids don't give good results
    # '2020-06-19_16h39m01s',
    '2020-06-19_16h39m02s',
    '2020-06-19_16h39m04s',
    '2020-06-19_16h39m06s',
    '2020-06-19_16h39m08s',
    '2020-06-19_16h39m10s',
    '2020-06-19_16h39m12s',
    '2020-06-19_16h39m14s',
]
spotting_short_names = {
    # '2020-06-19_16h38m19s': 'ME Lipid Array DAN+',
    # '2020-06-19_16h39m01s': 'ME Lipid Array DAN-',
    '2020-06-19_16h39m02s': 'ME Spotted Array B DHB+',
    '2020-06-19_16h39m04s': 'ME SlideD DHB+',
    '2020-06-19_16h39m06s': 'ME SlideE DAN-',
    '2020-06-19_16h39m08s': 'MZ DHB+ (120-720)',
    '2020-06-19_16h39m10s': 'MZ DHB+ (60-360)',
    '2020-06-19_16h39m12s': 'MZ DAN- (50-300)',
    '2020-06-19_16h39m14s': 'MZ DAN- (140-800)'
}

# lipid_mol_ids = set(pd.read_csv('./spotting/lipid_spotted_mols.csv')[lambda df: df.cm_name.notna()].id)
jan_mol_ids = set(pd.read_csv('./spotting/jan_spotted_mols.csv')[lambda df: df.cm_name.notna()].id)
ourcon_mol_ids = set(pd.read_csv('./spotting/ourcon_spotted_mols.csv')[lambda df: df.cm_name.notna()].id)

spotting_mol_lists = {
    # '2020-06-19_16h38m19s': lipid_mol_ids,
    # '2020-06-19_16h39m01s': lipid_mol_ids,
    '2020-06-19_16h39m02s': jan_mol_ids,
    '2020-06-19_16h39m04s': jan_mol_ids,
    '2020-06-19_16h39m06s': jan_mol_ids,
    '2020-06-19_16h39m08s': ourcon_mol_ids,
    '2020-06-19_16h39m10s': ourcon_mol_ids,
    '2020-06-19_16h39m12s': ourcon_mol_ids,
    '2020-06-19_16h39m14s': ourcon_mol_ids,
}