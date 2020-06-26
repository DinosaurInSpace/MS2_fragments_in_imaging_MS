# pip install numpy pandas scipy sklearn enrichmentanalysis-dvklopfenstein metaspace2020
from pathlib import Path
import re

import matplotlib.pyplot as plt
import seaborn as sns
from MSMS_scoring_datasets import spotting_mol_lists, spotting_ds_ids, spotting_short_names, whole_body_ds_ids
from MSMS_scoring_metrics import get_ds_results

#%%
# Bigger plots

plt.rcParams['figure.figsize'] = 15, 10

#%%
# Plot distributions of various values
for ds_id in spotting_ds_ids[5:]:
    ds = get_ds_results(ds_id)
    sns.kdeplot(ds.ann_mols_df[lambda df: ~df.is_parent & df.is_detected].coloc_to_parent.rename(spotting_short_names[ds_id]), clip=(0.0,1.0), cut=0)

out = Path('./mol_scoring/stats/coloc_to_parent/spotting_MZ.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out))
plt.show()

#%%
# Plot distributions of various values
for ds_id in spotting_ds_ids[:5]:
    ds = get_ds_results(ds_id)
    sns.kdeplot(ds.ann_mols_df[lambda df: ~df.is_parent & df.is_detected].coloc_to_parent.rename(spotting_short_names[ds_id]), clip=(0.0,1.0), cut=0)

out = Path('./mol_scoring/stats/coloc_to_parent/spotting_ME.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out))
plt.show()

#%%
# Plot distributions of various values
# plt.hist(test_results.mols_df.global_enrich_uncorr, bins=10)
for ds_id in whole_body_ds_ids[:5]:
    ds = get_ds_results(ds_id)
    name = re.sub('_cloned_from.*', '', ds.name)
    sns.kdeplot(ds.ann_mols_df[lambda df: ~df.is_parent & df.is_detected].coloc_to_parent.rename(name), clip=(0.0,1.0), cut=0)

out = Path('./mol_scoring/stats/coloc_to_parent/whole_body_1.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out))
plt.show()

#%%
# Plot distributions of various values
fig, plots = plt.subplots(4, 5)
for i, ds_id in enumerate(spotting_ds_ids):
    y, x = divmod(i, 5)
    ds = get_ds_results(ds_id)
    name = re.sub('_cloned_from.*', '', ds.name)
    plots[y*2, x].hist(ds.mols_df[lambda df: df.is_detected].global_enrich_uncorr.rename(name + " global"))
    plots[y*2+1, x].hist(ds.ann_mols_df[lambda df: ~df.is_parent & df.is_detected & (df.group_enrich_uncorr != 1)].group_enrich_uncorr.rename(name), bins=30)

# out = Path('./mol_scoring/stats/coloc_to_parent/whole_body_1.png')
# out.parent.mkdir(parents=True, exist_ok=True)
# plt.savefig(str(out))
plt.show()

#%%
# Plot distributions of various values

fig, plots = plt.subplots(3, 3)
# fig.suptitle('Spotting datasets, frag-to-parent coloc, spotted mols')
fig.subplots_adjust()

for i, ds_id in list(enumerate(spotting_ds_ids)):
    y, x = divmod(i, 3)
    ds = get_ds_results(ds_id)
    name = re.sub('_cloned_from.*', '', ds.name)
    name = re.sub('_full_msms.*', '', name)
    expected_mol_ids = spotting_mol_lists.get(ds_id)
    if expected_mol_ids:
        print(name)
        plots[y, x].set_title(name, fontsize=8)
        # sns.kdeplot(ds.ann_mols_df[lambda df: ~df.is_parent & df.is_detected].coloc_to_parent.rename(name), clip=(0.0, 1.0), cut=0, ax=plots[y, x])
        plots[y, x].hist(ds.ann_mols_df[lambda df: ~df.is_parent & df.is_detected & df.parent_id.isin(expected_mol_ids)].coloc_to_parent, bins=30)

out = Path('./mol_scoring/stats/coloc_to_parent/spotting_expected_mols.png')
# out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out))
fig.show()

#%%
# Plot distributions of various values

fig, plots = plt.subplots(3, 3)
fig.subplots_adjust()
for i, ds_id in list(enumerate(whole_body_ds_ids))[:9]:
    y, x = divmod(i, 3)
    ds = get_ds_results(ds_id)
    name = re.sub('_cloned_from.*', '', ds.name)
    print(name)
    plots[y, x].set_title(name, fontsize=8)
    plots[y, x].hist(ds.ann_mols_df[lambda df: ~df.is_parent & df.is_detected].coloc_to_parent, bins=30)

out = Path('./mol_scoring/stats/coloc_to_parent/whole_body_all_mols.png')
# out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out))
fig.show()

#%%
# Plot distributions of various values
for mode in ['all', 'spotted', 'bad']:
    for src in ['global', 'group']:
        plt.close('all')
        fig, plots = plt.subplots(3, 3)
        fig.tight_layout()
        for i, ds_id in list(enumerate(spotting_ds_ids))[:9]:
            y, x = divmod(i, 3)
            ds = get_ds_results(ds_id)
            name = re.sub('_cloned_from.*', '', ds.name)
            print(name)
            expected_mol_ids = spotting_mol_lists.get(ds_id)
            if src == 'global':
                if mode == 'spotted':
                    data = ds.mols_df[lambda df: df.is_detected & df.index.isin(expected_mol_ids)].global_enrich_uncorr
                elif mode == 'bad':
                    data = ds.mols_df[lambda df: df.is_detected & ~df.index.isin(expected_mol_ids)].global_enrich_uncorr
                else:
                    data = ds.mols_df[lambda df: df.is_detected].global_enrich_uncorr
            else:
                # only include groups with more than 1 mol
                formulas = ds.ann_mols_df.groupby('formula').parent_id.count()
                formulas = set(formulas.index[formulas > 1])
                if mode == 'spotted':
                    filter = ds.ann_mols_df.parent_id.isin(expected_mol_ids)
                elif mode == 'bad':
                    filter = ~ds.ann_mols_df.parent_id.isin(expected_mol_ids)
                else:
                    filter = True
                data = ds.ann_mols_df[lambda df: df.is_detected & df.formula.isin(formulas) & filter].group_enrich_uncorr
            plots[y, x].set_title(name, fontsize=8)
            plots[y, x].hist(data, bins=30, range=(0,1))

        out = Path(f'./mol_scoring/stats/p_values_spotting_{src}_enrich_{mode}_mols.png')
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out))
        fig.show()

#%%
# Plot distributions of various values
for mode in ['all', 'spotted', 'bad']:
    plt.close('all')
    fig, plots = plt.subplots(3, 3)
    fig.tight_layout()
    for i, ds_id in list(enumerate(spotting_ds_ids))[:9]:
        y, x = divmod(i, 3)
        ds = get_ds_results(ds_id)
        name = re.sub('_cloned_from.*', '', ds.name)
        print(name)
        expected_mol_ids = spotting_mol_lists.get(ds_id)
        if mode == 'spotted':
            data = ds.mols_df[lambda df: df.is_detected & df.index.isin(expected_mol_ids)].tfidf
        elif mode == 'bad':
            data = ds.mols_df[lambda df: df.is_detected & ~df.index.isin(expected_mol_ids)].tfidf
        else:
            data = ds.mols_df[lambda df: df.is_detected].tfidf
        data[data > 3] = 3
        plots[y, x].set_title(name, fontsize=8)
        plots[y, x].hist(data, bins=30, range=(1,3))

    out = Path(f'./mol_scoring/stats/p_values_spotting_tfidf_{mode}_mols.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    fig.show()