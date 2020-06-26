# pip install numpy pandas scipy sklearn enrichmentanalysis-dvklopfenstein metaspace2020
import pickle
from functools import lru_cache
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from metaspace.sm_annotation_utils import SMInstance, SMDataset
from scipy.ndimage import median_filter
from sklearn.metrics import pairwise_kernels

#%%
sm = SMInstance()
#%%
PARSE_MOL_ID = re.compile(r'([^_]+)_(\d+)([pf])')


@lru_cache()  # Only load when needed, as it eats a bunch of memory
def get_msms_df():
    msms_df = pd.read_pickle('to_metaspace/cm3_msms_all_both.pickle')
    del msms_df['db_isobar']
    msms_df.rename(columns={'ion_mass': 'mz'}, inplace=True)
    msms_df['hmdb_id'] = msms_df.id.str.replace(PARSE_MOL_ID, lambda m: m[1])
    msms_df['frag_idx'] = msms_df.id.str.replace(PARSE_MOL_ID, lambda m: m[2]).astype(np.int32)
    msms_df['is_parent'] = msms_df.id.str.replace(PARSE_MOL_ID, lambda m: m[3]) == 'p'
    msms_df['mol_name'] = msms_df.name.str.replace("^[^_]+_[^_]+_", "")
    msms_df['mol_href'] = 'https://hmdb.ca/metabolites/' + msms_df.hmdb_id

    # Clean up results by converting everything to HMDB IDs and removing items that can't be converted
    msms_df.hmdb_id.rename({
        'msmls87': 'HMDB0006557',  # ADP-GLUCOSE -> ADP-glucose
    }, inplace=True)
    ids_to_drop = [
        'msmls65',  # 5-HYDROXYTRYPTOPHAN (Different stereochemistry to HMDB0000472 5-Hydroxy-L-tryptophan, which is also included)
        'msmls183',  # DEOXYGUANOSINE-MONOPHOSPHATE (Identical to HMDB0001044 2'-Deoxyguanosine 5'-monophosphate)
        'msmls189',  # DGDP (Identical to HMDB0000960 dGDP)
        'C00968',  # 3',5'-Cyclic dAMP (Not in HMDB at all)
        'msmls142',  # CORTISOL 21-ACETATE (Not in HMDB at all)
        'msmls192',  # DIDECANOYL-GLYCEROPHOSPHOCHOLINE (Not in HMDB at all)
    ]
    msms_df = msms_df[~msms_df.hmdb_id.isin(ids_to_drop)]
    # Add is_lipid column
    hmdb_mols = pickle.load(open('to_metaspace/hmdb_mols.pickle', 'rb'))
    lipid_ids = [mol['id'] for mol in hmdb_mols if mol['super_class'] == 'Lipids and lipid-like molecules']
    msms_df['is_lipid'] = msms_df.hmdb_id.isin(lipid_ids)

    return msms_df

# msms_df = get_msms_df()

#%%
class DSResults:
    ds_id: str
    sm_ds: SMDataset
    db_id: str
    name: str
    anns: pd.DataFrame
    ds_images: Dict[str, np.ndarray]
    ds_coloc: pd.DataFrame
    mols_df: pd.DataFrame
    ann_mols_df: pd.DataFrame
    anns_df: pd.DataFrame
    export_data: Dict[str, pd.DataFrame]

    def get_coloc(self, f1, f2):
        if f1 == f2:
            return 1
        if f1 not in self.ds_coloc.index or f2 not in self.ds_coloc.index:
            return 0
        return self.ds_coloc.loc[f1, f2]


def fetch_ds_results(_ds_id):
    res = DSResults()
    res.ds_id = _ds_id
    res.sm_ds = sm.dataset(id=_ds_id)
    res.db_id = [db for db in res.sm_ds.databases if re.match(r'^\d|^ls_cm3_msms_all_', db)][0]
    res.name = re.sub('[\W ]+', '_', res.sm_ds.name)
    res.name = re.sub('_cloned_from.*', '', res.name)
    res.anns = res.sm_ds.results(database=res.db_id)
    ann_images = res.sm_ds.all_annotation_images(
        fdr=1,
        database=res.db_id,
        only_first_isotope=True,
        scale_intensity=False,
    )
    res.ds_images = dict((imageset.formula, imageset[0]) for imageset in ann_images)
    res.export_data = {}
    return res


# test_results = fetch_ds_results('2020-05-26_17h58m22s')
#%%

def add_coloc_matrix(res: DSResults):
    keys = list(res.ds_images.keys())
    images = list(res.ds_images.values())
    cnt = len(keys)
    if cnt == 0:
        res.ds_coloc = pd.DataFrame(dtype='f')
    elif cnt == 1:
        res.ds_coloc = pd.DataFrame([[1]], index=keys, columns=keys)
    else:
        h, w = images[0].shape
        flat_images = np.vstack(images)
        flat_images[flat_images < np.quantile(flat_images, 0.5, axis=1, keepdims=True)] = 0
        filtered_images = median_filter(flat_images.reshape((cnt, h, w)), (1, 3, 3)).reshape((cnt, h * w))
        distance_matrix = pairwise_kernels(filtered_images, metric='cosine')
        ds_coloc = pd.DataFrame(distance_matrix, index=keys, columns=keys, dtype='f')
        ds_coloc.rename_axis(index='source', columns='target', inplace=True)
        res.ds_coloc = ds_coloc

# calc_coloc_matrix(test_results)
#%%

def add_result_dfs(res: DSResults):

    # Get detected IDs from dataset
    detected_frag_ids = set()
    detected_mol_ids = set()
    for mol_ids in res.anns.moleculeIds:
        detected_frag_ids.update(mol_ids)
        detected_mol_ids.update(PARSE_MOL_ID.match(mol_id).groups()[0] for mol_id in mol_ids)

    # Exclude fragments of the wrong polarity
    df = get_msms_df()
    df = df[df.polarity == res.sm_ds.polarity.lower()].copy()
    df['is_detected'] = df.id.isin(detected_frag_ids)
    df['parent_is_detected'] = df.hmdb_id.isin(df[df.is_parent & df.is_detected].hmdb_id)
    df['in_range'] = (df.mz >= res.anns.mz.min() - 0.1) & (df.mz <= res.anns.mz.max() + 0.1)

    df = df.merge(pd.DataFrame({
        'parent_formula': df[df.is_parent].set_index('hmdb_id').formula,
        'parent_n_detected': df.groupby('hmdb_id').is_detected.sum().astype(np.uint32),
        'parent_n_frags': df.groupby('hmdb_id').in_range.sum().astype(np.uint32),
        'parent_n_frags_unfiltered': df.groupby('hmdb_id').frag_idx.max(),
    }), how='left', left_on='hmdb_id', right_index=True)
    df['coloc_to_parent'] = [res.get_coloc(f1, f2) for f1, f2 in df[['formula', 'parent_formula']].itertuples(False, None)]

    # Exclude molecules with no matches at all
    df = df[df.hmdb_id.isin(detected_mol_ids)]
    # Exclude molecules where the parent isn't matched
    df = df[df.parent_is_detected]
    # Exclude fragments that are outside of detected mz range
    # 0.1 buffer added because centroiding can move the peaks slightly
    df = df[df.in_range]

    res.mols_df = df[df.is_parent][[
        'hmdb_id', 'mz', 'is_detected', 'mol_name', 'formula',
        'parent_n_detected', 'parent_n_frags', 'parent_n_frags_unfiltered', 'mol_href',
    ]].set_index('hmdb_id')

    res.ann_mols_df = df[[
        'id', 'hmdb_id', 'mz', 'is_parent', 'is_detected', 'mol_name', 'formula',
        'coloc_to_parent', 'parent_formula', 'frag_idx', 'mol_href',
        'parent_n_detected', 'parent_n_frags', 'parent_n_frags_unfiltered',
    ]].set_index('id')

    res.anns_df = df[df.is_detected][[
        'formula', 'mz', 'mol_href',
    ]].drop_duplicates().set_index('formula')

# make_result_dfs(test_results)
# %%

def get_msms_results_for_ds(ds_id):
    cache_path = Path(f'./mol_scoring/cache/ds_results/{ds_id}.pickle')
    if not cache_path.exists():
        res = fetch_ds_results(ds_id)
        add_coloc_matrix(res)
        add_result_dfs(res)
        res.ds_images = None  # Save memory/space as downstream analysis doesn't need this
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(res, cache_path.open('wb'))
    else:
        res = pickle.load(cache_path.open('rb'))
    return res


# test_results = get_msms_results_for_ds('2020-05-26_17h58m22s')

#%%