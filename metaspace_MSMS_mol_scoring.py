# pip install numpy pandas scipy sklearn enrichmentanalysis-dvklopfenstein metaspace2020
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import re
from getpass import getpass
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from metaspace.sm_annotation_utils import SMInstance, SMDataset
from scipy.ndimage import median_filter
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from enrichmentanalysis.enrich_run import EnrichmentRun
from sklearn.metrics import pairwise_kernels

# matplotlib.use('gtk3agg')

#%%
 # Bigger plots
# plt.rcParams['figure.figsize'] = 15, 10
# Wider maximum width of pandas columns (needed to see the full lists of molecules)

pd.set_option('max_colwidth', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
#%%
sm = SMInstance(host='https://beta.metaspace2020.eu', api_key=getpass())
#%%
msms_df = pd.read_pickle('./cm3_msms_all_both.pickle')
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


test_results = fetch_ds_results('2020-05-26_17h58m22s')
#%%

def calc_coloc_matrix(res: DSResults):
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

calc_coloc_matrix(test_results)
#%%

def make_result_dfs(res: DSResults):
    PARSE_MOL_ID = re.compile(r'([^_]+)_(\d+)([pf])')

    # Get detected IDs from dataset
    detected_frag_ids = set()
    detected_mol_ids = set()
    for mol_ids in res.anns.moleculeIds:
        detected_frag_ids.update(mol_ids)
        detected_mol_ids.update(PARSE_MOL_ID.match(mol_id).groups()[0] for mol_id in mol_ids)


    # Exclude fragments of the wrong polarity
    df = msms_df[msms_df.polarity == res.sm_ds.polarity.lower()]
    df = df.rename(columns={
        'ion_mass': 'mz',
    })

    df['parent_id'] = df.id.str.replace(PARSE_MOL_ID, lambda m: m[1])
    df['frag_idx'] = df.id.str.replace(PARSE_MOL_ID, lambda m: m[2]).astype(np.int32)
    df['is_parent'] = df.id.str.replace(PARSE_MOL_ID, lambda m: m[3]) == 'p'
    df['is_detected'] = df.id.isin(detected_frag_ids)
    df['parent_is_detected'] = df.parent_id.isin(df[df.is_parent & df.is_detected].parent_id)
    df['mol_name'] = df.name.str.replace("^[^_]+_[^_]+_", "")
    df['mol_href'] = 'https://hmdb.ca/metabolites/' + df.parent_id
    df['in_range'] = (df.mz >= res.anns.mz.min() - 0.1) & (df.mz <= res.anns.mz.max() + 0.1)

    df = df.merge(pd.DataFrame({
        'parent_formula': df[df.is_parent].set_index('parent_id').formula,
        'parent_n_detected': df.groupby('parent_id').is_detected.sum().astype(np.uint32),
        'parent_n_frags': df.groupby('parent_id').in_range.sum().astype(np.uint32),
        'parent_n_frags_unfiltered': df.groupby('parent_id').frag_idx.max(),
    }), how='left', left_on='parent_id', right_index=True)
    df['coloc_to_parent'] = [res.get_coloc(f1, f2) for f1, f2 in df[['formula', 'parent_formula']].itertuples(False, None)]

    # Exclude molecules with no matches at all
    df = df[df.parent_id.isin(detected_mol_ids)]
    # Exclude molecules where the parent isn't matched
    df = df[df.parent_is_detected]
    # Exclude fragments that are outside of detected mz range
    # 0.1 buffer added because centroiding can move the peaks slightly
    df = df[df.in_range]

    res.mols_df = df[df.is_parent][[
        'parent_id', 'mz', 'is_detected', 'mol_name', 'formula',
        'parent_n_detected', 'parent_n_frags', 'parent_n_frags_unfiltered', 'mol_href',
    ]].set_index('parent_id')

    res.ann_mols_df = df[[
        'id', 'parent_id', 'mz', 'is_parent', 'is_detected', 'mol_name', 'formula',
        'coloc_to_parent', 'parent_formula', 'frag_idx', 'mol_href',
        'parent_n_detected', 'parent_n_frags', 'parent_n_frags_unfiltered',
    ]].set_index('id')

    res.anns_df = df[df.is_detected][[
        'formula', 'mz', 'mol_href',
    ]].drop_duplicates().set_index('formula')

make_result_dfs(test_results)
#%%

def add_tfidf_score(res: DSResults):
    """
    TF-IDF where:
    * each parent molecule is a "document"
    * each annotated formula is a "term"
    * a term is only in a document when it is a parent ion or a predicted fragment ion
    * a term's "frequency" in a document is (colocalization to parent) / (number of predicted ions for this parent ion)

    Caveats:
    * The output value is just summed per document to produce a score. May not be optimal.
    """
    terms_df = (
        res.ann_mols_df
        # [lambda df: ~df.is_parent]
        # To use constant value:
        # .assign(value=1)
        # To use value based on 1/num_features:
        # Note that this equally scales the parent annotation
        .assign(value=res.ann_mols_df.coloc_to_parent / res.ann_mols_df.parent_n_frags)
        .pivot_table(index='parent_id', columns='formula', values='value', fill_value=0, aggfunc='sum')
    )
    # terms_df /= np.array([features_per_parent_s.reindex_like(terms_df).values]).T
    terms_matrix = csr_matrix(terms_df.values)

    tfidf_raw = TfidfTransformer().fit_transform(terms_matrix)
    tfidf_s = pd.Series(
        tfidf_raw.toarray().sum(axis=1),
        index=terms_df.index,
        name='tfidf_score',
    )
    res.mols_df['tfidf'] = tfidf_s
    res.ann_mols_df = res.ann_mols_df.merge(tfidf_s, left_on='parent_id', right_index=True)


add_tfidf_score(test_results)
#%%

def add_enrichment_analysis(res: DSResults):
    """
    Over-representation analysis where:
    * The "population" is all predicted ion formulas
    * The "study set" is all observed ion formulas
    * "Associations" are groups linking each ion formula to potential parent molecules

    Caveats:
    * Colocalization is not considered, even though it can be a compelling way to disprove fragments
      association with their parents
    * A bad P-value is not evidence that the candidate molecule is wrong. It only indicates the
      possibility that the observed fragment distribution was due to random chance.
    """

    def enrich(df, prefix=''):
        population = set(df.formula.unique())
        study_ids = set(df[df.is_detected].formula.unique())
        associations = df.groupby('formula').apply(lambda grp: set(grp.parent_id)).to_dict()

        enrichment_results = (
            EnrichmentRun(population, associations, alpha=0.05, methods=('sm_bonferroni',))
            .run_study(study_ids, study_name='results')
            .results
        )

        # HACK: r.get_nt_prt() converts lots of fields to strings, so manually grab all the interesting fields
        raw_data = pd.DataFrame([{
            'parent_id': r.termid,
            **r.ntpval._asdict(),
            **r.multitests._asdict(),
            'stu_items': r.stu_items
        } for r in enrichment_results if r.ntpval.study_cnt]).set_index('parent_id')

        return pd.DataFrame({
            # prefix + 'enrich_ratio': raw_data.study_ratio / raw_data.pop_ratio,
            prefix + 'enrich_p': raw_data.sm_bonferroni,
            prefix + 'enrich_uncorr': raw_data.pval_uncorr,
        }, index=raw_data.index, dtype='f')

    enrich_data = enrich(res.ann_mols_df, 'global_')
    res.mols_df = res.mols_df.join(enrich_data, how='left')
    res.ann_mols_df = res.ann_mols_df.merge(enrich_data, how='left', left_on='parent_id', right_index=True)

    mini_df = res.ann_mols_df[['parent_id', 'formula', 'is_detected']]
    detected_formulas = set(mini_df[mini_df.is_detected].formula)
    group_enrich_datas = []
    for formula, formula_df in mini_df[mini_df.formula.isin(detected_formulas)].groupby('formula'):
        related_df = pd.concat([formula_df, mini_df[mini_df.parent_id.isin(formula_df.parent_id)]])
        related_df['coloc'] = [res.get_coloc(formula, f) for f in related_df.formula]
        related_df['is_detected'] = related_df.is_detected & (related_df.coloc > 0.55)
        group_enrich_data = enrich(related_df, 'group_').assign(formula=formula).set_index('formula', append=True)
        group_enrich_datas.append(group_enrich_data)

    res.ann_mols_df = res.ann_mols_df.merge(
        pd.concat(group_enrich_datas),
        how='left',
        left_on=['parent_id', 'formula'],
        right_index=True
    )


add_enrichment_analysis(test_results)
#%%

def find_interesting_groups(res: DSResults):
    href_base = f'https://beta.metaspace2020.eu/annotations?ds={res.ds_id}&db={res.db_id}&fdr=1&q='

    # Build lookup of parent scores, indexed by fragment formulas
    df = res.ann_mols_df[res.ann_mols_df.is_detected].set_index('parent_id').drop(columns=['is_detected'])
    parents_df = df[df.is_parent]
    frags_df = df[~df.is_parent]

    # Summarize stats per group
    def get_isobar_summary(df):
        return pd.Series({
            'n_total': len(df),
            'n_confident': (df.global_enrich_uncorr <= 0.1).sum(),
            'n_unsure': ((df.global_enrich_uncorr > 0.1) & (df.global_enrich_uncorr <= 0.5)).sum(),
            'n_unlikely': (df.global_enrich_uncorr > 0.5).sum(),
        })

    parents_summary_df = parents_df.groupby('formula').apply(get_isobar_summary)
    frags_summary_df = frags_df.groupby('formula').apply(get_isobar_summary)

    summary_df = (
        parents_summary_df.add(frags_summary_df, fill_value=0)
        .merge(parents_summary_df, how='left', left_index=True, right_index=True, suffixes=('', '_p'))
        .merge(frags_summary_df, how='left', left_index=True, right_index=True, suffixes=('', '_f'))
        .fillna(0)
        .assign(href=lambda df: href_base + df.index)
    )

    # Pick interesting groups
    can_pick_one = summary_df[(summary_df.n_confident_p == 1) & (summary_df.n_confident_f == 0) & (summary_df.n_unsure == 0) & (summary_df.n_total > 1)]
    can_refine = summary_df[(summary_df.n_confident > 0) & (summary_df.n_unlikely > 0)]
    doubtful_annotation = summary_df[(summary_df.n_confident_p == 0) & (summary_df.n_unsure_p == 0) & (summary_df.n_unlikely_p > 0)]

    def candidates_matching(values):
        # col_order = [
        #     'parent_formula', 'is_parent',
        #     'enrich_ratio', 'enrich_p', 'enrich_p_uncorr', 'tfidf_score',
        #     'mol_name', 'feature_n', 'parent_num_features', 'href'
        # ]
        col_order = [
            'is_parent', 'frag_idx',
            'tfidf_score',
            'global_enrich_p', 'global_enrich_uncorr',
            'group_enrich_p', 'group_enrich_uncorr',
            'mol_name', 'parent_n_detected', 'parent_n_frags', 'parent_n_frags_unfiltered', 'href', 'mol_href']
        return (
            pd.concat([
                parents_df[parents_df.formula.isin(values)],
                frags_df[frags_df.formula.isin(values)],
            ])
            .assign(href=lambda df: href_base + df.formula)
            .sort_values(['formula', 'group_enrich_p', 'is_parent'])
            .rename_axis(index='parent_id')
            .reset_index()
            .set_index(['formula', 'parent_id'])
            [col_order]
        )

    can_pick_one_ids = candidates_matching(can_pick_one.index)
    can_refine_ids = candidates_matching(can_refine.index)
    doubtful_annotation_ids = candidates_matching(doubtful_annotation.index)
    summary_df_ids = candidates_matching(summary_df.index)
    summary_by_id = summary_df_ids.reset_index().set_index(['parent_id', 'formula']).sort_index()

    res.export_data['One good assignment anns'] = can_pick_one
    res.export_data['One good assignment mols'] = can_pick_one_ids
    res.export_data['Good split anns'] = can_refine
    res.export_data['Good split mols'] = can_refine_ids
    res.export_data['Doubtful annotation anns'] = doubtful_annotation
    res.export_data['Doubtful annotation mols'] = doubtful_annotation_ids
    res.export_data['All anns'] = summary_df
    res.export_data['All mols'] = summary_df_ids
    res.export_data['All mols by id'] = summary_by_id



find_interesting_groups(test_results)
#%%

def export(base_dir: str, res: DSResults):

    SCALE_ONE_TO_ZERO = {
        'type': '3_color_scale',
        'min_type': 'num', 'min_value': 0.0, 'min_color': '#63BE7B',
        'mid_type': 'num', 'mid_value': 0.5, 'mid_color': '#FFEB84',
        'max_type': 'num', 'max_value': 1.0, 'max_color': '#F8696B',
    }
    SCALE_DEFAULT = {
        'type': '3_color_scale',
        'min_color': '#FFFFFF',
        'mid_color': '#FFEB84',
        'max_color': '#63BE7B',
    }
    column_formats = {
        'global_enrich_p': SCALE_ONE_TO_ZERO,
        'global_enrich_uncorr': SCALE_ONE_TO_ZERO,
        'group_enrich_p': SCALE_ONE_TO_ZERO,
        'group_enrich_uncorr': SCALE_ONE_TO_ZERO,
        'feature_n': None,
        'parent_num_features': None,
    }

    def to_excel_colorize(writer, df, index=True, header=True, **kwargs):
        if not df.empty:
            df.to_excel(writer, index=index, header=header, **kwargs)

            worksheet = writer.book.worksheets()[-1]
            index_cols = df.index.nlevels if index else 0
            header_rows = df.columns.nlevels if header else 0
            indexes = [(name or i, df.index.get_level_values(i).dtype, df.index.get_level_values(i)) for i, name in enumerate(df.index.names)]
            columns = [(name, df.dtypes[name], df[name]) for name in df.columns]
            for col_i, (name, dtype, values) in enumerate([*indexes, *columns]):
                if np.issubdtype(dtype.type, np.number):
                    options = column_formats.get(name, SCALE_DEFAULT)
                    # options = {'type': '3_color_scale'}
                    if options:
                        worksheet.conditional_format(header_rows, col_i, worksheet.dim_rowmax, col_i, options)

                width = max(10, len(name))
                if not np.issubdtype(dtype.type, np.number):
                    for v in values:
                        width = min(max(width, len(str(v)) * 3 // 2), 50)
                worksheet.set_column(col_i, col_i, width=width)

    out_file = Path(f'./mol_scoring/{base_dir}/{res.ds_id}_{res.name}.xlsx')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_file, engine='xlsxwriter') as writer:
        for sheet_name, data in res.export_data.items():
            to_excel_colorize(writer, data, sheet_name=sheet_name)

export('test', test_results)
# %%

@lru_cache(maxsize=None)
def get_ds_results(_ds_id):
    res = fetch_ds_results(_ds_id)
    calc_coloc_matrix(res)
    make_result_dfs(res)
    add_tfidf_score(res)
    add_enrichment_analysis(res)
    find_interesting_groups(res)
    return res

# %%

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

spotting_ds_ids = [
    # '2020-05-14_16h32m01s',
    # '2020-05-14_16h32m04s',
    # '2020-05-14_16h32m07s',
    # '2020-05-14_16h32m10s',
    # '2020-05-14_16h32m14s',
    # '2020-05-14_16h32m16s',
    # '2020-05-14_16h32m19s',
    # '2020-05-14_16h32m22s',
    # '2020-05-14_16h32m26s',
    '2020-06-19_16h38m19s',
    '2020-06-19_16h39m01s',
    '2020-06-19_16h39m02s',
    '2020-06-19_16h39m04s',
    '2020-06-19_16h39m06s',
    '2020-06-19_16h39m08s',
    '2020-06-19_16h39m10s',
    '2020-06-19_16h39m12s',
    '2020-06-19_16h39m14s',
]
spotting_short_names = {
    '2020-06-19_16h38m19s': 'ME Lipid Array DAN+',
    '2020-06-19_16h39m01s': 'ME Lipid Array DAN-',
    '2020-06-19_16h39m02s': 'ME Spotted Array B DHB+',
    '2020-06-19_16h39m04s': 'ME SlideD DHB+',
    '2020-06-19_16h39m06s': 'ME SlideE DAN-',
    '2020-06-19_16h39m08s': 'MZ DHB+ (120-720)',
    '2020-06-19_16h39m10s': 'MZ DHB+ (60-360)',
    '2020-06-19_16h39m12s': 'MZ DAN- (50-300)',
    '2020-06-19_16h39m14s': 'MZ DAN- (140-800)'
}

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

# %%
# Batch processing

def run(_ds_id, base_dir):
    res = get_ds_results(_ds_id)
    export(base_dir, res)
# for ds_id in whole_body_ds_ids:
#     run(ds_id, 'whole_body')
# for ds_id in spotting_ds_ids:
#     run(ds_id, 'spotting')
# for ds_id in high_quality_ds_ids:
#     run(ds_id, 'high_quality')


#%%
# Plot distributions of various values
# plt.hist(test_results.mols_df.global_enrich_uncorr, bins=10)
plt.xlim(0, 1)
for ds_id in spotting_ds_ids[5:]:
    ds = get_ds_results(ds_id)
    sns.kdeplot(ds.ann_mols_df[lambda df: ~df.is_parent & df.is_detected].coloc_to_parent.rename(spotting_short_names[ds_id]), clip=(0.0,1.0))

out = Path('./mol_scoring/stats/coloc_to_parent/spotting_MZ.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out))
plt.show()

#%%
# Plot distributions of various values
# plt.hist(test_results.mols_df.global_enrich_uncorr, bins=10)
plt.xlim(0, 1)
for ds_id in spotting_ds_ids[:5]:
    ds = get_ds_results(ds_id)
    sns.kdeplot(ds.ann_mols_df[lambda df: ~df.is_parent & df.is_detected].coloc_to_parent.rename(spotting_short_names[ds_id]), clip=(0.0,1.0))

out = Path('./mol_scoring/stats/coloc_to_parent/spotting_ME.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out))
plt.show()

#%%
# Plot distributions of various values
# plt.hist(test_results.mols_df.global_enrich_uncorr, bins=10)
plt.xlim(0, 1)
for ds_id in whole_body_ds_ids[:5]:
    ds = get_ds_results(ds_id)
    sns.kdeplot(ds.ann_mols_df[lambda df: ~df.is_parent & df.is_detected].coloc_to_parent.rename(ds.name), clip=(0.0,1.0))

out = Path('./mol_scoring/stats/coloc_to_parent/whole_body_1.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out))
plt.show()

#%%
dict((ds_id, get_ds_results(ds_id).name) for ds_id in spotting_ds_ids)
#%%