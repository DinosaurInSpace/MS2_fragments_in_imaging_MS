# pip install numpy pandas scipy sklearn enrichmentanalysis-dvklopfenstein metaspace2020
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from enrichmentanalysis.enrich_run import EnrichmentRun

from MSMS_scoring_datasets import whole_body_ds_ids, spotting_ds_ids, high_quality_ds_ids
from MSMS_scoring_fetch_data import DSResults, get_msms_results_for_ds


#%%
# matplotlib.use('qt5agg')
# Bigger plots

plt.rcParams['figure.figsize'] = 15, 10
# Wider maximum width of pandas columns (needed to see the full lists of molecules)

pd.set_option('max_colwidth', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
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


# test_results = get_msms_results_for_ds('2020-05-26_17h58m22s')
# add_tfidf_score(test_results)
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


# add_enrichment_analysis(test_results)
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

# find_interesting_groups(test_results)
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

# export('test', test_results)
# %%

@lru_cache(maxsize=None)
def get_ds_results(ds_id):
    res = get_msms_results_for_ds(ds_id)
    add_tfidf_score(res)
    add_enrichment_analysis(res)
    find_interesting_groups(res)
    return res

# %%

# %%
# Batch processing

def run(_ds_id, base_dir):
    res = get_ds_results(_ds_id)
    export(base_dir, res)
for ds_id in whole_body_ds_ids:
    run(ds_id, 'whole_body')
for ds_id in spotting_ds_ids:
    run(ds_id, 'spotting')
for ds_id in high_quality_ds_ids:
    run(ds_id, 'high_quality')



#%%