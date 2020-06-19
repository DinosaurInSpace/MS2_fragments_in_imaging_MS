# pip install numpy pandas scipy sklearn enrichmentanalysis-dvklopfenstein metaspace2020
from collections import defaultdict
from pathlib import Path
import re
import numpy as np
import pandas as pd
from metaspace.sm_annotation_utils import SMInstance, SMDataset
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from enrichmentanalysis.enrich_run import EnrichmentRun
#%%
 # Bigger plots
# plt.rcParams['figure.figsize'] = 15, 10
# Wider maximum width of pandas columns (needed to see the full lists of molecules)
pd.set_option('max_colwidth', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
#%%
sm = SMInstance(host='https://beta.metaspace2020.eu')
#%%
ds_id: str = '2020-05-26_17h58m22s'
sm_ds: SMDataset = None
db_id: str = None
ds_name: str = None
ds_results: pd.DataFrame = None


def get_ds_results():
    global ds_id, sm_ds, db_id, ds_name, ds_results
    sm_ds = sm.dataset(id=ds_id)
    db_id = [db for db in sm_ds.databases if re.match(r'^\d', db)][0]
    ds_name = re.sub('[\W ]+', '_', sm_ds.name)
    ds_results = sm_ds.results(database=db_id)


# get_ds_results()
#%%
fragments_df: pd.DataFrame = None

def make_fragments_df():
    global fragments_df, df
    PARSE_MOL_ID = re.compile(r'([^_]+)_(\d+)([pf])')

    # Get detected IDs from dataset
    detected_ids = set()
    parent_ids = set()
    for mol_ids in ds_results.moleculeIds:
        detected_ids.update(mol_ids)
        parent_ids.update(PARSE_MOL_ID.match(mol_id).groups()[0] for mol_id in mol_ids)

    df = pd.read_pickle('./cm3_msms_all_both.pickle')
    # Exclude fragments of the wrong polarity
    df = df[df.polarity == sm_ds.polarity.lower()]

    df['parent_id'] = df.id.str.replace(PARSE_MOL_ID, lambda m: m[1])
    df['frag_idx'] = df.id.str.replace(PARSE_MOL_ID, lambda m: m[2]).astype(np.int32)
    df['is_parent'] = df.id.str.replace(PARSE_MOL_ID, lambda m: m[3]) == 'p'
    df['is_detected'] = df.id.isin(detected_ids)
    df['mol_name'] = df.name.str.replace("^[^_]+_[^_]+_", "")
    df['mol_href'] = 'https://hmdb.ca/metabolites/' + df.parent_id
    df['in_range'] = (df.ion_mass >= ds_results.mz.min() - 0.1) & (df.ion_mass <= ds_results.mz.max() + 0.1)

    df = df.merge(pd.DataFrame({
        'parent_n_detected': df.groupby('parent_id').is_detected.sum().astype(np.uint32),
        'parent_n_frags': df.groupby('parent_id').in_range.sum().astype(np.uint32),
        'parent_n_frags_unfiltered': df.groupby('parent_id').frag_idx.max(),
    }), on='parent_id')

    # Exclude molecules with no hits
    df = df[df.parent_id.isin(parent_ids)]
    # Exclude fragments that are outside of detected mz range
    # 0.1 buffer added because centroiding can move the peaks slightly
    df = df[df.in_range]

    fragments_df = df[[
        'id', 'parent_id', 'frag_idx', 'is_parent', 'is_detected', 'mol_name', 'formula',
        'parent_n_detected', 'parent_n_frags', 'parent_n_frags_unfiltered', 'db_n_isobar',
        'mol_href',
    ]].set_index('id')

# make_fragments_df()
#%%

def add_tfidf_score():
    """
    TF-IDF where:
    * each parent molecule is a "document"
    * each annotated formula is a "term"
    * a term is only in a document when it is a parent ion or a predicted fragment ion
    * a term's "frequency" in a document is either constant 1, or 1 / (number of predicted ions for this parent ion)

    Caveats:
    * Colocalization isn't taken into account. It could be included by using it to scale the "frequency"...
    * The output value is just summed per document to produce a score. May not be optimal.
    """
    global fragments_df
    terms_df = (
        fragments_df
        # [lambda df: ~df.is_parent]
        # To use constant value:
        # .assign(value=1)
        # To use value based on 1/num_features:
        # Note that this equally scales the parent annotation
        .assign(value=fragments_df.parent_n_frags.rdiv(1) * fragments_df.is_detected)
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

    fragments_df = fragments_df.merge(tfidf_s, how='left', left_on='parent_id', right_index=True)


# add_tfidf_score()
#%%

def add_enrichment_analysis():
    """
    Over-representation analysis where:
    * The "population" is all predicted ion formulas
    * The "study set" is all observed ion formulas
    * "Associations" are groups linking each ion formula to potential parent molecules

    Caveats:
    * P-values are too harshly corrected because they're corrected across the whole population
      instead of across groups of competing candidates
    * Too many fragments are considered - many will be below the threshold of detection or
      out of m/z range. This reduces enrichment power.
    * P-values aren't corrected enough because the data acquisition step threw out candidate molecules
      that don't have an annotated parent ion
    * Colocalization is not considered, even though it can be a compelling way to disprove fragments
      association with their parents
    * A bad P-value is not evidence that the candidate molecule is wrong. It only indicates the
      possibility that the observed fragment distribution was due to random chance.
    """
    global fragments_df, parents_df

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
            prefix + 'enrich_ratio': raw_data.study_ratio / raw_data.pop_ratio,
            prefix + 'enrich_p': raw_data.sm_bonferroni,
            prefix + 'enrich_uncorr': raw_data.pval_uncorr,
        }, index=raw_data.index, dtype='f')


    enrich_data = enrich(fragments_df, 'global_')
    fragments_df = fragments_df.merge(enrich_data, how='left', left_on='parent_id', right_index=True)

    mini_df = fragments_df[['parent_id', 'formula', 'is_detected']]
    detected_formulas = set(mini_df[mini_df.is_detected].formula)
    group_enrich_datas = []
    for formula, formula_df in mini_df[mini_df.formula.isin(detected_formulas)].groupby('formula'):
        related_df = pd.concat([formula_df, mini_df[mini_df.parent_id.isin(formula_df.parent_id)]])
        group_enrich_data = enrich(related_df, 'group_').assign(formula=formula).set_index('formula', append=True)
        group_enrich_datas.append(group_enrich_data)

    fragments_df = fragments_df.merge(
        pd.concat(group_enrich_datas),
        how='left',
        left_on=['parent_id', 'formula'],
        right_index=True
    )


# add_enrichment_analysis()
#%%
can_pick_one: pd.DataFrame = None
can_pick_one_ids: pd.DataFrame = None
can_refine: pd.DataFrame = None
can_refine_ids: pd.DataFrame = None
doubtful_annotation: pd.DataFrame = None
doubtful_annotation_ids: pd.DataFrame = None


def find_interesting_groups():
    global summary_df, summary_df_ids
    global can_pick_one, can_pick_one_ids
    global can_refine, can_refine_ids
    global doubtful_annotation, doubtful_annotation_ids

    href_base = f'https://beta.metaspace2020.eu/annotations?ds={ds_id}&db={db_id}&q='

    # Build lookup of parent scores, indexed by fragment formulas
    df = fragments_df[fragments_df.is_detected].set_index('parent_id').drop(columns=['is_detected'])
    parents_df = df[df.is_parent]
    frags_df = df[~df.is_parent]

    # Summarize stats per group
    def get_isobar_summary(df):
        return pd.Series({
            'n_total': len(df),
            'n_confident': (df.global_enrich_p <= 0.1).sum(),
            'n_unsure': ((df.global_enrich_p > 0.1) & (df.global_enrich_p <= 0.5)).sum(),
            'n_unlikely': (df.global_enrich_p > 0.5).sum(),
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
            'global_enrich_ratio', 'global_enrich_p', 'global_enrich_uncorr',
            'group_enrich_ratio', 'group_enrich_p', 'group_enrich_uncorr',
            'mol_name', 'parent_n_detected', 'parent_n_frags', 'parent_n_frags_unfiltered', 'db_n_isobar', 'href', 'mol_href']
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


# find_interesting_groups()
#%%

def export(base_dir):

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
        'global_enrich_p_uncorr': SCALE_ONE_TO_ZERO,
        'group_enrich_p': SCALE_ONE_TO_ZERO,
        'group_enrich_p_uncorr': SCALE_ONE_TO_ZERO,
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


    out_file = Path(f'./mol_scoring/{base_dir}/{ds_id}_{ds_name}.xlsx')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_file, engine='xlsxwriter') as writer:
        to_excel_colorize(writer, can_pick_one, sheet_name='One good assignment anns')
        to_excel_colorize(writer, can_pick_one_ids, sheet_name='One good assignment mols')
        to_excel_colorize(writer, can_refine, sheet_name='Good split anns')
        to_excel_colorize(writer, can_refine_ids, sheet_name='Good split mols')
        to_excel_colorize(writer, doubtful_annotation, sheet_name='Doubtful annotation anns')
        to_excel_colorize(writer, doubtful_annotation_ids, sheet_name='Doubtful annotation mols')
        to_excel_colorize(writer, summary_df, sheet_name='All anns')
        to_excel_colorize(writer, summary_df_ids, sheet_name='All mols')
        summary_by_id = summary_df_ids.reset_index().set_index(['parent_id', 'formula']).sort_index()
        to_excel_colorize(writer, summary_by_id, sheet_name='All mols by id')

# export('test')
# %%
# Batch processing

def run(_ds_id, base_dir):
    global ds_id
    ds_id = _ds_id
    get_ds_results()
    make_fragments_df()
    add_tfidf_score()
    add_enrichment_analysis()
    find_interesting_groups()
    export(base_dir)

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

for ds_id in whole_body_ds_ids:
    run(ds_id, 'whole_body')
for ds_id in spotting_ds_ids:
    run(ds_id, 'spotting')
for ds_id in high_quality_ds_ids:
    run(ds_id, 'high_quality')


#%%