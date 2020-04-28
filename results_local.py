
'''
Lachlan  2:44 PM
It looks like the python client can't handle cases where a candidate molecule has a None url.
As an interim workaround, I'd suggest making a local copy of the results() method that you're using from here:
https://github.com/metaspace2020/metaspace/blob/master/metaspace/python-client/metaspace/sm_annotation_utils.py#L666-L711

Four changes are needed:
* Remove the self argument on line 666 so that you can call it from outside of a class
* Change records = self._gqclient.getAnnotations( to sm._gqclient.getAnnotations( on line 681 to remove the dependency on the SMDataset class
* Change self.id to the dataset id on line 683
* Remove the moleculeIds= assignment on line 695 - that's where the bug is.

'''

import pandas as pd
from metaspace.sm_annotation_utils import SMInstance
sm = SMInstance()


def results(database, fdr=None, coloc_with=None):
    if coloc_with:
        assert fdr
        coloc_coeff_filter = {
            'database': database,
            'colocalizedWith': coloc_with,
            'fdrLevel': fdr,
        }
        annotation_filter = coloc_coeff_filter.copy()
    else:
        coloc_coeff_filter = None
        annotation_filter = {'database': database, 'hasHiddenAdduct': True

}
        if fdr:
            annotation_filter['fdrLevel'] = fdr

    records = sm._gqclient.getAnnotations(
        annotationFilter=annotation_filter,
        datasetFilter={'ids': '2017-05-17_19h49m04s'}  # Hardcoded dsid!
        #colocFilter=coloc_coeff_filter,
    )
    if not records:
        return pd.DataFrame()

    df = pd.io.json.json_normalize(records)
    return (
        df.assign(
            moleculeNames=df.possibleCompounds.apply(
                lambda lst: [item['name'] for item in lst]
            ),
            intensity=df.isotopeImages.apply(lambda imgs: imgs[0]['maxIntensity']),
        )
            .drop(columns=['possibleCompounds', 'dataset.id', 'dataset.name', 'offSampleProb' ,])
            .rename(
            columns={
                'sumFormula': 'formula',
                'msmScore': 'msm',
                'rhoChaos': 'moc',
                'fdrLevel': 'fdr',
                'colocalizationCoeff': 'colocCoeff',
            }
        )
            .set_index(['formula', 'adduct'])
    )