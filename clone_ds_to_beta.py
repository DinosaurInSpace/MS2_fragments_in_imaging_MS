import json, getpass, time

from metaspace.sm_annotation_utils import get_config, GraphQLClient

email = 'x@embl.de' # Put your email address here. This assumes the same email/password on both production and staging
password = getpass.getpass(prompt='Password: ', stream=None) # This will prompt for your password in the console
base_dataset_id = '2019-10-30_09h15m17s' # Get this from the url in the "Annotations" page when filtering for the dataset. It should look like this: 2019-11-04_15h23m33s

prod_gql = GraphQLClient(get_config('https://metaspace2020.eu', email, password))
beta_gql = GraphQLClient(get_config('https://beta.metaspace2020.eu', email, password))

result = prod_gql.query(
    """
    query editDatasetQuery($id: String!) {
      dataset(id: $id) {
        id
        name
        metadataJson
        configJson
        isPublic
        inputPath
        group { id }
        submitter { id }
        principalInvestigator { name email }
        molDBs
        adducts
      }
    }
    """,
    {'id': base_dataset_id}
)
ds = result['dataset']
config = json.loads(ds['configJson'])
metadata = json.loads(ds['metadataJson'])

beta_user_id = beta_gql.query(
    """
    query {
      currentUser {id}
    }
    """)['currentUser']['id']

adducts = ds['adducts']
# Uncomment if you want to add the `[M]+`/`[M]-` adduct
# new_adduct = '[M]+' if metadata['MS_Analysis']['Polarity'] == 'Positive' else '[M]-'
# if new_adduct not in adducts:
#     adducts.append(new_adduct)
# Or alternatively, if you only want [M]+/[M]-:
# new_adduct = '[M]+' if metadata['MS_Analysis']['Polarity'] == 'Positive' else '[M]-'
# adducts = [new_adduct]

beta_gql.query(
    """
    mutation ($input: DatasetCreateInput!) {
      createDataset(input: $input)
    }
    """,
    {
        'input': {
            'name': ds['name'] + f" (cloned from {base_dataset_id})",
            'inputPath': ds['inputPath'],
            'metadataJson': ds['metadataJson'],
            'molDBs': ['core_metabolome_v3'],
            'adducts': adducts,
            'submitterId': beta_user_id,
            'principalInvestigator': ds['principalInvestigator'],
            'isPublic': False,
            # You can also add a project ID if you want, e.g.
            # 'projectIds': ['d3c7dc98-013e-11ea-9e6f-b79984c7f8d4'],
        }
    }
)
# In case you put this in a loop to copy multiple datasets, always sleep 1 second between new datasts
# or else METASPACE may throw an error
time.sleep(1)