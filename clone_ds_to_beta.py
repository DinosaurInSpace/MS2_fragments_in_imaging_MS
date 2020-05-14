import json, time, ast
from metaspace.sm_annotation_utils import get_config, GraphQLClient
# from metaspace.sm_annotation_utils import SMInstance

def copy_beta(ds_id_in, adducts_in, input_db):
    # Copies dataset from production to beta, searches adducts and db.
    # adducts = 'HNaK', 'HNaKM', 'M'
    # input_db = 'core_metabolome_v3'

    #sm = SMInstance()
    #sm

    f = open('/Users/dis/.metaspace.json', "r")
    secret = (f.read())
    secret = secret.replace('\n', '')
    secret = ast.literal_eval(secret)
    f.close()

    email = secret['email']
    password = secret['password']
    base_dataset_id = ds_id_in # Get this from the url in the "Annotations" page when filtering for the dataset. It should look like this: 2019-11-04_15h23m33s

    #sm.login(secret['email'], secret['password'])

    prod_gql = GraphQLClient(get_config(host='https://metaspace2020.eu', email=email, password=password))
    beta_gql = GraphQLClient(get_config(host='https://beta.metaspace2020.eu', email=email, password=password))

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
    if ds['principalInvestigator'] and not ds['principalInvestigator'].get('email'):
        ds['principalInvestigator']['email'] = 'some@email.address'

    beta_user_id = beta_gql.query(
        """
        query {
          currentUser {id}
        }
        """)['currentUser']['id']

    if adducts_in == 'HNaK':
        adducts = ds['adducts']
    elif adducts_in == 'HNaKM':
        adducts = ds['adducts']
        new_adduct = '[M]+' if metadata['MS_Analysis']['Polarity'] == 'Positive' else '[M]-'
        if new_adduct not in adducts:
            adducts.append(new_adduct)
    else:
        new_adduct = '[M]+' if metadata['MS_Analysis']['Polarity'] == 'Positive' else '[M]-'
        adducts = [new_adduct]

    result = beta_gql.query(
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
                'molDBs': [input_db],
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

    # Should print and return new dsid on betaserver
    ds_id_out = dict(result)['createDataset']
    ds_id_out = ds_id_out.split(":")[1].split(",")[0].replace('"', '')
    output = {'ds_id_in': ds_id_in, 'ds_id_out': ds_id_out}
    print(output)
    return output
    time.sleep(1)