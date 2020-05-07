from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import copyfileobj
from typing import Optional, List

import pandas as pd
import numpy as np
import requests, json, os, pprint
from copy import deepcopy
from io import BytesIO
from PIL import Image

try:
    from typing import TypedDict  # Requires Python 3.8
except ImportError:
    TypedDict = dict


class DatasetDownloadLicense(TypedDict):
    code: str
    name: str
    link: Optional[str]


class DatasetDownloadContributor(TypedDict):
    name: Optional[str]
    institution: Optional[str]


class DatasetDownloadFile(TypedDict):
    filename: str
    link: str


class DatasetDownload(TypedDict):
    license: DatasetDownloadLicense
    contributors: List[DatasetDownloadContributor]
    files: List[DatasetDownloadFile]


def _extract_data(res):
    if not res.headers.get('Content-Type').startswith('application/json'):
        raise Exception('Wrong Content-Type: {}'.format(res.headers.get('Content-Type')))
    res_json = res.json()
    if 'data' in res_json and 'errors' not in res_json:
        return res_json['data']
    else:
        if 'errors' in res_json:
            pprint.pprint(res_json['errors'])
            raise Exception(res_json['errors'][0]['message'])
        elif 'message' in res_json:
            raise Exception(res_json['message'])
        else:
            pprint.pprint(res_json)
            raise Exception('Invalid response from server')


def get_config(
    host=None, verify_certificate=True, email=None, password=None, api_key=None, config_path=None
):
    default_config_path = Path.home() / '.metaspace'
    try:
        from configparser import ConfigParser

        if config_path:
            config_file = Path(config_path).read_text()
        else:
            config_file = default_config_path.read_text()

        config_parser = ConfigParser()
        config_parser.read_string("[metaspace]\n" + config_file)
        config = config_parser['metaspace']

        if not host:
            host = config.get('host')
        if not email and not api_key:
            email = config.get('email')
            password = config.get('password')
            api_key = config.get('api_key')
    except Exception as ex:
        if config_path:
            raise
        if default_config_path.exists():
            print('Error processing ~/.metaspace config file')
            raise

    host = host or 'https://metaspace2020.eu'
    return {
        'host': host,
        'graphql_url': '{}/graphql'.format(host),
        'moldb_url': '{}/mol_db/v1'.format(host),
        'signin_url': '{}/api_auth/signin'.format(host),
        'gettoken_url': '{}/api_auth/gettoken'.format(host),
        'usr_email': email,
        'usr_pass': password,
        'usr_api_key': api_key,
        'verify_certificate': verify_certificate,
    }


class GraphQLClient(object):
    def __init__(self, config):
        self._config = config
        self.host = self._config['host']
        self.session = requests.Session()
        self.session.verify = self._config['verify_certificate']
        self.logged_in = False

        if self._config.get('usr_api_key'):
            self.logged_in = self.query("query { currentUser { id } }") is not None
        elif self._config['usr_email']:
            login_res = self.session.post(
                self._config['signin_url'],
                params={"email": self._config['usr_email'], "password": self._config['usr_pass']},
            )
            if login_res.status_code == 401:
                print('Login failed. Only public datasets will be accessible.')
            elif login_res.status_code:
                self.logged_in = True
            else:
                login_res.raise_for_status()

    def query(self, query, variables={}):
        api_key = self._config.get('usr_api_key')
        if api_key:
            headers = {'Authorization': f'Api-Key {api_key}'}
        else:
            headers = {'Authorization': 'Bearer ' + self.get_jwt()}

        res = self.session.post(
            self._config['graphql_url'],
            json={'query': query, 'variables': variables},
            headers=headers,
            verify=self._config['verify_certificate'],
        )
        return _extract_data(res)

    def get_jwt(self):
        res = self.session.get(
            self._config['gettoken_url'], verify=self._config['verify_certificate']
        )
        res.raise_for_status()
        return res.text

    def get_submitter_id(self):
        query = """
        query {
          currentUser {
            id
          }
        }
        """
        return self.query(query)['currentUser']['id']

    def iterQuery(self, query, variables={}, batch_size=50000):
        """
        Assumes query has $offset and $limit parameters,
        and yields query results with these set to (k*batch_size, batch_size)
        """
        # FIXME: use scroll api?
        assert batch_size >= 100, "too small batch size, must be at least 100"
        offset = 0
        v = deepcopy(variables)
        while True:
            v['limit'] = batch_size
            v['offset'] = offset
            yield self.query(query, v)
            offset += batch_size

    def listQuery(self, field_name, query, variables={}, batch_size=50000):
        """
        Gets all results of an iterQuery as a list.
        Field name must be provided in addition to the query (e.g. 'allDatasets')
        """
        records = []
        for res in self.iterQuery(query, variables, batch_size):
            if not res[field_name]:
                break
            records.extend(res[field_name])
        return records

    DATASET_FIELDS = """
        id
        name
        uploadDT
        submitter {
          name
        }
        group {
          id
          name
          shortName
        }
        principalInvestigator {
          name
        }
        projects {
            name
        }
        polarity
        ionisationSource
        analyzer {
          type
          resolvingPower(mz: 400)
        }
        organism
        organismPart
        condition
        growthConditions
        maldiMatrix
        configJson
        metadataJson
        isPublic
        molDBs
        adducts
        acquisitionGeometry
        metadataType
        status
        inputPath
    """

    ANNOTATION_FIELDS = """
        sumFormula
        neutralLoss
        adduct
        ionFormula
        ion
        mz
        msmScore
        rhoSpatial
        rhoSpectral
        rhoChaos
        fdrLevel
        offSample
        offSampleProb
        dataset { id name }
        possibleCompounds { name information { url databaseId } }
        isotopeImages { mz url maxIntensity totalIntensity }
    """

    DEFAULT_ANNOTATION_FILTER = {
        'database': 'HMDB-v4',
        'hasNeutralLoss': False,
        'hasChemMod': False,
        'hasHiddenAdduct': False,
    }

    def getDataset(self, datasetId):
        query = (
            """
        query datasetInfo($id: String!) {
          dataset(id: $id) {
        """
            + self.DATASET_FIELDS
            + """
          }
        }
        """
        )
        match = self.query(query, {'id': datasetId})['dataset']
        if not match:
            if self.logged_in:
                raise DatasetNotFound("No dataset found with id {}.".format(datasetId))
            else:
                raise DatasetNotFound(
                    "No dataset found with id {}. You are not logged in. "
                    "If the dataset is set to private, you need to log in to access it.".format(
                        datasetId
                    )
                )
        else:
            return match

    def getDatasetByName(self, datasetName):
        query = (
            """
        query datasetInfo($filter: DatasetFilter!) {
          allDatasets(filter: $filter) {
        """
            + self.DATASET_FIELDS
            + """
          }
        }
        """
        )
        matches = self.query(query, {'filter': {'name': datasetName}})['allDatasets']
        if not matches:
            if self.logged_in:
                raise DatasetNotFound("No dataset found with name '{}'.".format(datasetName))
            else:
                raise DatasetNotFound(
                    "No dataset found with name '{}'. You are not logged in. "
                    "If the dataset is set to private, you need to log in to access it.".format(
                        datasetName
                    )
                )
        elif len(matches) > 1:
            print('Found datasets:')
            for dataset in matches:
                print(dataset['name'], 'id={}'.format(dataset['id']), '\n')
            raise Exception(
                'More than 1 dataset were found with the same name, '
                'please run your code with the dataset ID instead.'
            )
        else:
            return matches[0]

    def getAnnotations(self, annotationFilter=None, datasetFilter=None, colocFilter=None):
        query_arguments = [
            "$filter: AnnotationFilter",
            "$dFilter: DatasetFilter",
            "$orderBy: AnnotationOrderBy",
            "$sortingOrder: SortingOrder",
            "$offset: Int",
            "$limit: Int",
            "$colocalizationCoeffFilter: ColocalizationCoeffFilter",
        ]

        query = """
            query getAnnotations(
                %s
            ) {
                allAnnotations(
                    filter: $filter,
                    datasetFilter: $dFilter,
                    orderBy: $orderBy,
                    sortingOrder: $sortingOrder,
                    offset: $offset,
                    limit: $limit,
                ) {
                %s
                colocalizationCoeff(colocalizationCoeffFilter: $colocalizationCoeffFilter)
                }
            }""" % (
            ','.join(query_arguments),
            self.ANNOTATION_FIELDS,
        )
        if datasetFilter is None:
            datasetFilter = {}

        if annotationFilter is None:
            annot_filter = {}
        else:
            annot_filter = deepcopy(annotationFilter)
            for key, val in self.DEFAULT_ANNOTATION_FILTER.items():
                annot_filter.setdefault(key, val)

        if colocFilter:
            annot_filter.update(colocFilter)
            order_by = 'ORDER_BY_COLOCALIZATION'
        else:
            order_by = 'ORDER_BY_MSM'

        vars = {
            'filter': annot_filter,
            'dFilter': datasetFilter,
            'colocalizationCoeffFilter': colocFilter,
            'orderBy': order_by,
        }
        return self.listQuery(field_name='allAnnotations', query=query, variables=vars)

    def countAnnotations(self, annotationFilter=None, datasetFilter=None):
        query = """
            query getAnnotationCount($filter: AnnotationFilter,
                                 $dFilter: DatasetFilter) {
              countAnnotations(
                filter: $filter,
                datasetFilter: $dFilter
              )
            }"""
        if datasetFilter is None:
            annotationFilter = {}
        if annotationFilter is None:
            annotFilter = {}
        else:
            annotFilter = deepcopy(annotationFilter)
            for key, val in self.DEFAULT_ANNOTATION_FILTER.items():
                annotFilter.setdefault(key, val)

        return self.query(query=query, variables={'filter': annotFilter, 'dFilter': datasetFilter})

    def getDatasets(self, datasetFilter={}):
        query = (
            """
        query getDatasets($filter: DatasetFilter,
                          $offset: Int, $limit: Int) {
          allDatasets(
            filter: $filter,
            offset: $offset,
            limit: $limit
          ) {
        """
            + self.DATASET_FIELDS
            + """
          }
        }"""
        )
        return self.listQuery('allDatasets', query, {'filter': datasetFilter})

    def getRawOpticalImage(self, dsid):
        query = """
            query getRawOpticalImages($datasetId: String!){ 
                rawOpticalImage(datasetId: $datasetId) 
                {
                    url, transform
                }
            }
        """
        variables = {"datasetId": dsid}
        return self.query(query, variables)

    def getRegisteredImage(self, dsid, zoom_level=8):
        query = """
            query getRawOpticalImages($datasetId: String!, $zoom: Int){ 
                rawOpticalImage(datasetId: $datasetId) 
            }
        """
        variables = {"datasetId": dsid}
        return self.query(query, variables)

    def create_dataset(
        self,
        data_path,
        metadata,
        priority=0,
        ds_name=None,
        is_public=None,
        mol_dbs=None,
        adducts=None,
        ppm=None,
        ds_id=None,
    ):
        submitter_id = self.get_submitter_id()
        query = """
            mutation createDataset($id: String, $input: DatasetCreateInput!, $priority: Int) {
                createDataset(
                  id: $id,
                  input: $input,
                  priority: $priority
                )
            }
        """
        variables = {
            'id': ds_id,
            'input': {
                'name': ds_name,
                'inputPath': data_path,
                'isPublic': is_public,
                'molDBs': mol_dbs,
                'adducts': adducts,
                'ppm': ppm,
                'submitterId': submitter_id,
                'metadataJson': metadata,
            },
        }
        return self.query(query, variables)

    def delete_dataset(self, ds_id, force=False):
        query = """
                mutation customDeleteDataset ($dsId: String!, $force: Boolean) {
                      deleteDataset(
                        id: $dsId
                        force: $force
                      )
                 }
                """
        variables = {'dsId': ds_id}
        return self.query(query, variables)

    def update_dataset(
        self,
        ds_id,
        name=None,
        mol_dbs=None,
        adducts=None,
        ppm=None,
        reprocess=False,
        force=False,
        priority=1,
    ):
        query = """
            mutation updateMetadataDatabases($id: String!, $reprocess: Boolean, 
                $input: DatasetUpdateInput!, $priority: Int, $force: Boolean) {
                    updateDataset(
                      id: $id,
                      input: $input,
                      priority: $priority,
                      reprocess: $reprocess,
                      force: $force
                    )
            }
        """
        input_field = {}
        if name:
            input_field['name'] = name
        if mol_dbs:
            input_field['molDBs'] = mol_dbs
        if adducts:
            input_field['adducts'] = adducts
        if ppm:
            input_field['ppm'] = ppm
        variables = {
            'id': ds_id,
            'input': input_field,
            'priority': priority,
            'reprocess': reprocess,
            'force': force,
        }

        return self.query(query, variables)

    def get_molecular_databases(self, names=None):
        query = '''{
          molecularDatabases {
            id name version
          }
        }'''
        resp = self.query(query)
        if 'molecularDatabases' not in resp:
            raise Exception('GraphQL error: {}'.format(resp))
        mol_dbs = resp['molecularDatabases']
        if not names:
            return mol_dbs
        else:
            return [d for d in mol_dbs if d['name'] in names]


def ion(r):
    from pyMSpec.pyisocalc.tools import normalise_sf

    return (r.ds_id, normalise_sf(r.sf), r.adduct, 1)


class IsotopeImages(object):
    def __init__(self, images, sf, adduct, centroids, urls):
        self._images = images
        # Keeping the private self._sf, self._adduct fields for backwards compatibility. There is no other easy way
        # to get that data, so it's probable that there is code somewhere that depends on the private fields.
        self.formula = self._sf = sf
        self.adduct = self._adduct = adduct
        self._centroids = centroids
        self._urls = urls

    def __getitem__(self, index):
        return self._images[index]

    def __repr__(self):
        return "IsotopeImages({}{})".format(self._sf, self._adduct)

    def __len__(self):
        return len(self._images)

    def __bool__(self):
        return len(self._images) > 0

    def peak(self, index):
        return self._centroids[index]

    def plot(self, n_images=-1):
        import matplotlib.pyplot as plt

        if n_images > 0:
            n_images = min(n_images, len(self))
        else:
            n_images = len(self)
        for i in range(n_images):
            plt.subplot(1, len(self._images), i + 1)
            plt.title(round(self.peak(i), 4))
            plt.axis('off')
            plt.imshow(self._images[i], interpolation='none', cmap='viridis')


class OpticalImage(object):
    def __init__(self, image, registered_image):
        self._images = [image]
        self._transforms = [registered_image]
        self._itransforms = [np.linalg.inv(t) for t in self._transforms]

    def __getitem__(self, index):
        return self._images[index]

    def __len__(self):
        return len(self._images)

    def _tform_to_tuple(self, tform):
        return (tform[0, 0], tform[0, 1], tform[0, 2], tform[1, 0], tform[1, 1], tform[1, 2])

    def _to_rgb(self, image):
        return Image.fromarray(255 * image / image.max()).convert('RGB')

    def _transform(self, image, transform, target_image_shape):
        """
        :param image:
        :param transform:
        :return:
        """
        im_t = image.transform(
            target_image_shape,
            Image.AFFINE,
            self._tform_to_tuple(transform),
            resample=Image.NEAREST,
        )
        return im_t

    def to_ion_image(self, index, ion_image_shape):
        return self._transform(
            Image.fromarray(self[index]),
            self._transforms[index],
            (ion_image_shape[1], ion_image_shape[0]),
        )

    def ion_image_to_optical(self, ion_image, index=0):
        return self._transform(
            self._to_rgb(ion_image),
            self._itransforms[index],
            (self._images[index].shape[1], self._images[index].shape[0]),
        )


class Metadata(object):
    def __init__(self, json_metadata):
        self._json = json_metadata

    @property
    def json(self):
        return self._json


NYI = Exception("NOT IMPLEMENTED YET")


class SMDataset(object):
    def __init__(self, _info, gqclient):
        self._info = _info
        self._gqclient = gqclient
        self._config = json.loads(self._info['configJson'])
        self._metadata = Metadata(self._info['metadataJson'])
        self._session = requests.session()

    @property
    def id(self):
        return self._info['id']

    @property
    def name(self):
        return self._info['name']

    @property
    def s3dir(self):
        return self._info['inputPath']

    def __repr__(self):
        return "SMDataset({} | ID: {})".format(self.name, self.id)

    def annotations(
        self, fdr=0.1, database=None, return_vals=('sumFormula', 'adduct'), **annotation_filter
    ):
        annotation_filter.setdefault('fdrLevel', fdr)
        if database:
            annotation_filter['database'] = database
        dataset_filter = {'ids': self.id}

        records = self._gqclient.getAnnotations(annotation_filter, dataset_filter)
        return [list(r[val] for val in return_vals) for r in records]

    def results(self, database, fdr=None, coloc_with=None):
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
            annotation_filter = {'database': database}
            if fdr:
                annotation_filter['fdrLevel'] = fdr

        records = self._gqclient.getAnnotations(
            annotationFilter=annotation_filter,
            datasetFilter={'ids': self.id},
            colocFilter=coloc_coeff_filter,
        )
        if not records:
            return pd.DataFrame()

        df = pd.io.json.json_normalize(records)
        return (
            df.assign(
                moleculeNames=df.possibleCompounds.apply(
                    lambda lst: [item['name'] for item in lst]
                ),
#               moleculeIds=df.possibleCompounds.apply(
#                   lambda lst: [item['information'][0]['databaseId'] for item in lst]
#                ),
                intensity=df.isotopeImages.apply(lambda imgs: imgs[0]['maxIntensity']),
            )
            .drop(columns=['possibleCompounds', 'dataset.id', 'dataset.name', 'offSampleProb',])
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

    @property
    def metadata(self):
        return self._metadata

    @property
    def config(self):
        return self._config

    @property
    def adducts(self):
        return self._config['isotope_generation']['adducts']

    @property
    def polarity(self):
        return 'Positive' if self._config['isotope_generation']['charge'] > 0 else 'Negative'

    @property
    def databases(self):
        return self._config['databases']

    @property
    def database(self):
        return self.databases[0]

    @property
    def status(self):
        return self._info['status']

    @property
    def _baseurl(self):
        return self._gqclient.host

    def isotope_images(self, sf, adduct, only_first_isotope=False, scale_intensity=True):
        """
        Retrieve ion images for a specific sf and adduct
        :param str   sf:
        :param str   adduct:
        :param bool  only_first_isotope: Only retrieve the first (most abundant) isotopic ion image.
                                         Typically this is all you need for data analysis, as the less abundant isotopes
                                         are usually lower quality copies of the first isotopic ion image.
        :param bool  scale_intensity:    When True, the output values will be scaled to the intensity range of the original data.
                                         When False, the output values will be in the 0.0 to 1.0 range.
        :return IsotopeImages:
        """
        records = self._gqclient.getAnnotations(
            dict(sumFormula=sf, adduct=adduct, database=None), dict(ids=self.id)
        )

        import matplotlib.image as mpimg

        def fetchImage(url):
            if not url:
                return None
            url = self._baseurl + url
            im = mpimg.imread(BytesIO(self._session.get(url).content))
            mask = im[:, :, 3]
            data = im[:, :, 0]
            data[mask == 0] = 0
            assert data.max() <= 1
            return data

        if records:
            image_metadata = records[0]['isotopeImages']
        else:
            image_metadata = []
        if only_first_isotope:
            image_metadata = image_metadata[:1]

        images = [fetchImage(r.get('url')) for r in image_metadata]
        image_mzs = [r['mz'] for r in image_metadata]
        image_urls = [r['url'] for r in image_metadata]

        if scale_intensity:
            non_empty_images = [i for i in images if i is not None]
            shape = non_empty_images[0].shape
            for i in range(len(images)):
                if images[i] is None:
                    images[i] = np.zeros(shape, dtype=non_empty_images[0].dtype)
                else:
                    images[i] *= float(image_metadata[i]['maxIntensity'])

        return IsotopeImages(images, sf, adduct, image_mzs, image_urls)

    def all_annotation_images(
        self,
        fdr=0.1,
        database=None,
        only_first_isotope=False,
        scale_intensity=True,
        **annotation_filter,
    ):
        """
        Retrieve all ion images for the dataset and given annotation filters.
        :param float fdr:                Maximum FDR level of annotations
        :param str   database:           Molecular database to use (e.g. HMDBv4)
        :param bool  only_first_isotope: Only retrieve the first (most abundant) isotopic ion image for each annotation.
                                         Typically this is all you need for data analysis, as the less abundant isotopes
                                         are usually lower quality copies of the first isotopic ion image.
        :param bool  scale_intensity:    When True, the output values will be scaled to the intensity range of the original data.
                                         When False, the output values will be in the 0.0 to 1.0 range.
        :param       annotation_filter:  Additional filters passed to `SMDataset.annotations`
        :return list[IsotopeImages]:
        """
        with ThreadPoolExecutor() as pool:

            def get_annotation_images(row):
                sf, adduct = row
                return self.isotope_images(
                    sf,
                    adduct,
                    only_first_isotope=only_first_isotope,
                    scale_intensity=scale_intensity,
                )

            return list(
                pool.map(
                    get_annotation_images,
                    self.annotations(fdr=fdr, database=database, **annotation_filter),
                )
            )

    def optical_images(self):
        def fetch_image(url):
            from PIL import Image

            if not url:
                return None
            url = self._baseurl + url
            im = Image.open(BytesIO(self._session.get(url).content))
            return np.asarray(im)

        raw_im = self._gqclient.getRawOpticalImage(self.id)['rawOpticalImage']
        return OpticalImage(fetch_image(raw_im['url']), np.asarray(raw_im['transform']))

    def download_links(self) -> Optional[DatasetDownload]:
        """Returns a data structure containing links to download the dataset's input files"""
        result = self._gqclient.query(
            '''query ($id: String!) {
            dataset (id: $id) { downloadLinkJson }
        }''',
            {'id': self.id},
        )
        download_link_json = result and result.get('dataset', {}).get('downloadLinkJson')
        if download_link_json:
            return json.loads(download_link_json)

    def download_to_dir(self, path, base_name=None):
        """
        Downloads the dataset's input files to the specified directory
        :param path: Destination directory
        :param base_name: If specified, overrides the base name (excluding extension) of each file.
                          e.g. `base_name='foo'` will name the files as 'foo.imzML' and 'foo.ibd'
        :return:
        """

        def download_link(file: DatasetDownloadFile):
            prefix, suffix = os.path.splitext(file['filename'])
            dest_path = dest_root / ((base_name or prefix) + suffix)
            if not dest_path.exists():
                response = requests.get(file['link'], stream=True)
                with dest_path.open('wb') as dest:
                    copyfileobj(response.raw, dest)
                print(f'Wrote {dest_path}')
            else:
                print(f'File already exists: {dest_path}. Skipping.')

        dest_root = Path(path)
        if not dest_root.exists():
            print(f'Making directory {dest_root}')
            dest_root.mkdir(parents=True)

        link = self.download_links()
        if link:
            # Download in parallel to minimize risk that the imzML link expires before the ibd file
            # has finished downloading
            with ThreadPoolExecutor() as ex:
                ex.map(download_link, link['files'])


class SMInstance(object):
    def __init__(self, *args, **kwargs):
        self._config = get_config(*args, **kwargs)
        if not self._config['verify_certificate']:
            import warnings
            from urllib3.exceptions import InsecureRequestWarning

            warnings.filterwarnings('ignore', category=InsecureRequestWarning)
        self.reconnect()

    def __repr__(self):
        return "SMInstance({})".format(self._config['graphql_url'])

    def login(self, email=None, password=None, api_key=None):
        assert (
            email and password
        ) or api_key, 'Either email and password, or api_key must be provided'
        self._config['usr_email'] = email
        self._config['usr_pass'] = password
        self._config['usr_api_key'] = api_key
        self.reconnect()
        assert self._gqclient.logged_in, 'Login failed'

    def reconnect(self):
        self._gqclient = GraphQLClient(self._config)
        self._es_client = None

    def logged_in(self):
        return self._gqclient.logged_in

    @property
    def projects(self):
        from metaspace.projects_client import ProjectsClient

        return ProjectsClient(self._gqclient)

    def dataset(self, name=None, id=None) -> SMDataset:
        if id:
            return SMDataset(self._gqclient.getDataset(id), self._gqclient)
        elif name:
            return SMDataset(self._gqclient.getDatasetByName(name), self._gqclient)
        else:
            raise Exception("either name or id must be provided")

    def datasets(self, nameMask='', idMask='', **kwargs) -> List[SMDataset]:
        datasetFilter = kwargs.copy()
        if nameMask != '':
            datasetFilter['name'] = nameMask
        if idMask != '':
            datasetFilter['ids'] = idMask if isinstance(idMask, str) else "|".join(idMask)

        return [
            SMDataset(info, self._gqclient) for info in self._gqclient.getDatasets(datasetFilter)
        ]

    def all_adducts(self):
        raise NYI

    def database(self, name, version=None):
        assert self._moldb_client, 'provide moldb_url in the config'
        return self._moldb_client.getDatabase(name, version)

    def databases(self):
        assert self._moldb_client, 'provide moldb_url in the config'
        return self._moldb_client.getDatabaseList()

    def metadata(self, datasets):
        """
        Pandas dataframe for a subset of datasets
        where rows are flattened metadata JSON objects
        """
        df = pd.io.json.json_normalize([d.metadata.json for d in datasets])
        df.index = [d.name for d in datasets]
        return df

    def get_annotations(self, fdr=0.1, db_name="HMDB-v4", datasetFilter={}):
        """
        Returns: a table of booleans indicating which ions were annotated in a
        particular dataset at the specified fdr.
        Only returns datasets with at least one anntotation at the given FDR level.
        Use in conjunction with get_metadata() to get a full dataset list.
        :param fdr: fdr level to export annotations at
        :param db_name: database to search against
        :return: pandas dataframe indexed by dataset ids,
                 with multi-index adduct / molecular formula on columns
        """
        records = self._gqclient.getAnnotations(
            annotationFilter={'database': db_name, 'fdrLevel': fdr}, datasetFilter=datasetFilter
        )
        df = pd.io.json.json_normalize(records)
        return pd.DataFrame(
            dict(
                formula=df['sumFormula'],
                adduct=df['adduct'],
                msm=df['msmScore'],
                moc=df['rhoChaos'],
                rhoSpatial=df['rhoSpatial'],
                rhoSpectral=df['rhoSpectral'],
                fdr=df['fdrLevel'],
                mz=df['mz'],
                dataset_id=df['dataset.id'],
                dataset_name=df['dataset.name'],
                moleculeNames=[[item['name'] for item in lst] for lst in df['possibleCompounds']],
            )
        )

    def get_metadata(self, datasetFilter={}):
        datasets = self._gqclient.getDatasets(datasetFilter=datasetFilter)
        df = pd.concat(
            [
                pd.DataFrame(pd.io.json.json_normalize(json.loads(dataset['metadataJson'])))
                for dataset in datasets
            ]
        )
        df.index = [dataset['id'] for dataset in datasets]
        return df
        # return pd.DataFrame.from_records([pd.io.json.json_normalize(json.loads(dataset['metadataJson'])) for dataset in datasets])

    def submit_dataset(
        self,
        imzml_fn,
        ibd_fn,
        metadata,
        dsid=None,
        folder_uuid=None,
        dsName=None,
        isPublic=None,
        molDBs=None,
        adducts=None,
        s3bucket=None,
        priority=0,
    ):
        """
        Submit a dataset for processing on the SM Instance
        :param imzml_fn: file path to imzml
        :param ibd_fn: file path to ibd
        :param metadata: a properly formatted metadata json string
        :param s3bucket: this should be a bucket that both the user has write permission to and METASPACE can access
        :param folder_uuid: a unique key for the dataset
        :return:
        """
        try:
            import boto3
        except ImportError:
            raise ImportError('Please install boto3 to use the submit_dataset function')

        s3 = boto3.client('s3')
        buckets = s3.list_buckets()
        if not s3bucket in [b['Name'] for b in buckets['Buckets']]:
            s3.create_bucket(
                Bucket=s3bucket, CreateBucketConfiguration={'LocationConstraint': 'eu-west-1'}
            )
        if not folder_uuid:
            import uuid

            folder_uuid = str(uuid.uuid4())
        for fn in [imzml_fn, ibd_fn]:
            key = "{}/{}".format(folder_uuid, os.path.split(fn)[1])
            s3.upload_file(fn, s3bucket, key)
        folder = "s3a://" + s3bucket + "/" + folder_uuid
        return self._gqclient.create_dataset(
            data_path=folder,
            metadata=metadata,
            priority=priority,
            ds_name=dsName,
            is_public=isPublic,
            mol_dbs=molDBs,
            adducts=adducts,
            ds_id=dsid,
        )

    def submit_dataset_v2(
        self,
        imzml_fn,
        ibd_fn,
        ds_name,
        metadata,
        s3_bucket,
        is_public=None,
        moldbs=None,
        adducts=None,
        priority=0,
    ):
        """
        Submit a dataset for processing on the SM Instance
        :param imzml_fn: file path to imzml
        :param ibd_fn: file path to ibd
        :param ds_name: dataset name
        :param metadata: a properly formatted metadata json string
        :param s3_bucket: boto3 s3 bucket object, both the user has write permission to
        and METASPACE can access
        :param is_public: make dataset public
        :param moldbs: list molecular databases
        :param adducts: list of adducts
        :return:
        """
        import uuid

        dataset_key = str(uuid.uuid4())
        for fn in [imzml_fn, ibd_fn]:
            file_key = f'{dataset_key}/{Path(fn).name}'
            s3_bucket.upload_file(fn, file_key)
        return self._gqclient.create_dataset(
            data_path=f's3a://{s3_bucket.name}/{dataset_key}',
            metadata=metadata,
            priority=priority,
            ds_name=ds_name,
            is_public=is_public,
            mol_dbs=moldbs,
            adducts=adducts,
        )

    def update_dataset_dbs(self, datasetID, molDBs=None, adducts=None, priority=1):
        return self._gqclient.update_dataset(
            ds_id=datasetID, mol_dbs=molDBs, adducts=adducts, priority=priority, reprocess=True
        )

    def reprocess_dataset(self, dataset_id, force=False):
        return self._gqclient.update_dataset(ds_id=dataset_id, reprocess=True, force=force)

    def delete_dataset(self, ds_id, **kwargs):
        return self._gqclient.delete_dataset(ds_id, **kwargs)

    def current_user_id(self):
        result = self._gqclient.query("""query { currentUser { id } }""")
        return result['currentUser'] and result['currentUser']['id']

    def add_dataset_external_link(
        self, dataset_id: str, provider: str, link: str, replace_existing=False
    ):
        """
        Note that the current user must be the submitter of the dataset being edited.
        :param dataset_id:
        :param provider: Must be a known 3rd party link provider name.
                         Contact us if you're interested in integrating with METASPACE.
        :param link:
        :param replace_existing: pass True to overwrite existing links from the same provider
        :return: The updated list of external links
        """

        result = self._gqclient.query(
            """mutation($datasetId: String!, $provider: String!, $link: String!, 
                        $replaceExisting: Boolean!) {
                addDatasetExternalLink(datasetId: $datasetId, provider: $provider, link: $link, 
                                       replaceExisting: $replaceExisting) {
                    externalLinks { provider link }
                } 
            }""",
            {
                'datasetId': dataset_id,
                'provider': provider,
                'link': link,
                'replaceExisting': replace_existing,
            },
        )
        return result['addDatasetExternalLink']['externalLinks']

    def remove_dataset_external_link(
        self, dataset_id: str, provider: str, link: Optional[str] = None
    ):
        """
        Note that the current user must be the submitter of the dataset being edited.
        :param dataset_id:
        :param provider:
        :param link: If None, all links from the provider will be removed
        :return: The updated list of external links
        """

        result = self._gqclient.query(
            """mutation($datasetId: String!, $provider: String!, $link: String!) {
                removeDatasetExternalLink(datasetId: $datasetId, provider: $provider, link: $link) {
                    externalLinks { provider link }
                } 
            }""",
            {'datasetId': dataset_id, 'provider': provider, 'link': link},
        )
        return result['removeDatasetExternalLink']['externalLinks']


class MolecularDatabase:
    def __init__(self, metadata, client):
        self._metadata = metadata
        self._id = self._metadata['id']
        self._client = client

    @property
    def name(self):
        return self._metadata['name']

    @property
    def version(self):
        return self._metadata['version']

    def __repr__(self):
        return "MolDB({} [{}])".format(self.name, self.version)

    def sum_formulas(self):
        return self._client.getMolFormulaList(self._id)

    def names(self, sum_formula):
        return self._client.getMolFormulaNames(self._id, sum_formula)

    def ids(self, sum_formula):
        return self._client.getMolFormulaIds(self._id, sum_formula)

    def molecules(self, limit=100):
        return self._client.get_molecules(self._id, ['sf', 'mol_id', 'mol_name'], limit=limit)


def plot_diff(ref_df, dist_df, t='', xlabel='', ylabel='', col='msm'):
    import plotly.graph_objs as go
    from plotly.offline import iplot

    plot_df = dist_df.join(ref_df, rsuffix='_ref', how='inner').dropna()

    text_tmpl = (
        '{}{}<br>X: moc={:.3f} spat={:.3f} spec={:.3f}' '<br>Y: moc={:.3f} spat={:.3f} spec={:.3f}'
    )

    traces = []
    adducts = plot_df.index.get_level_values('adduct').unique()
    for adduct in adducts:
        df = plot_df.xs(adduct, level='adduct')
        txt = df.reset_index().apply(
            lambda r: text_tmpl.format(
                r.sf, adduct, r.moc_ref, r.spat_ref, r.spec_ref, r.moc, r.spat, r.spec
            ),
            axis=1,
        )

        if df.empty:
            continue

        traces.append(
            go.Scatter(
                x=df['{}_ref'.format(col)],
                y=df['{}'.format(col)],
                text=txt,
                mode='markers',
                name=adduct,
            )
        )

    data = go.Data(traces)
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            autosize=False,
            height=500,
            hovermode='closest',
            title=t + ' \'{}\' values'.format(col),
            width=500,
            xaxis=go.XAxis(
                autorange=False,
                range=[-0.05675070028979684, 1.0323925590539844],  # what the fuck is this?
                title=xlabel,
                type='linear',
            ),
            yaxis=go.YAxis(
                autorange=False,
                range=[-0.0015978995361995152, 1.0312345837176764],  # what the fuck is this?
                title=ylabel,
                type='linear',
            ),
        ),
    )
    iplot(fig, filename='ref_dist_msm_scatter')
    tmp_df = plot_df.dropna()
    return tmp_df


class DataframeTree(object):
    """
    Class for hierarchical clustering of Pandas dataframes.
    The intended usage is for making sense out of data returned by SMInstance.msm_scores
    """

    def __init__(self, df, method='ward', metric='euclidean'):
        import scipy.cluster.hierarchy as sch

        self._df = df
        self._Z = sch.linkage(self._df, method=method, metric=metric)
        self._root = DataframeNode(self._df, sch.to_tree(self._Z))

    @property
    def root(self):
        return self._root

    @property
    def df(self):
        """
        Dataframe reordered according to pre-order tree traversal.
        """
        return self.root.df

    @property
    def left(self):
        return self.root.left

    @property
    def right(self):
        return self.root.right

    def row_names(self):
        return list(self.df.index)

    def column_names(self):
        return list(self.df.columns)


class DataframeNode(object):
    def __init__(self, df, node):
        self._df = df
        self._node = node
        self._node_df = None
        self._left_node = None
        self._right_node = None

    @property
    def is_leaf(self):
        return self._node.is_leaf()

    @property
    def left(self):
        if self._left_node is None:
            self._left_node = DataframeNode(self._df, self._node.get_left())
        return self._left_node

    @property
    def right(self):
        if self._right_node is None:
            self._right_node = DataframeNode(self._df, self._node.get_right())
        return self._right_node

    @property
    def df(self):
        if self._node_df is None:
            self._node_df = self._df.iloc[self._node.pre_order(lambda x: x.get_id())]
        return self._node_df

    def row_names(self):
        return list(self.df.index)

    def column_names(self):
        return list(self.df.columns)


class DatasetNotFound(Exception):
    pass