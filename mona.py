import requests
import time
import json

import importlib.util
spec = importlib.util.spec_from_file_location("private", "/Users/dis/private.py")
private = importlib.util.module_from_spec(spec)
spec.loader.exec_module(private)


EMAIL = private.mona_user
PASSWORD = private.mona_pw
MAX_CALLS_PER_SECOND = 1
last_api_call = 0


def get_token(email, password):
  """
  Gets a new authentication token for your email/password. Tokens appear to expire after 7 days.
  Documentation: https://bitbucket.org/fiehnlab/mona/wiki/REST%20Authentication%20and%20Uploading
  They also changed it from HTTP to HTTPS without updating the docs. I only figured that out because
  I was lucky enough to notice one of the requests returned a "301 Permanent Redirect"
  """
  r = requests.post(
    'https://mona.fiehnlab.ucdavis.edu/rest/auth/login', 
    json={"username": email, "password": password}
  )
  if r.status_code == 200:
    return r.json()['token']
  else:
    raise Exception(f'{r.status_code} {r.reason} {r.text}')


def search_spectra(token, formula=None, name=None, 
                  smiles=None, inchi=None, inchi_key=None, 
                  polarity=None, ms_level=None, 
                  page_size=100, page=0):
  """
  All parameters are optional, but at least one should be provided.
  polarity can be 'positive' or 'negative'
  ms_level can be 'MS1', 'MS2', 'MS3', 'MS4'
  """
  global last_api_call
  query=[]
  if formula is not None:
    query.append(f"compound.metaData=q='name==\"molecular formula\" and value==\"{formula}\"'")
  if name is not None:
    query.append(f"compound.names=q='name=like=\"{name}\"'")
  if smiles is not None:
    query.append(f"compound.metaData=q='name==\"SMILES\" and value==\"{smiles}\"'")
  if inchi is not None:
    query.append(f"compound.metaData=q='name==\"InChI\" and value==\"{inchi}\"'")
  if inchi_key is not None:
    query.append(f"compound.metaData=q='name==\"InChIKey\" and value==\"{inchi_key}\"'")
  if polarity is not None:
    assert polarity in ('positive', 'negative')
    query.append(f"metaData=q='name==\"ionization mode\" and value==\"{polarity}\"'")
  if ms_level is not None:
    assert ms_level in ('MS1', 'MS2', 'MS3', 'MS4')
    query.append(f"metaData=q='name==\"ms level\" and value==\"{ms_level}\"'")
  merged_query = " and ".join(query)
  
  # Wait if the API is being hit too often
  min_interval = 1 / MAX_CALLS_PER_SECOND  
  time_since_last_call = time.perf_counter() - last_api_call
  time_to_wait = min_interval - time_since_last_call
  if time_to_wait > 0:
    time.sleep(time_to_wait)
  last_api_call = time.perf_counter()

  r = requests.get(
    'https://mona.fiehnlab.ucdavis.edu/rest/spectra/search', 
    params={"query": merged_query, "size": page_size, "page": page},
    headers={"Authorization": f"Bearer {token}"}
  )
  if r.status_code == 200:
    return r.json()
  else:
    raise Exception(f'{r.status_code} {r.reason} {r.text}')


def tests():
  TOKEN = get_token(EMAIL, PASSWORD)

  # Examples/tests:
  spectra = search_spectra(TOKEN, formula="C47H74O12S")
  spectra = search_spectra(TOKEN, name="Mannose")
  spectra = search_spectra(TOKEN, smiles="OCC1OC(O)C(O)C(O)C1O")
  spectra = search_spectra(TOKEN, inchi="InChI=1S/C6H12O6/c7-1-2-3(8)4(9)5(10)6(11)12-2/h2-11H,1H2/t2-,3-,4+,5+,6?/m1/s1")
  spectra = search_spectra(TOKEN, inchi_key="YCJAISVBRZFFQG-PMDVYHIWSA-N")
  spectra = search_spectra(TOKEN, name="Triadimefon", polarity="negative")
  spectra = search_spectra(TOKEN, name="Triadimefon", ms_level="MS4")

  print(spectra)

def mona_main(inchi_in, ms_level, search_polarity):
  TOKEN = get_token(EMAIL, PASSWORD)
  spectra = search_spectra(TOKEN, inchi=inchi_in, ms_level=ms_level, polarity=search_polarity)
  return spectra