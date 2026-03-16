import requests
import os
import pandas as pd
from itertools import zip_longest
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.colors as c
import numpy as np
from io import BytesIO
import seaborn as sns
from app.config import Config
from flask import flash
from typing import List
import glob
import matplotlib
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
import os
import glob
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
from typing import List
from itertools import zip_longest

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy.spatial.distance import pdist, squareform

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import joblib

import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from app.session_manager import SessionManager


# Path to the project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TMP_FILES_DIR = os.path.join(ROOT_DIR, 'data/tmp')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Set random seed for reproducibility - same data gives same results every time
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DB_PATH =  os.path.join(ROOT_DIR, 'instance/processed_data.db')

# def bioassay_post(identifier_list, identifier='cid', output='csv'):
#     identifier_list = list(map(str, identifier_list))
#     url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{identifier}/assaysummary/{output}'
#     headers = {'Content-Type': 'multipart/form-data'}
#     data = {identifier: ','.join(identifier_list)}
#     response = requests.post(url, data=data)
#     return response

import sqlite3
import pandas as pd
def bioassay_post(identifier_list, identifier='cid', limit=10000):
    """
    Retrieve bioassay data directly from the local database instead of making PubChem API calls

    Parameters:
    -----------
    identifier_list : list
        List of specific CIDs to retrieve bioassay data for
    identifier : str, default='cid'
        Type of identifier (currently only supports 'cid')
    limit : int, default=10000
        Maximum number of records to return

    Returns:
    --------
    DataFrame containing bioassay data from the local database
    """
    conn = sqlite3.connect(DB_PATH)

    try:
        if identifier != 'cid':
            raise ValueError("Only 'cid' is supported as identifier type")

        identifier_list = list(map(str, identifier_list))
        placeholders = ','.join(['?'] * len(identifier_list))

        query = f"SELECT * FROM data WHERE CID IN ({placeholders})"
        df = pd.read_sql_query(query, conn, params=identifier_list)

        if limit is not None and len(df) > limit:
            df = df.sample(n=limit, random_state=42)  # random sampling in memory

        # Normalize and rename columns
        df.columns = [col.strip() for col in df.columns]
        column_renames = {
            'ActivityOutcome': 'Activity Outcome',
            'activity_outcome': 'Activity Outcome',
            'activity outcome': 'Activity Outcome'
        }
        df.rename(columns=column_renames, inplace=True)

        print(f"Query returned {len(df)} rows with {len(df.columns)} columns")
        print("Columns:", df.columns.tolist())
        print(f"First few rows:\n{df.head(3)}")

        return df

    finally:
        conn.close()
 

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)



def make_bioprofile_matrix_new(df, min_actives=0, merge_dups_by='max'):
    """ will turn assay data file into wide (i.e., a matrix) format but performs all filtering in
        long format to save RAM"""

    df['Activity Outcome'] = df['Activity Outcome'] \
        .replace('Inactive', -1) \
        .replace('Active', 1) \
        .replace('Probe', 1) \
        .replace('Inconclusive', 0) \
        .replace('Unspecified', 0) \
        .fillna(0)

    df['Activity Transformed'] = df.groupby(['CID', 'AID'])['Activity Outcome'].transform(merge_dups_by)

    # for the num actives count, a CID is considered active
    # for a given AID if it has an active response for any of its
    # bioactivity outcomes for that AID.
    df['Activity Outcome Max'] = df.groupby(['CID', 'AID'])['Activity Outcome'].transform('max')

    # take only one response for a
    # CID/AID pair, ie., the transformed
    # bioactivity value
    df = df.drop_duplicates(['CID', 'AID', 'Activity Transformed'])
    # CID/AID/Bioactivity Outcome should be unique
    # just like CID/AID/Bioactivity Outcome Maz
    df_tmp = df.groupby('AID')['Activity Outcome Max'].apply(lambda x: (x == 1).sum()).reset_index(name='Num_Active')
    df = df.merge(df_tmp)
    df = df[df.Num_Active >= min_actives]

    # turn into wide format
    matrix = df.pivot(index='CID', columns='AID', values='Activity Transformed').fillna(0)
    matrix.reset_index(inplace=True)

    if 'CID' not in matrix.columns:
        print("Error: 'CID' matrix column is missing.")

    return matrix

def make_bioprofile_matrix(df):
    # turn into wide format
    matrix = df.pivot(index='CID', columns='AID', values='Activity_Transformed').fillna(0)
    return matrix


def make_matrix(data_file, min_actives=0, merge_dups_by='max', outfile='bioprofile_matrix.csv'):
    df = pd.read_csv(data_file, usecols=['AID', 'CID', 'Activity Outcome'])
    df['Activity Outcome'] = df['Activity Outcome']\
                                     .replace('Inactive', -1)\
                                     .replace('Active', 1)\
                                     .replace('Probe', 1)\
                                     .replace('Inconclusive', 0)\
                                     .replace('Unspecified', 0)\
                                     .fillna(0)
    df['Activity Transformed'] = df.groupby(['CID', 'AID'])['Activity Outcome'].transform(merge_dups_by)
    df['Activity Outcome Max'] = df.groupby(['CID', 'AID'])['Activity Outcome'].transform('max')
    df = df.drop_duplicates(['CID', 'AID', 'Activity Transformed'])
    df_tmp = df.groupby('AID')['Activity Outcome Max'].apply(lambda x: (x == 1).sum()).reset_index(name='Num_Active')
    df = df.merge(df_tmp)
    df = df[df.Num_Active >= min_actives]
    matrix = df.pivot(index='CID', columns='AID', values='Activity Transformed').fillna(0)
    matrix.to_csv(outfile)
    return matrix

def generate_bioprofile_new(identifier_list, identifier='cid', chunk=False):

    outfile = Config.BIOASSAY_OUT

    num_compounds = len(identifier_list)

    # check to see whether
    # the list should be queried
    # in chunks of data or
    # processed as a whole
    if not chunk:
        chunk_size = num_compounds
    else:
        chunk_size = chunk

    counter = 0

    # f = open(outfile, 'w')
    f = open(outfile, 'w', encoding='utf-8')
    header_written = False

    for gp in grouper(identifier_list, chunk_size):
        batch = [cid for cid in gp if cid]
        # response = bioassay_post(batch, identifier=identifier, output='csv')
        response = bioassay_post(batch, identifier=identifier , limit=100)


        if response.status_code == 200:
            text = response.text
            if not header_written:
                f.write(text)
                header_written = True
            else:
                header = text.split('\n')[0]
                text = text.replace(header, '')
                f.write(text)
            counter = counter + len(batch)
            print("Retrieved data for {} out of {} compounds.".format(counter, num_compounds))
        else:
            f.close()
            os.remove('data/assay_data.csv')
            print("Error: {}".format(response.status_code))
            return
    f.close()

    df = pd.read_csv(outfile)
    os.remove(outfile)
    return df

def generate_bioprofile(dataset_name, identifier_list, identifier='cid', chunk=False):
    num_compounds = len(identifier_list)

    # check to see whether the list should be queried
    # in chunks of data or processed as a wholeI put
    if not chunk:
        chunk_size = num_compounds
    else:
        chunk_size = chunk

    outfile = Config.BIOASSAY_OUT
    f = open(outfile, 'w', encoding='utf-8')
    header_written = False

    # for gp in grouper(identifier_list, chunk_size):
    #     batch = [cid for cid in gp if cid]
    #     response = bioassay_post(batch, identifier=identifier )

    #     if response.status_code == 200:
    #         text = response.text
    #         if not header_written:
    #             f.write(text)
    #             header_written = True
    #         else:
    #             header = text.split('\n')[0]
    #             text = text.replace(header, '')
    #             f.write(text)
    #     else:
    #         continue
    # f.close()

    # df = pd.read_csv(outfile)
    # os.remove(outfile)

    df =bioassay_post(batch, identifier=identifier )
    return df

import pandas as pd
from itertools import zip_longest

def grouper(iterable, n, fillvalue=None):
    """Helper function to split list into chunks of size n"""
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
def bioprofile(identifier_list, identifier='cid', outfile='bioprofile_long.csv', chunk=False):
    """
    Fetches bioassay data for a given list of CIDs, returns it as a DataFrame,
    and also writes the combined data to a CSV file.
    """
    #identifier_list = identifier_list[:200] if len(identifier_list) > 200 else identifier_list
    if len(identifier_list) > 500:
        error = f"Sorry, the selected file contains {len(identifier_list)} CIDs. The maximum allowed is 500."
        raise ValueError(error)
    num_compounds = len(identifier_list)
    chunk_size = num_compounds if not chunk else chunk
    counter = 0
    all_dfs = []
    header_written = False

    with open(outfile, 'w', encoding='utf-8') as f:
        for gp in grouper(identifier_list, chunk_size):
            batch = [cid for cid in gp if cid]
            df = bioassay_post(batch, identifier=identifier)
            print(df.columns.tolist())

            if df is not None and not df.empty:
                # 🔁 Rename BEFORE writing to file
                df.rename(columns={"Activity_Outcome": "Activity Outcome"}, inplace=True)

                all_dfs.append(df)
                df.to_csv(f, index=False, header=not header_written, mode='a')
                header_written = True

                counter += len(batch)
                print(f"Retrieved data for {counter} out of {num_compounds} compounds.")
            else:
                print("Warning: Received empty DataFrame for batch.")

    if all_dfs:
        result_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Saved bioassay data to {outfile}")
        print(f"Total records returned: {result_df.shape[0]}")
        return result_df
    else:
        print("No data retrieved.")
        return pd.DataFrame()


def confusion_matrix(df, activity_class):
    """ this function calculates the confusion matrix for an assay, toxicity pair """
    df[activity_class] = pd.to_numeric(df[activity_class], errors='coerce')
    df = df[df[activity_class].notnull()]

    tps = ((df[activity_class] == 1) & (df.Activity_Transformed == 1)).sum()
    fps = ((df[activity_class] == 0) & (df.Activity_Transformed == 1)).sum()
    tns = ((df[activity_class] == 0) & (df.Activity_Transformed == 0)).sum()
    fns = ((df[activity_class] == 1) & (df.Activity_Transformed == 0)).sum()

    return tps, fps, tns, fns


def render_bioprofile(profile, dataset_type):
    med = (
        profile[['CID', 'TOXICITY']]
        .drop_duplicates()
        ['TOXICITY']
        .median()
    )

    if dataset_type == 'Median':
        profile['TOXICITY_copy'] = profile['TOXICITY'].copy()
        profile.loc[profile['TOXICITY'] < med, 'TOXICITY_copy'] = 1
        profile.loc[profile['TOXICITY'] >= med, 'TOXICITY_copy'] = 0
        profile['TOXICITY'] = profile['TOXICITY_copy']

    elif dataset_type == 'Other':
        profile['TOXICITY_copy'] = profile['TOXICITY'].copy()
        profile.loc[profile['TOXICITY'] < med, 'TOXICITY_copy'] = 0
        profile.loc[profile['TOXICITY'] >= med, 'TOXICITY_copy'] = 1
        profile['TOXICITY'] = profile['TOXICITY_copy']

    elif dataset_type == 'Binary':
        profile['TOXICITY_copy'] = profile['TOXICITY'].copy()
        profile.loc[profile['TOXICITY'] < med, 'TOXICITY_copy'] = 0
        profile.loc[profile['TOXICITY'] >= med, 'TOXICITY_copy'] = 1
        profile['TOXICITY'] = profile['TOXICITY_copy']

    matrix = (
        profile.groupby('AID')
        .apply(lambda x: confusion_matrix(x, 'TOXICITY_copy'))
        .apply(pd.Series)
        .set_axis(['TP', 'FP', 'TN', 'FN'], axis=1)
        .reset_index()
        .sort_values('TP', ascending=False)
    )
    matrix['PPV'] = matrix.TP / (matrix.TP + matrix.FP)
    matrix['Sensitivity'] = matrix.TP / (matrix.TP + matrix.FN)

    profile = profile.pivot(index='CID', columns='AID', values='Activity_Transformed').fillna(0)
    tmp = matrix.set_index('AID')['PPV']
    tmp = tmp[profile.columns]
    supervised_profile = profile.loc[:, tmp > 0.6]
    supervised_profile = supervised_profile.fillna(0)

    return supervised_profile


def get_pca(supervised_profile, profile):
    pca = PCA(n_components=3).fit_transform(supervised_profile)
    pca = pd.DataFrame(pca, index=supervised_profile.index, columns=['PCA1', 'PCA2', 'PCA3'])
    y = profile[['CID', 'TOXICITY']].drop_duplicates().set_index('CID')
    pca = pd.merge(pca.join(y), profile, on='CID', how='inner')

    return pca


# Use a non-interactive backend
matplotlib.use('Agg')

from matplotlib.colors import ListedColormap, BoundaryNorm

def get_heat_map(matrix, output_file='clustermap.png'):
    """
    Generates a heatmap using seaborn's clustermap:
    -1 = Inactive
     0 = Inconclusive
     1 = Active

    Parameters:
    matrix (pd.DataFrame): Bioprofile matrix.
    output_file (str): File to save the heatmap.

    Returns:
    fig (matplotlib.figure.Figure): Generated heatmap figure.
    """

    # Limit the matrix to the first 2000 columns if more exist
    if matrix.shape[1] > 2000:
        matrix = matrix.iloc[:, :2000]

    print(f"Generating heatmap with {matrix.shape[1]} columns.")

    # Generate clustermap with coolwarm color scheme
    heatmap = sns.clustermap(
        matrix,
        cmap='coolwarm',
        cbar_kws={'orientation': 'horizontal', 'fraction': 0.046, 'pad': 0.04}
    )

    # Adjust the color bar position to top
    heatmap.cax.set_position([0.3, 1.02, 0.4, 0.02])

    # Customize color bar ticks and labels
    cbar = heatmap.ax_heatmap.collections[0].colorbar
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Inactive', 'Inconclusive', 'Active'])

    # Set axis labels
    heatmap.ax_heatmap.set_xlabel('PubChem Assay ID', fontsize=12)
    heatmap.ax_heatmap.set_ylabel('PubChem Compound ID', fontsize=12)

    # Tilt x-axis labels for readability
    heatmap.ax_heatmap.set_xticklabels(
        heatmap.ax_heatmap.get_xticklabels(),
        rotation=45,
        ha='right'
    )

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    return heatmap.fig


def get_active_bioassays(pca):
    bioassay = os.getenv('BIOASSAYS')
    bio_info = pd.read_table(bioassay)

    biodict = dict(zip(bio_info['AID'], bio_info['BioAssay Name']))
    pca1 = pca.copy()
    pca1['BioAssay Name'] = pca1['AID'].map(biodict)

    table = pca1.groupby(['AID', 'BioAssay Name'])['Activity_Transformed'].value_counts().unstack(fill_value=0)
    table = table.rename(columns={-1.0: 'Inactive', 0.0: 'Inconclusive', 1.0: 'Active'})
    table['Active rate'] = table['Active'] / (table['Active'] + table['Inactive'])
    table1 = table.loc[table['Inactive'] + table['Active'] > 10]
    table1 = table1.loc[table1['Inconclusive'] < (table1['Active'] + table1['Inactive'])].sort_values(by='Active rate',
                                                                                                      ascending=False)
    table1 = table1.reset_index()

    return table1


def select_top_assays(matrix_file='bioprofile_matrix.csv', output_file='SelectedAssays.csv', top_n=35):
    df = pd.read_csv(matrix_file, index_col=0)

    # <<< CRUCIAL: make a binary view for MI >>> 
    df_binary = df.replace({-1: 0})

    # Drop AIDs with no variation (now evaluated on 0/1 view)
    df_filtered = df_binary.loc[:, df_binary.nunique() > 1]

    # MI on binary features; y = any activity across AIDs
    X = df_filtered.T
    y = (df_filtered.sum(axis=1) > 0).astype(int)

    mi_scores = mutual_info_classif(X.T, y, discrete_features=True)
    mi_dict = dict(zip(df_filtered.columns, mi_scores))
    top_aids = sorted(mi_dict, key=mi_dict.get, reverse=True)[:top_n]

    with open(output_file, 'w') as f:
        for aid in top_aids:
            f.write(f"{aid}\n")
    return top_aids


def get_raw_bioactivity_data(aid: int):
    """
    Fetches bioactivity data for a specific AID from the local database.
    
    Parameters:
    -----------
    aid : int
        The AID to fetch data for
        
    Returns:
    --------
    DataFrame of CID and Activity Outcome, or None and error message
    """
    # Connect to the database
    db_path = DB_PATH
    print(f"Connecting to database at {db_path} to fetch bioactivity data for AID {aid}")
    conn = sqlite3.connect(db_path)
    
    try:
        # Query to get CID and Activity_Outcome for the specified AID
        query = "SELECT CID, Activity_Outcome as 'Activity Outcome' FROM data WHERE AID = ? LIMIT 10000"
        #query = "SELECT CID, Activity_Outcome as 'Activity Outcome' FROM data WHERE AID = ?"
        print(f"Executing SQL query: {query}")
        
        df = pd.read_sql_query(query, conn, params=(aid,))
        
        if df.empty:
            return None, f"No data found for the given AID: {aid}."
        
        print(f"Retrieved {len(df)} records for AID {aid}")
        print(f"Sample data:\n{df.head(3)}")
        
        return df[['CID', 'Activity Outcome']], None
        
    except Exception as e:
        return None, f"Database error for AID {aid}: {str(e)}"
        
    finally:
        # Ensure connection is closed even if an error occurs
        print("Closing database connection")
        conn.close()

# Function to clean bioactivity data
def clean_bioactivity_frame(df, aid):
    df['Activity Outcome'] = df['Activity Outcome'] \
                            .replace({'Inactive': 0, 'Active': 1, 'Probe': 1, 
                                      'Inconclusive': np.nan, 'Unspecified': np.nan})
    df = df.dropna(subset=['Activity Outcome'])
    if df.empty:
        return None, f"No valid activity outcomes after cleaning for AID {aid}."
    if not df['Activity Outcome'].isin([1]).any():
        return None, f"No active chemicals found in the AID {aid}."
    df = df.sort_values('Activity Outcome').drop_duplicates('CID', keep='last')
    df['Activity Outcome'] = df['Activity Outcome'].astype(int)
    return df, None

def get_inchi_from_cids(cids: List, batch_size: int = 1000):
    """
    Fetches InChI values for a list of CIDs from the local database.
    
    Parameters:
    -----------
    cids : List
        List of CIDs to fetch InChI values for
    batch_size : int, default=1000
        Size of batches to process (for large lists)
        
    Returns:
    --------
    DataFrame with CID and InChI columns, or None and error message
    """
    print(f"Fetching InChI values for {len(cids)} CIDs from local database")
    
    # Connect to the database
    db_path =DB_PATH
    print(f"Connecting to database at {db_path}")
    conn = sqlite3.connect(db_path)
    
    all_data = []
    
    try:
        # Process in batches for memory efficiency
        for i in range(0, len(cids), batch_size):
            batch = cids[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}: {len(batch)} compounds")
            
            # Create placeholders for SQL IN clause
            placeholders = ','.join(['?'] * len(batch))
            
            # Query to get CID and InChI_string for the batch
            query = f"""
                SELECT DISTINCT CID, InChI_string as InChI 
                FROM data 
                WHERE CID IN ({placeholders})
            """
            
            df = pd.read_sql_query(query, conn, params=batch)
            
            if not df.empty:
                all_data.append(df)
                print(f"Batch {i//batch_size + 1}: Retrieved {len(df)} records")
            else:
                print(f"Batch {i//batch_size + 1}: No matching records found")
        
        if not all_data:
            return None, "No InChI data retrieved from database."
            
        # Combine all batch results
        result_df = pd.concat(all_data, ignore_index=True)
        print(f"Total InChI records retrieved: {len(result_df)}")
        
        return result_df, None
        
    except Exception as e:
        return None, f"Database error while fetching InChI data: {str(e)}"
        
    finally:
        # Ensure connection is closed even if an error occurs
        print("Closing database connection")
        conn.close()
def import_pubchem_aid(aid: int):
    """
    Import bioassay data for a specific AID from the local database.
    
    Parameters:
    -----------
    aid : int
        The AID to fetch data for
        
    Returns:
    --------
    DataFrame containing compound_id, inchi, and activity, or None and error message
    """
    print(f"Importing data for AID {aid} from local database")
    
    # Step 1: Fetch raw bioactivity data from local DB
    raw_data, error = get_raw_bioactivity_data(aid)
    if error:
        return None, error
    print("Raw bioactivity data fetched successfully.")
    
    # Check if there are any Active or Probe compounds
    if not raw_data['Activity Outcome'].isin(['Active', 'Probe']).any():
        return None, f"No actives found in the AID {aid}."
    
    missing_values = raw_data['Activity Outcome'].isnull().sum()
    print(f"Missing values in 'Activity Outcome': {missing_values}")
    
    print("Unique values in 'Activity Outcome':", raw_data['Activity Outcome'].unique())

    # Step 2: Clean the bioactivity data
    bioactivity, clean_error = clean_bioactivity_frame(raw_data, aid)
    if clean_error:
        return None, clean_error
    print("Bioactivity data cleaned successfully.")

    # Step 3: Fetch InChI data for CIDs from local DB
    # Convert CIDs to strings to avoid int conversion errors with float values
    cids = bioactivity['CID'].dropna().astype(str).tolist()
    if not cids:
        return None, "No CIDs available to fetch InChI data."
    print(f"Fetching InChI data for {len(cids)} compounds...")

    # Use our local DB function instead of API call
    inchi_dict = fetch_inchi_for_cids(cids)
    if not inchi_dict:
        return None, "Failed to retrieve InChI data from database."
    print("InChI data fetched successfully.")

    # Step 4: Create DataFrame with InChI data
    inchi_data = pd.DataFrame({
        'CID': list(inchi_dict.keys()),
        'InChI': list(inchi_dict.values())
    })

    # Step 5: Merge bioactivity and InChI data
    # Convert CID to string in both DataFrames to ensure proper merging
    bioactivity['CID'] = bioactivity['CID'].astype(str)
    result = pd.merge(inchi_data, bioactivity, on='CID', how='inner')
    if result.empty:
        return None, "No matching CIDs found between bioactivity and InChI data."
    
    result = result.rename(columns={
        'CID': 'compound_id',
        'InChI': 'inchi',
        'Activity Outcome': 'activity'
    })
    
    return result, None
def fetch_and_save_aid_data(selected_aid_file='SelectedAssays.csv', session_dir=None):
    """
    Fetches data for selected AIDs and ensures that no more than 10000 CIDs
    are collected per AID.

    Args:
        selected_aid_file: Path to file containing selected AIDs
        session_dir: Session directory path (if None, will create new session)
    """
    if session_dir is None:
        session_dir = SessionManager.get_session_folder()  # Unique per request

    # Read selected AIDs from file
    with open(selected_aid_file, 'r') as f:
        aid_list = [int(line.strip()) for line in f.readlines()]

    for aid in aid_list:
        print(f"Processing AID: {aid}")

        # Fetch data for the given AID
        result, error = import_pubchem_aid(aid)

        if error:
            print(f"Skipping AID {aid} due to error: {error}")
            continue

        # Limit to 10000 CIDs if the dataset is larger
        if len(result) > 10000:
            print(f"⚠️ AID {aid} has {len(result)} compounds. Limiting to 10000.")
            result = result.sample(n=10000, random_state=42)

        output_file = os.path.join(session_dir, f"AID{aid}_chemicals.csv")

        # Save the result in a file
        result.to_csv(output_file, index=False)
        print(f"✅ Saved data for AID {aid} to {output_file} ({len(result)} compounds)")

        # Curate chemicals - pass session_dir
        curate_chemicals(aid, session_dir=session_dir)
        

def curate_chemicals(aid, session_dir=None):
    """
    Ensure equal number of actives and inactives, skip if one class is missing or has < 100 compounds.

    Args:
        aid: Assay ID
        session_dir: Session directory path (if None, will create new session)
    """
    if session_dir is None:
        session_dir = SessionManager.get_session_folder()  # Unique per request

    input_file = os.path.join(session_dir, f"AID{aid}_chemicals.csv")
    output_file = os.path.join(session_dir, f"AID{aid}_diverse.csv")

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Error: {input_file} not found.")
        return

    # Remove duplicate InChI entries
    df = df.drop_duplicates(subset=['inchi'])

    # Separate actives and inactives
    active_df = df[df['activity'] == 1]
    inactive_df = df[df['activity'] == 0]

    # Skip if only one class is present or either has < 100 samples
    if len(active_df) < 100 or len(inactive_df) < 100:
        print(f"⚠️ Skipping AID {aid}: Insufficient class balance (actives={len(active_df)}, inactives={len(inactive_df)}).")
        return

    # Sample equal number of actives and inactives (up to 500 each)
    sample_size = min(len(active_df), len(inactive_df), 500)
    active_df = active_df.sample(n=sample_size, random_state=42)
    inactive_df = inactive_df.sample(n=sample_size, random_state=42)

    curated_df = pd.concat([active_df, inactive_df]).reset_index(drop=True)

    # Save curated data
    curated_df.to_csv(output_file, index=False)
    print(f"✅ Saved curated chemicals to {output_file} ({sample_size} actives + {sample_size} inactives)")


def generate_rf_model_and_metrics(file_path, max_depth=10, class_weight='balanced', n_estimators=100, session_dir=None):
    if session_dir is None:
        session_dir = SessionManager.get_session_folder()  # Unique per request

    df = pd.read_csv(file_path)

    fingerprints = []
    for inchi in df['inchi']:
        mol = Chem.MolFromInchi(inchi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
        else:
            fingerprints.append(np.zeros((2048,)))

    X = np.array(fingerprints)
    y = df['activity'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(max_depth=max_depth, class_weight=class_weight, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred, average='binary'), 2)
    recall = round(recall_score(y_test, y_pred, average='binary'), 2)
    f1 = round(f1_score(y_test, y_pred, average='binary'), 2)
    roc_auc = round(roc_auc_score(y_test, y_pred), 2)

    aid = os.path.basename(file_path).split('_')[0][3:]  
    model_filename = os.path.join(session_dir, f"AID{aid}_rf_model.pkl")  

    joblib.dump(rf, model_filename)
    print(f"✅ Saved Random Forest model as {model_filename}")

    # ✅ Create metrics dictionary
    metrics = {
        'Model': f"AID{aid}_rf_model",
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

    # ✅ Ensure RF_metrics.csv is separate from AID*_diverse.csv
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(session_dir, 'RF_metrics.csv')

    metrics_df.to_csv(metrics_file, mode='a', header=not os.path.exists(metrics_file), index=False)

    print(f"📊 Appended new metrics to {metrics_file}")

    return metrics


def generate_all_rf_models_and_save_metrics(output_file='RF_metrics.csv', session_dir=None):
    """
    Generate RF models from CSV files and save the metrics in the temp directory.
    """
    if session_dir is None:
        session_dir = SessionManager.get_session_folder()  # Unique per request

    # Define the full path for the output file inside the temp directory
    output_file_path = os.path.join(session_dir, output_file)

    # Get all files matching AID{aid}_diverse.csv pattern in the temp directory
    curated_files = glob.glob(os.path.join(session_dir, 'AID*_diverse.csv'))

    if not curated_files:
        print("Error: No curated datasets found for training. Ensure AID*_diverse.csv files exist!")
        return

    all_metrics = []

    for file_path in curated_files:
        try:
            print(f"🔄 Training model for: {file_path}")
            metrics = generate_rf_model_and_metrics(file_path, session_dir=session_dir)

            if metrics is not None:
                all_metrics.append(metrics)
            else:
                print(f"⚠️ Warning: No metrics returned for {file_path}. Possible training failure.")
        except Exception as e:
            print(f"Error training model for {file_path}: {e}")

    # Convert the metrics to a DataFrame
    if not all_metrics:
        print("No valid metrics generated. Check training errors.")
        return

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values("Model", ascending=True)

    # Save the metrics to CSV inside the temp directory
    metrics_df.to_csv(output_file_path, index=False)
    print(f"Saved evaluation metrics to {output_file_path}")


 
# def generate_all_rf_models_and_save_metrics(output_file='RF_metrics.csv'):
#     # Get all files matching AID{aid}_diverse.csv pattern in the current directory
#     curated_files = glob.glob(os.path.join(TMP_FILES_DIR, 'AID*_diverse.csv'))
    
#     all_metrics = []
#     expected_columns = 4  # The correct number of columns expected

#     for file_path in curated_files:
#         try:
#             # Read CSV with pandas, enforcing the expected number of columns
#             df = pd.read_csv(file_path)

#             # Skip the file if it has extra or missing columns
#             if df.shape[1] != expected_columns:
#                 print(f"Skipping {file_path} due to incorrect column count ({df.shape[1]} found, expected {expected_columns}).")
#                 continue
            
#             # Process data and generate metrics
#             metrics = generate_rf_model_and_metrics(file_path)

#             if metrics is not None:  # Ensure only valid results are collected
#                 all_metrics.append(metrics)

#         except Exception as e:
#             print(f"Error processing {file_path}: {e}")

#     # Convert collected metrics to DataFrame
#     if all_metrics:
#         metrics_df = pd.DataFrame(all_metrics)

#         # Write all data at once (No append mode)
#         metrics_df.to_csv(output_file, index=False)

#         print(f"Saved evaluation metrics to {output_file}")
#     else:
#         print("No valid CSV files found for processing.")


def plot_rf_metrics(metrics_file='RF_metrics.csv', session_dir=None):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if session_dir is None:
        session_dir = SessionManager.get_session_folder()  # Unique per request

    # Load the metrics file
    df = pd.read_csv(metrics_file)
    if df.empty:
        print("The metrics file is empty. No data to plot.")
        return

    # Clean model names for display
    df['Model'] = df['Model'].str.replace('_rf_model', '', regex=False)

    # Extract models and metric columns
    models = df['Model'].tolist()
    metric_cols = [c for c in df.columns if c != "Model"]

    # Prepare data for boxplots: each model's box = distribution of its metrics
    data_by_model = [df.loc[df['Model'] == m, metric_cols].values.flatten()
                     for m in models]

    # Create figure
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    # Draw neutral boxes
    bp = ax.boxplot(
        data_by_model,
        positions=np.arange(len(models)),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(linestyle="--", linewidth=2, color="black"),
        boxprops=dict(linewidth=1.6, facecolor="#a6cee3", alpha=0.4, edgecolor="#333333"),
        whiskerprops=dict(linewidth=1.2, color="#333333"),
        capprops=dict(linewidth=1.2, color="#333333"),
    )

    # Colors for metrics (tab10 palette)
    tab10 = plt.get_cmap("tab10")
    metric_colors = {m: tab10(i % 10) for i, m in enumerate(metric_cols)}

    # Overlay colored dots for each metric per model
    legend_added = set()
    for i, m in enumerate(models):
        scores = df.loc[df["Model"] == m, metric_cols].iloc[0].to_dict()
        for metric, score in scores.items():
            x = i + (np.random.rand() - 0.5) * 0.18  # jitter
            ax.scatter(
                [x], [score],
                s=70, marker="o",
                alpha=0.95,
                facecolor=metric_colors[metric],
                edgecolor="black", linewidth=0.6,
                zorder=3,
                label=metric if metric not in legend_added else None
            )
            legend_added.add(metric)

    # Titles and labels
    ax.set_title("Metrics Across Assays", fontsize=20, pad=14, weight="bold")
    ax.set_xlabel("Models", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.axhline(0.70, linestyle="--", linewidth=1.5, color="#888888")

    # Legend for metrics
    handles = [Line2D([0], [0], marker="o", linestyle="",
                      markerfacecolor=metric_colors[m], markeredgecolor="black",
                      markeredgewidth=0.6, markersize=9, label=m)
               for m in metric_cols]
    ax.legend(handles=handles, title="Metric",
              bbox_to_anchor=(1.01, 1.0), loc="upper left", frameon=True)

    # Save and show
    plt.tight_layout()
    RF_metrics_plot = os.path.join(session_dir, 'RF_metrics_plot.png')
    plt.savefig(RF_metrics_plot, dpi=300, bbox_inches='tight')
    plt.show()
    

def filter_selected_aids(matrix_file='bioprofile_matrix.csv', output_file='Selected_bioprofile_matrix.csv', session_dir=None):
    """
    Filters the bioprofile matrix to include only AIDs for which we have trained RF models.

    Parameters:
    matrix_file (str): Path to the bioprofile matrix file.
    output_file (str): Path to save the filtered bioprofile matrix.
    session_dir (str): Session directory path (if None, will create new session)

    Returns:
    selected_df (DataFrame): The filtered matrix.
    """
    if session_dir is None:
        session_dir = SessionManager.get_session_folder()  # Unique per request

    try:
        # Load the bioprofile matrix
        df = pd.read_csv(matrix_file, index_col=0)
        print(f"✅ Loaded bioprofile matrix with shape: {df.shape}")

        # Corrected model file search pattern
        model_files = glob.glob(os.path.join(session_dir, "AID*_rf_model.pkl"))
        model_aids = [os.path.basename(f).split('_')[0][3:] for f in model_files]  # Extract AID numbers
        
        if not model_aids:
            raise ValueError("⚠️ No trained RF models found. Ensure models are saved as 'AID{aid}_rf_model.pkl'.")

        # ✅ Filter the matrix to include only relevant AIDs
        selected_cols = [aid for aid in model_aids if aid in df.columns]
        
        if not selected_cols:
            raise ValueError("⚠️ No matching AIDs found between bioprofile matrix and trained RF models.")

        selected_df = df[selected_cols]

        # ✅ Save the filtered matrix
        output_file_path = os.path.join(session_dir, output_file)
        selected_df.to_csv(output_file_path)
        print(f"📁 Filtered matrix saved as {output_file_path} with shape: {selected_df.shape}")

        return selected_df

    except Exception as e:
        print(f"❌ Error in filtering selected AIDs: {e}")
        return None


def generate_ecfp6_fingerprint(inchi):
    """
    Generates an ECFP6 fingerprint for a given InChI.
    
    Parameters:
    inchi (str): InChI string of the molecule.
    
    Returns:
    np.array: ECFP6 fingerprint array.
    """
    mol = Chem.MolFromInchi(inchi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        return np.zeros((2048,))  # Return zero vector if molecule is invalid


import sqlite3
import pandas as pd
from typing import Dict, List, Union

def fetch_inchi_for_cids(cids: List[Union[str, int]]) -> Dict[str, str]:
    """
    Fetches InChI values for a list of CIDs from the local database.
    
    Parameters:
    cids (List[Union[str, int]]): List of CIDs to fetch InChIs.
    
    Returns:
    Dict[str, str]: Dictionary mapping CID to InChI.
    """
    print(f"Fetching InChI values for {len(cids)} CIDs from local database")
    
    # Convert all CIDs to strings for consistency
    cids = list(map(str, cids))
    print(f"CIDs to fetch (first 5): {cids[:5]}{'...' if len(cids) > 5 else ''}")
    
    # Connect to the database
    db_path = DB_PATH
    print(f"Connecting to database at {db_path}")
    conn = sqlite3.connect(db_path)
    
    inchi_dict = {}
    
    try:
        # Create placeholders for SQL IN clause
        placeholders = ','.join(['?'] * len(cids))
        
        # Query to get distinct CID and InChI pairs
        # Using DISTINCT to avoid duplicates if the same CID appears multiple times
        query = f"""
            SELECT DISTINCT CID, InChI_string 
            FROM data 
            WHERE CID IN ({placeholders})
        """
        print(f"Executing SQL query: {query}")
        
        # Execute the query
        df = pd.read_sql_query(query, conn, params=cids)
        print(f"Query returned {len(df)} rows")
        
        # Convert the DataFrame to a dictionary
        if not df.empty:
            # Check if the InChI column name is exactly 'InChI' or 'InChI_string'
            inchi_column = 'InChI_string' if 'InChI_string' in df.columns else 'InChI'
            
            # Create dictionary mapping CID to InChI
            for _, row in df.iterrows():
                cid_str = str(row['CID'])
                inchi_dict[cid_str] = row[inchi_column]
            
            print(f"Successfully retrieved InChI values for {len(inchi_dict)} CIDs")
            print(f"Sample of retrieved data (first 2 entries):")
            items = list(inchi_dict.items())[:2]
            for cid, inchi in items:
                print(f"CID: {cid}, InChI: {inchi[:50]}...")
        else:
            print("No matching CIDs found in the database")
        
        # Report CIDs that were not found
        missing_cids = set(cids) - set(inchi_dict.keys())
        if missing_cids:
            print(f"Warning: {len(missing_cids)} CIDs not found in database")
            print(f"Missing CIDs (first 5): {list(missing_cids)[:5]}{'...' if len(missing_cids) > 5 else ''}")
            
        return inchi_dict
        
    except Exception as e:
        print(f"ERROR in fetch_inchi_for_cids: {str(e)}")
        return inchi_dict
        
    finally:
        # Ensure connection is closed even if an error occurs
        print("Closing database connection")
        conn.close()

def replace_zeroes_with_predictions(matrix_file='Selected_bioprofile_matrix.csv', output_file='DataFilled_bioprofile_matrix.csv', session_dir=None):
    """
    Replaces 0 values in the bioprofile matrix with predictions from trained RF models.
    """
    if session_dir is None:
        session_dir = SessionManager.get_session_folder()  # Unique per request

    try:
        df = pd.read_csv(matrix_file, index_col=0)
        df.index = df.index.astype(str)  # ✅ Ensure index is string for consistency
        print(f"✅ Loaded selected matrix with shape: {df.shape}")

        model_files = glob.glob(os.path.join(session_dir, "AID*_rf_model.pkl"))
        model_aids = [os.path.basename(f).split('_')[0][3:] for f in model_files]

        if not model_aids:
            print("⚠️ No trained RF models found.")
            return None

        log_data = []
        skipped_log = []

        for aid in model_aids:
            if aid not in df.columns:
                continue

            model_file = os.path.join(session_dir, f"AID{aid}_rf_model.pkl")

            if not os.path.exists(model_file):
                print(f"❌ Model file missing: {model_file}")
                continue

            try:
                model = joblib.load(model_file)
                print(f"✅ Loaded model: {model_file}")
            except Exception as e:
                print(f"❌ Error loading model for AID {aid}: {e}")
                continue

            cids_to_predict = df[df[aid] == 0].index.tolist()
            if not cids_to_predict:
                continue

            # 🔁 Ensure inchi_dict keys are strings
            inchi_dict = fetch_inchi_for_cids(cids_to_predict)
            inchi_dict = {str(k): v for k, v in inchi_dict.items()}

            fingerprints = []
            valid_cids = []

            for cid in cids_to_predict:
                cid_str = str(cid)
                inchi = inchi_dict.get(cid_str)

                if not inchi:
                    skipped_log.append([cid_str, aid, 'Missing InChI'])
                    continue

                mol = Chem.MolFromInchi(inchi)
                if mol is None:
                    skipped_log.append([cid_str, aid, 'RDKit parse failed'])
                    continue

                fp = generate_ecfp6_fingerprint(inchi)
                if fp.sum() == 0:
                    skipped_log.append([cid_str, aid, 'Empty fingerprint'])
                    continue

                fingerprints.append(fp)
                valid_cids.append(cid_str)

            if not fingerprints:
                print(f"⚠️ No valid fingerprints for AID {aid}, skipping.")
                continue

            fingerprints = np.array(fingerprints)
            predictions = model.predict(fingerprints).astype(int)

            for cid_str, pred in zip(valid_cids, predictions):
                if cid_str in df.index:
                    # Map binary model output to ternary matrix encoding
                    filled_value = 1 if int(pred) == 1 else -1
                    df.at[cid_str, aid] = filled_value
                    log_data.append([cid_str, aid, filled_value])
                else:
                    skipped_log.append([cid_str, aid, 'CID not in matrix'])

            print(f"✅ Replaced {len(valid_cids)} values for AID {aid}.")

        # ✅ Sort and save updated matrix
        df = df.sort_index(axis=0)
        df = df.sort_index(axis=1)
        output_file_path = os.path.join(session_dir, output_file)
        df.to_csv(output_file_path)
        print(f"📁 Matrix saved as {output_file_path}")

        # ✅ Save prediction log
        log_file = os.path.join(session_dir, 'replaced_values_log.csv')
        pd.DataFrame(log_data, columns=['CID', 'AID', 'Predicted_Value']).to_csv(log_file, index=False)
        print(f"📁 Prediction log saved as {log_file}")

        # ✅ Save skipped values for diagnosis
        skipped_file = os.path.join(session_dir, 'skipped_predictions_log.csv')
        pd.DataFrame(skipped_log, columns=['CID', 'AID', 'Reason']).to_csv(skipped_file, index=False)
        print(f"📁 Skipped prediction log saved as {skipped_file}")

        return df

    except Exception as e:
        print(f"❌ Error during prediction replacement: {e}")
        return None


# def main():
#     data_file = 'sample_dataset.txt'

#     target_compounds = []
#     with open(data_file, "r") as csv_file:
#         for line in csv_file:
#             target_compounds.append(line.strip())

#     bioprofile(target_compounds, chunk=100, outfile='bioprofile_long.csv')

#     matrix = make_matrix('bioprofile_long.csv', min_actives=1, outfile='bioprofile_matrix.csv')

#     get_heat_map(matrix)
    
#     select_top_assays('bioprofile_matrix.csv', 'SelectedAssays.csv')
    
#     fetch_and_save_aid_data()
    
#     generate_all_rf_models_and_save_metrics()
    
#     plot_rf_metrics()
    
#     filter_selected_aids()
    
#     replace_negative_one_with_predictions()
    
# if __name__ == "__main__":
#     main()