"""
Interface to interact with local database containing PubChem data
"""
import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple
import os

# Database path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(ROOT_DIR, 'instance/processed_data.db')

# Fetch raw bioactivity data
def get_raw_bioactivity_data(aid: int):
    print(f"Fetching bioactivity data for AID {aid} from local database")
    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT CID, Activity_Outcome as 'Activity Outcome' FROM data WHERE AID = ? ORDER BY RANDOM() LIMIT 10000"
        print(f"Executing query: {query}")
        df = pd.read_sql_query(query, conn, params=(aid,))
        if df.empty:
            return None, f"No data found for the given AID: {aid}."
        print(f"Retrieved {len(df)} records for AID {aid}")
        print(f"Sample data:\n{df.head(3)}")
        return df[['CID', 'Activity Outcome']], None
    except Exception as e:
        return None, f"Database error for AID {aid}: {str(e)}"
    finally:
        conn.close()
        print("Database connection closed")

# Clean bioactivity data
def clean_bioactivity_frame(df, aid):
    df['Activity Outcome'] = df['Activity Outcome'].replace({
        'Inactive': 0, 'Active': 1, 'Probe': 1, 
        'Inconclusive': np.nan, 'Unspecified': np.nan
    })
    df = df.dropna(subset=['Activity Outcome'])
    if df.empty:
        return None, f"No valid activity outcomes after cleaning for AID {aid}."
    if not df['Activity Outcome'].isin([1]).any():
        return None, f"No active chemicals found in AID {aid}."
    
    df['Activity Outcome'] = df['Activity Outcome'].astype(int)
    
    max_activity = df.groupby('CID')['Activity Outcome'].transform('max')
    df = df[df['Activity Outcome'] == max_activity]
    
    df = df.drop_duplicates(subset='CID')
    
    return df, None

# Fetch InChI values
def get_inchi_from_cids(cids: List, batch_size: int = 1000):
    print(f"Fetching InChI values for {len(cids)} CIDs from local database")
    conn = sqlite3.connect(DB_PATH)
    all_data = []
    try:
        for i in range(0, len(cids), batch_size):
            batch = cids[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}: {len(batch)} compounds")
            placeholders = ','.join(['?'] * len(batch))
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
        result_df = pd.concat(all_data, ignore_index=True)
        print(f"Total InChI records retrieved: {len(result_df)}")
        return result_df, None
    except Exception as e:
        return None, f"Database error while fetching InChI data: {str(e)}"
    finally:
        conn.close()
        print("Database connection closed")

# Fetch assay name
def get_assay_name(aid: int):
    print(f"Fetching assay name for AID {aid} from local database")
    conn = sqlite3.connect(DB_PATH)
    try:
        query = "SELECT DISTINCT assay_name FROM assay_info WHERE AID = ? LIMIT 1"
        cursor = conn.cursor()
        cursor.execute(query, (aid,))
        result = cursor.fetchone()
        if result and result[0]:
            return result[0], None
        return f"Bioassay {aid}", None
    except sqlite3.OperationalError as e:
        if "no such table: assay_info" in str(e):
            return f"Bioassay {aid}", None
        return None, f"Database error: {str(e)}"
    except Exception as e:
        return None, f"Database error: {str(e)}"
    finally:
        conn.close()
        print("Database connection closed")

# Import AID data
def import_pubchem_aid(aid: int):
    print(f"Importing data for AID {aid} from local database")
    raw_data, error = get_raw_bioactivity_data(aid)
    if error:
        return None, error
    print("Raw bioactivity data fetched successfully")

    bioactivity, clean_error = clean_bioactivity_frame(raw_data, aid)
    if clean_error:
        return None, clean_error
    print("Bioactivity data cleaned successfully")

    cids = bioactivity['CID'].dropna().astype(str).tolist()
    if not cids:
        return None, "No CIDs available to fetch InChI data."
    print(f"Fetching InChI data for {len(cids)} compounds...")

    inchi_data, inchi_error = get_inchi_from_cids(cids)
    if inchi_error:
        return None, inchi_error
    print("InChI data fetched successfully")

    bioactivity['CID'] = bioactivity['CID'].astype(str)
    result = pd.merge(inchi_data, bioactivity, on='CID', how='inner')
    if result.empty:
        return None, "No matching CIDs found between bioactivity and InChI data."

    result = result.rename(columns={
        'CID': 'compound_id',
        'InChI': 'inchi',
        'Activity Outcome': 'activity'
    })
    
    result['activity'] = result['activity'].astype(int)

    # Select random compounds
    result = select_diverse_compounds(result, num_actives=500, num_inactives=500)

    return result, None

# Randomly select compounds
def select_diverse_compounds(df, num_actives=500, num_inactives=500):
    active_df = df[df['activity'] == 1].copy()
    inactive_df = df[df['activity'] == 0].copy()

    # If small dataset and not many actives/inactives, return as it is
    if len(df) <= 1000 and len(active_df) <= num_actives and len(inactive_df) <= num_inactives:
        return df

    if not active_df.empty:
        active_df = active_df.sample(n=min(num_actives, len(active_df)), random_state=42)

    if not inactive_df.empty:
        inactive_df = inactive_df.sample(n=min(num_inactives, len(inactive_df)), random_state=42)

    final_df = pd.concat([active_df, inactive_df])

    return final_df
