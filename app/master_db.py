import sqlite3 as sql
import pandas as pd
import app.config as config
from rdkit.Chem import PandasTools

def make_connection():
    return sql.connect(config.Config.MASTER_DB_FILE)

def get_master() -> pd.DataFrame:
    con = make_connection()
    return pd.read_sql('select * from Master_database', con=con)

def make_query(query: str) -> pd.DataFrame:
    con = make_connection()
    return pd.read_sql(query, con=con)


#def get_database(db_name: str) -> pd.DataFrame:
    """Gets all compounds in the database associated with a given table."""

    q = """
    SELECT ml.[Master-ID] as [Master-ID],
           cl.CID as CID_COL,
           db.*
    FROM master_lookup ml
    INNER JOIN cid_lookup cl ON ml.[Master-ID] = cl.[Master-ID]
    INNER JOIN {} db ON db.[Dataset-ID] = ml.[Dataset-ID]
    WHERE ml.db = '{}'
    """.format(db_name, db_name)

    return make_query(q)

def get_database(db_name: str) -> pd.DataFrame:
    """
    Gets all compounds in the given curated dataset and their Master-ID from Master_database,
    keeping the Master-ID column from curated tables if present.
    """
    q = f"""
    SELECT md.[Master-ID] as MasterID_from_master,
           db.*
    FROM Master_database md
    INNER JOIN [{db_name}] db ON db.[Dataset-ID] = md.[Dataset-ID]
    """
    df = make_query(q)

    # If curated table already has Master-ID, use it
    if 'Master-ID' in df.columns:
        df.drop(columns=['MasterID_from_master'], inplace=True)
    else:
        df.rename(columns={'MasterID_from_master': 'Master-ID'}, inplace=True)

    return df

def get_raw_table(db_name: str) -> pd.DataFrame:
    """Gets all rows directly from the given curated table (no joins)."""
    q = f"SELECT * FROM [{db_name}]"
    return make_query(q)


TABLE_ACTIVITES = {
    'Hepatotoxicity_curated': ['PC_HT_class', 'H_HT_class'],
    'DART_curated': ['Oral-dev', 'Oral-mat'],
    'LD50_curated': ['LD50_mgkg'],
    'Estrogen_curated': ['ER-alpha log10 of 100*RBA (relative binding affinity vs estrogen)','Agonist Class','Agonist Potency','Antagonist Class','Antagonist Potency','Binding Class','Binding Potency','Uterotrophic Class']
}

# Lazy loading of CURRENT_DATABASES to avoid import-time errors
CURRENT_DATABASES = None

def get_current_databases():
    """Get list of current databases with error handling."""
    global CURRENT_DATABASES
    if CURRENT_DATABASES is None:
        try:
            CURRENT_DATABASES = [record for record in make_query('select distinct db from master_lookup').db.values
                                 if '_curated' in record]
        except Exception as e:
            print(f"Warning: Could not load current databases from master_lookup table: {e}")
            CURRENT_DATABASES = []
    return CURRENT_DATABASES

if __name__ == '__main__':
    hep = get_database('Hepatotoxicity_curated')
    print(hep)


