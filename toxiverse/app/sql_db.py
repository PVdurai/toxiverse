import sqlite3 as sql
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem import rdBase
from rdkit.Chem import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdFMCS
import chem, time


import config
import requests
from typing import List
import main
import chem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob, os, ntpath

TABLE_ACTIVITES = {
    'Hepatotoxicity_curated': ['PC_HT_class', 'H_HT_class'],
    'DART_curated': ['Oral-dev', 'Oral-mat'],
    'LD50_curated': ['LD50_mgkg'],
    'Estrogen_curated': ['ER-alpha log10 of 100*RBA (relative binding affinity vs estrogen)','Agonist Class','Agonist Potency','Antagonist Class','Antagonist Potency','Binding Class','Binding Potency','Uterotrophic Class']
}


def make_connection():
    return sql.connect(config.Config.MASTER_DB_FILE)

#bdb_con = sql.connect(config.Config.BINDING_DB)
# ID_COLUMNS = ['AlogS-ID', 'BBB-ID', 'Herbal-ID', 'DRUGBANK_ID',
#               'HPV-ID', 'natural-ID', 'AMES-ID', 'BCRP-ID',
#               'BSEP-ID', 'EB-ID']
#ID_COLUMNS = ['DART-ID', 'heptox-ID', 'LD50-ID',  'BBB-ID', 'AlogS-ID'][0:4]
#TABLE_ID = ['DART_curated', 'Hepatotoxicity_curated', 'LD50_curated', 'BBB_curated', 'AlogS_curated'][0:4]

def make_sql():
    """ load all the SDFs and in the DATA_DIR folder and create a master sql file """
    con = make_connection()
    for db in main.ALL_DATABASES:
        df = main.load_db(db)
        df.to_sql(db, con, if_exists='replace', index=False)
        print(f"loaded {db}")

    master = main.load_master(mol_col=True)
    master['InChiKey'] = [Chem.MolToInchiKey(mol) for mol in master.ROMol]

    cids = get_cids_from_inchikeys(master['InChiKey'].values.tolist())
    master = master.merge(cids, how='left', on='InChiKey')

    master = master.drop(columns=['ROMol'])

    master.to_sql('Master_database', con, if_exists='replace', index=False)

    create_master_lookup()
    create_cid_lookup()

def create_master_lookup():
    """ creates a master look up data to map all chemicals
     to their respective databases """
    con = make_connection()
    df = pd.read_sql('select * from Master_database', con)

    ID_COLUMNS = [col for col in df.columns if col.endswith('-ID') and col != 'Master-ID']

    df_melt = df.melt(id_vars=['Master-ID', 'CleanedInChI'],
                                        value_vars=ID_COLUMNS)
    df_melt = df_melt.rename(columns={'variable': 'db', 'value': 'Dataset-ID'})

    # rename columns to table names
    #TODO: We need to either add new datasets to this mapper or find a way to automate this by reading the sdf
    # mapper = {
    #     'DART-ID': 'DART_curated',
    #     'heptox-ID': 'Hepatotoxicity_curated',
    #     'LD50-ID': 'LD50_curated',
    #     'BBB-ID': 'BBB_curated',
    #     'AlogS-ID':  'AlogS_curated',
    # }

    mapper = create_mapper()

    df_melt['db'] = df_melt.db.map(mapper)
    df_null = df_melt[df_melt['Dataset-ID'].isnull()]
    df_null.to_csv('null.csv')

    df_melt = df_melt[df_melt['Dataset-ID'].notnull()]

    df_melt.to_sql('master_lookup', con, if_exists='replace', index=False)


def get_database(db_table: str):
    """ exports all the data as a pandas dataframe """
    if db_table not in ['Hepatotoxicity_curated', 'LD50_curated', 'DART_curated']:
        raise Exception(f"Sorry, {db_table} is not a valid table in the database.")

    con = make_connection()
    df = pd.read_sql(f'select * from {db_table}', con=con)
    return df

def get_cids_from_inchikeys(identifier_list: List[str]) -> pd.DataFrame:
    data = []

    num_chemicals = len(identifier_list)
    counter = 0

    for inchikey in identifier_list:
        url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{}/cids/txt'.format(inchikey)
        response = requests.get(url)
        time.sleep(0.2)
        if response.status_code == 200:
            cids = '|'.join(response.text.split('\n')[:-1])
            data.append((inchikey, cids))
        else:
            continue
        if counter % 100 == 0:
            print(f"Got CIDs for {counter} compounds...this could take a while....")
        counter = counter + 1

    return pd.DataFrame(data, columns=['InChiKey', 'CID'])


def create_cid_lookup():
    con = make_connection()
    df = pd.read_sql('select * from Master_database', con)
    df = df[['Master-ID', 'CleanedInChI', 'CID']]
    df.loc[:, 'CID'] = df.CID.str.split('|')
    df = df[df.CID.notnull()]
    df = df.explode('CID')
    df.to_sql('cid_lookup', con, if_exists='replace', index=False)


def add_bioprofile(from_cache=False):
    """ adds the bioprofile as a table to the database.  Requires the bioprofile
     notebook to be run.  """
    if from_cache:

        df = pd.read_csv(config.Config.BIOPROFILE_LONG)
        con = make_connection()
        df.to_sql('bioprofile', con, if_exists='replace', index=False)

    else:
        cids = make_query('select CID from cid_lookup').CID.unique().tolist()
        from bioprofile import bioprofile as bp
        con = make_connection()
        df = bp(cids, chunk=100)
        df.to_sql('bioprofile', con, if_exists='replace', index=False)



def add_chemical_space_table():
    """ creates chemical structure table from master database """
    from mordred import Calculator, descriptors
    con = make_connection()
    df = pd.read_sql(f'select * from master_lookup', con)
    cids = make_query('select * from cid_lookup')
    cids['CID'] = cids.CID.astype(int)
    active_stats = make_query('select * from active_stats')
    d = cids.merge(active_stats, how='left').fillna(0)
    df2 = df.merge(d, how='left')

    pc_stats = df2.groupby('Master-ID')['n_actives', 'n_aids'].sum().reset_index()

    calc = Calculator(descriptors, ignore_3D=True)

    mols = [Chem.MolFromInchi(inchi) for inchi in df.CleanedInChI.values.tolist()]
    desc = calc.pandas(mols)

    print(desc.head())

    desc_scl = StandardScaler().fit_transform(desc)
    desc_scl = pd.DataFrame(desc_scl, columns=desc.columns)
    desc_scl = desc_scl.dropna(axis=1)

    comps = PCA(n_components=3).fit_transform(desc_scl)

    comps = pd.DataFrame(comps, columns=['PCA1', 'PCA2','PCA3'])

    comps.loc[:, 'Master-ID'] = df['Master-ID']
    comps.loc[:, 'CleanedInChI'] = df['CleanedInChI']
    comps.loc[:, 'svg'] = [Chemical(inchi).get_svg() for inchi in comps.CleanedInChI]

    comps = comps.merge(pc_stats)

    master = make_query('select * from Merged')

    ID_COLUMNS = [col for col in master.columns if col.endswith('-ID') and col != 'Master-ID']

    dbs = master.copy()

    dbs[ID_COLUMNS] = master[ID_COLUMNS].applymap(lambda x: False if x == None else True)
    dbs['Master-ID'] = master['Master-ID']

    comps = comps.merge(dbs[ID_COLUMNS + ['Master-ID']])

    comps.to_sql('chemical_space', if_exists='replace', index=False, con=con)


def add_meta_stats_bioprofile():
    """ add distribution stats to the the database to easily calculate precentiles
     from the profile """
    bioprofile = make_query('select AID, CID, [Activity Outcome] from bioprofile')

    cid_x_actives = (
        bioprofile
        .query('`Activity Outcome` == "Active"')
        .groupby(['CID'])
        ['AID'].count()
        .reset_index()
        .rename(columns={'AID': 'n_actives'})

    )
    cid_x_aid = (
        bioprofile
        .groupby(['CID'])
        ['AID'].count()
        .reset_index()
        .rename(columns={'AID': 'n_aids'})

    )

    active_stats = cid_x_actives.merge(cid_x_aid, on='CID')
    active_stats['hit_rate'] = (active_stats.n_actives / active_stats.n_aids * 100).round(2)
    con = make_connection()
    active_stats.to_sql('active_stats', con, if_exists='replace', index=False)


def add_cluster_table():
    """ cluster the chemicals in the main database and two tables to the database
     the first table stores the clusters and the maximum common substructure for those
     chemicals the second table display"""
    toxdb = make_query('select CleanedInChI, [Master-ID] from Master_database')


    toxdb['ROMol'] = [Chem.MolFromInchi(inchi) for inchi in toxdb.CleanedInChI]

    print(f"{toxdb.shape[0]} chemicals for clustering")

    toxdb['fp'] = chem.get_fps(toxdb.ROMol.values.tolist(), 'maccs')
    toxdb['Cluster'] = chem.cluster_mols(toxdb.fp.values.tolist())



    cluster_rows = []
    for cluster, c_data in toxdb.groupby("Cluster"):
        if len(c_data) > 1:
            mcs = rdFMCS.FindMCS(c_data.ROMol.values.tolist(), timeout=10).smartsString
        else:
            mcs = 'None'

        cluster_rows.append([cluster, mcs, len(c_data)])
    cluster_df = pd.DataFrame(cluster_rows, columns=["Cluster", "MCS_Smarts", "NumChemicals"])

    toxdb = toxdb.drop(['ROMol', 'fp', 'CleanedInChI'], axis=1)

    toxdb.to_sql('ClusterMembership', if_exists='replace', con=make_connection(), index=False)
    cluster_df.to_sql('Clusters', if_exists='replace', con=make_connection(), index=False)


def add_saagar(saagar_file="~/data/tox21/Saagar_v1_834_SMARTS.txt"):
    chemicals = make_query('select CleanedInChI, [Master-ID] from Master_database')
    mols = [Chem.MolFromInchi(inchi) for inchi in chemicals.ClearnedInChI]
    pass


def get_chemical(inchi: str) -> pd.DataFrame:
    con = make_connection()
    df = pd.read_sql(f'select * from master_lookup where CleanedInChI == "{inchi}"', con).replace("heptox", "Hepatotoxicity")

    # the databses are listed as DB-ID, so we just
    # need to split and get the DB
    dbs = [db.split('-')[0] for db in df.db.values.tolist()]
    ids = df['Dataset-ID'].values.tolist()

    frames = []
    for db, _id in zip(dbs, ids):
        frame = pd.read_sql(f'select * from {db} where [Dataset-ID] == "{_id}"', con)
        frames.append(frame)

    return pd.concat(frames)


def get_chemical_space() -> pd.DataFrame:
    con = make_connection()
    return pd.read_sql('select * from chemical_space', con=con)

def get_master() -> pd.DataFrame:
    con = make_connection()
    return pd.read_sql('select * from Master_database', con=con)

def make_query(query: str) -> pd.DataFrame:
    con = make_connection()
    return pd.read_sql(query, con=con)

class Chemical:

    def __init__(self, inchi):
        self.inchi = inchi

    def get_databases(self):
        con = make_connection()
        df = pd.read_sql(f'select * from master_lookup where CleanedInChI == "{self.inchi}"', con).replace("heptox",
                                                                                                      "Hepatotoxicity")
        dbs = [db.split('-')[0] for db in df.db.values.tolist()]
        return dbs

    def get_cids(self):
        con = make_connection()
        df = pd.read_sql(f'select * from cid_lookup where CleanedInChI == "{self.inchi}"', con)

        cids = df.CID.values.tolist()
        return cids

    def get_master_id(self):
        con = make_connection()
        df = pd.read_sql(f'select [Master-ID] from Master_database where CleanedInChI == "{self.inchi}"', con)
        return df['Master-ID'].iloc[0]


    def get_assays(self):
        con = make_connection()
        cids = self.get_cids()
        cid_q = "({})".format(','.join(cids))

        assays = pd.read_sql(f'select AID, [Activity Outcome] from'
                             f' bioprofile where CID in {cid_q}', con)

        return assays

    def get_tox_data(self):
        con = make_connection()
        df = pd.read_sql(f'select * from master_lookup where CleanedInChI == "{self.inchi}"', con).replace("heptox",
                                                                                                      "Hepatotoxicity")

        # the databses are listed as DB-ID, so we just
        # need to split and get the DB
        dbs = [db.split('-')[0] for db in df.db.values.tolist()]
        ids = df['Dataset-ID'].values.tolist()

        frames = []
        for db, _id in zip(dbs, ids):
            frame = pd.read_sql(f'select * from {db} where [Dataset-ID] == "{_id}"', con)
            frames.append(frame)

        return pd.concat(frames)


    def get_active_stats(self):
        cids = [ str(int(cid)) for cid in self.get_cids()]
        cid_q = "({})".format(','.join(cids))
        query = f"select cid, n_actives, n_aids from active_stats where CID in {cid_q}"
        df = make_query(query)
        if df.empty:
            df[['n_actives', 'n_aids']] = 0
        counts = df[['n_actives', 'n_aids']].sum()
        return counts

    def get_svg(self, size=(200, 200)):
        mol = Chem.MolFromInchi(self.inchi)
        rdDepictor.Compute2DCoords(mol)
        mc = Chem.Mol(mol.ToBinary())
        Chem.Kekulize(mc)
        drawer = Draw.MolDraw2DSVG(size[0], size[1])
        drawer.DrawMolecule(mc)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace('svg:', '')
        return svg


def build_db():
    make_sql()
    add_bioprofile()

    add_meta_stats_bioprofile()
    add_chemical_space_table()
    add_cluster_table()

def create_mapper():
    """ a helper function that will create a dictionary that maps the ID
     column in the master database to what table it belongs.  """

    mapper = {}

    # first get all the sdfs
    sdfiles = [f for f in glob.glob(os.path.join(config.Config.DATA_DIR, '*.sdf')) if
               ntpath.basename(f) != 'Merged.sdf']
    for f in glob.glob(os.path.join(config.Config.DATA_DIR, '*.sdf')):
        if ntpath.basename(f) != 'Merged.sdf':

            tablename = ntpath.basename(f).split('.')[0]

            df = PandasTools.LoadSDF(f)

            # get DB-ID name
            db_name_id = df['Dataset-ID'].iloc[0].split('-')[0] + '-ID'
            mapper[db_name_id] = tablename

    return mapper

if __name__ == '__main__':
    build_db()
    test = True

    if test:
        aspirin_inchi = "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
        df = get_chemical(aspirin_inchi)
        print(df)

#%%
