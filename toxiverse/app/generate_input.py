import os

import pandas as pd

from numpy import nan

from rdkit.Chem import MolFromSmarts, SDMolSupplier
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Chem.rdmolops import RDKFingerprint


def get_assay_list(path):
    """
    :param path: File path to assay organization file
    :return: List of assays in order
    """

    return list(pd.read_csv(path, index_col=0, header=0).sort_values('Level', ascending=True).index)


def generate_molecules(path):
    """
    :param path: File path to molecule sdf file
    :return: List of rdkit mols
    """

    return [mol for mol in SDMolSupplier(path) if mol is not None]


def calc_maccs(mols, identifier='Code'):
    """
    :param mols: List of rdkit mols
    :param identifier: Name of the field to index the resulting DataFrame. Needs to be a valid property of all molecules
    :return: Matrix of MACCS keys as pandas DataFrame
    """

    data = [[int(x) for x in GenMACCSKeys(mol)] for mol in mols]
    index = [mol.GetProp(identifier) for mol in mols]
    cols = [f'M{i}' for i in range(len(data[0]))]
    fp = pd.DataFrame(data, index=index, columns=cols)

    # return fp.loc[:, (fp != 0).any(axis=0)]
    return fp


def calc_fcfp6(mols, identifier='Code'):
    """
    :param mols: List of rdkit mols
    :param identifier: Name of the field to index the resulting DataFrame. Needs to be a valid property of all molecules
    :return: Matrix of FCFPs as pandas DataFrame
    """

    data = [[int(x) for x in GetMorganFingerprintAsBitVect(mol, 3, 1024, useFeatures=True)] for mol in mols]
    index = [mol.GetProp(identifier) for mol in mols]
    cols = [f'F{i}' for i in range(len(data[0]))]
    fp = pd.DataFrame(data, index=index, columns=cols)

    return fp


def calc_rdkf(mols, identifier='Code'):
    """
    :param mols: List of rdkit mols
    :param identifier: Name of the field to index the resulting DataFrame. Needs to be a valid property of all molecules
    :return: Matrix of rdkit fingerprints as pandas DataFrame
    """

    index = [mol.GetProp(identifier) for mol in mols]
    data = pd.DataFrame([[int(x) for x in RDKFingerprint(mol, fpSize=1024)] for mol in mols], index=index)
    data.columns = [f'R{i}' for i in range(data.shape[1])]
    data = data.dropna(axis=1)

    return data.loc[:, (data != 0).any(axis=0)]


def calc_fcfp_er(mols, identifier='Code'):
    """
    :param mols: List of rdkit mols
    :param identifier: Name of the field to index the resulting DataFrame. Needs to be a valid property of all molecules
    :return: Matrix of all available fingerprints as pandas DataFrame
    """

    fcfp6 = calc_fcfp6(mols, identifier=identifier)
    er_alerts = calc_er_alerts(mols, identifier=identifier)

    return pd.concat([fcfp6, er_alerts], axis=1)


def calc_er_alerts(mols, identifier='Code'):
    """
    :param mols: List of rdkit mols
    :param identifier: Name of the field to index the resulting DataFrame. Needs to be a valid property of all molecules
    :return: Matrix of ER structural alerts as pandas DataFrame
    """

    fragments = ['[#6]1~[#6]~[#6]~[#6]2~[#6](~[#6]~1)~[#6]~[#6]~[#6]1~[#6]~2~[#6]~[#6]~[#6]2~[#6]~1~[#6]~[#6]~[#6]~2',
                 '[#8H]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1',
                 '[#6]1:[#6]:[#6]:[#6](:[#6]:[#6]:1)-[#6]~[#6]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1']
    submols = [MolFromSmarts(smarts) for smarts in fragments]
    matches = [[1 if mol.HasSubstructMatch(fragment) else 0 for mol in mols] for fragment in submols]
    index = [mol.GetProp(identifier) for mol in mols]

    return pd.DataFrame(matches, index=['steroid_skel', 'phenol', 'des_skel'], columns=index).T


def calc_fingerprint(mols, fragments, identifier='Code'):
    """
    :param mols: List of rdkit mols
    :param fragments: Molecular descriptor set to be used for modeling
    :param identifier: Name of the field to index the resulting DataFrame. Needs to be a valid property of all molecules
    :return: Matrix of fingerprints as pandas DataFrame
    """

    fragment_fxs = {
        'FCFP6': lambda mols: calc_fcfp6(mols, identifier=identifier),
        'MACCS': lambda mols: calc_maccs(mols, identifier=identifier),
        'rdkf': lambda mols: calc_rdkf(mols, identifier=identifier),
        'FCFP_ER': lambda mols: calc_fcfp_er(mols, identifier=identifier),
    }

    return fragment_fxs[fragments](mols)


def get_invivo(mols, endpoint, identifier='Code'):
    """
    :param mols: List of rdkit mols
    :param endpoint: Mol property containing activity of interest
    :param identifier: Name of the field to index the resulting DataFrame. Needs to be a valid property of all molecules
    :return: Activity vector as pandas Series
    """

    index = [mol.GetProp(identifier) for mol in mols]

    if mols[0].GetProp(endpoint) in ['Active', 'Inactive']:
        return pd.Series([1 if mol.GetProp(endpoint) in ['Active'] else 0 for mol in mols], index=index)

    elif mols[0].GetProp(endpoint) in ['0.0', '0', '1.0', '1']:
        return pd.Series([int(float(mol.GetProp(endpoint))) for mol in mols], index=index)

    else:
        raise Exception('Function get_invivo() needs to be modified to accept this format of activity.')


def get_activity_matrix(mols, endpoints, identifier='Code', mask_value=-1):
    """
    Takes in a list of rdkit molecules and returns a matrix with desired activities
    :param mols: List of rdkit mols
    :param endpoints: Mol properties containing activities of interest
    :param identifier: Name of the field to index the resulting DataFrame. Needs to be a valid property of all molecules
    :param mask_value: Value to use for missing activities (defaults to -1.00)
    :return: Activity matrix as pandas DataFrame
    """

    ys = pd.DataFrame(index=[mol.GetProp(identifier) for mol in mols])

    for endpoint in endpoints:
        y = []

        for mol in mols:
            if mol.HasProp(endpoint):
                y.append(float(mol.GetProp(endpoint)))
            else:
                y.append(mask_value)

        ys[endpoint] = y

    return ys


def cache_data(profiles, in_vivo, directory, dataset, endpoint, assays, fragments):
    """
    :param profiles: pandas DataFrame of calculated chemical fingerprints and in vitro activity information
    :param in_vivo: pandas Series of in vivo activity information
    :param directory: File path pointing to project directory
    :param dataset: Name of molecule sdf file
    :param endpoint: Mol property containing activity of interest
    :param assays: Name of file containing assay list of interest
    :param fragments: Molecular descriptor set to be used for modeling
    """

    cache_matrix = pd.concat([profiles, in_vivo], axis=1)
    cache_matrix.to_csv(os.path.join(directory, 'caches', f'{dataset}_{endpoint}_{assays}_{fragments}.csv'))


def load_cache_data(directory, dataset, endpoint, assays, fragments):
    """
    :param directory: File path pointing to project directory
    :param dataset: Name of molecule sdf file
    :param endpoint: Mol property containing activity of interest
    :param assays: Name of file containing assay list of interest
    :param fragments: Molecular descriptor set to be used for modeling
    :return: one pandas DataFrame containing chemical fingerprints and in vitro data, and one pandas Series containing
    in vivo data
    """

    cache_dir = os.path.join(directory, 'caches', f'{dataset}_{endpoint}_{assays}_{fragments}.csv')
    data = pd.read_csv(cache_dir, header=0, index_col=0)

    return data.iloc[:, :-1], data.iloc[:, -1]


def check_data_cache(directory, dataset, endpoint, assays, fragments):
    """
    :param directory: File path pointing to project directory
    :param dataset: Name of molecule sdf file
    :param endpoint: Mol property containing activity of interest
    :param assays: Name of file containing assay list of interest
    :param fragments: Molecular descriptor set to be used for modeling
    :return: Boolean about whether cache files exist
    """

    return os.path.exists(os.path.join(directory, 'caches', f'{dataset}_{endpoint}_{assays}_{fragments}.csv'))


def load_data(dataset, directory, assay_file, endpoint, fragments='MACCS', identifier='Code'):
    """
    :param dataset: Name of molecule sdf file
    :param directory: File path pointing to project directory
    :param assay_file: Name of file containing relevant assay names
    :param endpoint: Mol property containing activity of interest
    :param fragments: Molecular descriptor set to be used for modeling (defaults to MACCS keys)
    :param identifier: Name of the field to index the resulting DataFrame. Needs to be a valid property of all molecules
    :return: one pandas DataFrame containing chemical fingerprints and in vitro data, and one pandas Series containing
    in vivo data
    """

    if check_data_cache(directory, dataset, endpoint, assay_file, fragments):
        return load_cache_data(directory, dataset, endpoint, assay_file, fragments)

    else:
        mols = generate_molecules(os.path.join(directory, f'{dataset}.sdf'))
        X = calc_fingerprint(mols, fragments, identifier=identifier)
        assays = get_assay_list(os.path.join(directory, f'{assay_file}.csv'))
        in_vitro = get_activity_matrix(mols, assays, identifier=identifier)
        in_vivo = get_invivo(mols, endpoint)
        X = X[X.index.isin(in_vitro.index)]
        in_vivo = in_vivo[in_vivo.index.isin(in_vitro.index)]
        profiles = pd.concat([X, in_vitro], axis=1)

        assert not (nan in profiles or -1.0 in profiles)

        cache_data(profiles, in_vivo, directory, dataset, endpoint, assay_file, fragments)

        return profiles, in_vivo


def cache_dataset_fragments(dataset, directory, fragments, fragment_name):
    """
    Cache QSAR dataset features
    :param dataset: Name of molecule sdf file
    :param directory: File path pointing to project directory
    :param fragments: Matrix containing fragments of interest
    :param fragment_name: Molecular descriptor set to be used for modeling
    """

    fragments.to_csv(os.path.join(directory, 'caches', f'{dataset}_{fragment_name}.csv'))


def load_dataset_fragments(dataset, directory, fragment_name):
    """
    :param dataset: Name of molecule sdf file
    :param directory: File path pointing to project directory
    :param fragment_name: Molecular descriptor set to be used for modeling
    :return: one pandas DataFrame containing chemical fingerprints
    """

    return pd.read_csv(os.path.join(directory, 'caches', f'{dataset}_{fragment_name}.csv'), header=0, index_col=0)


def check_fragment_cache(dataset, directory, fragment_name):
    """
    :param dataset: Name of molecule sdf file
    :param directory: File path pointing to project directory
    :param fragment_name: Molecular descriptor set to be used for modeling
    :return: Boolean about whether fragment cache files exist
    """

    return os.path.exists(os.path.join(directory, 'caches', f'{dataset}_{fragment_name}.csv'))


def get_fragments(dataset, directory, features, identifier='Code'):
    """
    :param dataset: Name of molecule sdf file
    :param directory: File path pointing to project directory
    :param features: Molecular descriptor set to be used for modeling
    :return: Descriptor matrix as pandas DataFrame
    """

    if check_fragment_cache(dataset, directory, features):
        return load_dataset_fragments(dataset, directory, features)

    else:
        return calc_fingerprint(generate_molecules(os.path.join(directory, f'{dataset}.sdf')), features,
                                identifier=identifier)


def load_qsar_dataset(dataset, directory, endpoints, features='MACCS', identifier='Code'):
    """
    :param dataset: Name of molecule sdf file
    :param directory: File path pointing to project directory
    :param endpoints: Mol properties containing activities of interest
    :param features: Molecular descriptor set to be used for modeling
    :param identifier: Name of the field to index the resulting DataFrame. Needs to be a valid property of all molecules
    :return: (X, y): (Feature matrix as a pandas DataFrame, Activity matrix as a pandas DataFrame)
    """

    X = get_fragments(dataset, directory, features, identifier=identifier)
    ys = get_activity_matrix(generate_molecules(os.path.join(directory, f'{dataset}.sdf')), endpoints, identifier)

    return X, ys

