from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import pandas as pd

from sklearn.preprocessing import StandardScaler

def calc_descriptors_from_frame(df: pd.DataFrame, scale=False, desc_set=None) -> pd.DataFrame:
    """ calculates rdkit descriptors from a smiles.txt file """

    df['ROMol'] = [Chem.MolFromInchi(inchi) for inchi in df.inchi]

    if desc_set:
        desc_set = [desc[0] for desc in Descriptors.descList if desc[0] in desc_set]
    else:
        desc_set = [desc[0] for desc in Descriptors.descList]

    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_set)

    X = pd.DataFrame([list(calc.CalcDescriptors(mol)) for mol in df['ROMol']],
                     columns=list(calc.GetDescriptorNames()),
                     index=df.compound_id)
    X = X.loc[X.notnull().all(1), :]

    if scale:
        X = pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=X.columns)
    return X

def calc_fingerprints_from_frame(df: pd.DataFrame, kind='ECFP6') -> pd.DataFrame:

    df['ROMol'] = [Chem.MolFromInchi(inchi) for inchi in df.inchi]

    data = []

    for mol in df['ROMol'].values:
        fps = [float(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol,
                                                                         3,
                                                                         1024,
                                                                         useFeatures=True if kind == 'FCFP6' else False)]
        data.append(fps)

    return pd.DataFrame(data, index=df.compound_id)


def get_desc(df: pd.DataFrame, kind) -> pd.DataFrame:
        if kind == 'RDKit':
            return calc_descriptors_from_frame(df)
        else:
            return calc_fingerprints_from_frame(df, kind)