from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit import Chem
from rdkit.ML.Cluster import Butina
import pandas as pd
from rdkit.Chem import PandasTools
from typing import List
import math



def calc_descriptors_from_mol(mol):
    """
    Encode a molecule from a RDKit Mol into a set of descriptors.

    Parameters
    ----------
    mol : RDKit Mol
        The RDKit molecule.

    Returns
    -------
    list
        The set of chemical descriptors as a list.

    """
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([desc[0] for desc in Descriptors.descList])
    return list(calc.CalcDescriptors(mol))



def make_descriptors(inchis: List[str]) -> pd.DataFrame:
    mols = [Chem.MolFromInchi(inchi) for inchi in inchis]
    desc_list = []

    for mol in mols:
        desc = calc_descriptors_from_mol(mol)
        desc_list.append(desc)

    chemical_descriptors = pd.DataFrame(desc_list,
                                        columns=[desc[0] for desc in Descriptors.descList],
                                        index=inchis)
    chemical_descriptors = chemical_descriptors.drop('Ipc', axis=1)
    chemical_descriptors = chemical_descriptors.loc[:, chemical_descriptors.notnull().all()]
    return chemical_descriptors.fillna(chemical_descriptors.mean())

def get_fps(mol_list, fp_type):
    fp_dict = {
        "morgan2": [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in mol_list],
        "rdkit": [Chem.RDKFingerprint(x) for x in mol_list],
        "maccs": [MACCSkeys.GenMACCSKeys(x) for x in mol_list],
    }

    return fp_dict[fp_type]


def cluster_mols(fps, sim_cutoff=0.8):
    dist_cutoff = 1 - sim_cutoff
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    mol_clusters = Butina.ClusterData(dists, nfps, dist_cutoff, isDistData=True)
    cluster_id_list = [0] * nfps
    for idx, cluster in enumerate(mol_clusters, 1):
        for member in cluster:
            cluster_id_list[member] = idx
    return [x - 1 for x in cluster_id_list]


def z_score(x_bar_0, x_bar_1, n_0, n_1, var_0):
    """

    :param x_bar_0: mean of full set
    :param x_bar_1: mean of subset
    :param n_0: size of full set
    :param n_1: size of sub set
    :param var_0: variance of full set
    :return: z score


    tkae from finding discriminating structrual feature sby reassembling common building blocks.  Cross Kevin
    """
    z = (x_bar_0 - x_bar_1)*math.sqrt((n_0*n_1) / (var_0*(n_0-n_1)))
    return z