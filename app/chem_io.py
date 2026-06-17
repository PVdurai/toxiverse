from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


def _get_compound_id(row, fallback_index):
    """
    Safely get compound_id for error messages.
    """
    if "compound_id" in row and pd.notna(row["compound_id"]):
        return row["compound_id"]
    if "CID" in row and pd.notna(row["CID"]):
        return row["CID"]
    return fallback_index


def _mol_from_inchi_checked(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert InChI strings to RDKit molecules and stop with a clear error
    if any molecule cannot be parsed.
    """
    if "inchi" not in df.columns:
        raise ValueError("Missing required column: inchi")

    df = df.copy()
    mols = []
    invalid_rows = []

    for idx, row in df.iterrows():
        inchi = row.get("inchi")
        compound_id = _get_compound_id(row, idx)

        if pd.isna(inchi) or str(inchi).strip() == "":
            mols.append(None)
            invalid_rows.append((idx, compound_id, inchi))
            continue

        mol = Chem.MolFromInchi(str(inchi).strip())
        mols.append(mol)

        if mol is None:
            invalid_rows.append((idx, compound_id, inchi))

    if invalid_rows:
        examples = []
        for row_index, compound_id, inchi in invalid_rows[:10]:
            examples.append(
                f"row/index={row_index}, compound_id={compound_id}, inchi={inchi}"
            )

        msg = (
            f"Invalid molecule structure found in {len(invalid_rows)} row(s). "
            "RDKit could not convert these InChI values to molecules. "
            "Please correct or remove them before QSAR building.\n"
            + "\n".join(examples)
        )

        if len(invalid_rows) > 10:
            msg += f"\n... and {len(invalid_rows) - 10} more invalid row(s)."

        raise ValueError(msg)

    df["ROMol"] = mols
    return df


def calc_descriptors_from_frame(
    df: pd.DataFrame,
    scale=False,
    desc_set=None
) -> pd.DataFrame:
    """
    Calculate RDKit molecular descriptors from InChI values.
    """

    df = _mol_from_inchi_checked(df)

    if desc_set:
        desc_set = [desc[0] for desc in Descriptors.descList if desc[0] in desc_set]
    else:
        desc_set = [desc[0] for desc in Descriptors.descList]

    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_set)

    X = pd.DataFrame(
        [list(calc.CalcDescriptors(mol)) for mol in df["ROMol"]],
        columns=list(calc.GetDescriptorNames()),
        index=df.compound_id
    )

    # Replace infinite values and remove rows with invalid descriptor values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[X.notnull().all(axis=1), :]

    if scale:
        X = pd.DataFrame(
            StandardScaler().fit_transform(X),
            index=X.index,
            columns=X.columns
        )

    return X


def calc_fingerprints_from_frame(
    df: pd.DataFrame,
    kind="ECFP6"
) -> pd.DataFrame:
    """
    Calculate Morgan fingerprints from InChI values.

    ECFP6: radius 3, useFeatures=False
    FCFP6: radius 3, useFeatures=True
    """

    df = _mol_from_inchi_checked(df)

    data = []

    use_features = True if kind == "FCFP6" else False

    for mol in df["ROMol"].values:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=3,
            nBits=1024,
            useFeatures=use_features
        )
        fps = [float(x) for x in fp]
        data.append(fps)

    return pd.DataFrame(data, index=df.compound_id)


def get_desc(df: pd.DataFrame, kind) -> pd.DataFrame:
    if kind == "RDKit":
        return calc_descriptors_from_frame(df)
    else:
        return calc_fingerprints_from_frame(df, kind)