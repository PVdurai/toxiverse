import pandas as pd
import numpy as np
import app.curator.checker as chkr
import app.curator.standardizer as stdr
from rdkit import Chem
from typing import List
from itertools import groupby

class Curator:


    def __init__(self, df: pd.DataFrame, exclude_level=7):
        """

        :param df: a dataframe consisting of 3 columns activity, inchi and compound_id
        """
        # original df
        self.o_df = df
        self.el = 7

    def prep_frame(self):
        # make mol objects then convert to 2d

        self.o_df['o_mols'] = [Chem.MolFromInchi(inchi) for inchi in self.o_df.inchi]
        self.o_df['new_mols'] = [Chem.MolFromSmiles(Chem.MolToSmiles(mol).replace('@', '')) for mol in self.o_df.o_mols]


    def check_cmps(self):
        """ check if there are any structural errors, the errors must be greater than
        or equal to the exclusion level
        """
        flags = []

        for mol in self.o_df.new_mols:
            molblock = Chem.MolToMolBlock(mol)
            errors = chkr.check_molblock(molblock)
            has_flag = any(num >= self.el for num, error in errors)
            flags.append(has_flag)

        self.o_df['flags'] = flags
        self.o_df = self.o_df[self.o_df['flags'] == False]


    def standardize(self):

        standardized_mols = []
        for i, row in self.o_df.iterrows():
            standardized_mols.append(stdr.standardize_mol(row.new_mols))
        self.o_df['new_mols'] = standardized_mols

    def get_parents(self):

        children_mol = []
        for i, row in self.o_df.iterrows():
            child, _ = stdr.get_parent_mol(row.new_mols)
            children_mol.append(child)
        self.o_df['new_mols'] = children_mol

    def handle_duplicates(self, how='higher'):
        self.o_df['new_inchi'] = [Chem.MolToInchi(mol) for mol in self.o_df.new_mols]

        if how in ['higher', 'lower', 'remove']:
            ascending = True if how == 'lower' else False
            self.o_df.sort_values('activity', ascending=ascending, inplace=True)
            keep = 'first' if how in ['higher', 'lower'] else False
            self.o_df = self.o_df.drop_duplicates(subset='new_inchi', keep=keep)

        elif how == 'average':
            # Group by new_inchi and aggregate activity + first instance of inchi/compound_id
            grouped = self.o_df.groupby('new_inchi', as_index=False).agg({
                'activity': 'mean',
                'compound_id': 'first',  # or join list if needed
                'new_mols': 'first'
            })
            self.o_df = grouped


    def curate(self, duplicates):
        self.prep_frame()
        self.check_cmps()
        self.standardize()
        self.get_parents()
        self.handle_duplicates(how=duplicates)

        self.new_df = self.o_df.copy()
        self.new_df['inchi'] = self.new_df.new_inchi
        self.new_df = self.new_df[['activity', 'inchi', 'compound_id']]





if __name__ == '__main__':
    from rdkit.Chem import PandasTools

    df = PandasTools.LoadSDF("../../data/AID_743122.sdf")

    df['inchi'] = [Chem.MolToInchi(mol) for mol in df.ROMol]
    df['activity'] = df.ACTIVITY
    df['compound_id'] = df.CID
    df = df[['inchi', 'activity', 'compound_id']]

    print("Loaded {} molecules".format(len(df)))


    curator = Curator(df)
    curator.curate(duplicates='higher')
    print()
    print()
