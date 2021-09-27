#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools created with rdkit.
"""

from rdkit import Chem
from rdkit.Chem import rdChemReactions 
from urllib.request import urlopen
from grakel import Graph
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import RDKFingerprint
from rdkit import DataStructs

def mol_from_sdf(sdf_file):
    mol = Chem.SDMolSupplier(sdf_file)
    return mol

def mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol

def mol_to_smiles(mol):
    smiles = Chem.MolToSmiles(mol)
    return smiles

def smiles_enumeration(reaction, reactant1, reactant2):
    """Create a list of SMILES products from the reaction of reactant1 and reactant2 
    SMILES.
    
    Parameters:
    -----------
    reactions: reaction SMARTS string.
    reactant1: SMILES string.
    reactant2: SMILES string.
    """
    rxn = rdChemReactions.ReactionFromSmarts(reaction)
    reacts = (Chem.MolFromSmiles(reactant1), Chem.MolFromSmiles(reactant2))
    products = rxn.RunReactants(reacts)
    products_smiles = [Chem.MolToSmiles(mol) for i in products for mol in i]
    
    return products_smiles

def name_to_smiles(mol_name):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + mol_name + '/smiles'
        smiles = urlopen(url).read().decode('utf8')
        return smiles
    except:
        return float('nan')
    
def name_from_smiles(smiles):
    if '#' in smiles:
        smiles = smiles.replace('#', '%23')
    try:
        url = 'https://cactus.nci.nih.gov/chemical/structure/' + smiles + '/iupac_name'
        name = urlopen(url).read().decode('utf8')
        return name
    except:
        return float('nan')
    
def molg_from_smi(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_with_idx = { i:atom.GetSymbol() for i, atom in enumerate(mol.GetAtoms())}
    bond_with_idx = {i:bond.GetBondTypeAsDouble() for i,bond in enumerate(mol.GetBonds())}
    adj_m = Chem.GetAdjacencyMatrix(mol, useBO=True).tolist()

    return Graph(adj_m, atom_with_idx, bond_with_idx)

# def fps_from_smi(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     fps = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)#.ToBitString()
    
#     return fps

def fps_from_smi(smiles, fps_type=RDKFingerprint, fps_kw={}):
    mol = Chem.MolFromSmiles(smiles)
    fp = fps_type(mol, **fps_kw)
    return fp

def caluclate_tanimoto(fp1, fp2):
    tanimoto = DataStructs.FingerprintSimilarity(
        fp1, fp2, 
        metric=DataStructs.TanimotoSimilarity)
    return tanimoto
