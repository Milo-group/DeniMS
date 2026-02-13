import torch
import numpy as np
import networkx as nx
from rdkit import Chem

import itertools
from rdkit.Chem import MolToSmiles, MolFromSmiles

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdkit_functions import build_molecule, mol2smiles, check_valency

["H", "C", "N", "O", "F", "S", "Cl", "Br", "I"]  
allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'S': 4, 'Cl': 1, 'Br': 1, 'I': 1}
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
ATOM_VALENCY_2 = {2: 3, 3: 2, 6:3, 7:2}


from rdkit.Chem import rdDetermineBonds

def fix_colon(mol) -> str:
    """
    Fixes valence, charge, and bond issues in a molecule with ':' bond notation.
    """
    # Assign missing bond orders and valences
    rdDetermineBonds.DetermineBonds(mol)
    
    # Try sanitization
    Chem.SanitizeMol(mol)
    
    # Return the fixed SMILES
    return mol


def fix_valence_issues(mol, verbose=False):
    while True:
        flag, atomid_valence = check_valency(mol)
        if verbose:
            print("Valence check:", flag, atomid_valence)
        
        if flag:
            break  # Valid molecule
        
        assert len(atomid_valence) == 2
        idx = atomid_valence[0]
        v = atomid_valence[1]
        an = mol.GetAtomWithIdx(idx).GetAtomicNum()
        
        if verbose:
            print("Fixing valence for atom:", idx, "Atomic num:", an, "Valence:", v)
        
        if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
            mol.GetAtomWithIdx(idx).SetFormalCharge(1)
        else:
            # Additional valence correction strategies can be added here
            break  # If no fix is found, exit to avoid infinite loops



def build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder =  ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I'],
                                        charges=None, Hs=None, verbose=False):
    if verbose:
        print("\nBuilding new molecule")
    
    mol = Chem.RWMol()
    for idx, atom in enumerate(atom_types):
        a = Chem.Atom(atom_decoder[atom.item()])
        if charges is not None:
            a.SetFormalCharge(charges[idx])
        if Hs is not None:
            a.SetNumExplicitHs(Hs[idx])
        mol.AddAtom(a)
        if verbose:
            print("Atom added:", atom.item(), atom_decoder[atom.item()])
    
    edge_types = torch.triu(edge_types)
    all_bonds = torch.nonzero(edge_types)
    
    for bond in all_bonds:
        if bond[0].item() != bond[1].item():
            mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[edge_types[bond[0], bond[1]].item()])
            if verbose:
                print("Bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item())
    
    # If no charges are provided, attempt to iteratively fix valence issues
    if charges is None:
        fix_valence_issues(mol, verbose=verbose)
     
    return mol


def read_molecule_file(filename):
    samples = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if lines[i].startswith("N="):
            N = int(lines[i].split('=')[1].strip())
            i += 1
            
            if lines[i].startswith("X:"):
                i += 1
                atoms = torch.tensor(list(map(int, lines[i].split())), dtype=torch.int32)
                i += 1
                
            if lines[i].startswith("E:"):
                i += 1
                bonds = []
                while i < len(lines) and lines[i].strip():
                    bond_row = list(map(int, lines[i].split()))
                    bonds.append(bond_row)
                    i += 1
                bonds = torch.tensor(bonds, dtype=torch.int32)
                
            samples.append([atoms, bonds])
        i += 1
    
    return samples

def convert_to_nx_graph(atoms, bonds):
    G = nx.Graph()
    num_nodes = len(atoms)
    
    for node in range(num_nodes):
        G.add_node(node, atom_type=int(atoms[node]))
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if bonds[i, j] > 0:  # If there is a bond
                G.add_edge(i, j, bond_type=int(bonds[i, j]))
    
    return G

def smiles_to_graph(smiles):
    types = {'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F':4, 'Si':5, 'P':6, 'S':7, 'Cl':8, 'Br':9, 'I':10, 'H':11}
    bonds = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    num_atoms = mol.GetNumAtoms()
    atom_types = torch.tensor([types.get(mol.GetAtomWithIdx(i).GetSymbol(), -1) for i in range(num_atoms)], dtype=torch.int32)
    bond_matrix = torch.zeros((num_atoms, num_atoms), dtype=torch.int32)
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bonds.get(bond.GetBondType(), -1)
        bond_matrix[i, j] = bond_type
        bond_matrix[j, i] = bond_type
    
    return [atom_types, bond_matrix]


def read_smiles_file(filename):
    samples = []
    with open(filename, 'r') as f:
        for line in f:
            smiles = line.split()[0]  # Take only the first SMILES in each row
            try:
                atoms, bonds = smiles_to_graph(smiles)
                samples.append([atoms, bonds])
            except ValueError:
                print(f"Skipping invalid SMILES: {smiles}")
    return samples

def are_graphs_isomorphic(bonds1, bonds2):
    G1 = nx.Graph(bonds1)
    G2 = nx.Graph(bonds2)
    return nx.is_isomorphic(G1, G2)

def spectral_similarity(G1, G2):
    adj1 = nx.to_numpy_array(G1)
    adj2 = nx.to_numpy_array(G2)
    
    eigvals1 = np.sort(np.linalg.eigvals(adj1)).real.reshape(1, -1)
    eigvals2 = np.sort(np.linalg.eigvals(adj2)).real.reshape(1, -1)
    
    min_length = min(eigvals1.shape[1], eigvals2.shape[1])
    eigvals1 = eigvals1[:, :min_length]
    eigvals2 = eigvals2[:, :min_length]
    
    similarity = cosine_similarity(eigvals1, eigvals2)[0, 0]
    return similarity

from scipy.stats import wasserstein_distance

def degree_distribution_similarity(G1, G2):
    degrees1 = sorted([d for _, d in G1.degree()])
    degrees2 = sorted([d for _, d in G2.degree()])
    return wasserstein_distance(degrees1, degrees2)

def spectral_similarity(G1, G2, k=5):
    L1 = nx.laplacian_matrix(G1).toarray()
    L2 = nx.laplacian_matrix(G2).toarray()
    eigvals1 = np.sort(np.linalg.eigvalsh(L1))[:k]
    eigvals2 = np.sort(np.linalg.eigvalsh(L2))[:k]
    return 1 - np.linalg.norm(eigvals1 - eigvals2)


def fix_aromatic_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol:  # If already valid, return it as is
        return smiles
    
    # If invalid, attempt to fix aromatic atoms
    fixed_smiles = None
    for i in range(len(smiles)):
        if smiles[i].islower() and smiles[i].isalpha() and smiles[i] != 'c':  # Identifying aromatic atoms
            modified_smiles = smiles[:i] + f'[{smiles[i]}H]' + smiles[i+1:]
            mol = Chem.MolFromSmiles(modified_smiles)
            
            if mol != None:
                fixed_smiles = modified_smiles  # Return canonical form
                break
    
    return fixed_smiles if fixed_smiles else "Could not fix SMILES"







