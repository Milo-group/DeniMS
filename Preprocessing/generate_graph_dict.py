import argparse
import os

import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from tqdm.auto import tqdm


atom_types = {'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F':4, 'Si':5, 'P':6, 'S':7, 'Cl':8, 'Br':9, 'I':10, 'H':11}
bond_types = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3} 


def mol_to_graph(mol):
    """
    Convert an RDKit molecule to a PyTorch Geometric Data object.
    """
    N = mol.GetNumAtoms()

    type_idx = []
    for atom in mol.GetAtoms():
        type_idx.append(atom_types[atom.GetSymbol()])

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond_types[bond.GetBondType()] + 1]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bond_types)+1).to(torch.float)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    x = F.one_hot(torch.tensor(type_idx), num_classes=len(atom_types)).float()
    type_idx = torch.tensor(type_idx).long()
    to_keep = type_idx <= 11
    edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                                        num_nodes=len(to_keep))
    x = x[to_keep]
    x = x[:, :-1]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)



def generate_graph_dict(input_parquet, output_dict, input_smiles=None):
    """
    Generate graph dictionary from filtered parquet and SMILES file.
    
    Args:
        input_parquet: Path to input parquet file
        output_dict: Path to output dictionary file
        input_smiles: Optional path to SMILES file. If None, extracts unique SMILES from parquet.
    """
    print("\n" + "=" * 30)
    print("Graph dictionary generation")
    print("=" * 30)

    print(f"\n[1/3] Loading filtered parquet: {input_parquet}")
    full_data = pq.read_table(input_parquet, use_threads=True)
    full_data = full_data.to_pandas()
    print(f"    Parquet shape: {full_data.shape}")
    
    if input_smiles is None:
        print(f"\n[2/3] Extracting unique SMILES from parquet...")
        if 'smiles' not in full_data.columns:
            raise ValueError("'smiles' column not found in parquet file")
        smiles = full_data['smiles'].unique().tolist()
        print(f"    Unique SMILES count: {len(smiles)}")
    else:
        print(f"\n[2/3] Loading SMILES file: {input_smiles}")
        with open(input_smiles, 'r') as file:
            smiles = file.read().splitlines()
        print(f"    SMILES count: {len(smiles)}")
    print(f"\n[3/3] Building graph dictionary...")
    data_dict = {}
    total = 0
    num_invalid = 0
    
    for i, smi in tqdm(list(enumerate(smiles)), total=len(smiles), desc="  Processing SMILES"):
            
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            num_invalid += 1
            continue
            
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fingerprint = torch.tensor(np.array(fingerprint, dtype=np.float32)).unsqueeze(0)
        graph = mol_to_graph(mol)

        indices = np.where(
            (full_data['smiles'].values == smi)
        )[0]

        total += len(indices)
        
        if len(indices) == 0:
            num_invalid += 1
            continue
        else:
            data_dict[smi] = [indices, graph, fingerprint]
    
    print(f"\nSaving graph dictionary to: {output_dict}")
    torch.save(data_dict, output_dict)
    print("\n" + "=" * 30)
    print("Graph dictionary summary")
    print("=" * 30)
    print(f"  Total indices: {total}")
    print(f"  Invalid SMILES (no matches or parsing errors): {num_invalid}")
    print(f"  Valid entries in dictionary: {len(data_dict)}")
    print("=" * 30 + "\n")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-input_parquet",required=True,type=str,help="Path to input parquet file")
    parser.add_argument("-output_dict", required=True, type=str, help="Path to output dictionary file (.pt)")
    parser.add_argument("-input_smiles", default=None, type=str, help="Optional path to SMILES file. If not provided, extracts unique SMILES from parquet.")

    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.input_parquet):
        print(f"Error: Input parquet file not found at {args.input_parquet}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_dict)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    generate_graph_dict(
        input_parquet=args.input_parquet,
        output_dict=args.output_dict,
        input_smiles=args.input_smiles
    )


if __name__ == "__main__":
    main()
