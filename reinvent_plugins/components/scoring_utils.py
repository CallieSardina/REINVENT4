from rdkit import Chem
#import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem.QED import qed
from rdkit.Chem.Crippen import MolLogP
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import glob
import csv
import subprocess
import shutil
import math

from meeko import MoleculePreparation
from meeko import rdkitutils
from meeko import PDBQTWriterLegacy
import uuid
import subprocess
import re
import numpy as np
import random
from SAScore import sascorer


def QED(smiles: list):
    """Calculate QED score for a list of SMILES strings."""
    qed_scores = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            qed_scores.append(qed(mol))
        else:
            qed_scores.append(0.0)
   
    return qed_scores

def LogP(smiles: list):
    """Calculate LogP score for a list of SMILES strings."""
    logp_scores = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            logp_val = MolLogP(mol)
        else:
            logp_val = 0.0
        logp_scores.append(logp_val)
    
    return logp_scores

def SA_score(smiles: list):
    """Calculate SA (Synthetic Accessibility) score for a list of SMILES strings."""
    sa_scores = []
    
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            sa_scores.append(sascorer.calculateScore(mol))
        else:
            sa_scores.append(10.0)  # Assign highest difficulty score to invalid molecules

    return sa_scores

def smiles_to_fp(smiles, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius)

def shannon_entropy_morgan(smiles_list, radius=2, n_bits=2048):
    fps = [smiles_to_fp(s, radius) for s in smiles_list]
    fps = [fp for fp in fps if fp is not None]

    if len(fps) < 2:
        return 0.0

    bin_matrix = np.array([fp.ToBitString() for fp in fps])
    bin_matrix = np.array([[int(c) for c in row] for row in bin_matrix])

    feature_counts = bin_matrix.sum(axis=0)
    prob_dist = feature_counts / feature_counts.sum()

    return entropy(prob_dist)  # uses natural log by default

def scaffold_uniqueness(smiles_list):
    scaffolds = []
    for s in smiles_list:
        try:
            if not s:
                continue
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                print(f"Could not parse SMILES: {s}")
                continue
            scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s)
            scaffolds.append(scaffold)
        except Exception as e:
            print(f"Error processing SMILES {s}: {e}")
    return len(set(scaffolds)) / len(smiles_list) if smiles_list else 0.0


def Docking(smiles: list):
    rewards =  []
    meeko_prep = MoleculePreparation()
    cnt = 1
   
    
    for mol in smiles:
        outfile = os.path.join("pdbqt_outputs_thompson_sigma", f"drug_{cnt}.pdbqt") 
        try:
            # if 'CG0' in mol:
            #     print("SMILES string has 'CGO' which throws Auto parse error later.")
            #     continue  # skip this file

            # Create ligand object from SMILES string
            lig = Chem.MolFromSmiles(mol)
            if lig is None:
                print("Error 1")
                raise ValueError(f"Invalid SMILES string: {mol}")
            # Protonate and embed ligand
            protonated_lig = Chem.AddHs(lig)
            embedding_status = AllChem.EmbedMolecule(protonated_lig)
            if embedding_status != 0:
                print("Error 2")
                raise ValueError("Embedding failed")
            # Conformation energy minimization
            # energy_min_status = AllChem.UFFOptimizeMolecule(protonated_lig)
            # Prepare ligand for PDBQT output
            molsetups = meeko_prep.prepare(protonated_lig)
            pdbqt_string, success, error_msg = PDBQTWriterLegacy.write_string(molsetups[0])
            # Write output PDBQT
            if success:
                print(pdbqt_string, end='', file=open(outfile, 'w'))
            else:
                print("Error 3")
                raise ValueError("Preparation failed")
        except ValueError as e:
            pass
        cnt += 1
   
    docking_result_path = os.path.join("pdbqt_outputs_thompson_sigma", "docking_results.dlg")
    error_path = os.path.join("pdbqt_outputs_thompson_sigma", "docking_error.dlg")

    random_seed = random.randint(1, 999999)
   
    os.system(f"~/bin/Vina-GPU-2.1/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1 --config ./nac_hex_input/nac_hex_config.txt --seed {random_seed} > {docking_result_path} 2> {error_path}")
    print("here, QuickVina")
    # Calculate rewards
    affinities = [0.0 for i in range(len(smiles))]
    cur_i = None
    with open(docking_result_path, "r") as f:
        for line in f:
        
            if line.startswith("Refining"):
                cur_i = int(re.findall(r'\d+', line)[-1]) -1
               
            if line != "\n" and line[3] == "1":
                #affinities.append(-1 * float(line.split()[1]))
                affinities[cur_i] = -1 * float(line.split()[1])
                # affinities.append(float(line.split()[1]))
    #rewards = affinities
    print('Docking scores:', affinities)
    ####################
    non_zero = []
    for r in affinities:
        if r != 0:
            non_zero.append(r)
    print("LEN:", len(non_zero),sum(non_zero)/len(non_zero))
    
    files = glob.glob("pdbqt_outputs_thompson_sigma" + '/*')
    # Delete each file
    for file in files:
        if os.path.isdir(file):  # If it's a directory, delete it with shutil
            shutil.rmtree(file)
        else:
            os.remove(file)
    
    return affinities, random_seed