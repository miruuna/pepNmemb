import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA

from utils.classes import Peptide

import warnings
warnings.filterwarnings('ignore') 


step_size = 500 

peptide = Peptide(
    "WF2_8_1", 
    "/Volumes/miru_backup/pleu_conc/pleu8/pleu8/md_0_1.xtc",
    "/Volumes/miru_backup/pleu_conc/pleu8/pleu8/md_0_1.tpr",
    peptide_number=8, amino_acid_count=25, step_size=step_size)



def hbonds_per_res(obj):

    hbonds = []

    u =  obj.u
    peptide_name = obj.pep_name
    pep_num_dict = obj.pep_dict

    for pep, residues in tqdm(pep_num_dict.items()):
        res_count = 1

        for res_id in tqdm(range(residues[0], residues[1]+1)):

            h_bonds = HBA(u, update_selections=True, between=["resname POPE or resname POPG", f"resid {res_id}"])
            protein_hydrogens_sel = h_bonds.guess_hydrogens(f"resid {res_id}")
            protein_acceptors_sel = h_bonds.guess_acceptors(f"resid {res_id}")

            membrane_hydrogens_sel = h_bonds.guess_hydrogens("resname POPE or resname POPG")
            membrane_acceptors_sel = h_bonds.guess_acceptors("resname POPE or resname POPG")

            h_bonds.hydrogens_sel = f"({protein_hydrogens_sel}) or ({membrane_hydrogens_sel})"
            h_bonds.acceptors_sel = f"({protein_acceptors_sel}) or ({membrane_acceptors_sel})"

            h_bonds.run(step=step_size)
            
            hydrogen_count = h_bonds.count_by_time()
            time_count = h_bonds.times
            for i in range(len(time_count)):
                hbonds.append((pep, res_count, hydrogen_count[i], time_count[i]))
            res_count += 1

    df = pd.DataFrame(hbonds) 
    df.columns = ["Peptide", "Resid", "Hbonds", "Time"]
    df['Time'] = df['Time'].astype(float)/1000
    df['Time'] = df['Time'].astype(int)
    df = df.rename(columns={"Time": "Time (ns)"})
    df.to_csv(f"hbonds_{peptide_name}.csv")
    return df


if __name__ == "__main__":
    hbonds_per_res(peptide)