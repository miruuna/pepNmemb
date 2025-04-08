from __future__ import annotations

import argparse
import warnings
from typing import List, Tuple

import pandas as pd
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from tqdm import tqdm

from pepNmemb.scripts.classes import Peptide

# Suppress warnings
warnings.filterwarnings("ignore")


def hbonds_per_res(obj: Peptide, step_size: int = 500) -> pd.DataFrame:
    """
    Analyze hydrogen bonds for peptide residues in membrane.

    Args:
        obj (Peptide): Peptide object containing trajectory and properties.
        step_size (int, optional): Trajectory analysis step size. Defaults to 500.

    Returns:
        pd.DataFrame: DataFrame with hydrogen bond information.
    """
    hbonds: List[Tuple[str, int, float, float]] = []

    u = obj.u
    peptide_name = obj.pep_name
    pep_num_dict = obj.pep_dict

    for pep, residues in tqdm(pep_num_dict.items()):
        res_count = 1

        for res_id in tqdm(range(residues[0], residues[1] + 1)):
            h_bonds = HBA(
                u,
                update_selections=True,
                between=["resname POPE or resname POPG", f"resid {res_id}"],
            )

            # Guess hydrogen and acceptor selections
            protein_hydrogens_sel = h_bonds.guess_hydrogens(f"resid {res_id}")
            protein_acceptors_sel = h_bonds.guess_acceptors(f"resid {res_id}")
            membrane_hydrogens_sel = h_bonds.guess_hydrogens("resname POPE or resname POPG")
            membrane_acceptors_sel = h_bonds.guess_acceptors("resname POPE or resname POPG")

            # Update hydrogen and acceptor selections
            h_bonds.hydrogens_sel = f"({protein_hydrogens_sel}) or ({membrane_hydrogens_sel})"
            h_bonds.acceptors_sel = f"({protein_acceptors_sel}) or ({membrane_acceptors_sel})"

            # Run hydrogen bond analysis
            h_bonds.run(step=step_size)

            # Collect hydrogen bond count and time data
            hydrogen_count = h_bonds.count_by_time()
            time_count = h_bonds.times
            for i in range(len(time_count)):
                hbonds.append((pep, res_count, hydrogen_count[i], time_count[i]))
            res_count += 1

    # Create DataFrame and process time data
    df = pd.DataFrame(hbonds)
    df.columns = ["Peptide", "Resid", "Hbonds", "Time"]
    df["Time"] = (df["Time"].astype(float) / 1000).astype(int)
    df = df.rename(columns={"Time": "Time (ns)"})

    # Save to CSV
    df.to_csv(f"hbonds_{peptide_name}.csv", index=False)
    return df


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Run hydrogen bond analysis for peptide in membrane"
    )

    parser.add_argument("-xtc", "--xtc", type=str, help="Input xtc file path")
    parser.add_argument("-tpr", "--tpr", type=str, help="Input tpr file path")
    parser.add_argument("-pname", "--pep_name", type=str, help="Peptide name used for saving data")
    parser.add_argument("-pnum", "--pep_num", type=int, help="Number of peptides")
    parser.add_argument("-res", "--res_num", type=int, help="Number of residues in each peptide")
    parser.add_argument("-ss", "--step_size", type=int, default=500, help="Step size")

    args = parser.parse_args()

    peptide = Peptide(
        args.pep_name,
        args.xtc,
        args.tpr,
        peptide_number=args.pep_num,
        amino_acid_count=args.res_num,
        step_size=args.step_size,
    )

    print(
        f"Starting hydrogen bond analysis for peptide {args.pep_name} "
        f"found at {args.xtc}, {args.tpr}"
    )

    hbonds_per_res(peptide, step_size=args.step_size)


if __name__ == "__main__":
    main()
