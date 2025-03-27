import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

from .classes import Peptide


def get_index_shortest_distance(point_p: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Finds the index of the shortest distance between a point and a set of points.

    Parameters:
    - point_p: Find the shortest distance to this point
    - b: Set of points

    """

    point_p = point_p[0]
    distances_array = np.empty(len(b))

    for i in range(len(b)):
        p = b[i]
        distances_array[i] = np.sqrt(
            (point_p[1] - p[1]) ** 2 + (point_p[0] - p[0]) ** 2 + (point_p[2] - p[2]) ** 2
        )
    distances_array = distances_array[distances_array != 0]

    return (b[np.argmin(distances_array)], np.min(distances_array))


def get_closest_lipid_z(
    res_pos: np.ndarray, lipids: np.ndarray, p_up: float, p_low: float
) -> Optional[float]:
    """
    Finds the closest lipid in the Z direction to the given residue position.

    Parameters:
    - res_pos: Residue position
    - lipids: Array of lipid indexes
    - p_up: Mean z-position of P atoms in the upper bilayer
    - p_low: Mean z-position of P atoms in the lower bilayer

    """

    all_up = lipids
    kdtree = KDTree(lipids)

    for radius in [
        (float(p_up) - float(p_low)) / 2,
        (float(p_up) - float(p_low)),
        20,
        40,
        80,
        120,
    ]:

        x = kdtree.query_ball_point(res_pos, radius, return_sorted=True)

        if len(x[0]) == 0:
            continue

        else:
            closest_lipid_index, _ = get_index_shortest_distance(res_pos, all_up[x[0]])
            return closest_lipid_index[2]


def get_peptide_insertion(
    peptide_name: str,
    aminoacid_count: int,
    peptide_number: int,
    membrane_lipids: int,
    step_size: int,
    xtc_file_path: str,
    tpr_file_path: str,
) -> pd.DataFrame:
    """
    Analyzes peptide insertion into the membrane and saves results as a CSV file.

    Parameters:
    - peptide_name: Name of the peptide
    - aminoacid_count: Number of amino acids per peptide
    - peptide_number: Number of peptides
    - membrane_lipids: Number of lipids in the membrane
    - step_size: Step size for trajectory analysis
    - xtc_file_path: Path to the XTC trajectory file
    - tpr_file_path: Path to the TPR topology file
    """

    peptide = Peptide(
        pep_name=peptide_name,
        xtc_file_path=xtc_file_path,
        tpr_file_path=tpr_file_path,
        peptide_number=peptide_number,
        amino_acid_count=aminoacid_count,
        step_size=step_size,
    )
    u = peptide.u
    z_pos_list = []
    frames, n_frames = peptide.load_traj()
    for _, _ in tqdm(enumerate(u.trajectory[frames]), total=n_frames):
        u.select_atoms("resname POPG and name P").positions
        p_memb = u.select_atoms("resname POPG and name P").positions
        p_up = np.mean(p_memb[: int(membrane_lipids / 2)])
        p_low = np.mean(p_memb[int(membrane_lipids / 2) :])

        for pep, res_id_range in peptide.pep_dict.items():
            res_count = 1
            for res_id in range(res_id_range[0], res_id_range[1] + 1):
                # Calculate the z pos of each residue one at a time
                res_z_pos = np.mean(
                    u.select_atoms(f"name CA and resid {res_id}").positions[:, [2]].astype(float)
                )
                res_pos = u.select_atoms(f"name CA and resid {res_id}").positions

                # Determine the actual insertion of each residue
                pbc_crossed = (
                    1 if res_z_pos > 0 and res_z_pos < u.dimensions[2] / 2 else 0
                )  # This assumes that peptides don't insert more than half than the membrane height

                if pbc_crossed == 0:
                    lipid_z = get_closest_lipid_z(
                        res_pos, p_memb[: int(membrane_lipids / 2)], p_up, p_low
                    )

                    if lipid_z:
                        pep_insertion = res_z_pos - lipid_z

                        z_pos_list.append(
                            (
                                u.trajectory.time,
                                pep,
                                res_id,
                                res_count,
                                pep_insertion,
                                p_up,
                                p_low,
                                lipid_z,
                                res_z_pos,
                                pbc_crossed,
                            )
                        )
                        res_count += 1

    df = pd.DataFrame(
        z_pos_list,
        columns=[
            "Time (ns)",
            "Peptide_num",
            "Resid",
            "Residue",
            "CA Z position",
            "P_up",
            "P low",
            "lipid_z",
            "res_z_pos",
            "Crossed",
        ],
    )

    df["Residue"] = df["Residue"].astype("str")
    df["Time (ns)"] = df["Time (ns)"].astype(float) / 1000
    df["Time (ns)"] = df["Time (ns)"].astype(int)
    df.to_csv(f"pepNmemb/data/insertion_{peptide.pep_name}_insertion_curv.csv")

    return df


def main():
    parser = argparse.ArgumentParser(description="Run insertion into the membranr analysis")

    parser.add_argument("-xtc", "--xtc", type=str, help="Input xtc file path")
    parser.add_argument("-tpr", "--tpr", type=str, help="Input xtpr file path")
    parser.add_argument("-pname", "--pep_name", type=str, help="Peptide name used for saving data")
    parser.add_argument("-pnum", "--pep_num", type=int, help="Number of peptides")
    parser.add_argument("-res", "--res_num", type=int, help="Number of residues in each peptide")
    parser.add_argument("-mlipids", "--memb_lipids", type=int, help="Number of membrane lipids")
    parser.add_argument("-ss", "--step_size", type=int, help="Step size")

    args = parser.parse_args()

    peptide_name = args.pep_name
    xtc_file_path = args.xtc
    tpr_file_path = args.tpr
    peptide_number = args.pep_num
    aminoacid_count = args.res_num
    membrane_lipids = args.memb_lipids
    step_size = args.step_size

    print(f"Starting analysis for peptide {peptide_name} found at {xtc_file_path}, {tpr_file_path}")
    get_peptide_insertion(
        peptide_name,
        aminoacid_count,
        peptide_number,
        membrane_lipids,
        step_size,
        xtc_file_path,
        tpr_file_path,
    )


if __name__ == "__main__":
    main()
