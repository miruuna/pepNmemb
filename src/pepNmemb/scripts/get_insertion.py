from __future__ import annotations

import argparse
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from classes import Peptide
from tqdm import tqdm

from .scipy.spatial import KDTree
from .utils import NoneValueError, process_file, validate_dict_values


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

    for radius in [(float(p_up) - float(p_low)) / 2, (float(p_up) - float(p_low)), 20, 40, 80, 120]:

        x = kdtree.query_ball_point(res_pos, radius, return_sorted=True)

        if len(x[0]) == 0:
            continue

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
    output_folder: str,
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
    df.to_csv(f"{output_folder}/insertion_{peptide.pep_name}_insertion_curv.csv")

    return df


def main():
    args_dict = {}

    for k in [
        "peptide_name",
        "aminoacid_count",
        "peptide_number",
        "membrane_lipids",
        "step_size",
        "xtc_file_path",
        "tpr_file_path",
    ]:
        args_dict[k] = None

    parser = argparse.ArgumentParser(
        description="Process a file with key-value pairs or command line arguments."
    )

    # Add file argument (optional)
    parser.add_argument("-f", "--file", help="Path to input file with key=value pairs")

    parser.add_argument("-xtc", "--xtc", type=str, help="Input xtc file path")
    parser.add_argument("-tpr", "--tpr", type=str, help="Input xtpr file path")
    parser.add_argument("-pname", "--pep_name", type=str, help="Peptide name used for saving data")
    parser.add_argument("-pnum", "--pep_num", type=int, help="Number of peptides")
    parser.add_argument("-res", "--res_num", type=int, help="Number of residues in each peptide")
    parser.add_argument("-mlipids", "--memb_lipids", type=int, help="Number of membrane lipids")
    parser.add_argument("-ss", "--step_size", type=int, help="Step size")
    parser.add_argument("-o", "--output_folder", type=int, help="Output folder")

    args = parser.parse_args()

    if args.file:
        file_dict = process_file(args.file)
        args_dict.update(file_dict)
        args_dict["step_size"] = int(args_dict["step_size"])
        args_dict["aminoacid_count"] = int(args_dict["aminoacid_count"])
        args_dict["membrane_lipids"] = int(args_dict["membrane_lipids"])
        args_dict["peptide_number"] = int(args_dict["peptide_number"])
    else:
        args_dict["peptide_name"] = args.pep_name
        args_dict["xtc_file_path"] = args.xtc
        args_dict["tpr_file_path"] = args.tpr
        args_dict["peptide_number"] = args.pep_num
        args_dict["aminoacid_count"] = args.res_num
        args_dict["membrane_lipids"] = args.memb_lipids
        args_dict["step_size"] = args.step_size
        args_dict["output_folder"] = args.output_folder
    print(args_dict)
    try:
        validate_dict_values(args_dict)
        print("All values validated successfully.")
    except NoneValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(
        f"Starting analysis for peptide {args_dict['peptide_name']} \
            found at {args_dict['xtc_file_path']}, {args_dict['tpr_file_path']}"
    )
    get_peptide_insertion(
        args_dict["peptide_name"],
        args_dict["aminoacid_count"],
        args_dict["peptide_number"],
        args_dict["membrane_lipids"],
        args_dict["step_size"],
        args_dict["xtc_file_path"],
        args_dict["tpr_file_path"],
        args_dict["output_folder"],
    )


if __name__ == "__main__":
    main()
