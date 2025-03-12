from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from MDAnalysis.analysis import leaflet
from skspatial.objects import Line
from skspatial.objects import Points
import math as m

import argparse

from classes import Peptide


def find_consecutive_sublists(integers: List[int]) -> List[List[int]]:
    """
    Find all consecutive sublists within a list

    """
    if not integers:
        return []

    # Sort the list
    integers.sort()

    # Initialize variables
    result = []
    current_sublist = [integers[0]]

    # Iterate through the sorted list
    for i in range(1, len(integers)):
        # Check if the current integer is consecutive to the previous one
        if integers[i] == integers[i - 1] + 1:
            current_sublist.append(integers[i])
        else:
            # If the current sublist has at least 5 consecutive integers, add it to the result
            if len(current_sublist) >= 5:
                result.append(current_sublist)
            # Start a new sublist
            current_sublist = [integers[i]]

    # Check the last sublist
    if len(current_sublist) >= 5:
        result.append(current_sublist)

    return result


def get_vector(
    coord1: Tuple[float, float, float], coord2: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return (coord1[0] - coord2[0], coord1[0] - coord2[1], coord1[0] - coord2[2])


def vector_length(v: Tuple[float, float, float]) -> float:
    return m.sqrt(sum((c ** 2) for c in v))


def dot(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
    return sum([c1 * c2 for c1, c2 in zip(v1, v2)])


def angle_between_vectors(
    v1: Tuple[float, float, float], v2: Tuple[float, float, float]
) -> float:
    return m.acos(dot(v1, v2) / (vector_length(v1) * vector_length(v2))) * (180 / m.pi)


def get_coords_spanning_memb(
    u, residues: Tuple[int, int], p_up: float
) -> Tuple[List[List[List[float]]], List[int]]:
    pep_coords: List[List[List[float]]] = []
    selected_res: List[int] = []
    """
    Compute the coordinates of the peptide residues that span the membrane. It only takes into accout residues found within a range of -6 and + 10 above the averaveg Z-position of the P atoms in the upper layer of the membrane.

    """
    pep_coords = []
    selected_res = []
    for res_id in range(residues[0], residues[1] + 1):
        peptide_atoms = u.select_atoms(f"name CA and resid {res_id}").positions

        z_pep = peptide_atoms[:, [2]].astype(float)
        # inside the radius
        if z_pep < p_up + 10 and z_pep > p_up - 6:
            selected_res.append(res_id + 1)

    final_residues = find_consecutive_sublists(selected_res)
    pep_coords = [
        [
            u.select_atoms(f"name CA and resid {res_id}")
            .positions.reshape(-1)
            .astype(float)
            .tolist()
            for res_id in sub_list
            if len(
                u.select_atoms(f"name CA and resid {res_id}")
                .positions.reshape(-1)
                .astype(float)
                .tolist()
            )
            > 0
        ]
        for sub_list in final_residues
    ]

    return pep_coords, selected_res


def relative_tilt(obj, membrane_lipids: int) -> pd.DataFrame:
    """
    Compute relative tilt angle only for the upper bilayer
    """
    u = obj.u
    peptide_name = obj.pep_name
    pep_num_dict = obj.pep_dict
    frames, n_frames = obj.load_traj()
    angle_list = []

    for _, ts in tqdm(enumerate(u.trajectory[frames]), total=n_frames):
        timecount1 = u.trajectory.time
        if timecount1 != 0:
            L = leaflet.LeafletFinder(u, "name P")
            memb_vector = (0, 0, 1)

            p_memb = (
                u.select_atoms(f"resname POPG and name P")
                .positions[:, [2]]
                .astype(float)
            )
            p_up = np.mean(p_memb[: int(membrane_lipids / 2)])

            for pep, residues in pep_num_dict.items():

                pep_coords, _ = get_coords_spanning_memb(u, residues, p_up)
                if len(pep_coords) == 0 or len(pep_coords) > 1:
                    angle_list.append((timecount1, pep, None, len(pep_coords)))
                    continue

                points = Points(pep_coords[0])

                line_fit = Line.best_fit(points)
                pep_vector = line_fit.vector
                pep_vector = (pep_vector[0], pep_vector[1], pep_vector[2])
                pep_points = line_fit.point
                pep_points = (pep_points[0], pep_points[1], pep_points[2])

                tilt_angle = angle_between_vectors(memb_vector, pep_vector)
                angle_list.append(
                    (timecount1, pep, abs(90 - tilt_angle), len(pep_coords))
                )

    df = pd.DataFrame(
        angle_list, columns=["Time (ns)", "Peptide_num", "Angle", "len coords"]
    )
    df["Angle"] = df["Angle"].astype(float)
    df["Time (ns)"] = df["Time (ns)"].astype(float) / 1000
    df["Time (ns)"] = df["Time (ns)"].astype(int)

    df.to_csv(f"pepNmemb/data/tilt_angle_{peptide_name}.csv")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run insertion into the membrane analysis"
    )

    parser.add_argument("-xtc", "--xtc", type=str, help="Input xtc file path")
    parser.add_argument("-tpr", "--tpr", type=str, help="Input xtpr file path")
    parser.add_argument(
        "-pname", "--pep_name", type=str, help="Peptide name used for saving data"
    )
    parser.add_argument("-pnum", "--pep_num", type=int, help="Number of peptides")
    parser.add_argument(
        "-res", "--res_num", type=int, help="Number of residues in each peptide"
    )
    parser.add_argument(
        "-mlipids", "--memb_lipids", type=int, help="Number of membrane lipids"
    )
    parser.add_argument("-ss", "--step_size", type=int, help="Step size")

    args = parser.parse_args()

    peptide_name = args.pep_name
    xtc_file_path = args.xtc
    tpr_file_path = args.tpr
    peptide_number = args.pep_num
    aminoacid_count = args.res_num
    membrane_lipids = args.memb_lipids
    step_size = args.step_size

    print(
        f"Starting analysis for peptide {peptide_name} found at {xtc_file_path}, {tpr_file_path}"
    )

    peptide = Peptide(
        peptide_name,
        xtc_file_path,
        tpr_file_path,
        peptide_number=peptide_number,
        amino_acid_count=aminoacid_count,
        step_size=step_size,
    )

    relative_tilt(peptide, membrane_lipids)


if __name__ == "__main__":
    main()
