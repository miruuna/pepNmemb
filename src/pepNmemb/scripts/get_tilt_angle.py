""" Get angle of insertion of peptides relative to the membrane plane"""

from __future__ import annotations

import argparse
import math as m
import sys
from typing import List, Tuple

import MDAnalysis as mda
import numpy as np
import pandas as pd
from skspatial.objects import Line, Points
from tqdm import tqdm

from .classes import Peptide
from .utils import NoneValueError, process_file, validate_dict_values


def find_consecutive_sublists(integers: List[int]) -> List[List[int]]:
    """
    Find all consecutive sublists within a list.

    Args:
        integers (List[int]): Input list of integers to find consecutive sublists.

    Returns:
        List[List[int]]: List of consecutive sublists with at least 5 elements.
    """
    if not integers:
        return []

    # Sort the list
    integers.sort()

    # Initialize variables
    result: List[List[int]] = []
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
    """
    Calculate the vector between two coordinates.

    Args:
        coord1 (Tuple[float, float, float]): First coordinate.
        coord2 (Tuple[float, float, float]): Second coordinate.

    Returns:
        Tuple[float, float, float]: Vector from coord2 to coord1.
    """
    return (coord1[0] - coord2[0], coord1[1] - coord2[1], coord1[2] - coord2[2])


def vector_length(v: Tuple[float, float, float]) -> float:
    """
    Calculate the length of a vector.

    Args:
        v (Tuple[float, float, float]): Input vector.

    Returns:
        float: Length of the vector.
    """
    return m.sqrt(sum((c**2) for c in v))


def dot(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
    """
    Calculate the dot product of two vectors.

    Args:
        v1 (Tuple[float, float, float]): First vector.
        v2 (Tuple[float, float, float]): Second vector.

    Returns:
        float: Dot product of the two vectors.
    """
    return sum([c1 * c2 for c1, c2 in zip(v1, v2)])


def angle_between_vectors(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
    """
    Calculate the angle between two vectors in degrees.

    Args:
        v1 (Tuple[float, float, float]): First vector.
        v2 (Tuple[float, float, float]): Second vector.

    Returns:
        float: Angle between vectors in degrees.
    """
    return m.acos(dot(v1, v2) / (vector_length(v1) * vector_length(v2))) * (180 / m.pi)


def get_coords_spanning_memb(
    u: mda.Universe, residues: Tuple[int, int], p_up: float
) -> Tuple[List[List[List[float]]], List[int]]:
    """
    Compute coordinates of peptide residues spanning the membrane.

    Selects residues within -6 to +10 of the average Z-position
    of P atoms in the upper membrane layer.

    Args:
        u (mda.Universe): MDAnalysis Universe object.
        residues (Tuple[int, int]): Range of residue IDs.
        p_up (float): Average Z-position of upper membrane layer.

    Returns:
        Tuple[List[List[List[float]]], List[int]]:
        - List of coordinate lists for selected residues
        - List of selected residue IDs
    """
    pep_coords: List[List[List[float]]] = []
    selected_res: List[int] = []

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


def relative_tilt(obj: Peptide, membrane_lipids: int, output_folder: str) -> pd.DataFrame:
    """
    Compute relative tilt angle for the upper bilayer.

    Args:
        obj (Peptide): Peptide object containing trajectory and properties.
        membrane_lipids (int): Number of membrane lipids.

    Returns:
        pd.DataFrame: DataFrame with tilt angle information.
    """
    u = obj.u
    peptide_name = obj.pep_name
    pep_num_dict = obj.pep_dict
    frames, n_frames = obj.load_traj()
    angle_list = []

    for _, _ in tqdm(enumerate(u.trajectory[frames]), total=n_frames):
        timecount1 = u.trajectory.time
        if timecount1 != 0:
            memb_vector = (0, 0, 1)

            p_memb = u.select_atoms("resname POPG and name P").positions[:, [2]].astype(float)
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

                tilt_angle = angle_between_vectors(memb_vector, pep_vector)
                angle_list.append((timecount1, pep, abs(90 - tilt_angle), len(pep_coords)))

    df = pd.DataFrame(angle_list, columns=["Time (ns)", "Peptide_num", "Angle", "len coords"])
    df["Angle"] = df["Angle"].astype(float)
    df["Time (ns)"] = (df["Time (ns)"].astype(float) / 1000).astype(int)

    df.to_csv(f"{output_folder}/tilt_angle_{peptide_name}.csv")
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

    peptide = Peptide(
        args_dict["peptide_name"],
        args_dict["xtc_file_path"],
        args_dict["tpr_file_path"],
        args_dict["peptide_number"],
        args_dict["aminoacid_count"],
        args_dict["step_size"],
    )

    print(f"Starting analysis for peptide {args.pep_name} " f"found at {args.xtc}, {args.tpr}")

    relative_tilt(peptide, args_dict["membrane_lipids"], args_dict["output_folder"])

    print(
        f"Starting analysis for peptide {args_dict['peptide_name']} \
            found at {args_dict['xtc_file_path']}, {args_dict['tpr_file_path']}"
    )

    print(f"File saved at {args_dict['output_folder']}")


if __name__ == "__main__":
    main()
