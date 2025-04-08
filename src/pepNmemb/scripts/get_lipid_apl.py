"""
Compute Lipid APL using Voronoi Tesselation

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import freud
import MDAnalysis as mda
import MDAnalysis.core.groups
import MDAnalysis.core.universe
import numpy as np
import pandas as pd


def remove_overlapping(positions: np.ndarray) -> np.ndarray:
    """
    Adjust xy positions if any pair of xy coordinates are identical.

    Given an Nx3 array of atomic positions, make minor adjustments to xy
    positions if any pair of xy coordinates are identical. If atoms are
    overlapping in xy, Freud will complain when attempting to perform
    the Voronoi tessellation.

    Args:
        positions (numpy.ndarray): Array of atomic positions with shape (N, 3).

    Returns:
        numpy.ndarray: Adjusted positions to remove xy overlaps.
    """
    # Check whether any atoms are overlapping in the xy-plane
    unique, indices, counts = np.unique(positions, return_index=True, return_counts=True, axis=0)

    # If so, add a small distance between the two atoms (1e-3 A)
    # in the x-dimension
    if max(counts > 1):
        for duplicate_index in indices[counts > 1]:
            positions[duplicate_index, 0] += 0.001

    return positions


def get_all_areas(positions: np.ndarray, box: Union[np.ndarray, Tuple[float, float]]) -> np.ndarray:
    """
    Perform Voronoi tessellation and return area of each Voronoi cell.

    Args:
        positions (numpy.ndarray): XY coordinates of atomic positions.
        box (numpy.ndarray or tuple): Dimensions of the system (x, y).

    Returns:
        numpy.ndarray: Areas of Voronoi cells.
    """
    voro = freud.locality.Voronoi()
    areas = voro.compute(
        system={"Lx": box[0], "Ly": box[1], "dimensions": 2}, positions=positions
    ).volumes

    return areas


def get_area_per_lipid(
    unique_resnames: List[str],
    atom_group: mda.core.groups.AtomGroup,
    all_areas: np.ndarray,
    num_seeds: Dict[str, int],
    area_array: np.ndarray,
) -> np.ndarray:
    """
    Calculate area per lipid given Voronoi cell areas.

    Args:
        unique_resnames (List[str]): Lipid species in the membrane.
        atom_group (MDAnalysis.core.groups.AtomGroup): Atoms for Voronoi tessellation.
        all_areas (numpy.ndarray): Areas of each cell in tessellation.
        num_seeds (Dict[str, int]): Number of Voronoi seeds per lipid species.
        area_array (numpy.ndarray): Array to store area per lipid.

    Returns:
        numpy.ndarray: Modified area_array with area per lipid.
    """
    for res in unique_resnames:
        lipid_indices = np.where(atom_group.resnames == res)
        lipid_apl = all_areas[lipid_indices]

        # Sum area contribution of each cell for a given lipid
        lipid_apl = np.sum(
            lipid_apl.reshape(atom_group[lipid_indices].residues.n_residues, num_seeds[res]), axis=1
        )

        # Store area per lipid for current lipid species
        lipid_resindices = atom_group.select_atoms(f"resname {res}").residues.resindices
        area_array[lipid_resindices] = lipid_apl

    return area_array


def calculate_apl(path: str) -> pd.DataFrame:
    """
    Calculate area per lipid for a given molecular dynamics trajectory.

    Args:
        path (str): Path to molecular dynamics simulation files.

    Returns:
        pandas.DataFrame: DataFrame with area per lipid for each frame.
    """
    start, stop, step = 15000, None, 500

    apl_per_res_dict: Dict[str, np.ndarray] = {}

    all_lipid_sel = "(resname POPG and name C2 C21 C31) or (resname POPE and name C2 C21 C31)"
    all_protein_and_lipid = (
        "(resname POPG and name C2 C21 C31) or protein or " "(resname POPE and name C2 C21 C31)"
    )

    # Load universe
    u = mda.Universe(f"{path}/md_0_1.tpr", f"{path}/md_0_1_combined_first500ns_pbc.xtc")

    # Determine unique lipid species and Voronoi seeds
    membrane = u.select_atoms(all_lipid_sel).residues
    all_select = u.select_atoms(all_protein_and_lipid).residues
    unique_resnames = np.unique(membrane.resnames)
    num_residues = {lipid: sum(membrane.resnames == lipid) for lipid in unique_resnames}
    num_seeds = {
        lipid: int(
            u.select_atoms(f"({all_lipid_sel}) and resname {lipid}").n_atoms / num_residues[lipid]
        )
        for lipid in unique_resnames
    }

    # Output array initialization
    all_apl = np.full(
        (len([res.resid for res in all_select]), int(np.ceil(u.trajectory.n_frames / float(step)))),
        fill_value=np.nan,
        dtype=np.float32,
    )

    for ts in u.trajectory[start:stop:step]:
        # Atoms must be within the unit cell
        membrane.atoms.wrap(inplace=True)
        box = ts.dimensions

        frame_apl = np.asarray([res.resid for res in all_select], dtype=np.float32)

        midpoint = np.mean(membrane.atoms.select_atoms("name P").positions[:, 2])

        # Calculate area per lipid for lower and upper leaflets
        for leaflet_sign in ["<", ">"]:
            # Freud.order.Voronoi requires z positions set to 0
            leaflet = membrane.atoms.select_atoms(
                f"({all_lipid_sel}) and prop z {leaflet_sign} {midpoint}"
            ).residues
            atoms = leaflet.atoms.select_atoms(all_lipid_sel)
            pos = atoms.positions
            pos[:, 2] = 0

            # Check whether any atoms are overlapping in xy-plane
            pos = remove_overlapping(pos)

            # Voronoi tessellation to get area per cell
            areas = get_all_areas(pos, box)

            # Calculate area per lipid
            frame_apl = get_area_per_lipid(
                unique_resnames=unique_resnames,
                atom_group=atoms,
                all_areas=areas,
                num_seeds=num_seeds,
                area_array=frame_apl,
            )

        # Store data for this frame
        all_apl[:, ts.frame // step] = frame_apl

    for res in unique_resnames:
        apl_per_res_dict[res] = all_apl[u.select_atoms(f"resname {res}").residues.resindices]

    lipid_data = all_apl[u.select_atoms("resname POPG").residues.resindices]
    df_lipid = pd.DataFrame.from_records(lipid_data)
    df_lipid.index = range(1, 513)
    df_lipid.columns = range(1, np.size(df_lipid, 1) + 1)

    return df_lipid


def calc_and_write_to_file(path: str, membrane_type: str, results_directory: str) -> None:
    """
    Calculate area per lipid and write results to a CSV file.

    Args:
        path (str): Path to molecular dynamics simulation files.
        membrane_type (str): Type of membrane.
        results_directory (str): Directory to save results.

    Returns:
        None
    """
    Path(results_directory).mkdir(parents=True, exist_ok=True)
    all_lipid: Dict[str, pd.DataFrame] = {}

    if os.path.isdir(path):
        peptide_name = os.path.basename(path)
        print(f"Starting calculations for single peptide -- {peptide_name}")
        df_lipid = calculate_apl(path)
        df_lipid.to_csv(f"{results_directory}/apl_lipid_{peptide_name}_{membrane_type}.csv")
        print(f"{peptide_name} --- DONE")
        all_lipid[peptide_name] = df_lipid
    else:
        print("Invalid path provided")

    return


def main() -> None:
    b_dir = "path"
    output_dir = "APL"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    p = b_dir  # Replace with actual path if needed
    calc_and_write_to_file(p, "pg", output_dir)


if __name__ == "__main__":
    main()
