import MDAnalysis as mda
import MDAnalysis.transformations as trans
from typing import Dict, Tuple, List
import numpy as np


class Peptide:
    def __init__(
        self,
        pep_name: str,
        xtc_file_path: str,
        tpr_file_path: str,
        peptide_number: int,
        amino_acid_count: int,
        step_size: int,
    ) -> None:
        """
        Initializes a Peptide object for molecular dynamics analysis.

        Parameters:
        - pep_name: Name of the peptide
        - xtc_file_path: Path to the XTC trajectory file
        - tpr_file_path: Path to the TPR topology file
        - peptide_number: Number of peptides
        - amino_acid_count: Number of amino acids per peptide
        - step_size: Step size for trajectory analysis
        """

        if not all(
            isinstance(arg, (str, int)) and arg
            for arg in [pep_name, xtc_file_path, tpr_file_path]
        ):
            raise ValueError(
                "Invalid input: pep_name, xtc_file_path, \
                    and tpr_file_path must be non-empty strings."
            )
        if not all(
            isinstance(arg, int) and arg > 0
            for arg in [peptide_number, amino_acid_count, step_size]
        ):
            raise ValueError(
                "Invalid input: peptide_number, amino_acid_count,\
                      and step_size must be positive integers."
            )

        self.xtc_file_path = xtc_file_path
        self.tpr_file_path = tpr_file_path
        self.peptide_number = peptide_number
        self.amino_acid_count = amino_acid_count
        self.pep_name = pep_name
        self.step_size = step_size

        try:
            self.u = mda.Universe(tpr_file_path, xtc_file_path)
            self.u.trajectory.add_transformations(
                trans.unwrap(self.u.select_atoms(f"backbone"))
            )
        except Exception as e:
            raise RuntimeError(f"Error loading MDAnalysis Universe: {e}")

        self.protein_atoms = self.u.select_atoms("protein")
        self.prot_residues = self.protein_atoms.residues
        self.res_names = self.prot_residues.resnames
        self.res_ids = self.prot_residues.residues.resids
        self.resid_maps = {i + 1: j for i, j in enumerate(self.res_ids)}
        self.pep_dict = {
            k: ((k - 1) * amino_acid_count + 1, k * amino_acid_count)
            for k in range(1, peptide_number + 1)
        }
        self.pep_dict = {
            k: (self.resid_maps[i[0]], self.resid_maps[i[1]])
            for k, i in self.pep_dict.items()
        }

        self.start_id = self.pep_dict[1][0]
        self.end_id = self.pep_dict[1][1]

    def load_traj(self) -> Tuple[np.ndarray, int]:
        """
        Loads the trajectory and determines frame indices.

        Returns:
        - frames (np.ndarray): Array of selected frame indices
        - n_frames (int): Number of frames
        """

        try:
            start, stop_sim, _ = self.u.trajectory.check_slice_indices(None, None, None)
            frames = np.arange(start, stop_sim, self.step_size)
            n_frames = frames.size

            return frames, n_frames
        except Exception as e:
            raise RuntimeError(f"Error loading trajectory frames: {e}")
