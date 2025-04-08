from __future__ import annotations

from typing import Tuple

import MDAnalysis as mda
import MDAnalysis.transformations as trans
import numpy as np


class Peptide:
    """
    Main Peptide Class for molecular dynamics analysis.

    Manages trajectory loading, peptide selection, and frame processing.
    """

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
        Initialize a Peptide object for molecular dynamics analysis.

        Args:
            pep_name (str): Name of the peptide.
            xtc_file_path (str): Path to the XTC trajectory file.
            tpr_file_path (str): Path to the TPR topology file.
            peptide_number (int): Number of peptides.
            amino_acid_count (int): Number of amino acids per peptide.
            step_size (int): Step size for trajectory analysis.

        Raises:
            ValueError: If input arguments are invalid.
            RuntimeError: If there's an error loading the MDAnalysis Universe.
        """
        self._validate_inputs(
            pep_name, xtc_file_path, tpr_file_path, peptide_number, amino_acid_count, step_size
        )

        self.xtc_file_path = xtc_file_path
        self.tpr_file_path = tpr_file_path
        self.peptide_number = peptide_number
        self.amino_acid_count = amino_acid_count
        self.pep_name = pep_name
        self.step_size = step_size

        self._initialize_universe()
        self._process_protein_data()
        self._create_peptide_dictionary()

    def _validate_inputs(
        self,
        pep_name: str,
        xtc_file_path: str,
        tpr_file_path: str,
        peptide_number: int,
        amino_acid_count: int,
        step_size: int,
    ) -> None:
        """
        Validate input arguments for the Peptide class.

        Args:
            pep_name (str): Name of the peptide.
            xtc_file_path (str): Path to the XTC trajectory file.
            tpr_file_path (str): Path to the TPR topology file.
            peptide_number (int): Number of peptides.
            amino_acid_count (int): Number of amino acids per peptide.
            step_size (int): Step size for trajectory analysis.

        Raises:
            ValueError: If input arguments are invalid.
        """
        if not all(
            isinstance(arg, str) and arg for arg in [pep_name, xtc_file_path, tpr_file_path]
        ):
            raise ValueError(
                "pep_name, xtc_file_path, and tpr_file_path must be " "non-empty strings."
            )
        if not all(
            isinstance(arg, int) and arg > 0
            for arg in [peptide_number, amino_acid_count, step_size]
        ):
            raise ValueError(
                "peptide_number, amino_acid_count, and step_size must " "be positive integers."
            )

    def _initialize_universe(self) -> None:
        """
        Initialize MDAnalysis Universe and add trajectory transformations.

        Raises:
            RuntimeError: If there's an error loading the MDAnalysis
              Universe.
        """
        try:
            self.u = mda.Universe(self.tpr_file_path, self.xtc_file_path)
            self.u.trajectory.add_transformations(trans.unwrap(self.u.select_atoms("backbone")))
        except Exception as e:
            raise RuntimeError(f"Error loading MDAnalysis Universe: {e}")

    def _process_protein_data(self) -> None:
        """
        Process protein atoms, residues, and related information.

        """
        self.protein_atoms = self.u.select_atoms("protein")
        self.prot_residues = self.protein_atoms.residues
        self.res_names = self.prot_residues.resnames
        self.res_ids = self.prot_residues.residues.resids
        self.resid_maps = {i + 1: j for i, j in enumerate(self.res_ids)}

    def _create_peptide_dictionary(self) -> None:
        """
        Create a dictionary mapping peptide indices to residue ID
        ranges.
        """
        temp_pep_dict = {
            k: ((k - 1) * self.amino_acid_count + 1, k * self.amino_acid_count)
            for k in range(1, self.peptide_number + 1)
        }
        self.pep_dict = {
            k: (self.resid_maps[i[0]], self.resid_maps[i[1]]) for k, i in temp_pep_dict.items()
        }

        self.start_id = self.pep_dict[1][0]
        self.end_id = self.pep_dict[1][1]

    def load_traj(self) -> Tuple[np.ndarray, int]:
        """
        Load the trajectory and determine frame indices.

        Returns:
            Tuple[np.ndarray, int]:
            - frames: Array of selected frame indices
            - n_frames: Number of frames

        Raises:
            RuntimeError: If there's an error loading trajectory frames.
        """
        try:
            start, stop_sim, _ = self.u.trajectory.check_slice_indices(None, None, None)
            frames = np.arange(start, stop_sim, self.step_size)
            n_frames = frames.size

            return frames, n_frames
        except Exception as e:
            raise RuntimeError(f"Error loading trajectory frames: {e}")
