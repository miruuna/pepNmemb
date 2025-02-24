import MDAnalysis as mda
import MDAnalysis.transformations as trans
import numpy as np

class Peptide:

    def __init__(self, pep_name, xtc_file_path, tpr_file_path, peptide_number, amino_acid_count, step_size):
        
        self.xtc_file_path = xtc_file_path
        self.tpr_file_path = tpr_file_path
        self.peptide_number = peptide_number
        self.amino_acid_count = amino_acid_count
        self.pep_name = pep_name
        self.step_size = step_size


        self.u = mda.Universe(tpr_file_path, xtc_file_path)
        self.u.trajectory.add_transformations(trans.unwrap(self.u.select_atoms(f"backbone")))
        self.protein_atoms = self.u.select_atoms("protein")
        self.prot_residues = self.protein_atoms.residues
        self.res_names = self.prot_residues.resnames
        self.res_ids = self.prot_residues.residues.resids
        self.resid_maps = {i+1:j for i, j in enumerate(self.res_ids)}
        self.pep_dict = {
            k: ((k-1)*amino_acid_count+1, k*amino_acid_count) for k in range(1, peptide_number+1)
        }
        self.pep_dict = {k: (self.resid_maps[i[0]], self.resid_maps[i[1]]) for k,i in self.pep_dict.items()}

        self.start_id = self.pep_dict[1][0]
        self.end_id = self.pep_dict[1][1]


    def load_traj(self):

        stop_sim = None
        start, stop_sim, _ = self.u.trajectory.check_slice_indices(None, None, None)
        frames = np.arange(start, stop_sim, self.step_size)
        n_frames = frames.size

        return frames, n_frames
    