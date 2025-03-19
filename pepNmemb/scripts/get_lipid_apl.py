# Modified by MSerian
from tqdm import tqdm
import numpy as np
import MDAnalysis as mda
import freud
import pandas as pd
from pathlib import Path
from importlib import reload
import os



b_dir = os.getenv('DATA_FOLDER')
output_dir = f"APL"
Path(output_dir).mkdir(parents=True, exist_ok=True)

def remove_overlapping(positions):
    """Given an Nx3 array of atomic positions,
    make minor adjustments to xy positions
    if any pair of xy coordinates are
    identical.
    
    If atoms are overlapping in xy, Freud will 
    complain when attempting to perform the 
    Voronoi tessellation.
    """
    
    # Check whether any atoms are overlapping in the xy-plane
    # This may be an issue in CG sims with cholesteorl flip-flop
    # but is unlikely to be so in all-atom sims
    unique, indices, counts = np.unique(
            positions, return_index=True, return_counts=True, axis=0)
    
    # If so, add a small distance between the two atoms (1e-3 A)
    # in the x-dimension
    if max(counts > 1):
        for duplicate_index in indices[counts > 1]:
            positions[duplicate_index, 0] += 0.001
            
    return positions

def get_all_areas(positions, box):
    
    """Given xy coordinates of atomic positions
    and the dimensions of the system, perform a Voronoi
    tessellation and return the area in xy occupied
    by each Voronoi cel.
    """
    
    voro = freud.locality.Voronoi()
    areas = voro.compute(
        system=({"Lx": box[0], "Ly": box[1], "dimensions": 2}, positions)
        ).volumes

    return areas

def get_area_per_lipid(unique_resnames, atom_group, all_areas, num_seeds, area_array):
    """Calculate the area per lipid given the areas of
    every Voronoi cell in get_aaa tessellation.
    
    This requires summing contributions from each cell of
    a given lipid.
    
    unique_resnames: list of lipid species in the membrane
    atom_group: MDAnalysis atom group for which the Voronoi 
                tessellation was performed.
    all_areas: numpy array of areas of each cell in the
               tessellation.
    num_seeds: dictionary containing the number of Voronoi
               seeds used for each lipid species
    area_array: numpy array in which the area per lipid will 
                be stored
                
    returns: area_array
            the modified area_array contains the area per lipid
            for the leaflet to which atom_group corresponds.
    """
    
    for res in unique_resnames:
        lipid_indices = np.where(atom_group.resnames==res)
        lipid_apl = all_areas[lipid_indices]

        # We need to sum the area contribution of each cell for a given lipid
        lipid_apl = np.sum(lipid_apl.reshape(atom_group[lipid_indices].residues.n_residues, num_seeds[res]), axis=1)

        # store apl for current lipid species
        lipid_resindices = atom_group.select_atoms(f"resname {res}").residues.resindices
        area_array[lipid_resindices] = lipid_apl

    return area_array

def calculate_apl(path):
    start, stop, step = 15000, None, 500

    apl_per_res_dict = {}



    all_lipid_sel = f"(resname POPG and name C2 C21 C31) or (resname POPE and name C2 C21 C31)"
    all_protein_and_lipid = f"(resname POPG and name C2 C21 C31) or protein or (resname POPE and name C2 C21 C31)"
    
    new_gram_neg = ["PMPE", "POPE", "PYPG", "QMPE", "PMPG", "OYPE", "PVCL2"]
    resname_new_gram_neg = [ f"resname {l}" for l in new_gram_neg]
    str_new_gram_neg = " or ".join(resname_new_gram_neg)
    new_gram_pos = ["MAIPE", "AIPE", "AIPG", "MAIPG", "PAIPG", "PAICL2", "PAIPE", "DPPE", "DPPG"]
    resname_new_gram_pos = [ f"resname {l}" for l in new_gram_pos]

    # Load universe
    u = mda.Universe(f"{path}/md_0_1.tpr", f"{path}/md_0_1_combined_first500ns_pbc.xtc")
    
    # We need to know the resname of each unique lipid species
    # as well as how many Voronoi seeds are used for each lipid species
    membrane = u.select_atoms(all_lipid_sel).residues
    all_select = u.select_atoms(all_protein_and_lipid).residues
    unique_resnames = np.unique(membrane.resnames)
    num_residues = {lipid: sum(membrane.resnames==lipid) for lipid in unique_resnames}
    num_seeds = {
        lipid: int(
            u.select_atoms(f"({all_lipid_sel}) and resname {lipid}").n_atoms / num_residues[lipid]
        ) for lipid in unique_resnames
    }

    # Output array
    all_apl = np.full(
        (len([res.resid for res in all_select]),
         int(np.ceil(u.trajectory.n_frames/float(step)))),
        fill_value=np.NaN, dtype=np.float32
    )

    for ts in u.trajectory[start:stop:step]:

        # Atoms must be within the unit cell
        membrane.atoms.wrap(inplace=True)
        box = ts.dimensions

        frame_apl = np.asarray([res.resid for res in all_select],dtype=np.float32)

        midpoint = np.mean(membrane.atoms.select_atoms("name P").positions[:,2])

        # calculate area per lipid for the lower (<) and upper (>) leaflets
        for leaflet_sign in ["<", ">"]:

            # freud.order.Voronoi requires z positions set to 0
            leaflet = membrane.atoms.select_atoms(f"({all_lipid_sel}) and prop z {leaflet_sign} {midpoint}").residues
            atoms = leaflet.atoms.select_atoms(all_lipid_sel)
            pos = atoms.positions
            pos[:,2] = 0

            # Check whether any atoms are overlapping in the xy-plane
            pos = remove_overlapping(pos)

            # Voronoi tessellation to get area per cell
            areas = get_all_areas(pos, box)

            # Calculate area per lipid in the upper leaflet by considering the contribution of each cell of a given lipid
            frame_apl = get_area_per_lipid(
                unique_resnames=unique_resnames,
                atom_group=atoms,
                all_areas=areas,
                num_seeds=num_seeds,
                area_array=frame_apl
            )

        # Store data for this frame
        all_apl[:, ts.frame//step] = frame_apl

    for res in unique_resnames:
        apl_per_res_dict[res] = all_apl[u.select_atoms(f"resname {res}").residues.resindices]


    lipid_data = all_apl[u.select_atoms(f"resname POPG").residues.resindices]
    df_lipid= pd.DataFrame.from_records(lipid_data)
    df_lipid.index = range(1,513)
#     df_lipid.columns=range(1,102)
    df_lipid.columns=range(1,np.size(df_lipid,1)+1)

    
    return df_lipid

def calc_and_write_to_file(path, membrane_type, results_directory):
    Path(results_directory).mkdir(parents=True, exist_ok=True)
    #create selection
    all_lipid = {}
    if os.path.isdir(path):
        peptide_name = os.path.basename(path)
        print(f"Starting calculations for single peptide -- {peptide_name}")
        peptide_path = path
        df_lipid = calculate_apl(peptide_path)
        df_lipid.to_csv(f"{results_directory}/apl_lipid_{peptide_name}_{membrane_type}.csv")
        print(f"{peptide_name} --- DONE")
        all_lipid[peptide_name] = df_lipid
    else:
        print("NOO")


    calc_and_write_to_file(p, "pg", output_dir)

