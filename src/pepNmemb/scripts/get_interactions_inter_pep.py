from __future__ import annotations

import argparse

import pandas as pd
from classes import Peptide
from MDAnalysis.lib.distances import distance_array
from tqdm import tqdm


def get_interactions(obj):
    u = obj.u
    obj_name = obj.pep_name
    pep_num = obj.peptide_number

    frames, n_frames = obj.load_traj()

    all_list = []
    for _, _ in tqdm(enumerate(u.trajectory[frames]), total=n_frames):
        pair_done = []
        for pep in range(1, pep_num):
            res_range = obj.pep_dict[pep]
            for pep2 in range(pep + 1, pep_num + 1):
                res_range2 = obj.pep_dict[pep2]
                for res in range(res_range[0], res_range[1] + 1):
                    for res2 in range(res_range2[0], res_range2[1] + 1):
                        selection1 = u.select_atoms("resid %s" % res).center_of_mass()
                        selection2 = u.select_atoms("resid %s" % res2).center_of_mass()
                        dist = distance_array(
                            selection1, selection2, box=u.dimensions, result=None, backend="serial"
                        )
                        min_dist = dist
                        all_list.append(
                            (
                                res,
                                res2,
                                pep,
                                pep2,
                                min_dist[0].astype(float)[0],
                                int(u.trajectory.time / 1000),
                            )
                        )
            pair_done.append((pep, pep2))

    df = pd.DataFrame(
        all_list, columns=["Res1", "Res2", "Peptide1", "Peptide2", "mindist", "Time(ns)"]
    )
    df.to_csv(f"pepNmemb/data/interactions_{obj_name}.csv")
    return df


def main():
    parser = argparse.ArgumentParser(description="Run insertion into the membranr analysis")

    parser.add_argument("-xtc", "--xtc", type=str, help="Input xtc file path")
    parser.add_argument("-tpr", "--tpr", type=str, help="Input xtpr file path")
    parser.add_argument("-pname", "--pep_name", type=str, help="Peptide name used for saving data")
    parser.add_argument("-pnum", "--pep_num", type=int, help="Number of peptides")
    parser.add_argument("-res", "--res_num", type=int, help="Number of residues in each peptide")

    parser.add_argument("-ss", "--step_size", type=int, help="Step size")

    args = parser.parse_args()

    peptide_name = args.pep_name
    xtc_file_path = args.xtc
    tpr_file_path = args.tpr
    peptide_number = args.pep_num
    aminoacid_count = args.res_num
    step_size = args.step_size

    print(f"Starting analysis for peptide {peptide_name} found at {xtc_file_path}, {tpr_file_path}")

    peptide = Peptide(
        peptide_name,
        xtc_file_path,
        tpr_file_path,
        peptide_number=peptide_number,
        amino_acid_count=aminoacid_count,
        step_size=step_size,
    )

    get_interactions(peptide)


if __name__ == "__main__":
    main()
