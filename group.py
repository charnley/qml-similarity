
import get_kernels as get

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import TorsionFingerprints
from rdkit.ML.Cluster import Butina

import numpy as np

def view_upper(matrix):

    N = matrix.shape[0]
    iu1 = np.triu_indices(N, 1)

    return matrix[iu1]


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdf', action='store', help='', metavar="file")
    parser.add_argument('--method', action='store', type=str, help='Which kernels to generate')
    parser.add_argument('--rdkit', nargs="+", action='store', help='Rdkit specific arguments', default=[""])
    parser.add_argument('--clustering_threshold', action='store', help='Threshold for Butina clustering algorithm', default=1.0)

    args = parser.parse_args()

    if not args.sdf:
        print("No SDF set")
        quit()


    # Load SDF file
    if "removehs" in args.rdkit:
        ms = [x for x in Chem.SDMolSupplier(args.sdf)]
    else:
        ms = [x for x in Chem.SDMolSupplier(args.sdf, removeHs=False)]

    # Get the conformations
    m = ms[0]
    for i, mc in enumerate(ms[1:]):
        m.AddConformer(mc.GetConformer(), assignId=i+1)

    # Translate into real coordinates
    atoms_list, coordinates_list, natoms_list = get.get_coordinates(m)
    N = len(atoms_list)


    if args.method == "fchl":
        kernels = get.fchl_gaussian(atoms_list, coordinates_list)
        kernel = kernels[6]
        ksv = view_upper(kernel)
        ksv = -np.log(ksv)

    elif args.method == "slatm":
        kernels = get.slatm_gaussian(atoms_list, coordinates_list)
        kernel = kernels[10]
        ksv = view_upper(kernel)
        ksv = -np.log(ksv)

    elif args.method == "tfd":
        ksv = np.array(get.rdkit_tfd(m))

    elif args.method == "rms":
        ksv = np.array(get.rdkit_rms(m))

    elif args.method == "rmsd":
        kernel = get.rmsd_distance(atoms_list, coordinates_list, translation=False, rotation=True)
        ksv = view_upper(kernel)

    else:
        quit("error: unknown method")


    threshold = 1.0

    rms_clusters = Butina.ClusterData(ksv, N, threshold, isDistData=True, reordering=True)

    print("found",len(rms_clusters),"unique out of", N, "conformers")

    return


if __name__ == "__main__":
    main()
