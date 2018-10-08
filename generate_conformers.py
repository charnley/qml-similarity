
import itertools

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdMolDescriptors

import rmsd


def set_angles(m, smart, theta=120.0):
    """
    find smart and rotate the angle
    """


    Chem.SetAngleDeg(m, i, j, k, angle)


    return


def set_dehedral_angles(m, theta=120.0, rotate_general=True, rotate_ol=True, rotate_ine=True):
    """
    Systematic rotation of dihedral angles theta degrees

    Taken from Mads

    """

    rotate_idx_list = list()

    if rotate_general:
        smart = "[!#1]~[!$(*#*)&!D1]-!@[!$(*#*)&!D1]~[!#1]"
        rotate_idx_list += m.GetSubstructMatches(Chem.MolFromSmarts(smart))

    if rotate_ol:
        smart = "[*]~[*]-[O,S]-[#1]"
        rotate_idx_list += m.GetSubstructMatches(Chem.MolFromSmarts(smart))

    if rotate_ine:
        smart = "[*]~[*]-[NX3;H2]-[#1]"
        rotate_idx_list += m.GetSubstructMatches(Chem.MolFromSmarts(smart))


    # Find unique bonds and dihedral angles indexes

    idx_bonds = list()
    idx_dihedral = list()

    atoms = m.GetAtoms()

    for k, i, j, l in rotate_idx_list:

        if (i,j) in idx_bonds: continue
        idx_bonds.append((i,j))
        idx_dihedral.append((k,i,j,l))

        print("found", k,i,j,l)
        print(atoms[k].GetAtomicNum())
        print(atoms[i].GetAtomicNum())
        print(atoms[j].GetAtomicNum())
        print(atoms[l].GetAtomicNum())




    # find dihedrals of parent molecule and create all combinations
    # where the angles are rotated theta degrees.
    parent = m.GetConformer()

    # List of alle moveable angles
    dihedrals = list()

    for k, i, j, l in idx_dihedral:
        parent_dihedral = rdMolTransforms.GetDihedralDeg(parent, k, i, j, l)

        new_dihedrals = [ x*theta for x in range(int(360./theta))]

        print(new_dihedrals)

        dihedrals.append(new_dihedrals)



    # make all possible combinations of dihedral angles
    dihedral_combinations = list(itertools.product(*dihedrals))

    # Create the conformations according to angle combinations
    for dihedrals in dihedral_combinations:

        for (k,i,j,l), angle in zip(idx_dihedral, dihedrals):

            print(k,i,j,l, angle)

            rdMolTransforms.SetDihedralDeg(parent, k, i, j, l, angle)

        # translate mol to centroid
        rdMolTransforms.CanonicalizeConformer(parent)
        m.AddConformer(parent, assignId=True)

    return m



def set_conformers_random(m, n_conformers):
    """
    """

    AllChem.EmbedMultipleConfs(m, n_conformers, AllChem.ETKDG())

    return


def get_conformers(smiles, n_conformers):

    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    set_conformers_random(m, n_conformers)

    return m



def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--smiles', action='store', help='', metavar="str")
    parser.add_argument('-f', '--filename', action='store', help='', metavar="filename", default="hash.sdf")
    parser.add_argument('-a', '--align', action='store_true', help='align molecules')
    parser.add_argument('--scan', action='store_true', help='scan angles')
    parser.add_argument('--scan_theta', action='store', type=float, help='', metavar="angle", default=170.0)
    parser.add_argument('--add_hydrogens', action='store_true', help='Add hydrogens', default=True)

    args = parser.parse_args()

    if not args.smiles:
        print("No smiles set")
        quit()

    m = Chem.MolFromSmiles(args.smiles)

    smart = "~".join(["[*]"]*4)
    idx_list = m.GetSubstructMatches(Chem.MolFromSmarts(smart))

    m = Chem.AddHs(m)

    # if args.add_hydrogens:
    #     m = Chem.AddHs(m)
    #     AllChem.EmbedMolecule(m, AllChem.ETKDG())

    AllChem.EmbedMultipleConfs(m, numConfs=1, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
    # AllChem.EmbedMultipleConfs(m, numConfs=1, useBasicKnowledge=True)

    atoms = m.GetAtoms()

    # for i, a in enumerate(atoms):
    #     print(i, a.GetAtomicNum())


    cm = m.GetConformer()

    if args.scan:
        for x in range(0, 1080, 10):
            rdMolTransforms.SetDihedralDeg(cm, 0, 1, 2, 3, x)
            m.AddConformer(cm, assignId=True)

    # if args.scan:
    #     set_conformers_random(m, 1)
    #     set_dehedral_angles(m, theta=args.scan_theta)

    else:
        n_conformers = 70
        # n_conformers = rdMolDescriptors.CalcNumRotatableBonds(m)
        m = get_conformers(args.smiles, n_conformers)

    # Alignment
    if args.align:
        if True:
            Chem.rdMolAlign.AlignMolConformers(m)

        # if True:
        #     for i, m in enumerate(m.GetConformations()):
        #         if i == 0: continue
        #
        #         # m.AddConformer(parent, assignId=True)


    writer = Chem.SDWriter(args.filename)
    for i in range(m.GetNumConformers()):
        writer.write(m, i)

    return


if __name__ == "__main__":
    main()

