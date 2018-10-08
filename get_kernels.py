
import copy

import numpy as np
import time
import rmsd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import TorsionFingerprints
from rdkit.ML.Cluster import Butina

import matplotlib.pyplot as plt

import qml.fchl as fchl
import qml.representations as qml_representations
import qml.kernels.distance as qml_distance


global __ATOM_LIST__
__ATOM_LIST__ = [ x.strip() for x in ['h ','he', \
      'li','be','b ','c ','n ','o ','f ','ne', \
      'na','mg','al','si','p ','s ','cl','ar', \
      'k ','ca','sc','ti','v ','cr','mn','fe','co','ni','cu', \
      'zn','ga','ge','as','se','br','kr', \
      'rb','sr','y ','zr','nb','mo','tc','ru','rh','pd','ag', \
      'cd','in','sn','sb','te','i ','xe', \
      'cs','ba','la','ce','pr','nd','pm','sm','eu','gd','tb','dy', \
      'ho','er','tm','yb','lu','hf','ta','w ','re','os','ir','pt', \
      'au','hg','tl','pb','bi','po','at','rn', \
      'fr','ra','ac','th','pa','u ','np','pu'] ]


def get_atom(atom):
    global __ATOM_LIST__
    atom = atom.lower()
    return __ATOM_LIST__.index(atom) + 1


def kernel_from_lt(lt, N, fill=True):

    kernel = np.zeros((N,N))
    view = np.tril_indices(N, -1)
    kernel[view] = lt

    for i in range(N):
        for j in range(i):
            kernel[j,i] = kernel[i,j]

    return kernel


def get_coordinates(mol):

    atoms_list = []
    coordinates_list = []
    natoms_list = []

    for conf in mol.GetConformers():

        atoms = []
        coordinates = []

        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            xyz = [pos.x, pos.y, pos.z]
            coordinates.append(xyz)
            atoms.append(atom.GetSymbol())

        atoms = np.array(atoms)
        natoms = atoms.shape[0]
        coordinates = np.array(coordinates)

        natoms_list.append(natoms)
        atoms_list.append(atoms)
        coordinates_list.append(coordinates)

    return atoms_list, coordinates_list, natoms_list


def get_slatm_representations(atoms_list, coordinates_list):

    charge_list = []
    for atoms in atoms_list:
        charges = [get_atom(atom) for atom in atoms]
        charges = np.array(charges)

        charge_list.append(charges)

    mbtypes = qml_representations.get_slatm_mbtypes(charge_list)

    replist = [qml_representations.generate_slatm(coordinates, atoms, mbtypes, local=False) for coordinates, atoms in zip(coordinates_list, charge_list)]
    replist = np.array(replist)

    return replist


def slatm_gaussian(atoms_list, coordinates_list, dist="l2"):

    slatm_representations = get_slatm_representations(atoms_list, coordinates_list)

    l2distances = qml_distance.l2_distance(slatm_representations, slatm_representations)

    sigmas = [0.01 * 2**i for i in range(20)]
    d_lambda = float(10)**-6
    d_lambda = 0.0

    kernels = []

    # L2 kernel
    for sigma in sigmas:
        kernel = copy.deepcopy(l2distances)
        kernel = np.square(kernel)
        kernel *= -1
        kernel /= sigma**2
        kernel = np.exp(kernel)

        # diag_kernel = kernel[np.diag_indices_from(kernel)]
        kernel[np.diag_indices_from(kernel)] += d_lambda

        kernels.append(kernel)

    kernels = np.array(kernels)

    return kernels


def get_fchl_representations(atoms_list, coordinates_list, nmax, cut_distance=1e6):

    rep_list = []

    charge_list = []
    for atoms in atoms_list:
        charge_list.append([get_atom(atom) for atom in atoms])

    for atoms, coordinates in zip(charge_list, coordinates_list):
        rep = fchl.generate_representation(
            coordinates,
            atoms,
            max_size=nmax,
            neighbors=nmax,
            cut_distance=cut_distance)
        rep_list.append(rep)

    rep_list = np.array(rep_list)

    return rep_list


def fchl_linear(atoms_list, coordinates_list, cut_distance=1e6):

    nmax = max([atoms.shape[0] for atoms in atoms_list])

    rep_list = get_fchl_representations(atoms_list, coordinates_list, nmax,cut_distance=cut_distance)

    sigmas = [0.01 * 2**i for i in range(8)]

    kernel_args = {
        "kernel": "linear",
        "cut_distance": cut_distance,
        "alchemy":'off'
    }
    kernels = fchl.get_global_symmetric_kernels(rep_list, **kernel_args)

    # Felix stuff
    # diagonal = kernel[np.diag_indices_from(kernel)]
    # new_norm = np.sqrt(diagonal[np.newaxis]*diagonal[:,np.newaxis])
    # kernel /= new_norm

    # diagonal = kernel[np.diag_indices_from(kernel)]
    # new_norm = (diagonal[np.newaxis] + diagonal[:,np.newaxis])/2.0
    # kernel -= new_norm
    # kernel += 1

    return kernels


def fchl_multiquadratic(atoms_list, coordinates_list, cut_distance=1e6):

    nmax = max([atoms.shape[0] for atoms in atoms_list])

    rep_list = get_fchl_representations(atoms_list, coordinates_list, nmax,cut_distance=cut_distance)

    sigmas = [0.01 * 2**i for i in range(8)]

    kernel_args = {
        "kernel": "multiquadratic",
        "kernel_args": {
            "c":[0.0],
            },
        "cut_distance": cut_distance,
        "alchemy":'off'
    }
    kernels = fchl.get_global_symmetric_kernels(rep_list, **kernel_args)

    return kernels


def fchl_gaussian(atoms_list, coordinates_list, cut_distance=1e6):

    nmax = max([atoms.shape[0] for atoms in atoms_list])

    rep_list = get_fchl_representations(atoms_list, coordinates_list, nmax,cut_distance=cut_distance)

    sigmas = [0.01 * 2**i for i in range(20)]

    kernel_args = {
        "kernel": "gaussian",
        "kernel_args": {
            "sigma": sigmas,
            },
        "cut_distance": cut_distance,
        "alchemy":'off'
    }

    kernels = fchl.get_global_symmetric_kernels(rep_list, **kernel_args)

    return kernels


def rmsd_distance(atoms_list, coordinates_list, translation=False, rotation=False):

    N = len(atoms_list)

    kernel = np.zeros((N,N))

    # Lower triangular

    for i in range(N):
        for j in range(i):

            coord_i = coordinates_list[i]
            coord_j = coordinates_list[j]

            # unique pairs
            if translation:
                coord_i = coord_i - rmsd.centroid(coord_i)
                coord_j = coord_j - rmsd.centroid(coord_j)

            if rotation:
                kernel[i,j] = rmsd.kabsch_rmsd(coord_i, coord_j)
            else:
                kernel[i,j] = rmsd.rmsd(coord_i, coord_j)

            kernel[j,i] = kernel[i,j]


    # np.fill_diagonal(kernel, 0.0)
    # iu2 = np.triu_indices(N, 1)
    # il2 = np.tril_indices(N, -1)
    # kernel[iu2] = kernel[il2]

    return kernel


def rdkit_rms(mol):

    AllChem.AlignMolConformers(mol)
    kernel = AllChem.GetConformerRMSMatrix(mol, prealigned=True)

    return kernel


def rdkit_tfd(mol):

    kernel = TorsionFingerprints.GetTFDMatrix(mol)

    return kernel


