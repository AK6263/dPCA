import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
from sklearn.decomposition import PCA

def get_trajectory(PDB_FILE_NAME: str, XTC_FILE_NAME: str):
    """Loads the trajectory and topology to genedrate a Universe Object

    Args:
        PDB_FILE_NAME (str): The Topology File (Formats : PDB)
        XTC_FILE_NAME (str): The Trajectory File (Formats : XTC)

    Returns:
        MDAnalysis.Universe(): Universe Object containing our coordinates
    """
    univ = mda.Universe(PDB_FILE_NAME, XTC_FILE_NAME)
    return univ


def get_dihedrals(univ):
    """Extract the Dihedral Angles from the Universe Object

    Args:
        univ (MDAnalysis.Universe): The Universe Object containing our system

    Returns:
        numpy.ndarray: Dihedral Angles of the protein
    """
    sel_phi_angle = [res.phi_selection() for res in univ.residues[1:]]
    sel_psi_angle = [res.psi_selection() for res in univ.residues[:-1]]

    # The Following will extract the Dihedral angles from the selected atom coordinates
    PHI = Dihedral(sel_phi_angle).run()
    PSI = Dihedral(sel_psi_angle).run()

    # The phi and psi angles
    phi = PHI.angles
    psi = PSI.angles

    dihedrals = np.stack((phi, psi), axis=-1)
    return dihedrals


def transform_dihedrals(dihedrals):
    """Transforms the Periodic Dihedral Angles into sine and cosine values
    to avoid problems due to periodicity

    Args:
        dihedrals (numpy.ndarray): Dihedral Angles obtained from get_dihedrals()

    Returns:
        numpy.ndarray: Transformed Angles
    """

    # Converting the dihedrals to their Sin and Cosine Values
    psi_sin = np.sin(dihedrals[:, :, 1] * np.pi / 180)
    psi_cos = np.cos(dihedrals[:, :, 1] * np.pi / 180)
    phi_sin = np.sin(dihedrals[:, :, 0] * np.pi / 180)
    phi_cos = np.cos(dihedrals[:, :, 0] * np.pi / 180)

    sin_cos_dihedrals = np.concatenate([psi_sin, psi_cos, phi_sin, phi_cos], axis=1)

    return sin_cos_dihedrals


def get_components(transformed_dihedrals, components=2, random_state=420):
    """Performs Principal Component Analysis on the transformed Dihedrals

    Args:
        transformed_dihedrals (numpy.ndarray): The sine and cosine transformed dihedrals
        components (int, optional): The Number of Principal Components to Return. Defaults to 2.
        random_state (int, optional): A Random State for PCA. Defaults to 420.

    Returns:
        numpy.ndarray: Principal components of the transformed dihedrals
    """

    pca = PCA(n_components=components, random_state=random_state)
    principal_comps = pca.fit_transform(transformed_dihedrals)

    return principal_comps


def dPCA(PDB_FILE_NAME: str, XTC_FILE_NAME: str, components=2, random_state=420):
    """Main Function to call which return the principal components

    Args:
        PDB_FILE_NAME (str): The Topology File (Formats : PDB)
        XTC_FILE_NAME (str): The Trajectory File (Formats : XTC)
        components (int, optional): The Number of Principal Components to Return. Defaults to 2.
        random_state (int, optional): A Random State for PCA. Defaults to 420.

    Returns:
        numpy.ndarray: Principal components of the transformed dihedrals
    """
    univ = get_trajectory(PDB_FILE_NAME, XTC_FILE_NAME)
    dihedrals = get_dihedrals(univ)
    sin_cos_dihedrals = transform_dihedrals(dihedrals)
    comps = get_components(sin_cos_dihedrals, components, random_state)

    return comps
