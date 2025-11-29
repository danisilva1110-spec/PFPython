"""Symbolic kinematics helpers for open-chain robots."""
from sympy import Matrix

from .parameters import LinkParameters
from .transforms import transform_from_dh


def forward_kinematics(links, q):
    """Compute cumulative transforms for each link frame.

    Parameters
    ----------
    links : list[LinkParameters]
        Robot description.
    q : list
        Generalized coordinates matching the order of ``links``.
    """
    transforms = []
    T = Matrix.eye(4)
    for link, qi in zip(links, q):
        if link.joint_type == "revolute":
            theta = link.theta + qi
            d_val = link.d
        else:
            theta = link.theta
            d_val = link.d + qi
        A = transform_from_dh(link.a, link.alpha, d_val, theta)
        T = T * A
        transforms.append(T)
    return transforms


def center_of_mass_positions(transforms, links):
    """Compute COM positions in the base frame."""
    com_positions = []
    for T, link in zip(transforms, links):
        com_hom = Matrix.vstack(link.com, Matrix([1]))
        com_positions.append((T * com_hom)[:3, 0])
    return com_positions


def compute_jacobians(transforms, links, q):
    """Compute linear and angular Jacobians for each link's COM."""
    origins = [Matrix([0, 0, 0])]
    z_axes = [Matrix([0, 0, 1])]
    for T in transforms:
        origins.append(T[:3, 3])
        z_axes.append(T[:3, 2])

    com_positions = center_of_mass_positions(transforms, links)
    jacobians = []
    for i, (link, p_com) in enumerate(zip(links, com_positions)):
        Jv_cols = []
        Jw_cols = []
        for j, prev_axis in enumerate(z_axes[:-1]):
            if j > i:
                Jv_cols.append(Matrix([0, 0, 0]))
                Jw_cols.append(Matrix([0, 0, 0]))
                continue
            p_j = origins[j]
            if links[j].joint_type == "revolute":
                Jv_cols.append(prev_axis.cross(p_com - p_j))
                Jw_cols.append(prev_axis)
            else:
                Jv_cols.append(prev_axis)
                Jw_cols.append(Matrix([0, 0, 0]))
        jacobians.append((Matrix.hstack(*Jv_cols), Matrix.hstack(*Jw_cols)))
    return jacobians
