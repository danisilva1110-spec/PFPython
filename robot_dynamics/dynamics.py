"""Symbolic dynamics helpers for open-chain robots."""
from sympy import Matrix, zeros, diff

from .energies import kinetic_energy, potential_energy
from .builders import build_links_from_matrices
from .kinematics import forward_kinematics


def inertia_matrix(links, q, qd):
    """Compute the inertia matrix M(q)."""
    T_energy = kinetic_energy(links, q, qd)
    n = len(q)
    M = zeros(n)
    for i in range(n):
        for j in range(n):
            M[i, j] = diff(diff(T_energy, qd[i]), qd[j])
    return Matrix(M)


def coriolis_matrix(links, q, qd):
    """Compute the Coriolis matrix C(q, qdot) using Christoffel symbols."""
    n = len(q)
    M = inertia_matrix(links, q, qd)
    C = zeros(n)
    for i in range(n):
        for j in range(n):
            c_ij = 0
            for k in range(n):
                c_ijk = 0.5 * (
                    diff(M[i, j], q[k]) + diff(M[i, k], q[j]) - diff(M[j, k], q[i])
                )
                c_ij += c_ijk * qd[k]
            C[i, j] = c_ij
    return Matrix(C)


def gravity_vector(links, q, gravity):
    """Compute G(q) from potential energy."""
    V = potential_energy(links, q, gravity)
    return Matrix([diff(V, qi) for qi in q])


def centripetal_vector(C, qd):
    """Compute the centripetal/Coriolis contribution H(q, qdot) = C(q,qdot) * qdot."""
    return Matrix(C) * Matrix(qd)


def equations_of_motion_from_matrices(
    dh_params,
    joint_types,
    masses,
    excentricities,
    inertia_tensors,
    q,
    qd,
    gravity,
    axis_orders=None,
):
    """High-level wrapper that mirrors a matrix-based MATLAB interface.

    Parameters
    ----------
    dh_params : sequence
        Rows of (a, alpha, d, theta) for each joint.
    joint_types : sequence
        Joint type codes (``"R"`` or ``"P"``/``"D"``).
    masses : sequence
        Mass of each link.
    excentricities : sequence
        3x1 offsets from the frame origin to each link's COM.
    inertia_tensors : sequence
        3x3 inertia tensors about each link's COM.
    q, qd : sequence
        Generalized coordinates and their derivatives.
    gravity : Matrix
        Gravity vector expressed in the base frame.
    axis_orders : sequence, optional
        Axis labels (``x``, ``y`` or ``z``) describing each joint's motion axis.

    Returns
    -------
    dict
        Contains ``links`` (the constructed :class:`LinkParameters` list), the
        cumulative ``transforms`` and the symbolic matrices ``M``, ``C``, ``H``
        and ``G``.
    """

    links = build_links_from_matrices(
        dh_params,
        joint_types,
        masses,
        excentricities,
        inertia_tensors,
        axis_orders=axis_orders,
    )

    transforms = forward_kinematics(links, q)
    M = inertia_matrix(links, q, qd)
    C = coriolis_matrix(links, q, qd)
    G = gravity_vector(links, q, gravity)
    H = centripetal_vector(C, qd)

    return {
        "links": links,
        "transforms": transforms,
        "M": M,
        "C": C,
        "H": H,
        "G": G,
    }
