"""Symbolic dynamics helpers for open-chain robots."""
from sympy import Matrix, zeros, diff

from .energies import kinetic_energy, potential_energy
from .builders import (
    _filter_by_mask,
    build_links_from_matrices,
    build_state_symbols,
    parse_axis_order,
)
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


def equations_of_motion_from_order(
    dh_params,
    axis_order,
    masses,
    excentricities,
    inertia_tensors,
    gravity,
    active_mask=None,
    q=None,
    qd=None,
):
    """Wrapper that accepts a MATLAB-style ordem de juntas (Dx/Dy/Dz/x/y/z).

    Parameters
    ----------
    dh_params : sequence
        Rows of (a, alpha, d, theta) for each joint.
    axis_order : sequence
        Strings like ``"Dx"``, ``"Dy"``, ``"Dz"`` for prismáticos ou ``"x"``,
        ``"y"``, ``"z"`` (ou ``"Rx"``, ``"Ry"``, ``"Rz"``) para rotacionais.
    masses, excentricities, inertia_tensors : sequence
        Same as :func:`equations_of_motion_from_matrices`.
    gravity : Matrix
        Gravity vector expressed in the base frame.
    active_mask : sequence of int/bool, optional
        Entries with ``0`` são removidas do cálculo, permitindo reutilizar as
        tabelas originais apenas ligando/desligando graus de liberdade.
    q, qd : sequence, optional
        Generalized coordinates and their derivatives. If omitted, são gerados
        automaticamente a partir de ``axis_order`` e ``active_mask``.
    """

    joint_types, axis_labels = parse_axis_order(axis_order)

    if active_mask is not None:
        dh_params = _filter_by_mask(active_mask, dh_params)
        joint_types = _filter_by_mask(active_mask, joint_types)
        masses = _filter_by_mask(active_mask, masses)
        excentricities = _filter_by_mask(active_mask, excentricities)
        inertia_tensors = _filter_by_mask(active_mask, inertia_tensors)
        axis_labels = _filter_by_mask(active_mask, axis_labels)

    if q is None or qd is None:
        q_auto, qd_auto = build_state_symbols(axis_order, active_mask=active_mask)
        q = q if q is not None else q_auto
        qd = qd if qd is not None else qd_auto

    return equations_of_motion_from_matrices(
        dh_params,
        joint_types,
        masses,
        excentricities,
        inertia_tensors,
        q=q,
        qd=qd,
        gravity=gravity,
        axis_orders=axis_labels,
    )
