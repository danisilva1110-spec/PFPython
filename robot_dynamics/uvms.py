"""Helpers for UVMS (ROV + manipulator) symbolic dynamics based on axis lists."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from sympy import Matrix, Expr, cos, sin, zeros, diff


AxisChar = str


def _unit_axis(axis: AxisChar) -> Matrix:
    axis = axis.lower()
    if axis == "x":
        return Matrix([1, 0, 0])
    if axis == "y":
        return Matrix([0, 1, 0])
    if axis == "z":
        return Matrix([0, 0, 1])
    raise ValueError("axis must be one of 'x', 'y' or 'z'")


def rotation_matrix(axis: AxisChar, angle: Expr) -> Matrix:
    """Homogeneous rotation about the given axis."""
    c = cos(angle)
    s = sin(angle)
    axis = axis.lower()
    if axis == "x":
        R = Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        R = Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "z":
        R = Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError("axis must be one of 'x', 'y' or 'z'")

    return Matrix.vstack(
        Matrix.hstack(R, Matrix([[0], [0], [0]])), Matrix([[0, 0, 0, 1]])
    )


def translation_matrix(offset: Sequence[Expr]) -> Matrix:
    """Homogeneous translation by the provided 3-element vector."""
    if len(offset) != 3:
        raise ValueError("offset must have three elements")
    x, y, z = offset
    return Matrix(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )


def classify_joint(axis: AxisChar) -> str:
    return "prismatic" if axis.lower().startswith("d") else "revolute"


def uvms_kinematics(
    axis_order: Sequence[AxisChar],
    excentricities: Iterable[Iterable[Expr]],
    q: Sequence[Expr],
) -> Tuple[List[Matrix], List[Matrix], List[Matrix], List[Matrix], List[Matrix]]:
    """Compute transforms, joint origins, axes and COM guesses for an UVMS."""

    ex_list = [Matrix(v) for v in excentricities]
    if len(axis_order) != len(ex_list) or len(q) != len(ex_list):
        raise ValueError("axis_order, excentricities and q must have the same length")

    T = Matrix.eye(4)
    transforms: List[Matrix] = []
    origins: List[Matrix] = []
    axes_world: List[Matrix] = []
    com_positions: List[Matrix] = []
    link_orientations: List[Matrix] = []

    for axis, offset, qi in zip(axis_order, ex_list, q):
        if offset.shape != (3, 1):
            offset = offset.reshape(3, 1)
        origin_current = T[:3, 3]
        origins.append(origin_current)

        base_rot = T[:3, :3]
        axis_dir = _unit_axis(axis[-1]) if axis.lower().startswith("d") else _unit_axis(axis)
        axis_world = base_rot * axis_dir
        axes_world.append(axis_world)

        joint_type = classify_joint(axis)
        if joint_type == "revolute":
            T = T * rotation_matrix(axis[-1], qi)
        else:
            T = T * translation_matrix(axis_world * qi)

        R_after = T[:3, :3]
        link_orientations.append(R_after)

        T = T * translation_matrix(offset)
        transforms.append(T)

        com_positions.append((T * translation_matrix(offset * 0 + offset / 2))[:3, 3])

    return transforms, origins, axes_world, com_positions, link_orientations


def uvms_jacobians(
    origins: Sequence[Matrix],
    axes_world: Sequence[Matrix],
    com_positions: Sequence[Matrix],
    joint_types: Sequence[str],
):
    jacobians = []
    n = len(com_positions)
    for i in range(n):
        Jv_cols = []
        Jw_cols = []
        for j in range(n):
            if j > i:
                Jv_cols.append(Matrix([0, 0, 0]))
                Jw_cols.append(Matrix([0, 0, 0]))
                continue
            axis = axes_world[j]
            origin = origins[j]
            if joint_types[j] == "revolute":
                Jv_cols.append(axis.cross(com_positions[i] - origin))
                Jw_cols.append(axis)
            else:
                Jv_cols.append(axis)
                Jw_cols.append(Matrix([0, 0, 0]))
        jacobians.append((Matrix.hstack(*Jv_cols), Matrix.hstack(*Jw_cols)))
    return jacobians


def uvms_kinetic_energy(
    masses: Sequence[Expr],
    inertias: Sequence[Matrix],
    qd: Sequence[Expr],
    com_positions: Sequence[Matrix],
    link_orientations: Sequence[Matrix],
    jacobians,
):
    T_total = 0
    for mass, inertia, (Jv, Jw), R in zip(masses, inertias, jacobians, link_orientations):
        v = Jv * Matrix(qd)
        w = Jw * Matrix(qd)
        inertia_world = R * inertia * R.T
        T_total += 0.5 * mass * (v.T * v)[0] + 0.5 * (w.T * inertia_world * w)[0]
    return T_total.simplify()


def uvms_potential_energy(masses, com_positions, gravity: Matrix):
    V_total = 0
    for mass, pos in zip(masses, com_positions):
        V_total += -mass * gravity.dot(pos)
    return V_total.simplify()


def uvms_dynamics(
    axis_order: Sequence[AxisChar],
    excentricities: Iterable[Iterable[Expr]],
    q: Sequence[Expr],
    qd: Sequence[Expr],
    masses: Sequence[Expr],
    inertias: Sequence[Matrix],
    gravity: Matrix,
):
    transforms, origins, axes_world, com_positions, orientations = uvms_kinematics(
        axis_order, excentricities, q
    )

    joint_types = [classify_joint(axis) for axis in axis_order]
    jacobians = uvms_jacobians(origins, axes_world, com_positions, joint_types)

    T_energy = uvms_kinetic_energy(masses, inertias, qd, com_positions, orientations, jacobians)
    V_energy = uvms_potential_energy(masses, com_positions, gravity)

    n = len(q)
    M = zeros(n)
    for i in range(n):
        for j in range(n):
            M[i, j] = diff(diff(T_energy, qd[i]), qd[j])

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

    G = Matrix([diff(V_energy, qi) for qi in q])
    H = Matrix(C) * Matrix(qd)

    return transforms, T_energy, V_energy, Matrix(M), Matrix(C), H, G
