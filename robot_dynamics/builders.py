"""Helpers to build symbolic robot descriptions from matrix-like inputs.

The routines below mirror a common MATLAB workflow where the user provides
matrices/vectors of link data (DH parameters, excentricity offsets, inertia
tensors) together with the rotation order of each joint. They produce
``LinkParameters`` objects that can be fed directly into the kinematics and
dynamics functions defined in this package.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from sympy import Matrix, sympify

from .parameters import LinkParameters

_AXES = {"x", "y", "z"}


def _normalize_joint_type(joint_type: str) -> str:
    joint_type = joint_type.lower()
    if joint_type in {"r", "revolute"}:
        return "revolute"
    if joint_type in {"p", "d", "prismatic"}:
        return "prismatic"
    raise ValueError("joint_type must be 'revolute'/'r' or 'prismatic'/'p'/'d'")


def _normalize_axis_label(axis_label: str) -> str:
    axis_label = axis_label.lower()
    if len(axis_label) == 2 and axis_label[0] in {"r", "p", "d"}:
        axis_label = axis_label[1]
    if axis_label not in _AXES:
        raise ValueError("axis entries must be 'x', 'y' or 'z' (optionally prefixed by R/P/D)")
    return axis_label


def sanitize_inertia_tensor(inertia_matrix: Iterable[Iterable]) -> Matrix:
    """Return a symmetric 3x3 inertia tensor.

    Many spreadsheets store inertia tensors with mirrored off-diagonal terms; this
    helper enforces symmetry so the resulting matrix can be safely rotated and used
    in kinetic energy calculations.
    """

    inertia = Matrix(inertia_matrix)
    if inertia.shape != (3, 3):
        raise ValueError("inertia_tensors entries must be 3x3")
    inertia = inertia.applyfunc(sympify)
    return (inertia + inertia.T) / 2


def build_links_from_matrices(
    dh_params: Sequence[Sequence],
    joint_types: Sequence[str],
    masses: Sequence,
    excentricities: Sequence[Sequence],
    inertia_tensors: Sequence[Iterable[Iterable]],
    axis_orders: Sequence[str] | None = None,
) -> list[LinkParameters]:
    """Create :class:`LinkParameters` objects from matrix-style inputs.

    Parameters
    ----------
    dh_params : sequence of (a, alpha, d, theta)
        Each entry can be a numeric value or a SymPy expression.
    joint_types : sequence of str
        Order of joints; accepts ``"R"``, ``"P"``/``"D"`` or their long names.
    masses : sequence
        Mass of each link.
    excentricities : sequence of length-3 sequences
        Offsets from the frame origin to the center of mass.
    inertia_tensors : sequence of 3x3 arrays or nested lists
        Inertia tensor about the COM for each link.
    axis_orders : sequence of str, optional
        Axis labels (``x``, ``y`` or ``z``) describing the rotation/translation
        direction of each joint. Entries may optionally be prefixed with ``R``/``P``
        or ``D`` for readability. If omitted, all joints use the standard ``z``
        axis.
    """

    if axis_orders is None:
        axis_orders = ["z"] * len(dh_params)

    if not (len(dh_params) == len(joint_types) == len(masses) == len(excentricities) == len(inertia_tensors) == len(axis_orders)):
        raise ValueError("all inputs must have the same length")

    links: list[LinkParameters] = []
    for dh_row, joint_type, mass, excent, inertia, axis_label in zip(
        dh_params, joint_types, masses, excentricities, inertia_tensors, axis_orders
    ):
        if len(dh_row) != 4:
            raise ValueError("Each DH row must contain (a, alpha, d, theta)")
        a, alpha, d, theta = map(sympify, dh_row)
        com = Matrix([[sympify(excent[0])], [sympify(excent[1])], [sympify(excent[2])]])
        inertia_matrix = sanitize_inertia_tensor(inertia)

        links.append(
            LinkParameters(
                a=a,
                alpha=alpha,
                d=d,
                theta=theta,
                joint_type=_normalize_joint_type(joint_type),
                mass=sympify(mass),
                com=com,
                inertia=inertia_matrix,
                axis=_normalize_axis_label(axis_label),
            )
        )
    return links
