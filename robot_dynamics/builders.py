"""Helpers to build symbolic robot descriptions from matrix-like inputs.

The routines below mirror a common MATLAB workflow where the user provides
matrices/vectors of link data (DH parameters, excentricity offsets, inertia
tensors) together with the rotation order of each joint. They produce
``LinkParameters`` objects that can be fed directly into the kinematics and
dynamics functions defined in this package.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from sympy import Matrix, symbols, sympify

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


def _normalize_mask(mask: Sequence) -> list[bool]:
    """Convert a sequence of truthy/falsey entries into a bool list."""

    normalized = [bool(entry) for entry in mask]
    if not normalized:
        raise ValueError("mask must contain at least one entry")
    return normalized


def _filter_by_mask(mask: Sequence, sequence: Sequence):
    """Return a list with elements whose corresponding mask entry is truthy."""

    mask = _normalize_mask(mask)
    if len(mask) != len(sequence):
        raise ValueError("mask length must match the sequence being filtered")
    return [item for item, keep in zip(sequence, mask) if keep]


def parse_axis_order(axis_order: Sequence[str]) -> tuple[list[str], list[str]]:
    """Parse strings like ``"Dx"``/``"Dz"``/``"x"``/``"z"`` into joint types and axes.

    This mirrors the MATLAB workflow where a single cell array encodes the
    direction and whether the motion is prismatic (``D``/``P`` prefix) or
    revolute (``R`` prefix or no prefix). The output feeds directly into
    :func:`build_links_from_matrices`.
    """

    joint_types: list[str] = []
    axis_labels: list[str] = []
    for raw_code in axis_order:
        if not raw_code:
            raise ValueError("axis order entries cannot be empty")
        code = raw_code.lower()
        axis_label = _normalize_axis_label(code[-1])
        if code[0] in {"d", "p"}:
            joint_types.append("prismatic")
        else:
            joint_types.append("revolute")
        axis_labels.append(axis_label)
    return joint_types, axis_labels


def build_state_symbols(
    axis_order: Sequence[str],
    active_mask: Sequence | None = None,
    symbol_prefix: str = "q",
    derivative_prefix: str = "dq",
) -> tuple[list, list]:
    """Create ``q``/``qd`` symbols automatically from an axis order.

    Parameters
    ----------
    axis_order : sequence of str
        Entries like ``"Dx"``, ``"Dy"``, ``"Dz"``, ``"x"``, ``"y"`` or ``"z"``.
    active_mask : sequence of bool/int, optional
        Mask to drop joints from the calculation (``0`` removes the entry,
        ``1`` keeps it). This mirrors the MATLAB trick of multiplying by a
        matrix of ``0``/``1`` to disable degrees of freedom without editing
        the rest of the tables.
    symbol_prefix : str
        Fallback base name for generalized coordinates when an entry is not a
        valid Python identifier.
    derivative_prefix : str
        Prefix for velocity symbols. Defaults to ``dq`` (e.g., ``dqDx`` or
        ``dq1``).
    """

    if active_mask is None:
        active_mask = [True] * len(axis_order)

    keep_mask = _normalize_mask(active_mask)
    if len(axis_order) != len(keep_mask):
        raise ValueError("active_mask must have the same length as axis_order")

    q_symbols = []
    qd_symbols = []

    for idx, (code, keep) in enumerate(zip(axis_order, keep_mask)):
        if not keep:
            continue
        base_name = code if code.isidentifier() else f"{symbol_prefix}{idx + 1}"
        q_sym = symbols(base_name)
        qd_sym = symbols(f"{derivative_prefix}{base_name}")
        q_symbols.append(q_sym)
        qd_symbols.append(qd_sym)

    return q_symbols, qd_symbols


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
