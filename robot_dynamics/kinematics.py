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

    def _depends_on(expr, sym):
        return hasattr(expr, "has") and expr.has(sym)

    transforms = []
    T = Matrix.eye(4)
    for link, qi in zip(links, q):
        if link.joint_type == "revolute":
            theta = link.theta if _depends_on(link.theta, qi) else link.theta + qi
            d_val = link.d
        else:
            theta = link.theta
            d_val = link.d if _depends_on(link.d, qi) else link.d + qi
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


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def _axis_from_transform(transform: Matrix, axis_label: str) -> Matrix:
    """Select the requested joint axis (x/y/z) from a homogeneous transform."""

    axis_idx = AXIS_TO_INDEX[axis_label]
    return transform[:3, axis_idx]


def compute_jacobians(transforms, links, q):
    """Compute linear and angular Jacobians for each link's COM.

    Each joint can rotate/translate about ``x``, ``y`` or ``z``; this
    direction is read from :class:`LinkParameters.axis` so that excentric
    offsets or custom assembly orders can be respected without changing the
    DH chain itself.
    """

    if not links:
        return []

    origins = [Matrix([0, 0, 0])]
    axes = [_axis_from_transform(Matrix.eye(4), links[0].axis)]

    for transform, next_link in zip(transforms[:-1], links[1:]):
        origins.append(transform[:3, 3])
        axes.append(_axis_from_transform(transform, next_link.axis))

    origins.append(transforms[-1][:3, 3])

    com_positions = center_of_mass_positions(transforms, links)
    jacobians = []
    for i, (link, p_com) in enumerate(zip(links, com_positions)):
        Jv_cols = []
        Jw_cols = []
        for j, axis_vec in enumerate(axes):
            if j > i:
                Jv_cols.append(Matrix([0, 0, 0]))
                Jw_cols.append(Matrix([0, 0, 0]))
                continue
            p_j = origins[j]
            if links[j].joint_type == "revolute":
                Jv_cols.append(axis_vec.cross(p_com - p_j))
                Jw_cols.append(axis_vec)
            else:
                Jv_cols.append(axis_vec)
                Jw_cols.append(Matrix([0, 0, 0]))
        jacobians.append((Matrix.hstack(*Jv_cols), Matrix.hstack(*Jw_cols)))
    return jacobians
