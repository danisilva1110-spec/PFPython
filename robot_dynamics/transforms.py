import sympy as sp
from sympy import Matrix

from .types import AxisType


def dh_transform(theta, d, a, alpha) -> Matrix:
    ct, st = sp.cos(theta), sp.sin(theta)
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    return Matrix(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1],
        ]
    )


def axis_from_char(rotation: Matrix, axis: AxisType) -> Matrix:
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    return rotation[:, idx]
