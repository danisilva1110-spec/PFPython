"""Homogeneous transformation helpers based on Denavit-Hartenberg parameters."""
from sympy import Matrix, cos, sin


def transform_from_dh(a, alpha, d, theta):
    """Construct a homogeneous transform using standard DH parameters."""
    ca = cos(alpha)
    sa = sin(alpha)
    ct = cos(theta)
    st = sin(theta)

    return Matrix([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1],
    ])
