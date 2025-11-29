"""Ready-to-use symbolic robot definitions."""
from sympy import Matrix, symbols, zeros

from .parameters import LinkParameters


def planar_two_dof():
    """Create a planar 2-DOF manipulator using standard DH parameters."""
    q1, q2, dq1, dq2, l1, l2, m1, m2, g = symbols(
        "q1 q2 dq1 dq2 l1 l2 m1 m2 g"
    )

    I1 = zeros(3)
    I2 = zeros(3)
    I1[2, 2] = symbols("I1zz")
    I2[2, 2] = symbols("I2zz")

    link1 = LinkParameters(
        a=l1,
        alpha=0,
        d=0,
        theta=q1,
        joint_type="revolute",
        mass=m1,
        com=Matrix([[l1 / 2], [0], [0]]),
        inertia=I1,
    )

    link2 = LinkParameters(
        a=l2,
        alpha=0,
        d=0,
        theta=q2,
        joint_type="revolute",
        mass=m2,
        com=Matrix([[l2 / 2], [0], [0]]),
        inertia=I2,
    )

    gravity_vector = Matrix([0, -g, 0])
    return (link1, link2), (q1, q2), (dq1, dq2), gravity_vector
