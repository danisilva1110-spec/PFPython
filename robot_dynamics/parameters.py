"""Data structures for defining symbolic robot parameters."""
from dataclasses import dataclass
from sympy import Matrix, Expr


@dataclass
class LinkParameters:
    """Denavit-Hartenberg link parameters and inertial properties.

    Attributes
    ----------
    a : Expr
        Link length.
    alpha : Expr
        Link twist.
    d : Expr
        Link offset.
    theta : Expr
        Joint angle for the nominal configuration.
    joint_type : str
        Either ``"revolute"`` or ``"prismatic"``.
    mass : Expr
        Mass of the link.
    com : Matrix
        3x1 position of the center of mass expressed in the link frame.
    inertia : Matrix
        3x3 inertia tensor about the link frame origin, expressed in the
        link frame.
    """

    a: Expr
    alpha: Expr
    d: Expr
    theta: Expr
    joint_type: str
    mass: Expr
    com: Matrix
    inertia: Matrix

    def __post_init__(self) -> None:
        if self.joint_type not in {"revolute", "prismatic"}:
            raise ValueError("joint_type must be either 'revolute' or 'prismatic'")
        if not isinstance(self.com, Matrix) or self.com.shape != (3, 1):
            raise ValueError("com must be a 3x1 sympy Matrix")
        if not isinstance(self.inertia, Matrix) or self.inertia.shape != (3, 3):
            raise ValueError("inertia must be a 3x3 sympy Matrix")
