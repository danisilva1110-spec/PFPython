from dataclasses import dataclass
from typing import List

import sympy as sp
from sympy import Matrix

from .types import AxisType, JointType


@dataclass
class Joint:
    joint_type: JointType
    theta: sp.Symbol
    d: sp.Symbol
    a: sp.Symbol
    alpha: sp.Symbol
    axis: AxisType = "z"


@dataclass
class Link:
    joint: Joint
    mass: sp.Symbol
    com: Matrix
    inertia: Matrix


@dataclass
class RobotModel:
    links: List[Link]
    gravity: Matrix

    @property
    def dof(self) -> int:
        return len(self.links)
