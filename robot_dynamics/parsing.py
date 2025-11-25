from typing import List, Tuple

import sympy as sp
from sympy import Matrix

from .models import Joint, Link
from .types import AxisType, JointType


def validate_axes(axes: List[AxisType], dof: int) -> List[AxisType]:
    if len(axes) != dof:
        raise ValueError(f"Esperam-se {dof} eixos (x/y/z), mas recebi {len(axes)}.")
    invalid = [a for a in axes if a not in ("x", "y", "z")]
    if invalid:
        raise ValueError(f"Eixos inválidos: {invalid}. Use apenas x, y ou z.")
    return axes


def parse_axis_order(order: List[str]) -> Tuple[List[JointType], List[AxisType]]:
    joint_types, axes = [], []
    for token in order:
        token_lower = token.lower()
        if token_lower in ("dx", "dy", "dz"):
            joint_types.append("P")
            axes.append(token_lower[-1])
        elif token_lower in ("x", "y", "z"):
            joint_types.append("R")
            axes.append(token_lower)
        else:
            raise ValueError(
                "Entrada de eixo inválida: {token}. Use Dx/Dy/Dz para prismáticas ou x/y/z para rotacionais.".format(
                    token=token
                )
            )
    return joint_types, validate_axes(axes, len(order))


def build_links_from_data(
    qs: List[sp.Symbol],
    joint_types: List[JointType],
    axes: List[AxisType],
    dh_params: List[Tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol]],
    masses: List[sp.Symbol],
    coms: List[Matrix],
    inertias: List[Matrix],
) -> List[Link]:
    if not (
        len(qs)
        == len(joint_types)
        == len(axes)
        == len(dh_params)
        == len(masses)
        == len(coms)
        == len(inertias)
    ):
        raise ValueError(
            "Listas de juntas, parâmetros DH, massas, com e inércias devem ter o mesmo tamanho."
        )

    links = []
    for i, (jt, axis) in enumerate(zip(joint_types, axes)):
        theta_i, d_i, a_i, alpha_i = dh_params[i]
        theta = qs[i] if jt == "R" else theta_i
        d = qs[i] if jt == "P" else d_i
        links.append(
            Link(
                Joint(jt, theta, d, a_i, alpha_i, axis=axis),
                mass=masses[i],
                com=coms[i],
                inertia=inertias[i],
            )
        )
    return links
