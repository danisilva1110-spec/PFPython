from typing import List, Tuple

from sympy import Matrix

from .models import RobotModel
from .transforms import axis_from_char, dh_transform


def forward_kinematics(model: RobotModel, debug: bool = False) -> Tuple[List[Matrix], List[Matrix]]:
    Ts, origins = [], []
    T = Matrix.eye(4)
    for idx, link in enumerate(model.links):
        origins.append(T[:3, 3].copy())
        T = T * dh_transform(link.joint.theta, link.joint.d, link.joint.a, link.joint.alpha)
        Ts.append(T.copy())
        if debug:
            print(
                f"[DEBUG][Cinemática] Elo {idx + 1}/{model.dof} concluído (origem: {origins[-1].T})"
            )
    origins.append(T[:3, 3].copy())
    return Ts, origins


def spatial_jacobians(
    model: RobotModel, Ts: List[Matrix], origins: List[Matrix], debug: bool = False
) -> Tuple[List[Matrix], List[Matrix]]:
    motion_axes = []
    for j, link in enumerate(model.links):
        R_prev = Matrix.eye(3) if j == 0 else Ts[j - 1][:3, :3]
        motion_axes.append(axis_from_char(R_prev, link.joint.axis))

    Jvs, Jws = [], []
    for i, link in enumerate(model.links):
        origin_current = origins[i]
        o_com = origin_current + Ts[i][:3, :3] * link.com
        Jv_cols, Jw_cols = [], []
        for j in range(model.dof):
            if j > i:
                Jv_cols.append(Matrix([0, 0, 0]))
                Jw_cols.append(Matrix([0, 0, 0]))
                continue
            axis_vec = motion_axes[j]
            o_j = origins[j + 1]
            if model.links[j].joint.joint_type == "R":
                Jv_cols.append(axis_vec.cross(o_com - o_j))
                Jw_cols.append(axis_vec)
            else:
                Jv_cols.append(axis_vec)
                Jw_cols.append(Matrix([0, 0, 0]))
        Jvs.append(Matrix.hstack(*Jv_cols))
        Jws.append(Matrix.hstack(*Jw_cols))
        if debug:
            print(f"[DEBUG][Cinemática] Jacobianos do elo {i + 1}/{model.dof} calculados.")
    return Jvs, Jws
