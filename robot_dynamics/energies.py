"""Energy calculations for symbolic open-chain robots."""
from sympy import Matrix

from .kinematics import compute_jacobians, forward_kinematics


def kinetic_energy(links, q, qd):
    """Compute kinetic energy using Jacobians for each link."""
    transforms = forward_kinematics(links, q)
    jacobians = compute_jacobians(transforms, links, q)

    def _skew(v):
        return Matrix(
            [
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ]
        )

    T_total = 0
    for idx, (link, (Jv, Jw)) in enumerate(zip(links, jacobians)):
        v = Jv * Matrix(qd)
        w = Jw * Matrix(qd)
        R = transforms[idx][:3, :3]
        skew_r = _skew(link.com)
        inertia_com = link.inertia - link.mass * (skew_r.T * skew_r)
        inertia_world = R * inertia_com * R.T
        T_total += 0.5 * link.mass * (v.T * v)[0] + 0.5 * (w.T * inertia_world * w)[0]
    return T_total.simplify()


def potential_energy(links, q, gravity):
    """Compute potential energy given a gravity vector."""
    transforms = forward_kinematics(links, q)
    com_positions = [pos for pos in (T[:3, 0:3] * link.com + T[:3, 3] for T, link in zip(transforms, links))]

    V_total = 0
    for link, p in zip(links, com_positions):
        V_total += -link.mass * gravity.dot(p)
    return V_total.simplify()
