"""Symbolic dynamics helpers for open-chain robots."""
from sympy import Matrix, zeros, diff

from .energies import kinetic_energy, potential_energy


def inertia_matrix(links, q, qd):
    """Compute the inertia matrix M(q)."""
    T_energy = kinetic_energy(links, q, qd)
    n = len(q)
    M = zeros(n)
    for i in range(n):
        for j in range(n):
            M[i, j] = diff(diff(T_energy, qd[i]), qd[j])
    return Matrix(M)


def coriolis_matrix(links, q, qd):
    """Compute the Coriolis matrix C(q, qdot) using Christoffel symbols."""
    n = len(q)
    M = inertia_matrix(links, q, qd)
    C = zeros(n)
    for i in range(n):
        for j in range(n):
            c_ij = 0
            for k in range(n):
                c_ijk = 0.5 * (
                    diff(M[i, j], q[k]) + diff(M[i, k], q[j]) - diff(M[j, k], q[i])
                )
                c_ij += c_ijk * qd[k]
            C[i, j] = c_ij
    return Matrix(C)


def gravity_vector(links, q, gravity):
    """Compute G(q) from potential energy."""
    V = potential_energy(links, q, gravity)
    return Matrix([diff(V, qi) for qi in q])


def centripetal_vector(C, qd):
    """Compute the centripetal/Coriolis contribution H(q, qdot) = C(q,qdot) * qdot."""
    return Matrix(C) * Matrix(qd)
