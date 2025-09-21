"""
robot_model.py
----------------

This module defines classes and utility functions for computing
homogeneous transformation matrices, forward kinematics, energies
and the full equations of motion for an n‑degree‑of‑freedom
manipulator defined by a sequence of rotation/translation
operations.  Each joint may be either revolute (a rotation about
one of the principal axes) or prismatic (a translation along one
of the principal axes).

The core functionality is implemented using SymPy.  Symbolic
expressions for the mass/inertia matrix ``M(q)``, Coriolis matrix
``C(q, q̇)``, gravity vector ``G(q)``, kinetic and potential
energies are produced.  The final joint torques/forces required
to realise a given motion are obtained from ``τ = M(q) q̈ +
C(q, q̇) q̇ + G(q)``.  Optionally these symbolic expressions can
be converted to numerical functions using ``sympy.lambdify``.  If
an array library with GPU support (such as CuPy) is supplied to
``lambdify``, numerical evaluation will run on the GPU.  Note that
symbolic derivation itself always happens on the CPU.

The typical workflow is:

1. Construct a ``RobotModel`` passing a list of operation strings
   (``operations``) and the inertial parameters for each link.
2. Access the properties ``M``, ``C``, ``G`` to obtain the
   symbolic dynamic matrices.  Use ``equations_of_motion()`` to
   build the full vector of joint torques/forces.
3. Optionally call ``lambdify_dynamics()`` to obtain Python
   callables for fast numerical evaluation on CPU or GPU.

Example
~~~~~~~

```
import sympy as sp
from robot_model import RobotModel

# Define a planar 2‑DOF arm: rotation about z followed by translation along x
ops = ["Rz", "Tx"]
masses = [sp.symbols('m1'), sp.symbols('m2')]
coms   = [sp.Matrix([sp.symbols('c1'), 0, 0]), sp.Matrix([sp.symbols('c2'), 0, 0])]
inerts = [sp.diag(0, 0, sp.symbols('I1z')), sp.diag(0, 0, sp.symbols('I2z'))]
robot  = RobotModel(operations=ops, masses=masses, com_local=coms, inertia_local=inerts)

# Symbolic dynamics
M_sym  = robot.M
C_sym  = robot.C
G_sym  = robot.G

# Convert to numeric functions (using NumPy by default)
M_func, C_func, G_func = robot.lambdify_dynamics()

# Evaluate at a numeric configuration (q, q̇) and parameter values
import numpy as np
q_val   = np.array([0.3, 0.2])
qd_val  = np.array([0.1, 0.0])
params  = {robot.symbols['m1']: 1.0, robot.symbols['m2']: 0.5,
           robot.symbols['c1']: 0.25, robot.symbols['c2']: 0.2,
           robot.symbols['I1z']: 0.01, robot.symbols['I2z']: 0.005,
           robot.symbols['g']: 9.81}
M_num  = M_func(q_val, params)
C_num  = C_func(q_val, qd_val, params)
G_num  = G_func(q_val, params)
```
"""

from __future__ import annotations

import sympy as sp
from typing import Iterable, List, Optional, Tuple, Dict, Callable, Any


def skew(v: sp.Matrix) -> sp.Matrix:
    """Return the 3×3 skew‑symmetric matrix for a 3‑vector ``v``.

    The skew matrix satisfies ``skew(a) * b = a × b`` for any
    vectors ``a``, ``b`` in ℝ³.

    Parameters
    ----------
    v : sp.Matrix (3×1)
        Vector to convert into a skew‑symmetric matrix.

    Returns
    -------
    sp.Matrix (3×3)
        Skew‑symmetric matrix.
    """
    vx, vy, vz = v
    return sp.Matrix([[0, -vz,  vy],
                      [vz,   0, -vx],
                      [-vy, vx,   0]])


def rotation_matrix(axis: str, angle: sp.Symbol) -> sp.Matrix:
    """Return a 3×3 rotation matrix about one of the principal axes.

    Parameters
    ----------
    axis : {'x', 'y', 'z'}
        Axis of rotation.
    angle : sympy.Symbol
        Rotation angle.

    Returns
    -------
    sp.Matrix (3×3)
        Rotation matrix.
    """
    axis = axis.lower()
    c = sp.cos(angle)
    s = sp.sin(angle)
    if axis == 'x':
        return sp.Matrix([[1, 0, 0],
                          [0, c, -s],
                          [0, s,  c]])
    elif axis == 'y':
        return sp.Matrix([[ c, 0, s],
                          [ 0, 1, 0],
                          [-s, 0, c]])
    elif axis == 'z':
        return sp.Matrix([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]])
    else:
        raise ValueError(f"Invalid rotation axis '{axis}'. Use 'x', 'y' or 'z'.")


def translation_matrix(axis: str, distance: sp.Symbol) -> sp.Matrix:
    """Return a 4×4 homogeneous translation matrix along one axis.

    Parameters
    ----------
    axis : {'x', 'y', 'z'}
        Axis of translation.
    distance : sympy.Symbol
        Amount of translation.

    Returns
    -------
    sp.Matrix (4×4)
        Homogeneous translation matrix.
    """
    axis = axis.lower()
    T = sp.eye(4)
    if axis == 'x':
        T[0, 3] = distance
    elif axis == 'y':
        T[1, 3] = distance
    elif axis == 'z':
        T[2, 3] = distance
    else:
        raise ValueError(f"Invalid translation axis '{axis}'. Use 'x', 'y' or 'z'.")
    return T


def homogeneous_for_op(op: str, q: sp.Symbol) -> sp.Matrix:
    """Return a 4×4 homogeneous transform for a single operation.

    ``op`` should be of the form ``'Rx'``, ``'Ry'``, ``'Rz'`` for rotations
    or ``'Tx'``, ``'Ty'``, ``'Tz'`` for translations.  The supplied
    symbolic variable ``q`` is used as the rotation angle or the
    translation distance.
    """
    op = op.strip().lower()
    if len(op) != 2:
        raise ValueError("Operation strings must have length 2, e.g. 'Rx', 'Ty'.")
    kind, axis = op[0], op[1]
    if kind == 'r':
        R = rotation_matrix(axis, q)
        T = sp.eye(4)
        T[:3, :3] = R
        return T
    elif kind == 't':
        return translation_matrix(axis, q)
    else:
        raise ValueError(f"Unknown operation kind '{kind}'. Use 'R' or 'T'.")


def homogeneous_from_ops(ops: List[str], q_syms: Optional[List[sp.Symbol]] = None) -> sp.Matrix:
    """Compute the cumulative homogeneous transformation for a sequence of operations.

    Parameters
    ----------
    ops : list of str
        Sequence of operation codes.  Each element must be of the form
        ``'Rx'``, ``'Ry'``, ``'Rz'`` for rotations or ``'Tx'``, ``'Ty'``, ``'Tz'``
        for translations.  The order of operations is the order they are
        applied: the first operation is closest to the base.
    q_syms : list of sympy.Symbol, optional
        List of symbolic variables (length equal to ``len(ops)``)
        representing the rotation angles or translation distances for
        each operation.  If ``None`` (default), a list of new symbols
        ``q1, q2, ..., q_n`` is automatically created.

    Returns
    -------
    sp.Matrix (4×4)
        Homogeneous transformation matrix representing the net effect
        of the operations.
    """
    n = len(ops)
    if q_syms is None:
        q_syms = list(sp.symbols(f'q1:{n+1}', real=True))
    if len(q_syms) != n:
        raise ValueError("Number of symbolic variables must match number of operations.")
    T = sp.eye(4)
    for op, q in zip(ops, q_syms):
        T_op = homogeneous_for_op(op, q)
        T = T * T_op
    return sp.simplify(T)


class RobotModel:
    """A symbolic model of an n‑DOF manipulator defined by operations.

    Each joint is specified by a string code in ``operations``.  If the code
    starts with ``'R'`` it denotes a revolute joint about the axis given by
    the second character.  If it starts with ``'T'`` it denotes a
    prismatic joint (pure translation) along the axis given by the second
    character.  The sequence of operations defines the order in which
    transformations are applied from base to end effector.

    Inertial parameters must be supplied for each link.  Link ``i`` is the
    body immediately after joint ``i``.  Its mass ``m[i]``, its
    centre‑of‑mass position ``com_local[i]`` expressed in the link frame,
    and its inertia tensor ``inertia_local[i]`` expressed in the link
    frame must be provided.  The gravity vector can be supplied in the
    base frame; by default it is ``[0, 0, -g]`` with ``g`` a positive
    symbol.
    """

    def __init__(
        self,
        operations: List[str],
        masses: List[sp.Symbol],
        com_local: List[sp.Matrix],
        inertia_local: List[sp.Matrix],
        gravity_vector: Optional[sp.Matrix] = None,
        offsets: Optional[List[float]] = None,
    ):
        # Basic checks
        self.operations = [op.strip() for op in operations]
        self.n = len(self.operations)
        if len(masses) != self.n or len(com_local) != self.n or len(inertia_local) != self.n:
            raise ValueError("Length of masses, com_local and inertia_local must equal number of operations.")
        self.m = list(masses)
        self.com_local = [sp.Matrix(c) for c in com_local]
        self.I_local = [sp.Matrix(I) for I in inertia_local]
        # Create symbolic variables q, qd, qdd
        self.q  = sp.Matrix(sp.symbols(f'q1:{self.n+1}', real=True))
        self.qd = sp.Matrix(sp.symbols(f'qd1:{self.n+1}', real=True))
        self.qdd= sp.Matrix(sp.symbols(f'qdd1:{self.n+1}', real=True))
        # Offsets for joint variables (constant offsets added to q[i])
        self.offsets = offsets if offsets is not None else [0]*self.n
        # Gravity vector in base frame
        g = sp.symbols('g', positive=True, real=True)
        self.gvec = gravity_vector if gravity_vector is not None else sp.Matrix([0, 0, -g])
        # Collect parameters into dict for convenience when lambdifying
        self.symbols: Dict[str, sp.Symbol] = {}
        # Save mass/inertia symbols for lambdify convenience
        for i in range(self.n):
            # Ensure masses are sympy symbols
            self.symbols[f'm{i+1}'] = self.m[i]
            # Each com component may be symbolic
            cx, cy, cz = self.com_local[i]
            self.symbols[f'c{i+1}x'] = cx
            self.symbols[f'c{i+1}y'] = cy
            self.symbols[f'c{i+1}z'] = cz
            # inertia entries: Ixx, Iyy, Izz and maybe off diag
            I_mat = self.I_local[i]
            for row in range(3):
                for col in range(3):
                    if I_mat[row, col].free_symbols:
                        key = f'I{i+1}{chr(ord("x")+row)}{chr(ord("x")+col)}'
                        self.symbols[key] = I_mat[row, col]
        self.symbols['g'] = self.gvec[2] if isinstance(self.gvec[2], sp.Symbol) else g
        # Precompute transforms and kinematics
        self._compute_kinematics()
        # Compute dynamic matrices
        self.M = self._inertia_matrix()
        self.G = self._gravity_vector()
        self.C = self._coriolis_matrix()

    def _compute_kinematics(self):
        """Compute forward kinematics and Jacobians.

        This populates lists of transformation matrices ``self.T_list`` and
        rotation matrices ``self.R_list`` from the base to each frame,
        the origins ``self.origins``, the z‑axes ``self.z_axes`` used
        for joint screw axes, the CoM positions ``self.p_com`` in the
        base frame, and the Jacobians for each link's CoM.
        """
        T = sp.eye(4)
        self.T_list = [T]
        self.R_list = [T[:3, :3]]
        self.origins = [T[:3, 3]]
        # Build transforms sequentially using q + offset
        for i, op in enumerate(self.operations):
            q_i = self.q[i] + (self.offsets[i] if i < len(self.offsets) else 0)
            T_i = homogeneous_for_op(op, q_i)
            T = T * T_i
            self.T_list.append(T)
            self.R_list.append(T[:3, :3])
            self.origins.append(T[:3, 3])
        # z axes for each joint screw axis (in base frame)
        self.z_axes = [R[:, 2] for R in self.R_list[:-1]]  # exclude the final frame
        # Position of CoM of each link in base
        self.p_com = []
        for i in range(self.n):
            # CoM position: p_i + R_i * p_c_i (i+1 because link i is after joint i)
            p_i = self.origins[i+1]
            R_i = self.R_list[i+1]
            p_ci = p_i + R_i * self.com_local[i]
            self.p_com.append(p_ci)
        # Jacobians
        self.Jv: List[sp.Matrix] = []  # linear parts (3×n)
        self.Jw: List[sp.Matrix] = []  # angular parts (3×n)
        for i in range(self.n):
            Jv_i = sp.zeros(3, self.n)
            Jw_i = sp.zeros(3, self.n)
            p_i = self.p_com[i]
            for j in range(self.n):
                if j > i:
                    # joint j does not affect link i (since j > i means later in chain)
                    continue
                z_j = self.z_axes[j]
                o_j = self.origins[j]
                kind = self.operations[j][0].lower()
                if kind == 'r':
                    # revolute: angular axis and linear cross product
                    Jw_i[:, j] = z_j
                    Jv_i[:, j] = z_j.cross(p_i - o_j)
                elif kind == 't':
                    # prismatic: angular part zero, linear part equals z_j
                    Jw_i[:, j] = sp.Matrix([0, 0, 0])
                    Jv_i[:, j] = z_j
                else:
                    raise ValueError(f"Unknown joint kind '{kind}'.")
            self.Jv.append(Jv_i)
            self.Jw.append(Jw_i)

    def _inertia_matrix(self) -> sp.Matrix:
        """Compute the inertia (mass) matrix M(q)."""
        M = sp.zeros(self.n)
        for i in range(self.n):
            m_i = self.m[i]
            R_i = self.R_list[i+1]  # rotation of link i frame
            I_i = self.I_local[i]
            Jv_i = self.Jv[i]
            Jw_i = self.Jw[i]
            # translational kinetic energy: m * (Jv)^T * Jv
            M += m_i * (Jv_i.T * Jv_i)
            # rotational kinetic energy: (Jw)^T * (R I R^T) * Jw
            M += Jw_i.T * (R_i * I_i * R_i.T) * Jw_i
        return sp.simplify(M)

    def _potential_energy(self) -> sp.Expr:
        """Compute the potential energy V(q)."""
        V = 0
        for i in range(self.n):
            V += self.m[i] * self.gvec.dot(self.p_com[i])
        return sp.simplify(V)

    def _gravity_vector(self) -> sp.Matrix:
        """Compute the gravity torque/force vector G(q)."""
        V = self._potential_energy()
        G = sp.Matrix([sp.diff(V, qi) for qi in self.q])
        return sp.simplify(G)

    def _coriolis_matrix(self) -> sp.Matrix:
        """Compute the Coriolis matrix C(q, q̇)."""
        C = sp.zeros(self.n)
        # Use Christoffel symbols method
        for i in range(self.n):
            for j in range(self.n):
                cij = 0
                for k in range(self.n):
                    cij += sp.Rational(1, 2) * (
                        sp.diff(self.M[i, j], self.q[k]) +
                        sp.diff(self.M[i, k], self.q[j]) -
                        sp.diff(self.M[k, j], self.q[i])
                    ) * self.qd[k]
                C[i, j] = sp.simplify(cij)
        return C

    def equations_of_motion(self) -> sp.Matrix:
        """Compute the vector of generalized joint forces/torques τ."""
        return sp.simplify(self.M * self.qdd + self.C * self.qd + self.G)

    def kinetic_energy(self) -> sp.Expr:
        """Compute total kinetic energy T (translational + rotational)."""
        T = 0
        for i in range(self.n):
            m_i = self.m[i]
            R_i = self.R_list[i+1]
            I_i = self.I_local[i]
            v_i = self.Jv[i] * self.qd
            w_i = self.Jw[i] * self.qd
            T += sp.Rational(1,2) * m_i * v_i.dot(v_i)
            T += sp.Rational(1,2) * w_i.dot(R_i * I_i * R_i.T * w_i)
        return sp.simplify(T)

    def potential_energy(self) -> sp.Expr:
        """Compute total potential energy V."""
        return self._potential_energy()

    def lagrangian(self) -> sp.Expr:
        """Compute the Lagrangian L = T - V."""
        return sp.simplify(self.kinetic_energy() - self.potential_energy())

    def lambdify_dynamics(
        self,
        modules: Optional[Any] = None
    ) -> Tuple[Callable, Callable, Callable]:
        """Return callable functions to evaluate M(q), C(q, q̇) and G(q).

        Parameters
        ----------
        modules : optional
            Passed to ``sympy.lambdify``.  By default this is ``None`` which
            causes ``sympy.lambdify`` to choose ``math`` and ``numpy``.  To
            evaluate on a GPU using CuPy supply ``modules='cupy'`` (after
            ``pip install cupy``).  Alternatively, you may pass a custom
            dictionary mapping SymPy functions to your own numerical
            implementations.

        Returns
        -------
        tuple of callables: (M_func, C_func, G_func)
            Each callable takes the joint coordinates (``q_val``), optional
            joint velocities (for ``C``), and a dict of parameter values
            corresponding to the symbols in ``self.symbols``.  The return
            values are numerical arrays of appropriate shape.  See
            ``examples`` in the module docstring.
        """
        # Flatten parameter symbols for lambdify
        param_syms = list(self.symbols.values())

        # Functions of q only (M and G), and of q and qd (C)
        M_lambda = sp.lambdify((self.q, param_syms), self.M, modules=modules)
        G_lambda = sp.lambdify((self.q, param_syms), self.G, modules=modules)
        C_lambda = sp.lambdify((self.q, self.qd, param_syms), self.C, modules=modules)

        def M_func(q_val: Iterable, params: Dict[sp.Symbol, float]) -> Any:
            param_vals = [params[sym] for sym in param_syms]
            return M_lambda(q_val, param_vals)

        def G_func(q_val: Iterable, params: Dict[sp.Symbol, float]) -> Any:
            param_vals = [params[sym] for sym in param_syms]
            return G_lambda(q_val, param_vals)

        def C_func(q_val: Iterable, qd_val: Iterable, params: Dict[sp.Symbol, float]) -> Any:
            param_vals = [params[sym] for sym in param_syms]
            return C_lambda(q_val, qd_val, param_vals)

        return M_func, C_func, G_func

    # End of class RobotModel
