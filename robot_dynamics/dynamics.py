import sys
from concurrent.futures import ProcessPoolExecutor
from threading import Thread
from typing import List, Optional

import multiprocessing as mp
import sympy as sp
from sympy import Matrix

from .kinematics import forward_kinematics, spatial_jacobians
from .models import RobotModel


def _write_log(message: str):
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def _drain_queue(debug_queue):
    for msg in iter(debug_queue.get, None):
        _write_log(msg)


def _maybe_log(debug_queue, message: str):
    if debug_queue is None:
        _write_log(message)
    else:
        debug_queue.put(message)


def _energy_terms(args):
    (
        link,
        Jv_i,
        Jw_i,
        origin_current,
        T_i,
        dq_vec,
        gravity,
        idx,
        total,
        debug_queue,
        debug,
    ) = args
    R = T_i[:3, :3]
    v = Jv_i * dq_vec
    w = Jw_i * dq_vec
    I_world = R * link.inertia * R.T
    kinetic = 0.5 * link.mass * (v.T * v)[0] + 0.5 * (w.T * I_world * w)[0]
    potential = link.mass * gravity.dot(origin_current + R * link.com)
    if debug:
        _maybe_log(debug_queue, f"[DEBUG][Dinâmica] Energias do elo {idx}/{total} calculadas.")
    return kinetic, potential


def _lagrange_tau_term(args):
    L, qs, dqs, ddqs, idx, total, debug_queue, debug = args
    q, dq, ddq = qs[idx], dqs[idx], ddqs[idx]
    dL_dq = sp.diff(L, q)
    dL_ddq = sp.diff(L, dq)
    d_dt_dL_ddq = sum(sp.diff(dL_ddq, q_var) * dq_var for q_var, dq_var in zip(qs, dqs))
    d_dt_dL_ddq += sum(sp.diff(dL_ddq, dq_var) * ddq_var for dq_var, ddq_var in zip(dqs, ddqs))
    result = d_dt_dL_ddq - dL_dq
    if debug:
        _maybe_log(debug_queue, f"[DEBUG][Dinâmica] Equação de torque τ{idx + 1}/{total} derivada.")
    return result


def dynamics(
    model: RobotModel,
    qs: List[sp.Symbol],
    dqs: List[sp.Symbol],
    ddqs: List[sp.Symbol],
    parallel: bool = False,
    processes: Optional[int] = None,
    debug: bool = False,
):
    Ts, origins = forward_kinematics(model, debug=debug)
    Jvs, Jws = spatial_jacobians(model, Ts, origins, debug=debug)

    manager = debug_queue = listener = None
    if debug and parallel and model.dof > 1:
        manager = mp.Manager()
        debug_queue = manager.Queue()
        listener = Thread(target=_drain_queue, args=(debug_queue,), daemon=True)
        listener.start()

    dq_vec = Matrix(dqs)
    energy_args = [
        (
            link,
            Jvs[i],
            Jws[i],
            origins[i],
            Ts[i],
            dq_vec,
            model.gravity,
            i + 1,
            model.dof,
            debug_queue,
            debug,
        )
        for i, link in enumerate(model.links)
    ]

    energy_results = []
    try:
        if parallel and model.dof > 1:
            with ProcessPoolExecutor(max_workers=processes) as executor:
                for res in executor.map(_energy_terms, energy_args, chunksize=1):
                    energy_results.append(res)
        else:
            for arg in energy_args:
                energy_results.append(_energy_terms(arg))

        kinetic_terms, potential_terms = zip(*energy_results)
        kinetic_total = sp.together(sp.Add(*kinetic_terms))
        potential_total = sp.together(sp.Add(*potential_terms))
        L = sp.together(kinetic_total - potential_total)

        tau_args = [
            (L, qs, dqs, ddqs, idx, model.dof, debug_queue, debug) for idx in range(model.dof)
        ]
        tau_terms = []
        if parallel and model.dof > 1:
            with ProcessPoolExecutor(max_workers=processes) as executor:
                for res in executor.map(_lagrange_tau_term, tau_args, chunksize=1):
                    tau_terms.append(res)
        else:
            for arg in tau_args:
                tau_terms.append(_lagrange_tau_term(arg))

        tau_raw = Matrix(tau_terms)
        M_raw = tau_raw.jacobian(ddqs)
        zero_dd = {dd: 0 for dd in ddqs}
        zero_d = {dq: 0 for dq in dqs}
        Cg_raw = tau_raw.xreplace(zero_dd)
        C_raw = Cg_raw - Cg_raw.xreplace(zero_d)
        G_raw = Cg_raw.xreplace(zero_d)
        H_raw = C_raw + G_raw

        replacements, reduced_exprs = sp.cse(
            [kinetic_total, M_raw, C_raw, H_raw, G_raw, tau_raw], optimizations="basic"
        )
        kinetic_opt, M_opt, C_opt, H_opt, G_opt, tau_opt = (
            sp.Matrix(expr) if hasattr(expr, "shape") else expr for expr in reduced_exprs
        )
        return replacements, kinetic_opt, M_opt, C_opt, H_opt, G_opt, tau_opt
    finally:
        if debug_queue is not None:
            debug_queue.put(None)
            listener.join()
            manager.shutdown()
