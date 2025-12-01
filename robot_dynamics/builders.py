"""Helpers to build symbolic robot descriptions from matrix-like inputs.

The routines below mirror a common MATLAB workflow where the user provides
matrices/vectors of link data (DH parameters, excentricity offsets, inertia
tensors) together with the rotation order of each joint. They produce
``LinkParameters`` objects that can be fed directly into the kinematics and
dynamics functions defined in this package.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from sympy import Matrix, symbols, sympify

from .parameters import LinkParameters

_AXES = {"x", "y", "z"}


def _normalize_joint_type(joint_type: str) -> str:
    joint_type = joint_type.lower()
    if joint_type in {"r", "revolute"}:
        return "revolute"
    if joint_type in {"p", "d", "prismatic"}:
        return "prismatic"
    raise ValueError("joint_type must be 'revolute'/'r' or 'prismatic'/'p'/'d'")


def _normalize_axis_label(axis_label: str) -> str:
    axis_label = axis_label.lower()
    if len(axis_label) == 2 and axis_label[0] in {"r", "p", "d"}:
        axis_label = axis_label[1]
    if axis_label not in _AXES:
        raise ValueError("axis entries must be 'x', 'y' or 'z' (optionally prefixed by R/P/D)")
    return axis_label


def _normalize_mask(mask: Sequence) -> list[bool]:
    """Convert a sequence of truthy/falsey entries into a bool list."""

    normalized = [bool(entry) for entry in mask]
    if not normalized:
        raise ValueError("mask must contain at least one entry")
    return normalized


def _filter_by_mask(mask: Sequence, sequence: Sequence):
    """Return a list with elements whose corresponding mask entry is truthy."""

    mask = _normalize_mask(mask)
    if len(mask) != len(sequence):
        raise ValueError("mask length must match the sequence being filtered")
    return [item for item, keep in zip(sequence, mask) if keep]


def parse_axis_order(axis_order: Sequence[str]) -> tuple[list[str], list[str]]:
    """Parse strings like ``"Dx"``/``"Dz"``/``"x"``/``"z"`` into joint types and axes.

    This mirrors the MATLAB workflow where a single cell array encodes the
    direction and whether the motion is prismatic (``D``/``P`` prefix) or
    revolute (``R`` prefix or no prefix). The output feeds directly into
    :func:`build_links_from_matrices`.
    """

    joint_types: list[str] = []
    axis_labels: list[str] = []
    for raw_code in axis_order:
        if not raw_code:
            raise ValueError("axis order entries cannot be empty")
        code = raw_code.lower()
        axis_label = _normalize_axis_label(code[-1])
        if code[0] in {"d", "p"}:
            joint_types.append("prismatic")
        else:
            joint_types.append("revolute")
        axis_labels.append(axis_label)
    return joint_types, axis_labels


def build_state_symbols(
    axis_order: Sequence[str],
    active_mask: Sequence | None = None,
    symbol_prefix: str = "q",
    derivative_prefix: str = "dq",
) -> tuple[list, list]:
    """Create ``q``/``qd`` symbols automatically from an axis order.

    Parameters
    ----------
    axis_order : sequence of str
        Entries like ``"Dx"``, ``"Dy"``, ``"Dz"``, ``"x"``, ``"y"`` or ``"z"``.
    active_mask : sequence of bool/int, optional
        Mask to drop joints from the calculation (``0`` removes the entry,
        ``1`` keeps it). This mirrors the MATLAB trick of multiplying by a
        matrix of ``0``/``1`` to disable degrees of freedom without editing
        the rest of the tables.
    symbol_prefix : str
        Fallback base name for generalized coordinates when an entry is not a
        valid Python identifier.
    derivative_prefix : str
        Prefix for velocity symbols. Defaults to ``dq`` (e.g., ``dqDx`` or
        ``dq1``).
    """

    if active_mask is None:
        active_mask = [True] * len(axis_order)

    keep_mask = _normalize_mask(active_mask)
    if len(axis_order) != len(keep_mask):
        raise ValueError("active_mask must have the same length as axis_order")

    q_symbols = []
    qd_symbols = []

    for idx, (code, keep) in enumerate(zip(axis_order, keep_mask)):
        if not keep:
            continue
        base_name = code if code.isidentifier() else f"{symbol_prefix}{idx + 1}"
        q_sym = symbols(base_name)
        qd_sym = symbols(f"{derivative_prefix}{base_name}")
        q_symbols.append(q_sym)
        qd_symbols.append(qd_sym)

    return q_symbols, qd_symbols


def sanitize_inertia_tensor(inertia_matrix: Iterable[Iterable]) -> Matrix:
    """Return a symmetric 3x3 inertia tensor.

    Many spreadsheets store inertia tensors with mirrored off-diagonal terms; this
    helper enforces symmetry so the resulting matrix can be safely rotated and used
    in kinetic energy calculations.
    """

    inertia = Matrix(inertia_matrix)
    if inertia.shape != (3, 3):
        raise ValueError("inertia_tensors entries must be 3x3")
    inertia = inertia.applyfunc(sympify)
    return (inertia + inertia.T) / 2


def build_links_from_matrices(
    dh_params: Sequence[Sequence],
    joint_types: Sequence[str],
    masses: Sequence,
    excentricities: Sequence[Sequence],
    inertia_tensors: Sequence[Iterable[Iterable]],
    axis_orders: Sequence[str] | None = None,
    ) -> list[LinkParameters]:
    """Create :class:`LinkParameters` objects from matrix-style inputs.

    Parameters
    ----------
    dh_params : sequence of (a, alpha, d, theta)
        Each entry can be a numeric value or a SymPy expression.
    joint_types : sequence of str
        Order of joints; accepts ``"R"``, ``"P"``/``"D"`` or their long names.
    masses : sequence
        Mass of each link.
    excentricities : sequence of length-3 sequences
        Offsets from the frame origin to the center of mass.
    inertia_tensors : sequence of 3x3 arrays or nested lists
        Inertia tensor about the COM for each link.
    axis_orders : sequence of str, optional
        Axis labels (``x``, ``y`` or ``z``) describing the rotation/translation
        direction of each joint. Entries may optionally be prefixed with ``R``/``P``
        or ``D`` for readability. If omitted, all joints use the standard ``z``
        axis.
    """

    if axis_orders is None:
        axis_orders = ["z"] * len(dh_params)

    if not (len(dh_params) == len(joint_types) == len(masses) == len(excentricities) == len(inertia_tensors) == len(axis_orders)):
        raise ValueError("all inputs must have the same length")

    links: list[LinkParameters] = []
    for dh_row, joint_type, mass, excent, inertia, axis_label in zip(
        dh_params, joint_types, masses, excentricities, inertia_tensors, axis_orders
    ):
        if len(dh_row) != 4:
            raise ValueError("Each DH row must contain (a, alpha, d, theta)")
        a, alpha, d, theta = map(sympify, dh_row)
        com = Matrix([[sympify(excent[0])], [sympify(excent[1])], [sympify(excent[2])]])
        inertia_matrix = sanitize_inertia_tensor(inertia)

        links.append(
            LinkParameters(
                a=a,
                alpha=alpha,
                d=d,
                theta=theta,
                joint_type=_normalize_joint_type(joint_type),
                mass=sympify(mass),
                com=com,
                inertia=inertia_matrix,
                axis=_normalize_axis_label(axis_label),
            )
        )
    return links


def _masked_symbol(base_name: str, template_value):
    """Return a symbolic placeholder or a fixed value following a 0/1 mask.

    The MATLAB workflow often uses matrices de ``0``/``1`` para travar ou
    liberar parâmetros. Aqui replicamos essa ideia: ``0`` remove a variável do
    cálculo, ``1`` cria um símbolo com ``base_name`` e qualquer outro valor é
    retornado como está (após :func:`sympify`).
    """

    if template_value is None:
        return symbols(base_name)

    value = sympify(template_value)
    if value == 0:
        return 0
    if value == 1:
        return symbols(base_name)
    return value


def _normalize_template(length: int, template, fallback_factory):
    if template is None:
        return [fallback_factory(i) for i in range(length)]
    if len(template) != length:
        raise ValueError("template length must match axis_order length")
    return template


def auto_parameters_from_order(
    axis_order: Sequence[str],
    active_mask: Sequence | None = None,
    excentricity_template: Sequence[Sequence] | None = None,
    inertia_template: Sequence[Iterable[Iterable]] | None = None,
    mass_template: Sequence | None = None,
    gravity_symbol: str = "g",
):
    """Gere automaticamente tabelas simbólicas a partir do vetor de rotações.

    Este helper replica o fluxo do código MATLAB: com base no comprimento de
    ``axis_order`` criamos todas as variáveis simbólicas (``q``/``qd``, DH,
    massas, excentricidades e tensores de inércia). Matrizes de ``0``/``1``
    podem ser passadas como *templates* para travar parâmetros específicos,
    exatamente como no workflow original.

    Parameters
    ----------
    axis_order : sequence of str
        Vetor de juntas (``Dx``/``Dy``/``Dz``/``x``/``y``/``z``).
    active_mask : sequence of int/bool, optional
        Máscara 0/1 para ligar/desligar graus de liberdade. Controla tanto a
        criação de ``q``/``qd`` quanto a seleção do símbolo usado no termo
        variável de cada linha DH (``theta`` ou ``d``).
    excentricity_template : sequence of (ex, ey, ez), optional
        Matrizes de ``0``/``1`` (ou valores fixos) usadas como máscara para os
        offsets de centro de massa. ``0`` -> fixa em zero, ``1`` -> símbolo,
        outro valor -> valor fixo.
    inertia_template : sequence of 3x3, optional
        Máscara/valores para cada tensor de inércia. Segue a mesma regra de
        ``0``/``1``/valor.
    mass_template : sequence, optional
        Máscara/valores para as massas de cada elo.
    gravity_symbol : str
        Nome do símbolo escalar usado na gravidade (``g`` por padrão).

    Returns
    -------
    dict
        ``dh_params``, ``masses``, ``excentricities``, ``inertia_tensors``,
        ``q``, ``qd`` e ``gravity`` prontos para uso em
        :func:`equations_of_motion_from_order`.
    """

    if active_mask is None:
        active_mask = [True] * len(axis_order)

    keep_mask = _normalize_mask(active_mask)
    if len(keep_mask) != len(axis_order):
        raise ValueError("active_mask must have the same length as axis_order")

    q, qd = build_state_symbols(axis_order, active_mask=keep_mask)
    q_iter = iter(q)

    def _next_q(keep: bool):
        return next(q_iter) if keep else 0

    dh_params = []
    for idx, (code, keep) in enumerate(zip(axis_order, keep_mask)):
        a_sym = symbols(f"a{idx + 1}")
        alpha_sym = symbols(f"alpha{idx + 1}")
        d_sym = symbols(f"d{idx + 1}")
        theta_sym = symbols(f"theta{idx + 1}")

        # Variável DH de junta: theta para R, d para P/D
        code_lower = code.lower()
        if code_lower.startswith(("d", "p")):
            dh_params.append((a_sym, alpha_sym, _next_q(keep), theta_sym))
        else:
            dh_params.append((a_sym, alpha_sym, d_sym, _next_q(keep)))

    excentricity_template = _normalize_template(
        len(axis_order), excentricity_template, lambda _: (1, 1, 1)
    )
    excentricities = []
    for idx, excent in enumerate(excentricity_template):
        if len(excent) != 3:
            raise ValueError("each excentricity entry must have 3 components")
        excentricities.append(
            (
                _masked_symbol(f"c{idx + 1}x", excent[0]),
                _masked_symbol(f"c{idx + 1}y", excent[1]),
                _masked_symbol(f"c{idx + 1}z", excent[2]),
            )
        )

    inertia_template = _normalize_template(
        len(axis_order), inertia_template, lambda _: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    inertia_tensors = []
    for idx, inertia_mask in enumerate(inertia_template):
        inertia_matrix = Matrix(inertia_mask)
        if inertia_matrix.shape != (3, 3):
            raise ValueError("each inertia template must be 3x3")

        filled_inertia = Matrix.zeros(3)
        labels_matrix = [["xx", "xy", "xz"], ["yx", "yy", "yz"], ["zx", "zy", "zz"]]
        for r in range(3):
            for c in range(3):
                tmpl = inertia_matrix[r, c]
                label = labels_matrix[r][c]
                filled_inertia[r, c] = _masked_symbol(f"I{idx + 1}{label}", tmpl)

        inertia_tensors.append(sanitize_inertia_tensor(filled_inertia))

    mass_template = _normalize_template(len(axis_order), mass_template, lambda _: 1)
    masses = [_masked_symbol(f"m{idx + 1}", tmpl) for idx, tmpl in enumerate(mass_template)]

    gravity = Matrix([[0], [0], [-symbols(gravity_symbol)]])

    return {
        "dh_params": dh_params,
        "masses": masses,
        "excentricities": excentricities,
        "inertia_tensors": inertia_tensors,
        "q": q,
        "qd": qd,
        "gravity": gravity,
    }
