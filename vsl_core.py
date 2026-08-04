"""
VSL (Verilog Semantic Language) - a small notation for describing
register/FSM logic in a compact, unambiguous way instead of relying on
free-text JSON fields. Includes the parser, validator, Verilog renderer,
and the LLM agent that generates VSL from a natural-language prompt.
"""

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent

MODEL = "google-cloud:gemini-2.5-pro"

# --- IR types -------------------------------------------------------------

class PortDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class EdgeType(str, Enum):
    POSEDGE = "posedge"
    NEGEDGE = "negedge"
    SYNC = "sync"
    SYNC_NEG = "sync_neg"
    NONE = "none"


class OpKind(str, Enum):
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NOT = "NOT"
    NAND = "NAND"
    NOR = "NOR"
    XNOR = "XNOR"
    REDUCE_AND = "REDUCE_AND"
    REDUCE_OR = "REDUCE_OR"
    REDUCE_XOR = "REDUCE_XOR"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    SHIFT_LEFT = "SHIFT_LEFT"
    SHIFT_RIGHT = "SHIFT_RIGHT"
    ARITH_SHIFT_RIGHT = "ARITH_SHIFT_RIGHT"
    ROTATE_LEFT = "ROTATE_LEFT"
    ROTATE_RIGHT = "ROTATE_RIGHT"
    CONST = "CONST"
    SIGNAL_REF = "SIGNAL_REF"
    MUX2 = "MUX2"


class Signal(BaseModel):
    id: str
    direction: PortDirection
    width: int = Field(1, ge=1)
    is_register: bool = False
    is_port_declared: bool = Field(
        False,
        description="True if already declared as 'output reg' in the module "
                    "interface, so the renderer shouldn't redeclare it."
    )
    port_is_reg_typed: bool = Field(
        False,
        description="True if the fixed module interface actually declares "
                    "this port as 'output reg' (verified against the real "
                    "interface text, not just the VSL PORT marker). When "
                    "True, the renderer drives this register directly "
                    "(no internal '_r' shadow / assign), since Verilog "
                    "forbids a continuous assign onto a reg-typed port."
    )
    needs_always_block: bool = Field(
        False,
        description="True if this signal is combinational (not clock-driven) "
                    "but declared 'output reg' in the interface -- Verilog "
                    "requires driving such a signal from inside an always "
                    "block, not with a plain 'assign'."
    )


class OperandRef(BaseModel):
    signal: str = ""
    bit_index: Optional[int] = None
    bit_range: Optional[tuple[int, int]] = None
    dynamic_bit_index: Optional[str] = Field(
    None,
    description="Verilog expression used as dynamic bit-select index."
)
    raw_verilog: Optional[str] = Field(
        None,
        description="For a concatenation literal like {in, 16'b0} or "
                    "{out_bytes[23:16], in, 8'b0} -- passed through to the "
                    "renderer as-is rather than treated as a plain signal name."
    )


class Operation(BaseModel):
    target: str = Field("", description="Target signal (empty if target_concat is used)")
    target_concat: Optional[list[str]] = Field(
        None,
        description="For '{a, b} = expr' style assignments, e.g. target_concat=['cout', 'sum']"
    )
    target_bit_index: Optional[int] = Field(
        None,
        description="For a bit-indexed LHS assignment, e.g. 'next_state[3] = expr' "
                    "-- 'target' holds the base signal name ('next_state') and this "
                    "holds the bit index (3)."
    )
    target_bit_range: Optional[tuple[int, int]] = Field(
        None,
        description="For a bit-range LHS assignment, e.g. 'out[7:4] = expr' -- "
                    "'target' holds the base signal name and this holds (hi, lo)."
    )
    op: OpKind
    operands: list[OperandRef] = Field(default_factory=list)
    const_value: Optional[int] = None
    const_is_x: bool = Field(
        False,
        description="True if this CONST operation represents Verilog's "
                    "don't-care/unknown literal (rendered as 1'bx), rather "
                    "than a plain numeric constant."
    )


class ComparisonKind(str, Enum):
    EQ = "EQ"
    NEQ = "NEQ"
    LT = "LT"
    LTE = "LTE"
    GT = "GT"
    GTE = "GTE"


class SimpleCondition(BaseModel):
    signal: str
    bit_index: Optional[int] = None
    bit_range: Optional[tuple[int, int]] = None
    concat_signals: Optional[list[str]] = Field(
        None,
        description="For a concatenation on the left-hand side, e.g. "
                    "'{d,c,b,a}=0' -- holds ['d','c','b','a']. 'signal' is "
                    "unused (empty) when this is set."
    )
    comparison: ComparisonKind = ComparisonKind.EQ
    value: int = 0
    value_signal: Optional[str] = Field(
        None,
        description="For a signal-vs-signal comparison like A=B or a<b -- "
                    "if set, this overrides 'value' as the right-hand side. "
                    "Holds just the base signal name; any bit index/range on "
                    "the right-hand side is in value_signal_bit_index / "
                    "value_signal_bit_range."
    )
    value_signal_bit_index: Optional[int] = None
    value_signal_bit_range: Optional[tuple[int, int]] = None


class Condition(BaseModel):
    terms: list[SimpleCondition] = Field(..., min_length=1)

    @classmethod
    def equals(cls, signal: str, value: int) -> "Condition":
        return cls(terms=[SimpleCondition(signal=signal, comparison=ComparisonKind.EQ, value=value)])


class PriorityBranch(BaseModel):
    condition: Optional[Condition] = None
    result_op: Operation


class FSMStateEncoding(BaseModel):
    state_signal: str
    encoding: dict[str, int]


class RegisterUpdate(BaseModel):
    target_register: str
    clock: str
    edge: EdgeType = EdgeType.POSEDGE

    reset_signal: Optional[str] = None
    reset_edge: EdgeType = EdgeType.NONE
    reset_value: int = 0

    # NEW
    init_value: Optional[int] = Field(
        None,
        description="Optional power-up/initial value for the register."
    )

    init_is_x: bool = Field(
        False,
        description="If True, initialize to X instead of a numeric value."
    )

    next_signal: Optional[str] = Field(
        None,
        description="If set, this register just latches next_signal each "
                    "clock edge (used for the 'state, next' FSM pattern; "
                    "branches stays empty, the logic lives in a COMB block)."
    )

    branches: list[PriorityBranch] = Field(default_factory=list)

class CombBlock(BaseModel):
    """Combinational priority-branch block (e.g. FSM next-state logic, or
    a general-purpose case-like block for any wire -- not only signals fed
    into a REG's NEXT=). Same semantics as RegisterUpdate.branches but
    drives a wire, not a reg."""
    target_signal: str
    branches: list[PriorityBranch] = Field(default_factory=list)


class CircuitIR(BaseModel):
    module_name: str = "unknown"
    function: str = "parsed from VSL"
    signals: list[Signal] = Field(default_factory=list)
    combinational_ops: list[Operation] = Field(default_factory=list)
    comb_blocks: list[CombBlock] = Field(default_factory=list)
    register_updates: list[RegisterUpdate] = Field(default_factory=list)
    fsm_states: list[FSMStateEncoding] = Field(default_factory=list)


class VSLParseError(Exception):
    """syntax problem"""
    pass


class ValidationError(Exception):
    """semantic problem"""
    pass


# --- Parser: VSL text -> CircuitIR, no LLM involved ---------------------

_STATE_ENCODING: dict[str, int] = {}


_VERILOG_SIZED_LITERAL_RE = re.compile(r"^(\d+)'([bdhoBDHO])([0-9a-fA-F_xXzZ]+)$")
_VERILOG_BASE_TO_INT_BASE = {"b": 2, "d": 10, "h": 16, "o": 8}

_X_LITERAL_RE = re.compile(r"^(?:\d+'[a-zA-Z])?[xX]$")

def _verilog_literal_to_int(token: str) -> int:
    """Converts a Verilog sized literal like 2'b00, 8'hFF, or 4'd10 to a
    plain int. Raises ValueError if the token isn't in this format."""
    m = _VERILOG_SIZED_LITERAL_RE.match(token)
    if not m:
        raise ValueError(f"'{token}' is not a Verilog sized literal")
    _, base_char, digits = m.groups()
    base = _VERILOG_BASE_TO_INT_BASE[base_char.lower()]
    return int(digits.replace("_", ""), base)


def _resolve_value(token: str) -> int:
    token = token.strip()
    if token in _STATE_ENCODING:
        return _STATE_ENCODING[token]
    if _VERILOG_SIZED_LITERAL_RE.match(token):
        return _verilog_literal_to_int(token)
    try:
        return int(token, 0)
    except ValueError:
        raise VSLParseError(f"Cannot resolve value '{token}' (not an int and not a known state name)")


_SIZED_CONST_RE = re.compile(r"^\d+'[bdhoBDHO][0-9a-fA-F_xXzZ]+$")
_REPLICATION_RE = re.compile(r"^(\d+)\{(.+)\}$")


def _is_valid_concat_part(part: str) -> bool:
    part = part.strip()

    if _SIZED_CONST_RE.match(part):
        return True
    
    if re.match(r"^\w+(\[[^\]]+\])?$", part):
        return True


    if part.startswith("{") and part.endswith("}"):
        inner_stripped = part[1:-1].strip()
        inner_group_parts = _split_top_level_commas(inner_stripped)
        if len(inner_group_parts) > 1:
            return all(_is_valid_concat_part(p.strip()) for p in inner_group_parts)
        part = inner_stripped

    m = _REPLICATION_RE.match(part)
    if m:
        inner = m.group(2).strip()
        inner_parts = _split_top_level_commas(inner)
        return all(_is_valid_concat_part(p.strip()) for p in inner_parts)

    return False

def _parse_operand(token: str, extra_ops: Optional[list] = None) -> OperandRef:
    token = token.strip()

    # concatenation
    if token.startswith("{") and token.endswith("}"):
        inner = token[1:-1]
        parts = [p.strip() for p in _split_top_level_commas(inner)]

        if not parts:
            raise VSLParseError(f"Cannot parse concatenation '{token}'")

        rendered_parts = []

        for part in parts:
            if _is_valid_concat_part(part):
                rendered_parts.append(part)
                continue

            # a more complex expression inside the concat
            aux_name = _next_aux_signal()
            aux_op = _parse_expression(
                aux_name,
                part,
                extra_ops=extra_ops
            )

            if extra_ops is not None:
                extra_ops.append(aux_op)

            rendered_parts.append(aux_name)

        return OperandRef(
            raw_verilog="{" + ", ".join(rendered_parts) + "}"
        )

    # sized constant (8'hFF, 4'b1010, ...)
    if _SIZED_CONST_RE.match(token):
        return OperandRef(raw_verilog=token)

    # bit range: foo[7:0]
    m = re.match(r"^(\w+)\[(\d+):(\d+)\]$", token)
    if m:
        hi = int(m.group(2))
        lo = int(m.group(3))

        if hi < lo:
            hi, lo = lo, hi

        return OperandRef(
            signal=m.group(1),
            bit_range=(hi, lo)
        )

    # fixed bit index: foo[3]
    m = re.match(r"^(\w+)\[(\d+)\]$", token)
    if m:
        return OperandRef(
            signal=m.group(1),
            bit_index=int(m.group(2))
        )

    # dynamic/expression bit index
    m = re.match(r"^(\w+)\[(.+)\]$", token)
    if m:
        sig = m.group(1)
        idx_expr = m.group(2).strip()

        # simple index, just a signal name
        if re.match(r"^\w+$", idx_expr):
            return OperandRef(
                signal=sig,
                dynamic_bit_index=idx_expr
            )

        # index built from a more complex expression
        if extra_ops is not None:
            aux_name = _next_aux_signal()

            aux_op = _parse_expression(
                aux_name,
                idx_expr,
                extra_ops=extra_ops
            )

            extra_ops.append(aux_op)

            return OperandRef(
                signal=sig,
                dynamic_bit_index=aux_name
            )

        # fallback -- let the renderer handle the expression directly
        return OperandRef(
            signal=sig,
            dynamic_bit_index=idx_expr
        )

    # plain signal
    if re.match(r"^\w+$", token):
        return OperandRef(signal=token)

    raise VSLParseError(f"Cannot parse operand '{token}'")

_BINARY_OPS = {
    "<<r": OpKind.ROTATE_LEFT,
    ">>r": OpKind.ROTATE_RIGHT,
    "<<": OpKind.SHIFT_LEFT,
    ">>>": OpKind.ARITH_SHIFT_RIGHT,
    ">>": OpKind.SHIFT_RIGHT,
    "+": OpKind.ADD,
    "-": OpKind.SUB,
    "*": OpKind.MUL,
    "&": OpKind.AND,
    "|": OpKind.OR,
    "^": OpKind.XOR,
}
_BINARY_OP_TOKENS_BY_LENGTH = sorted(_BINARY_OPS.keys(), key=len, reverse=True)


_aux_signal_counter = 0


def _next_aux_signal() -> str:
    global _aux_signal_counter
    _aux_signal_counter += 1
    return f"__aux_{_aux_signal_counter}"


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        matches_at_end = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    matches_at_end = False
                    break
        if matches_at_end:
            s = s[1:-1].strip()
        else:
            break
    return s


def _split_top_level(s: str, seps: str) -> Optional[tuple[str, str, str]]:
    """
    Split on the first separator character found at top level
    (not inside parentheses or concatenations).
    """

    paren_depth = 0
    brace_depth = 0

    for i, ch in enumerate(s):
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
        elif paren_depth == 0 and brace_depth == 0 and ch in seps:
            return s[:i], ch, s[i + 1:]

    return None

def _split_top_level_token(s: str, token: str) -> Optional[tuple[str, str, str]]:
    """Like _split_top_level, but splits on a full (possibly multi-char)
    token such as '<<' or '!=', only at paren/brace-depth 0."""
    depth = 0
    i = 0
    n = len(token)
    while i < len(s):
        ch = s[i]
        if ch in "({":
            depth += 1
        elif ch in ")}":
            depth -= 1
        elif depth == 0 and s[i:i + n] == token:
            return s[:i], token, s[i + n:]
        i += 1
    return None


def _split_top_level_commas(s: str) -> list[str]:
    """Splits a concatenation's inner text on commas, ignoring commas that
    are nested inside parens or braces (e.g. a ternary member's own
    sub-expressions)."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts

def _split_top_level_ands(s: str) -> list[str]:
    parts = []
    depth = 0
    start = 0

    for i, ch in enumerate(s):
        if ch in "({":
            depth += 1
        elif ch in ")}":
            depth -= 1
        elif ch == "&" and depth == 0:
            parts.append(s[start:i].strip())
            start = i + 1

    parts.append(s[start:].strip())
    return parts

def _parse_expression(target: str, expr: str, extra_ops: Optional[list] = None) -> Operation:
    """Parses the right-hand side of '->' or '='. extra_ops collects
    auxiliary ops created for nested ternaries (each nested MUX becomes
    its own Operation on a synthetic signal, referenced by the outer one)."""
    expr = _strip_outer_parens(expr.strip())

    if expr in _STATE_ENCODING:
        return Operation(target=target, op=OpKind.CONST, operands=[], const_value=_STATE_ENCODING[expr])

    # don't-care literal: bare x or a sized form like 1'bx, 8'hx --
    # always renders as 1'bx no matter how it was written
    if _X_LITERAL_RE.match(expr):
        return Operation(target=target, op=OpKind.CONST, operands=[], const_is_x=True)

    # unary NOT
    if expr.startswith("~"):
        inner = expr[1:].strip()
        if not any(_split_top_level_token(inner, tok) for tok in _BINARY_OP_TOKENS_BY_LENGTH) \
                and _split_top_level(inner, "?") is None:
            stripped_inner = _strip_outer_parens(inner)
  
            if any(_split_top_level_token(stripped_inner, tok) for tok in _BINARY_OP_TOKENS_BY_LENGTH):
                aux_name = _next_aux_signal()
                aux_op = _parse_expression(aux_name, stripped_inner, extra_ops=extra_ops)
                if extra_ops is not None:
                    extra_ops.append(aux_op)
                return Operation(target=target, op=OpKind.NOT, operands=[OperandRef(signal=aux_name)])
            return Operation(target=target, op=OpKind.NOT, operands=[_parse_operand(stripped_inner)])

    # unary reduction: &<signal>, |<signal>, ^<signal> -- ANDs/ORs/XORs
    _REDUCE_OP_KIND = {"&": OpKind.REDUCE_AND, "|": OpKind.REDUCE_OR, "^": OpKind.REDUCE_XOR}
    if expr and expr[0] in _REDUCE_OP_KIND:
        inner = expr[1:].strip()
        if not any(_split_top_level_token(inner, tok) for tok in _BINARY_OP_TOKENS_BY_LENGTH) \
                and _split_top_level(inner, "?") is None:
            return Operation(target=target, op=_REDUCE_OP_KIND[expr[0]],
                              operands=[_parse_operand(_strip_outer_parens(inner))])

    # if the whole expression is a concatenation, e.g. {state=D?1:0,
    # state=C?1:0}, we need to catch that before checking for a ternary,
    # otherwise the ? inside the braces gets mistaken for a top-level one
    if expr.startswith("{") and expr.endswith("}"):
        depth = 0
        matches_at_end = True
        for i, ch in enumerate(expr):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and i != len(expr) - 1:
                    matches_at_end = False
                    break
        if matches_at_end:
            return Operation(target=target, op=OpKind.SIGNAL_REF,
                              operands=[_parse_operand(expr, extra_ops=extra_ops)])

    split = _split_top_level(expr, "?")
    if split is not None:
        cond_part, _, rest = split

        rest_split = _split_top_level(rest, ":")
        if rest_split is None:
            raise VSLParseError(f"Ternary missing ':' in '{expr}'")

        true_part, _, false_part = rest_split
        cond_part = cond_part.strip()

        term_strs = []

        for term_text in _split_top_level_ands(cond_part):
            term = _parse_comparison_term(term_text)

            term_strs.append(
                _simple_condition_to_verilog(term).replace(" ", "")
            )

        select_encoded = "&&".join(term_strs)

        def _branch_operand(branch_text: str) -> OperandRef:
            branch_text = _strip_outer_parens(branch_text.strip())

            if _split_top_level(branch_text, "?") is not None or any(
                op in branch_text for op in _BINARY_OP_TOKENS_BY_LENGTH
            ):
                aux_name = _next_aux_signal()
                aux_op = _parse_expression(
                    aux_name,
                    branch_text,
                    extra_ops=extra_ops
                )

                if extra_ops is not None:
                    extra_ops.append(aux_op)

                return OperandRef(signal=aux_name)

            if _X_LITERAL_RE.match(branch_text):
                return OperandRef(signal="__const_x")

            if re.match(r"^\d+$", branch_text):
                return OperandRef(signal=f"__const_{branch_text}")

            if branch_text in _STATE_ENCODING:
                return OperandRef(
                    signal=f"__const_{_STATE_ENCODING[branch_text]}"
                )

            return _parse_operand(
                branch_text,
                extra_ops=extra_ops
            )

        return Operation(
            target=target,
            op=OpKind.MUX2,
            operands=[
                OperandRef(signal=select_encoded),
                _branch_operand(true_part),
                _branch_operand(false_part),
            ],
        )

    for op_token in _BINARY_OP_TOKENS_BY_LENGTH:
        split = _split_top_level_token(expr, op_token)
        if split is not None:
            lhs, _, rhs = split
            # collect all top-level parts for chains like a+b+cin
            parts = [lhs]
            remainder = rhs
            while True:
                next_split = _split_top_level_token(remainder, op_token)
                if next_split is None:
                    parts.append(remainder)
                    break
                part, _, remainder = next_split
                parts.append(part)
            if len(parts) >= 2 and all(p.strip() for p in parts):
                operands = []
                for part in parts:
                    part = part.strip()
                    part = _strip_outer_parens(part)
                    if re.match(r"^\d+$", part):
                        operands.append(OperandRef(signal=f"__const_{part}"))
                    elif part.startswith("~"):
                        aux_name = _next_aux_signal()
                        aux_op = _parse_expression(aux_name, part, extra_ops=extra_ops)
                        if extra_ops is not None:
                            extra_ops.append(aux_op)
                        operands.append(OperandRef(signal=aux_name))
                    elif _split_top_level(part, "?") is not None or any(
                        _split_top_level_token(part, tok) is not None for tok in _BINARY_OP_TOKENS_BY_LENGTH
                    ):
                        aux_name = _next_aux_signal()
                        aux_op = _parse_expression(aux_name, part, extra_ops=extra_ops)
                        if extra_ops is not None:
                            extra_ops.append(aux_op)
                        operands.append(OperandRef(signal=aux_name))
                    else:
                        operands.append(_parse_operand(part, extra_ops=extra_ops))
                return Operation(target=target, op=_BINARY_OPS[op_token], operands=operands)

    return Operation(target=target, op=OpKind.SIGNAL_REF, operands=[_parse_operand(expr, extra_ops=extra_ops)])


_COMPARISON_TOKENS_BY_LENGTH = ["!=", "<=", ">=", "=", "<", ">"]
_COMPARISON_KIND_BY_TOKEN = {
    "=": ComparisonKind.EQ,
    "!=": ComparisonKind.NEQ,
    "<": ComparisonKind.LT,
    "<=": ComparisonKind.LTE,
    ">": ComparisonKind.GT,
    ">=": ComparisonKind.GTE,
}


def _parse_rhs_signal_ref(rhs: str) -> tuple[Optional[str], Optional[int], Optional[tuple[int, int]]]:
    """Parses the right-hand side of a signal-vs-signal comparison, which
    may itself carry a bit index or bit range, e.g. 'b', 'b[7]', or
    'b[7:4]'. Returns (signal, bit_index, bit_range), any of which may be
    None. Returns (None, None, None) if rhs isn't a signal reference at
    all (e.g. it's a plain number, handled elsewhere)."""
    m = re.match(r"^(\w+)\[(\d+):(\d+)\]$", rhs)
    if m:
        return m.group(1), None, (int(m.group(2)), int(m.group(3)))
    m = re.match(r"^(\w+)\[(\d+)\]$", rhs)
    if m:
        return m.group(1), int(m.group(2)), None
    if re.match(r"^\w+$", rhs):
        return rhs, None, None
    return None, None, None


def _parse_comparison_term(term_text: str) -> SimpleCondition:
    """Parses a single condition like 'signal=value', 'signal!=value',
    'signal[bit]<value', a signal-vs-signal comparison like 'a=b', 'x<y',
    'a[7]=b[7]', or a concatenation on the left like '{d,c,b,a}=0'.
    Longest-token-first so '<=' doesn't get split as '<' + '='."""
    term_text = term_text.strip()
    for token in _COMPARISON_TOKENS_BY_LENGTH:
        split = _split_top_level_token(term_text, token)

        if split is None:
            continue

        lhs, _, rhs = split
        comparison = _COMPARISON_KIND_BY_TOKEN[token]
        lhs = lhs.strip()
        rhs = rhs.strip()

        value: Optional[int] = None
        value_signal: Optional[str] = None
        value_signal_bit_index: Optional[int] = None
        value_signal_bit_range: Optional[tuple[int, int]] = None
        try:
            value = _resolve_value(rhs)
        except VSLParseError:
            ref_signal, ref_bit_index, ref_bit_range = _parse_rhs_signal_ref(rhs)
            if ref_signal is not None:
                value_signal = ref_signal
                value_signal_bit_index = ref_bit_index
                value_signal_bit_range = ref_bit_range
                value = 0
            else:
                raise VSLParseError(f"Cannot resolve right-hand side '{rhs}' in condition '{term_text}'")

        if lhs.startswith("{") and lhs.endswith("}"):
            names = [p.strip() for p in _split_top_level_commas(lhs[1:-1])]
            if not names or not all(re.match(r"^\w+$", n) for n in names):
                raise VSLParseError(f"Cannot parse concatenation '{lhs}' in condition '{term_text}'")
            return SimpleCondition(
                signal="", concat_signals=names,
                comparison=comparison, value=value, value_signal=value_signal,
                value_signal_bit_index=value_signal_bit_index,
                value_signal_bit_range=value_signal_bit_range,
            )

        m = re.match(r"^(\w+)\[(\d+):(\d+)\]$", lhs)
        if m:
            return SimpleCondition(
                signal=m.group(1), bit_range=(int(m.group(2)), int(m.group(3))),
                comparison=comparison, value=value, value_signal=value_signal,
                value_signal_bit_index=value_signal_bit_index,
                value_signal_bit_range=value_signal_bit_range,
            )
        m = re.match(r"^(\w+)\[(\d+)\]$", lhs)
        if m:
            return SimpleCondition(
                signal=m.group(1), bit_index=int(m.group(2)),
                comparison=comparison, value=value, value_signal=value_signal,
                value_signal_bit_index=value_signal_bit_index,
                value_signal_bit_range=value_signal_bit_range,
            )
        m = re.match(r"^\w+$", lhs)
        if m:
            return SimpleCondition(
                signal=lhs, comparison=comparison, value=value, value_signal=value_signal,
                value_signal_bit_index=value_signal_bit_index,
                value_signal_bit_range=value_signal_bit_range,
            )
    raise VSLParseError(f"Cannot parse condition term '{term_text}'")


def _parse_condition(cond_text: str) -> Optional[Condition]:
    cond_text = _strip_outer_parens(cond_text.strip())

    if cond_text == "*":
        return None

    terms = [
        _parse_comparison_term(term_text)
        for term_text in _split_top_level_ands(cond_text)
    ]

    return Condition(terms=terms)

def _is_new_block_start(line: str) -> bool:
    """True if `line` starts a new top-level VSL construct (REG/COMB/
    STATES), so a branch-continuation loop knows to stop even if the line
    happens to contain '->' (e.g. a REG line's reset clause)."""
    return line.startswith("REG ") or line.startswith("COMB ") or line.startswith("STATES:")


def _needs_signal_registration(operand: OperandRef) -> bool:
    """Whether this operand refers to a plain signal name that should be
    registered via _ensure_signal -- false for constants, encoded
    comparisons, and raw_verilog pass-through operands (concatenations),
    whose inner signal names are assumed to already exist."""
    if operand.raw_verilog is not None:
        return False
    if operand.signal.startswith("__const_"):
        return False
    if "==" in operand.signal:
        return False
    return True


_FENCE_LINE_RE = re.compile(r"^(```|'''|~~~)[\w-]*$")


def _strip_fence_lines(text: str) -> str:
    """Strips leading/trailing 'fence' lines the model sometimes wraps its
    VSL output in (```, ```verilog, ''', ~~~, etc.), even though VSLOutput
    is a structured field and shouldn't need it. This is intentionally
    narrow -- it only recognizes known fence markers (optionally followed
    by a language tag), so it can't accidentally eat a real VSL line."""
    lines = text.strip("\n").splitlines()
    while lines and _FENCE_LINE_RE.match(lines[0].strip()):
        lines.pop(0)
    while lines and _FENCE_LINE_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines)


_OUTPUT_REG_RE = re.compile(r"output\s+reg\s*(?:\[[^\]]*\]\s*)?(\w+)")
_PORT_DECL_RE = re.compile(
    r"(?:input|output)\s+(?:reg\s+)?(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*)?(\w+)"
)


def _find_output_reg_names(module_interface: str) -> set[str]:
    """Scans a raw Verilog module interface for 'output reg <name>' ports
    -- these can't be driven with 'assign', only from inside an always
    block, even for purely combinational logic."""
    return set(_OUTPUT_REG_RE.findall(module_interface or ""))


def _find_port_widths(module_interface: str) -> dict[str, int]:
    """Scans a raw Verilog module interface for every input/output port
    declaration and returns {port_name: width}. A port with no [hi:lo]
    range is 1 bit wide. This lets us correctly size signals that are
    only ever used as plain operands (e.g. inputs to a combinational
    expression) and never declared via a VSL REG line, which would
    otherwise default to width=1 regardless of their real interface width."""
    widths: dict[str, int] = {}
    for hi, lo, name in _PORT_DECL_RE.findall(module_interface or ""):
        if hi and lo:
            widths[name] = abs(int(hi) - int(lo)) + 1
        else:
            widths[name] = 1
    return widths


def parse_vsl(text: str, module_interface: str = "") -> CircuitIR:
    """Parses VSL text into a CircuitIR. Pure parsing, no LLM involved.
    module_interface, if given, is the raw fixed Verilog module interface
    text -- used to (1) detect which ports are declared 'output reg', so
    combinational assignments to them render as always @(*) blocks instead
    of illegal 'assign' statements, and (2) correctly size every port
    signal (input or output) from its real declared width, instead of
    defaulting plain operands to width=1."""
    global _STATE_ENCODING, _aux_signal_counter
    _STATE_ENCODING = {}
    _aux_signal_counter = 0
    output_reg_names = _find_output_reg_names(module_interface)
    port_widths = _find_port_widths(module_interface)

    text = _strip_fence_lines(text)

    ir = CircuitIR()
    signals_seen: dict[str, Signal] = {}
    next_signal_names: set[str] = set()

    def _ensure_signal(name: str, is_register: bool = False, width: int = 1):
        if name not in signals_seen:
            # a REG line always gives an explicit width; for plain
            # operands, prefer the real interface width if we know it,
            # instead of just defaulting to 1
            effective_width = width if width != 1 else port_widths.get(name, width)
            signals_seen[name] = Signal(id=name, direction=PortDirection.INPUT, width=effective_width, is_register=is_register)
        elif is_register:
            # this is an authoritative REG declaration 
            signals_seen[name].is_register = True
            signals_seen[name].width = width

    def _ensure_condition_signals(term: "SimpleCondition") -> None:
        """Registers every signal a condition term refers to, whether it's
        a plain signal, a concatenation like '{d,c,b,a}', or a signal (with
        optional bit index) on the right-hand side of a comparison."""
        if term.concat_signals is not None:
            for name in term.concat_signals:
                _ensure_signal(name)
        elif term.signal:
            _ensure_signal(term.signal)
        if term.value_signal is not None:
            _ensure_signal(term.value_signal)

    def _estimate_raw_verilog_width(raw: str) -> int:
        """Best-effort width for a raw_verilog operand (a concatenation or
        replication literal like '{5{a}}' or '{a, b, c}'), so ops that
        include one as an operand don't fall back to a silent 1-bit
        default. Falls back to 1 for anything it can't confidently size
        (e.g. a nested expression) rather than guessing wrong."""
        raw = raw.strip()
        if _SIZED_CONST_RE.match(raw):
            return int(raw.split("'")[0])
        if not (raw.startswith("{") and raw.endswith("}")):
            return 1
        inner = raw[1:-1].strip()
        m = _REPLICATION_RE.match(inner)
        if m:
            count = int(m.group(1))
            body = m.group(2).strip()
            body_width = _estimate_raw_verilog_width(body) if body.startswith("{") else _signal_or_1_width(body)
            return count * body_width
        total = 0
        for part in _split_top_level_commas(inner):
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                total += _estimate_raw_verilog_width(part)
            else:
                m2 = _REPLICATION_RE.match(part)
                if m2:
                    total += int(m2.group(1)) * _signal_or_1_width(m2.group(2).strip())
                else:
                    total += _signal_or_1_width(part)
        return total or 1

    def _signal_or_1_width(token: str) -> int:
        token = token.lstrip("~").strip()
        m = _SIZED_CONST_RE.match(token)
        if m:
            return int(token.split("'")[0])
        m = re.match(r"^(\w+)\[(\d+):(\d+)\]$", token)
        if m:
            return abs(int(m.group(2)) - int(m.group(3))) + 1
        m = re.match(r"^\w+$", token)
        if m:
            sig = signals_seen.get(token)
            if sig is not None:
                return sig.width
        return 1

    def _infer_op_width(op: Operation) -> int:
        """Infers an aux signal's width as the widest operand referenced
        in its Operation, so a ternary/binary-op result (e.g. picking
        between two 8-bit signals) isn't left at the default 1-bit width,
        which would silently truncate it in the rendered Verilog."""
        best = 1
        for operand in op.operands:
            if operand.raw_verilog is not None:
                best = max(best, _estimate_raw_verilog_width(operand.raw_verilog))
                continue
            if not operand.signal:
                continue
            sig = signals_seen.get(operand.signal)
            if sig is not None:
                best = max(best, sig.width)
        return best


    lines = [ln for ln in text.strip().splitlines() if ln.strip() and not ln.strip().startswith("#")]

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("STATES:"):
            body = line[len("STATES:"):].strip()
            for pair in body.split(","):
                name, val = pair.split("=")
                val = val.strip()
                try:
                    _STATE_ENCODING[name.strip()] = _verilog_literal_to_int(val)
                except ValueError:
                    _STATE_ENCODING[name.strip()] = int(val)
            i += 1
            continue

        if line.startswith("REG "):
            # PORT and NEXT=<name> can show up in either order in the
            # model's output ('... PORT NEXT=q_next' or '... NEXT=q_next
            # PORT') -- if PORT lands in the middle, move it to the end,
            # since the regex below only matches NEXT= followed by PORT
            m_port_mid = re.match(r"^(.*)\bPORT\b\s+(NEXT=\w+)\s*$", line)
            if m_port_mid:
                line = f"{m_port_mid.group(1).rstrip()} {m_port_mid.group(2)} PORT"

            m = re.match(
                r"^REG\s+(\w+)(?:\[(\d+)\])?\s+@(\w+)\.(pos|neg)"
                r"(?:\s+(\w+)\.(pos|neg|sync|syncpos|syncneg)->(\w+(?:'[bdhoBDHO][0-9a-fA-F_xXzZ]+)?))?"
                r"(?:\s+INIT=(x|\d+(?:'[bdhoBDHO][0-9a-fA-F_xXzZ]+)?))?"
                r"(?:\s+NEXT=(\w+))?"
                r"(\s+PORT)?$",
                line,
            )
            if not m:
                raise VSLParseError(f"Cannot parse REG line: '{line}'")
            (reg_name,width,clock,edge,rst_sig,rst_edge,rst_val_tok,init_val_tok,next_sig,port_flag,) = m.groups()
            width = int(width) if width else 1
            _ensure_signal(reg_name, is_register=True, width=width)
            if port_flag:
                signals_seen[reg_name].is_port_declared = True
                if reg_name in output_reg_names:
                    signals_seen[reg_name].port_is_reg_typed = True
            _ensure_signal(clock)
            if rst_sig:
                _ensure_signal(rst_sig)
            rst_val = _resolve_value(rst_val_tok) if rst_val_tok is not None else 0
            init_value = None
            init_is_x = False

            if init_val_tok is not None:
                if init_val_tok.lower() == "x":
                    init_is_x = True
                else:
                    init_value = _resolve_value(init_val_tok)
            def _edge_type_for(tok: Optional[str]) -> EdgeType:
                if tok == "pos":
                    return EdgeType.POSEDGE
                if tok == "neg":
                    return EdgeType.NEGEDGE
                if tok in ("sync", "syncpos"):
                    return EdgeType.SYNC
                if tok == "syncneg":
                    return EdgeType.SYNC_NEG
                return EdgeType.NONE

            if next_sig:
                # 'state, next' pattern: this register just latches
                # next_sig every clock edge. The actual priority logic for
                # next_sig lives in a separate COMB block parsed below.
                _ensure_signal(next_sig)
                next_signal_names.add(next_sig)
                ir.register_updates.append(RegisterUpdate(
                    target_register=reg_name,
                    clock=clock,
                    edge=EdgeType.POSEDGE if edge == "pos" else EdgeType.NEGEDGE,
                    reset_signal=rst_sig,
                    reset_edge=_edge_type_for(rst_edge) if rst_sig else EdgeType.NONE,
                    reset_value=rst_val,

                    init_value=init_value,
                    init_is_x=init_is_x,

                    next_signal=next_sig,
                    branches=[],
                ))
                i += 1
                continue

            branches: list[PriorityBranch] = []
            i += 1
            while i < len(lines):
                branch_line = lines[i].strip()
                if _is_new_block_start(branch_line):
                    break
                stripped = branch_line[1:].strip() if branch_line.startswith("|") else branch_line
                if "->" not in stripped:
                    break
                cond_text, expr_text = stripped.split("->", 1)
                condition = _parse_condition(cond_text)
                if condition is not None:
                    for term in condition.terms:
                        _ensure_condition_signals(term)
                branch_extra_ops: list = []
                result_op = _parse_expression(reg_name, expr_text.strip(), extra_ops=branch_extra_ops)
                for aux_op in branch_extra_ops:
                    for aux_operand in aux_op.operands:
                        if _needs_signal_registration(aux_operand):
                            _ensure_signal(aux_operand.signal)
                    _ensure_signal(aux_op.target, width=_infer_op_width(aux_op))
                    ir.combinational_ops.append(aux_op)
                for operand in result_op.operands:
                    if _needs_signal_registration(operand):
                        _ensure_signal(operand.signal)
                branches.append(PriorityBranch(condition=condition, result_op=result_op))
                i += 1

            ir.register_updates.append(RegisterUpdate(
                target_register=reg_name,
                clock=clock,
                edge=EdgeType.POSEDGE if edge == "pos" else EdgeType.NEGEDGE,
                reset_signal=rst_sig,
                reset_edge=_edge_type_for(rst_edge) if rst_sig else EdgeType.NONE,
                reset_value=rst_val,

                init_value=init_value,
                init_is_x=init_is_x,

                branches=branches,
            ))
            continue

        if line.startswith("COMB "):
            comb_target = line[len("COMB "):].strip()
            _ensure_signal(comb_target)

            comb_branches: list[PriorityBranch] = []
            i += 1
            while i < len(lines):
                branch_line = lines[i].strip()
                if _is_new_block_start(branch_line):
                    break
                stripped = branch_line[1:].strip() if branch_line.startswith("|") else branch_line
                if "->" not in stripped:
                    break
                cond_text, expr_text = stripped.split("->", 1)
                condition = _parse_condition(cond_text)
                if condition is not None:
                    for term in condition.terms:
                        _ensure_condition_signals(term)
                comb_extra_ops_local: list = []
                result_op = _parse_expression(comb_target, expr_text.strip(), extra_ops=comb_extra_ops_local)
                for aux_op in comb_extra_ops_local:
                    for aux_operand in aux_op.operands:
                        if _needs_signal_registration(aux_operand):
                            _ensure_signal(aux_operand.signal)
                    _ensure_signal(aux_op.target, width=_infer_op_width(aux_op))
                    ir.combinational_ops.append(aux_op)
                for operand in result_op.operands:
                    if _needs_signal_registration(operand):
                        _ensure_signal(operand.signal)
                comb_branches.append(PriorityBranch(condition=condition, result_op=result_op))
                i += 1

            ir.comb_blocks.append(CombBlock(target_signal=comb_target, branches=comb_branches))
            continue

        if "=" in line and "->" not in line and not line.startswith("|"):
            split_result = _split_top_level_token(line, "=")
            if split_result is None:
                raise VSLParseError(f"Cannot parse assignment line: '{line}'")
            target, _, expr_text = split_result
            target = target.strip()
            expr_text = expr_text.strip()

            concat_match = re.match(r"^\{([\w\s,]+)\}$", target)
            if concat_match:
                target_names = [t.strip() for t in concat_match.group(1).split(",")]
                for name in target_names:
                    _ensure_signal(name)
                    if name in output_reg_names or name in next_signal_names:
                        signals_seen[name].needs_always_block = True
                comb_extra_ops: list = []
                # Parse the RHS with a placeholder target -- the real
                # multi-signal target is attached via target_concat below.
                op = _parse_expression("", expr_text, extra_ops=comb_extra_ops)
                op = op.model_copy(update={"target": "", "target_concat": target_names})
                for aux_op in comb_extra_ops:
                    for aux_operand in aux_op.operands:
                        if _needs_signal_registration(aux_operand):
                            _ensure_signal(aux_operand.signal)
                    _ensure_signal(aux_op.target, width=_infer_op_width(aux_op))
                    ir.combinational_ops.append(aux_op)
                for operand in op.operands:
                    if _needs_signal_registration(operand):
                        _ensure_signal(operand.signal)
                ir.combinational_ops.append(op)
                i += 1
                continue

            # plain single-signal target, possibly with a bit index or range
            target_bit_index: Optional[int] = None
            target_bit_range: Optional[tuple[int, int]] = None
            base_name = target
            m_range = re.match(r"^(\w+)\[(\d+):(\d+)\]$", target)
            m_bit = re.match(r"^(\w+)\[(\d+)\]$", target)
            if m_range:
                base_name = m_range.group(1)
                hi = int(m_range.group(2))
                lo = int(m_range.group(3))
                if hi < lo:
                    hi, lo = lo, hi
                target_bit_range = (hi, lo)
            elif m_bit:
                base_name = m_bit.group(1)
                target_bit_index = int(m_bit.group(2))

            _ensure_signal(base_name)
            if base_name in output_reg_names or base_name in next_signal_names:
                signals_seen[base_name].needs_always_block = True
            comb_extra_ops: list = []
            op = _parse_expression(base_name, expr_text, extra_ops=comb_extra_ops)
            if target_bit_index is not None or target_bit_range is not None:
                op = op.model_copy(update={
                    "target_bit_index": target_bit_index,
                    "target_bit_range": target_bit_range,
                })
            for aux_op in comb_extra_ops:
                for aux_operand in aux_op.operands:
                    if _needs_signal_registration(aux_operand):
                        _ensure_signal(aux_operand.signal)
                _ensure_signal(aux_op.target, width=_infer_op_width(aux_op))
                ir.combinational_ops.append(aux_op)
            for operand in op.operands:
                if _needs_signal_registration(operand):
                    _ensure_signal(operand.signal)
            ir.combinational_ops.append(op)
            i += 1
            continue

        raise VSLParseError(f"Unrecognized VSL line: '{line}'")

    ir.signals = list(signals_seen.values())
    if _STATE_ENCODING:
        registers_using_states = set()
        for ru in ir.register_updates:
            for branch in ru.branches:
                if branch.condition:
                    for term in branch.condition.terms:
                        if term.signal == ru.target_register:
                            registers_using_states.add(ru.target_register)
        for reg_name in registers_using_states:
            ir.fsm_states.append(FSMStateEncoding(state_signal=reg_name, encoding=dict(_STATE_ENCODING)))

    return ir


def validate_circuit(ir: CircuitIR) -> list[str]:
    problems: list[str] = []
    signal_ids = {s.id for s in ir.signals}
    signals_by_id = {s.id: s for s in ir.signals}

    def check_signal_exists(sig_id: str, ctx: str):
        if not sig_id or sig_id.startswith("__const_") or _ENCODED_COMPARISON_RE.search(sig_id):
            return
        if sig_id not in signal_ids:
            problems.append(f"{ctx}: unknown signal '{sig_id}'")

    def check_condition_term_signals(term: "SimpleCondition", ctx: str):
        if term.concat_signals is not None:
            for sig in term.concat_signals:
                check_signal_exists(sig, ctx)
        else:
            check_signal_exists(term.signal, ctx)
        if term.value_signal is not None:
            check_signal_exists(term.value_signal, f"{ctx} value_signal")

    for op in ir.combinational_ops:
        if op.target_concat:
            for sig in op.target_concat:
                check_signal_exists(sig, "combinational_ops target_concat")
        elif op.target:
            check_signal_exists(op.target, "combinational_ops target")
        else:
            problems.append("combinational_ops entry has neither target nor target_concat set")
        for operand in op.operands:
            check_signal_exists(operand.signal, f"combinational_ops operand of {op.target or op.target_concat}")

    driven_registers = set()
    for ru in ir.register_updates:
        check_signal_exists(ru.target_register, "register_updates target")
        sig = signals_by_id.get(ru.target_register)
        if sig and not sig.is_register:
            problems.append(f"'{ru.target_register}' used as register target but Signal.is_register is False")
        if ru.target_register in driven_registers:
            problems.append(f"'{ru.target_register}' has more than one RegisterUpdate block (multiple drivers)")
        driven_registers.add(ru.target_register)

        if ru.next_signal is not None:
            check_signal_exists(ru.next_signal, f"'{ru.target_register}' next_signal")
            continue

        default_branches = [b for b in ru.branches if b.condition is None]
        if len(default_branches) > 1:
            problems.append(f"'{ru.target_register}' has {len(default_branches)} default branches; must have at most 1")
        if not default_branches:
            problems.append(f"'{ru.target_register}' has no default/hold branch")

        for i, branch in enumerate(ru.branches):
            if branch.condition is not None:
                for term in branch.condition.terms:
                    check_condition_term_signals(
                        term,
                        f"'{ru.target_register}' branch {i} condition"
                    )
            check_signal_exists(branch.result_op.target, f"'{ru.target_register}' branch {i} result target")

    next_signals_expected = {ru.next_signal for ru in ir.register_updates if ru.next_signal}
    comb_targets_declared = {cb.target_signal for cb in ir.comb_blocks}
    comb_targets_declared |= {
        op.target for op in ir.combinational_ops
        if op.target and (op.target_bit_index is not None or op.target_bit_range is not None)
    }
    missing_comb_blocks = next_signals_expected - comb_targets_declared
    if missing_comb_blocks:
        problems.append(f"REG uses NEXT= for {missing_comb_blocks} but no matching COMB block was found")

    for cb in ir.comb_blocks:
        check_signal_exists(cb.target_signal, "comb_blocks target")
        default_branches = [b for b in cb.branches if b.condition is None]
        if len(default_branches) > 1:
            problems.append(f"COMB '{cb.target_signal}' has {len(default_branches)} default branches; must have at most 1")
        if not default_branches:
            problems.append(f"COMB '{cb.target_signal}' has no default/hold branch")
        for i, branch in enumerate(cb.branches):
            if branch.condition is not None:
                for term in branch.condition.terms:
                    check_condition_term_signals(
                        term,
                        f"COMB '{cb.target_signal}' branch {i} condition"
                    )
            check_signal_exists(branch.result_op.target, f"COMB '{cb.target_signal}' branch {i} result target")

    comb_targets = set()
    for op in ir.combinational_ops:
        if op.target_concat:
            comb_targets.update(op.target_concat)
        elif op.target:
            comb_targets.add(op.target)
    comb_targets |= comb_targets_declared
    overlap = comb_targets & driven_registers
    if overlap:
        problems.append(f"signals driven both combinationally and sequentially: {overlap}")

    return problems


_ENCODED_COMPARISON_RE = re.compile(r"(!=|<=|>=|==|<|>)")


def _operand_to_verilog(ref: OperandRef, rename_map: Optional[dict] = None) -> str:
    rename_map = rename_map or {}
    if ref.raw_verilog is not None:
        text = ref.raw_verilog
        for old_name, new_name in rename_map.items():
            text = re.sub(rf"\b{re.escape(old_name)}\b", new_name, text)
        return text
    if ref.signal == "__const_x":
        return "1'bx"
    if ref.signal.startswith("__const_"):
        return ref.signal[len("__const_"):]
    if _ENCODED_COMPARISON_RE.search(ref.signal):
        terms = ref.signal.split("&&")
        rendered_terms = []
        for term in terms:
            m = _ENCODED_COMPARISON_RE.search(term)
            op_symbol = m.group(1)
            sig, val = term.split(op_symbol, 1)
            sig = rename_map.get(sig, sig)
            val = rename_map.get(val, val)
            rendered_terms.append(f"{sig} {op_symbol} {val}")
        return " && ".join(rendered_terms)
    sig = rename_map.get(ref.signal, ref.signal)
    if ref.bit_index is not None:
        return f"{sig}[{ref.bit_index}]"
    if ref.bit_range is not None:
        hi, lo = ref.bit_range
        return f"{sig}[{hi}:{lo}]"
    if ref.dynamic_bit_index is not None:
        idx_sig = rename_map.get(ref.dynamic_bit_index, ref.dynamic_bit_index)
        return f"{sig}[{idx_sig}]"
    return sig


def _render_lhs(target: str, target_bit_index: Optional[int], target_bit_range: Optional[tuple[int, int]],
                rename_map: Optional[dict] = None) -> str:
    """Renders the left-hand side of an assignment, honoring an optional
    bit-index/bit-range on the target (e.g. 'next_state[3]', 'out[7:4]')."""
    rename_map = rename_map or {}
    name = rename_map.get(target, target)
    if target_bit_index is not None:
        return f"{name}[{target_bit_index}]"
    if target_bit_range is not None:
        hi, lo = target_bit_range
        return f"{name}[{hi}:{lo}]"
    return name


def _op_to_verilog_expr(op: Operation, signals_by_id: dict, rename_map: Optional[dict] = None) -> str:
    rename_map = rename_map or {}
    if op.op == OpKind.CONST:
        return "1'bx" if op.const_is_x else str(op.const_value)
    if op.op == OpKind.SIGNAL_REF:
        return _operand_to_verilog(op.operands[0], rename_map)
    if op.op == OpKind.NOT:
        return f"~{_operand_to_verilog(op.operands[0], rename_map)}"
    if op.op == OpKind.REDUCE_AND:
        return f"&{_operand_to_verilog(op.operands[0], rename_map)}"
    if op.op == OpKind.REDUCE_OR:
        return f"|{_operand_to_verilog(op.operands[0], rename_map)}"
    if op.op == OpKind.REDUCE_XOR:
        return f"^{_operand_to_verilog(op.operands[0], rename_map)}"
    if op.op == OpKind.NAND:
        parts = [_operand_to_verilog(o, rename_map) for o in op.operands]
        return f"~({' & '.join(parts)})"
    if op.op == OpKind.NOR:
        parts = [_operand_to_verilog(o, rename_map) for o in op.operands]
        return f"~({' | '.join(parts)})"
    if op.op == OpKind.XNOR:
        parts = [_operand_to_verilog(o, rename_map) for o in op.operands]
        return f"~({' ^ '.join(parts)})"
    if op.op == OpKind.MUX2:
        if len(op.operands) != 3:
            raise ValidationError("MUX2 requires exactly 3 operands")
        sel, if_true, if_false = (_operand_to_verilog(o, rename_map) for o in op.operands)
        return f"({sel}) ? ({if_true}) : ({if_false})"
    if op.op in (OpKind.ROTATE_LEFT, OpKind.ROTATE_RIGHT):
        # Verilog has no rotate operator, so build it with concatenation:
        # rotate right by 1 -> {q[0], q[99:1]}, rotate left by 1 -> {q[98:0], q[99]}
        if len(op.operands) != 2:
            raise ValidationError(f"{op.op} requires exactly 2 operands: [signal, shift_amount]")
        sig_operand, amount_operand = op.operands
        if not amount_operand.signal.startswith("__const_"):
            raise ValidationError(f"{op.op} only supports a constant rotate amount, got '{amount_operand.signal}'")
        amount = int(amount_operand.signal[len("__const_"):])
        sig = rename_map.get(sig_operand.signal, sig_operand.signal)
        signal_def = signals_by_id.get(sig_operand.signal)
        if signal_def is None:
            raise ValidationError(f"{op.op}: unknown signal '{sig_operand.signal}'")
        width = signal_def.width
        if amount <= 0 or amount >= width:
            raise ValidationError(f"{op.op}: rotate amount {amount} out of range for {width}-bit signal '{sig}'")

        def _range(hi: int, lo: int) -> str:
            return f"{sig}[{hi}]" if hi == lo else f"{sig}[{hi}:{lo}]"

        if op.op == OpKind.ROTATE_RIGHT:
            return f"{{{_range(amount - 1, 0)}, {_range(width - 1, amount)}}}"
        else:
            return f"{{{_range(width - 1 - amount, 0)}, {_range(width - 1, width - amount)}}}"
    symbol = {
        OpKind.AND: "&", OpKind.OR: "|", OpKind.XOR: "^",
        OpKind.ADD: "+", OpKind.SUB: "-", OpKind.MUL: "*",
        OpKind.SHIFT_LEFT: "<<", OpKind.SHIFT_RIGHT: ">>",
        OpKind.ARITH_SHIFT_RIGHT: ">>>",
    }.get(op.op)
    if symbol is None:
        raise ValidationError(f"Renderer does not support op kind {op.op}")
    parts = [_operand_to_verilog(o, rename_map) for o in op.operands]
    return f" {symbol} ".join(parts)


_COMPARISON_SYMBOL = {
    ComparisonKind.EQ: "==", ComparisonKind.NEQ: "!=",
    ComparisonKind.LT: "<", ComparisonKind.LTE: "<=",
    ComparisonKind.GT: ">", ComparisonKind.GTE: ">=",
}


def _simple_condition_to_verilog(term: SimpleCondition, rename_map: Optional[dict] = None) -> str:
    rename_map = rename_map or {}
    if term.concat_signals is not None:
        names = [rename_map.get(n, n) for n in term.concat_signals]
        lhs = "{" + ", ".join(names) + "}"
    else:
        signal = rename_map.get(term.signal, term.signal)
        if term.bit_index is not None:
            lhs = f"{signal}[{term.bit_index}]"
        elif term.bit_range is not None:
            hi, lo = term.bit_range
            lhs = f"{signal}[{hi}:{lo}]"
        else:
            lhs = signal
    if term.value_signal is not None:
        rhs_signal = rename_map.get(term.value_signal, term.value_signal)
        if term.value_signal_bit_index is not None:
            rhs = f"{rhs_signal}[{term.value_signal_bit_index}]"
        elif term.value_signal_bit_range is not None:
            hi, lo = term.value_signal_bit_range
            rhs = f"{rhs_signal}[{hi}:{lo}]"
        else:
            rhs = rhs_signal
    else:
        rhs = term.value
    return f"{lhs} {_COMPARISON_SYMBOL[term.comparison]} {rhs}"


def _condition_to_verilog(cond: Condition, rename_map: Optional[dict] = None) -> str:
    return " && ".join(_simple_condition_to_verilog(t, rename_map) for t in cond.terms)


def render_verilog(ir: CircuitIR) -> str:
    problems = validate_circuit(ir)
    if problems:
        raise ValidationError("Cannot render invalid CircuitIR:\n" + "\n".join(problems))

    signals_by_id = {s.id: s for s in ir.signals}
    next_signals = {ru.next_signal for ru in ir.register_updates if ru.next_signal}

    # Any register that's a declared output port gets an internal shadow
    # register (name + '_r') that actually receives the <= assignments,
    # plus a final 'assign port = port_r;' -- but only when the fixed
    # interface declares the port as plain 'output' (implicit wire), since
    # that's the case where writing to the port directly with <= would be
    # illegal. If the interface instead declares it 'output reg' (verified
    # via port_is_reg_typed), a continuous assign onto it is what's
    # illegal, so we drive that port name directly instead.
    port_reg_names = {ru.target_register for ru in ir.register_updates
                      if signals_by_id.get(ru.target_register) and signals_by_id[ru.target_register].is_port_declared
                      and not signals_by_id[ru.target_register].port_is_reg_typed}
    internal_name_for = {name: f"{name}_r" for name in port_reg_names}

    def _render_name(name: str) -> str:
        return internal_name_for.get(name, name)

    # Any signal driven by a combinational_op or comb_block, that isn't a
    # register and isn't already declared as a port, needs an explicit
    # wire declaration -- otherwise it's an implicit net, which some
    # toolchains (default_nettype none) reject at compile time.
    combinationally_driven = set()
    for op in ir.combinational_ops:
        if op.target_concat:
            combinationally_driven.update(op.target_concat)
        elif op.target:
            combinationally_driven.add(op.target)
    for cb in ir.comb_blocks:
        combinationally_driven.add(cb.target_signal)

    # Only synthetic auxiliary signals (created by the parser for nested
    # ternaries) need an explicit wire declaration here -- named signals
    # (e.g. a module's own output port) are assumed to already be declared
    # outside the body via the fixed module interface.
    wires_to_declare = {s for s in combinationally_driven if s.startswith("__aux_")}

    lines: list[str] = []
    for s in ir.signals:
        if s.is_register and not s.is_port_declared:
            width_decl = f"[{s.width - 1}:0] " if s.width > 1 else ""
            lines.append(f"reg {width_decl}{s.id};")
    for port_name in sorted(port_reg_names):
        sig_def = signals_by_id[port_name]
        width_decl = f"[{sig_def.width - 1}:0] " if sig_def.width > 1 else ""
        lines.append(f"reg {width_decl}{internal_name_for[port_name]};")
    for next_sig in sorted(next_signals):
        # A 'next' signal driving a register must also be reg-typed in
        # Verilog, matching the canonical 'reg [N:0] state, next;' style.
        state_reg = next(ru.target_register for ru in ir.register_updates if ru.next_signal == next_sig)
        state_sig_def = signals_by_id.get(state_reg)
        width_decl = f"[{state_sig_def.width - 1}:0] " if state_sig_def and state_sig_def.width > 1 else ""
        lines.append(f"reg {width_decl}{next_sig};")
    for wire_name in sorted(wires_to_declare):
        sig_def = signals_by_id.get(wire_name)
        width_decl = f"[{sig_def.width - 1}:0] " if sig_def and sig_def.width > 1 else ""
        lines.append(f"wire {width_decl}{wire_name};")
    lines.append("")

    for ru in ir.register_updates:
        if ru.init_value is None and not ru.init_is_x:
            continue

        render_target = _render_name(ru.target_register)

        sig_def = signals_by_id.get(ru.target_register)
        width = sig_def.width if sig_def is not None else 1

        lines.append("initial begin")

        if ru.init_is_x:
            if width == 1:
                lines.append(f"    {render_target} = 1'bx;")
            else:
                lines.append(f"    {render_target} = {width}'bx;")
        else:
            lines.append(f"    {render_target} = {ru.init_value};")

        lines.append("end")
        lines.append("")
        
    for op in ir.combinational_ops:
        if op.target_concat:
            lhs = "{" + ", ".join(_render_name(t) for t in op.target_concat) + "}"
            needs_always = any(signals_by_id.get(t) and signals_by_id[t].needs_always_block for t in op.target_concat)
        else:
            lhs = _render_lhs(op.target, op.target_bit_index, op.target_bit_range, internal_name_for)
            sig_def = signals_by_id.get(op.target)
            needs_always = bool(sig_def and sig_def.needs_always_block)

        expr_str = _op_to_verilog_expr(op, signals_by_id, internal_name_for)
        if needs_always:
            lines.append("always @(*) begin")
            lines.append(f"    {lhs} = {expr_str};")
            lines.append("end")
        else:
            lines.append(f"assign {lhs} = {expr_str};")

    for port_name in sorted(port_reg_names):
        lines.append(f"assign {port_name} = {internal_name_for[port_name]};")

    for cb in ir.comb_blocks:
        lines.append(f"always @(*) begin")
        for i, branch in enumerate(cb.branches):
            if branch.condition is None:
                if i == 0:
                    lines.append(f"    {cb.target_signal} = {_op_to_verilog_expr(branch.result_op, signals_by_id, internal_name_for)};")
                else:
                    lines.append(f"    else")
                    lines.append(f"        {cb.target_signal} = {_op_to_verilog_expr(branch.result_op, signals_by_id, internal_name_for)};")
            else:
                keyword = "if" if i == 0 else "else if"
                lines.append(f"    {keyword} ({_condition_to_verilog(branch.condition, internal_name_for)})")
                lines.append(f"        {cb.target_signal} = {_op_to_verilog_expr(branch.result_op, signals_by_id, internal_name_for)};")
        lines.append("end")
        lines.append("")

    for ru in ir.register_updates:
        render_target = _render_name(ru.target_register)
        edge_kw = "posedge" if ru.edge == EdgeType.POSEDGE else "negedge"
        sens = f"{edge_kw} {ru.clock}"
        is_async_reset = ru.reset_signal and ru.reset_edge in (EdgeType.POSEDGE, EdgeType.NEGEDGE)
        is_sync_reset = ru.reset_signal and ru.reset_edge in (EdgeType.SYNC, EdgeType.SYNC_NEG)
        sync_reset_check = (
            ru.reset_signal if ru.reset_edge == EdgeType.SYNC else f"!{ru.reset_signal}"
        ) if is_sync_reset else None
        if is_async_reset:
            reset_edge_kw = "posedge" if ru.reset_edge == EdgeType.POSEDGE else "negedge"
            sens += f" or {reset_edge_kw} {ru.reset_signal}"

        lines.append(f"always @({sens}) begin")

        if is_async_reset:
            reset_check = ru.reset_signal if ru.reset_edge == EdgeType.POSEDGE else f"!{ru.reset_signal}"
            lines.append(f"    if ({reset_check})")
            lines.append(f"        {render_target} <= {ru.reset_value};")
            lines.append("    else begin")
            indent = "        "
        else:
            lines.append("    begin")
            indent = "        "

        if ru.next_signal is not None:
            # 'state, next' pattern: state simply latches next every edge.
            # A sync reset here still needs its own check inside the body.
            if is_sync_reset:
                lines.append(f"{indent}if ({sync_reset_check})")
                lines.append(f"{indent}    {render_target} <= {ru.reset_value};")
                lines.append(f"{indent}else")
                lines.append(f"{indent}    {render_target} <= {ru.next_signal};")
            else:
                lines.append(f"{indent}{render_target} <= {ru.next_signal};")
        else:
            branches = ru.branches
            if is_sync_reset:
                # render as "if (reset) ...; else if (...)" chain
                lines.append(f"{indent}if ({sync_reset_check})")
                lines.append(f"{indent}    {render_target} <= {ru.reset_value};")
                start_keyword = "else if"
            else:
                start_keyword = None

            for i, branch in enumerate(branches):
                effective_i = i if start_keyword is None else i + 1
                if branch.condition is None:
                    if effective_i == 0:
                        lines.append(f"{indent}{render_target} <= {_op_to_verilog_expr(branch.result_op, signals_by_id, internal_name_for)};")
                    else:
                        lines.append(f"{indent}else")
                        lines.append(f"{indent}    {render_target} <= {_op_to_verilog_expr(branch.result_op, signals_by_id, internal_name_for)};")
                else:
                    keyword = "if" if effective_i == 0 else (start_keyword if i == 0 else "else if")
                    lines.append(f"{indent}{keyword} ({_condition_to_verilog(branch.condition, internal_name_for)})")
                    lines.append(f"{indent}    {render_target} <= {_op_to_verilog_expr(branch.result_op, signals_by_id, internal_name_for)};")

        lines.append("    end")
        lines.append("end")

    return "\n".join(lines)


def diff_circuit_ir(old: CircuitIR, new: CircuitIR) -> list[str]:
    """Returns a list of concrete changes between two CircuitIR snapshots
    (which registers/branches changed), used in the revision loop."""
    changes: list[str] = []

    old_regs = {ru.target_register: ru for ru in old.register_updates}
    new_regs = {ru.target_register: ru for ru in new.register_updates}

    for reg_name in set(old_regs) | set(new_regs):
        old_ru = old_regs.get(reg_name)
        new_ru = new_regs.get(reg_name)

        if old_ru is None:
            changes.append(f"register '{reg_name}': added")
            continue

        if new_ru is None:
            changes.append(f"register '{reg_name}': removed")
            continue

        def _cond_key(cond):
            if cond is None:
                return None
            return tuple(
                (t.signal, t.bit_index, t.bit_range, t.comparison, t.value)
                for t in cond.terms
            )

        old_conditions = [_cond_key(b.condition) for b in old_ru.branches]
        new_conditions = [_cond_key(b.condition) for b in new_ru.branches]

        if old_conditions != new_conditions:
            changes.append(
                f"register '{reg_name}': branch/priority order changed "
                f"from {old_conditions} to {new_conditions}"
            )

        if old_ru.init_value != new_ru.init_value:
            changes.append(
                f"register '{reg_name}': init_value changed "
                f"from {old_ru.init_value} to {new_ru.init_value}"
            )

        if old_ru.init_is_x != new_ru.init_is_x:
            changes.append(
                f"register '{reg_name}': init_is_x changed "
                f"from {old_ru.init_is_x} to {new_ru.init_is_x}"
            )

        if old_ru.next_signal != new_ru.next_signal:
            changes.append(
                f"register '{reg_name}': next_signal changed "
                f"from {old_ru.next_signal} to {new_ru.next_signal}"
            )

        if old_ru.reset_signal != new_ru.reset_signal:
            changes.append(
                f"register '{reg_name}': reset_signal changed "
                f"from {old_ru.reset_signal} to {new_ru.reset_signal}"
            )

        if old_ru.reset_edge != new_ru.reset_edge:
            changes.append(
                f"register '{reg_name}': reset_edge changed "
                f"from {old_ru.reset_edge} to {new_ru.reset_edge}"
            )

        if old_ru.reset_value != new_ru.reset_value:
            changes.append(
                f"register '{reg_name}': reset_value changed "
                f"from {old_ru.reset_value} to {new_ru.reset_value}"
            )

        for i, (ob, nb) in enumerate(zip(old_ru.branches, new_ru.branches)):
            if ob.result_op.op != nb.result_op.op:
                changes.append(
                    f"register '{reg_name}' branch {i}: operation changed "
                    f"from {ob.result_op.op} to {nb.result_op.op}"
                )

    old_sig_ids = {s.id for s in old.signals}
    new_sig_ids = {s.id for s in new.signals}

    if old_sig_ids != new_sig_ids:
        changes.append(
            f"signals changed: added={new_sig_ids - old_sig_ids}, "
            f"removed={old_sig_ids - new_sig_ids}"
        )

    return changes

class VSLOutput(BaseModel):
    """Output type for gir_agent: just the raw VSL text it generates."""
    vsl_code: str = Field(..., description="Circuit logic written in VSL, following the grammar in the system prompt.")


VSL_GRAMMAR_AND_EXAMPLES = """
You translate a natural-language Verilog module description into VSL
(Verilog Semantic Language) -- a compact notation invented for this
project. Do not output JSON, and do not output an English description of
the logic. Output only VSL syntax, following the grammar below exactly.

GRAMMAR:

1. Register declaration (one line):
   REG <name>[<width>] @<clock>.<edge> [<reset>.<edge>-><reset_value>] [INIT=<value>] [PORT]
   - [<width>] is optional; omit it for a 1-bit signal.
   - <edge> is 'pos' or 'neg'.
   - the reset clause in [...] is optional. The reset's <edge> is one of:
       pos     -- ASYNCHRONOUS active-high reset (checked in the
                  sensitivity list, e.g. 'always @(posedge clk or posedge rst)')
       neg     -- ASYNCHRONOUS active-low reset (sensitivity list uses negedge)
       sync / syncpos -- SYNCHRONOUS active-high reset: checked ONLY
                  inside the clocked always block as 'if (reset_signal)',
                  NOT in the sensitivity list (sensitivity list stays
                  just 'posedge clk').
       syncneg -- SYNCHRONOUS active-low reset: same as above, but
                  checked as 'if (!reset_signal)' inside the body.
   - CRITICAL: for a sync reset, you MUST pick the polarity token that
     matches the signal's actual active level, exactly like you would for
     an async reset's pos/neg -- a signal named with a trailing 'n' (e.g.
     resetn, rstn) or one the description calls "active-low" needs
     'syncneg', not plain 'sync'/'syncpos'. Using the wrong polarity here
     produces a register that is always held in reset (or never reset).
   - Use a sync variant whenever the description says the reset takes
     effect "on the next clock edge", or shows the reset condition
     combined with other conditions inside the same clocked block (e.g.
     "reset OR counter reaches its max value, both checked every clock
     edge"). Use an async variant (pos/neg) only when the description
     explicitly says the reset is immediate/asynchronous.
   - CRITICAL: do not default to async ('pos'/'neg') out of habit -- check
     whether the description implies the reset is only sampled at the
     clock edge (sync) or overrides the clock immediately (async). When
     in doubt, and the description doesn't explicitly say "asynchronous",
     prefer a sync variant.
   - the optional INIT=<value> gives the register a power-up initial
     value (rendered as an 'initial' block), completely SEPARATE from any
     reset. Use this whenever a description specifies the register's
     value at time zero / on power-up / before the first clock edge,
     independent of whether a reset signal exists or has ever been
     asserted -- e.g. "the register starts at 0", "q is initially 0",
     or a testbench-style requirement that the very first sampled value
     (before any reset pulse) already be a specific number, not X/unknown.
     <value> can be a plain number or a sized literal (8'h00); use INIT=x
     if the description explicitly says the power-up value is unknown/
     don't-care (this is rarely needed -- only use it if stated).
     CRITICAL: if a register has NO reset signal at all in the
     description, but the description still requires a specific known
     value at time zero (not X), you MUST add INIT=<value> -- without it,
     the register starts at X and stays X until the first real update,
     which will mismatch any testbench checking the value at time zero.
   - the trailing PORT keyword is mandatory for ANY register whose name
     already appears as an output in the fixed module interface -- this
     applies whether the interface says 'output reg [7:0] q' OR just
     'output [7:0] q' (without the word 'reg'). In BOTH cases the signal
     is already declared once by the interface; adding 'reg ...;' again
     inside the body is a duplicate declaration and a compile error.
     PORT tells the renderer: "don't declare this signal again."
     Only omit PORT for a register that is purely internal (does NOT
     appear anywhere in the fixed module interface).
   Example (async reset): REG count[4] @clk.pos rst.pos->0
   Example (sync reset): REG q[10] @clk.pos rst.sync->0 PORT
   Example for an output port already declared as 'output reg [7:0] q'
   in the interface: REG q[8] @clk.pos PORT
   Example (no reset at all, but must start at a known value 0):
   REG q[8] @clk.pos INIT=0 PORT
     * -> d

2. Priority branches (immediately follow a REG line, one condition per line):
   <condition> -> <expression>
   | <condition> -> <expression>
   | * -> <expression>
   - Lines after the first are prefixed with '|'.
   - Branches are evaluated in the order written; the first matching
     condition wins.
   - The LAST line must always be '* -> <expression>' (the default/hold
     case) -- this is mandatory, never omit it.

3. Conditions:
   <signal>=<value>
   Comparisons other than equality are also supported (no spaces around
   the operator):
     count!=0    -- not equal
     count<5     -- less than
     count<=5    -- less than or equal
     count>5     -- greater than
     count>=5    -- greater than or equal
   The right-hand side can also be another signal instead of a fixed
   number, for comparisons between two signals: a<b, x=y, count<max_count.
   Multiple conditions can be combined with '&' (AND only, no OR):
   <signal1>=<value1> & <signal2>=<value2>

4. Expressions (right-hand side of -> or =):
   - a bare signal: data
   - a unary NOT: ~a, ~Q, ~data[3] (bitwise negation of a signal)
   - a binary operation: q<<1, q>>1, a+b, a-b, a&b, a|b, a^b
   - parenthesized sub-expressions to control grouping: (a|b) & (c|d)
   - a bit slice: a[7:0], a[3]
   - a concatenation: {sig1, sig2, ...}, using standard Verilog concatenation
     syntax -- can mix signals, bit slices, sized constants like 16'b0 or
     8'h00, and even full sub-expressions like a ternary, e.g.
     {out_bytes[15:0], in}, {in, 16'b0}, or {state=D?1:0, state=C?1:0}
     A member can also be a replication, {N{expr}}, meaning expr repeated
     N times, e.g. {24{in[7]}} for 24 copies of bit in[7] (commonly used
     for sign-extension): {{24{in[7]}}, in}
   - a ternary (combinational only): <condition> ? <if_true> : <if_false>
     where <condition> can be a single comparison ('signal=value',
     'signal!=value', 'signal<value', etc. -- same operators as section 3)
     or several joined with '&', e.g. p=1 & q=1 ? 0 : 1
   - the don't-care/unknown literal: x (or a sized form like 1'bx, 4'hx --
     any spelling renders the same way). Use this whenever the description
     says a case is "unspecified", "don't care", or explicitly gives
     1'bx as the result, e.g. '* -> x' or 'a=1 & b=1 ? 0 : x'.

   CRITICAL: "shift" and "rotate" are DIFFERENT operations -- do not
   confuse them. A shift (<<, >>) drops the bit that falls off the end and
   fills with 0. A rotate wraps the bit that falls off one end back onto
   the other end, so no information is lost. If the description says
   "rotate" (or describes bits wrapping around, circular shift, etc.), you
   MUST use the rotate operators below, not << / >>:
     q<<r1   -- rotate LEFT by 1 bit (the top bit wraps to the bottom)
     q>>r1   -- rotate RIGHT by 1 bit (the bottom bit wraps to the top)
   Only use a plain << or >> when the description genuinely means a
   non-circular shift (bits fall off and are lost, zeros fill in).

   For negated gates (NAND, NOR, XNOR), express them as a ternary using
   the equivalent truth table rather than inventing new operator symbols.
   Example: NAND of p and q -> p=1 & q=1 ? 0 : 1
   Example: NOR of p and q  -> p=0 & q=0 ? 1 : 0

5. Combinational assignment (no clock involved):
   <signal> = <expression>
   The left-hand side can also be a multi-signal concatenation, exactly
   like Verilog's own {a, b} = expr syntax, when an expression's result
   needs to be split across more than one signal (e.g. a carry-out and a
   sum from an adder):
     {sig1, sig2, ...} = <expression>
   Example: {cout, sum} = a+b   (cout gets the carry bit, sum gets the rest)

   The left-hand side can also be a single bit index or bit range of an
   existing signal, exactly like Verilog's own bit-select LHS, when a
   description assigns each bit (or a sub-range) of a vector separately
   rather than all at once:
     next_state[3] = a & b
     out[7:4] = data[3:0]
   Use this instead of trying to force a whole-vector expression when the
   description gives a separate formula per output bit.

   An expression's binary operator can also chain more than two operands
   when the description implies more than one term added/combined
   together, e.g. a full adder's three-input sum:
     {cout, sum} = a+b+cin

5b. Combinational priority block (a general-purpose "case", for a plain
   wire -- not only for FSM next-state logic):
     COMB <target>
       <condition> -> <expression>
     | <condition> -> <expression>
     | * -> <expression>
   This has the exact same syntax and semantics as a REG block's priority
   branches (section 2), except it drives a combinational signal instead
   of a clocked register. Use this whenever a description gives many
   (more than 2-3) distinct cases for a single output based on a
   selector's value -- e.g. a case statement, a lookup table, or a
   multi-way select -- instead of building a deeply nested ternary, which
   is harder to get right. The selector can be a single signal or a
   concatenation of several, matched with '=' exactly like section 3:
     COMB out
       sel=0 -> data0
     | sel=1 -> data1
     | sel=2 -> data2
     | * -> 4'hf
   Example with a multi-signal selector and don't-care outputs:
     COMB Y
       {a,b,c,d}=0 -> 0
     | {a,b,c,d}=1 -> 0
     | {a,b,c,d}=2 -> 1
     | {a,b,c,d}=4 -> x
     | * -> 1

6. FSM named states (must appear BEFORE the REG block that uses them):
   STATES: <name1>=<val1>, <name2>=<val2>, ...
   Then use the state names directly in conditions and results instead of
   raw numbers.

7. Two-register FSM pattern ('state, next'): many real designs describe an
   FSM with a separate combinational next-state signal computed by its own
   priority block, then latched into the state register every clock edge.
   Use this pattern (not inline priority branches on the REG line) whenever
   the description implies this two-signal structure (e.g. mentions a
   "next state" signal, or the reference style declares "reg state, next"):

   REG <state_name>[<width>] @<clock>.<edge> [<reset>.<edge>-><reset_value>] NEXT=<next_name>

   COMB <next_name>
     <condition> -> <expression>
   | <condition> -> <expression>
   | * -> <expression>

   - The REG line with NEXT=<next_name> has NO priority branches of its own
     -- it simply means "latch <next_name> into <state_name> every clock
     edge (after checking reset, if any)".
   - The separate COMB block computes <next_name> using the exact same
     priority-branch syntax as a normal REG block (same '|' separators,
     same mandatory '* -> ...' default line).
   - <next_name> does not need to be declared anywhere else -- just use it
     consistently in the REG line's NEXT= and the COMB block's target.
   - COMB blocks are not limited to this FSM pattern -- see section 5b for
     using COMB as a general-purpose case-like block for any wire.

8. Comments (optional, ignored by the parser): lines starting with #

STRICT RULES:
- Every REG block MUST end with a '* -> ...' default branch. A register
  with no default branch is invalid -- there is no such thing as
  "undefined behavior" in this notation. (Exception: a REG line using
  NEXT=<name> has no branches of its own -- the COMB block it points to
  must have the default branch instead.)
- Every COMB block MUST also end with a '* -> ...' default branch, same
  as a REG block.
- Only '=' is used for equality in conditions; do not use '=='.
- Do not invent new syntax beyond what's shown here. If a construct doesn't
  fit this grammar, keep the VSL as close as possible and let the fallback
  path handle it.
- CRITICAL: the [<width>] on a REG line (and any bit-width you infer for
  any signal) MUST come ONLY from the actual Verilog port declaration
  given in "The module interface is FIXED..." section of the prompt (e.g.
  `input [7:0] data` means width 8). NEVER infer a width from a number
  that appears in the task name, module name, or anywhere else in the
  prose description. If a signal's width cannot be determined from the
  fixed interface, default to matching whatever width that same signal
  has in the interface -- do not guess a width from surrounding text.
- CRITICAL: if a description gives more than 2-3 distinct cases for one
  output (a case statement, truth table, or lookup table), prefer a COMB
  block (section 5b) over a deeply nested ternary. A nested ternary more
  than 2-3 levels deep is much more error-prone to get right than a flat
  list of '<selector>=<value> -> <expr>' branches.
- CRITICAL: if a description says a case is unspecified/don't-care, or
  explicitly gives 1'bx as the result for some input combination, you
  MUST represent that with the 'x' literal (section 4) -- do not
  substitute 0 or 1 for it, that changes the circuit's behavior.

Do not wrap your output in code fences, quotes, or any other delimiter --
output raw VSL text only, starting directly with the first VSL line (e.g.
'STATES:' or 'REG').

EXAMPLES:

Description: "An 8-bit shift register. When load is high, q takes the
value of data. Otherwise, if ena[0] is set, shift left by 1 (this has
priority). Otherwise if ena[1] is set, shift right by 1. Otherwise hold.
Fixed module interface: output reg [7:0] q, input load, input [1:0] ena,
input [7:0] data, input clk."
VSL:
REG q[8] @clk.pos PORT
  load=1 -> data
| ena=1 -> q<<1
| ena=2 -> q>>1
| * -> q

Description: "A 4-bit counter that increments every clock cycle and resets
to 0 when rst is high."
VSL:
REG count[4] @clk.pos rst.pos->0
  * -> count+1

Description: "A 2-state FSM. In IDLE, if start is asserted, go to RUN. In
RUN, if done is asserted, go back to IDLE."
VSL:
STATES: IDLE=0, RUN=1

REG state[1] @clk.pos
  state=IDLE & start=1 -> RUN
| state=RUN & done=1 -> IDLE
| * -> state

Description: "Output y equals a when sel is 1, otherwise b."
VSL:
y = sel=1 ? a : b

Description: "Output out is the AND of a and b."
VSL:
out = a & b

Description: "Output y is the NAND of inputs p and q."
VSL:
y = p=1 & q=1 ? 0 : 1

Description: "A 100-bit register q, clocked on posedge clk. When load is
high, q takes data. Otherwise if ena is 1, ROTATE q right by 1 bit (the
bit that falls off the bottom wraps around to the top -- no bit is lost).
Otherwise if ena is 2, ROTATE q left by 1 bit (same wrap-around behavior).
Otherwise hold. Fixed module interface: output reg [99:0] q, input load,
input [1:0] ena, input [99:0] data, input clk."
VSL:
REG q[100] @clk.pos PORT
  load=1 -> data
| ena=1 -> q>>r1
| ena=2 -> q<<r1
| * -> q

Description: "A 2-state FSM (A, B) with a separate next-state signal.
Asynchronously resets to state B on areset. In state A, if in is 0 go to
B, else stay in A. In state B, if in is 0 go to A, else stay in B. Output
out is 1 when in state B."
VSL:
STATES: A=0, B=1

REG state @clk.pos areset.pos->B NEXT=next

COMB next
  state=A & in=0 -> B
| state=A & in=1 -> A
| state=B & in=0 -> A
| state=B & in=1 -> B
| * -> state

out = state=B ? 1 : 0

Description: "A 10-bit down-counter register count_value, clocked on
posedge clk. When load is high, count_value takes the value of data.
Otherwise, if count_value is not zero, decrement it by 1. Otherwise (it is
already zero) hold. Output tc is 1 when count_value equals zero."
VSL:
REG count_value[10] @clk.pos PORT
  load=1 -> data
| count_value!=0 -> count_value-1
| * -> count_value

tc = count_value=0 ? 1 : 0

Description: "A 10-bit up-counter register q, clocked on posedge clk.
Every clock cycle, q increments by 1. If reset is high, OR if q has
reached 999, q resets to 0 on the next clock edge (this reset is checked
every clock cycle along with the other conditions, not asynchronously)."
VSL:
REG q[10] @clk.pos rst.sync->0 PORT
  q=999 -> 0
| * -> q+1

Description: "A 2-bit state register, clocked on posedge clk. resetn is
active-low: when resetn is 0, the state resets to A on the next clock
edge (checked synchronously, not asynchronously -- resetn is sampled only
at the clock edge, along with the normal transition logic)."
VSL:
STATES: A=0, B=1

REG state[2] @clk.pos resetn.syncneg->A NEXT=next

COMB next
  state=A -> B
| * -> A

Description: "A full adder. Given three 1-bit inputs a, b, and cin,
compute a 1-bit sum output and a 1-bit carry-out output cout. sum and
cout together equal a+b+cin."
VSL:
{cout, sum} = a+b+cin

Description: "A 24-bit shift register out_bytes, clocked on posedge clk,
with a synchronous reset to 0. Each clock cycle it shifts in the new
8-bit input in from the right, keeping the previous 16 bits shifted left.
Fixed module interface: output [23:0] out_bytes, input [7:0] in, input
reset, input clk."
VSL:
REG out_bytes[24] @clk.pos reset.sync->0 PORT
  * -> {out_bytes[15:0], in}

Description: "Output z is 1 if the two 2-bit inputs A and B are equal,
0 otherwise."
VSL:
z = A=B ? 1 : 0

Description: "A JK flip-flop register Q, clocked on posedge clk. When j
and k are both 1, Q toggles (inverts). When only j is 1, Q becomes 1.
When only k is 1, Q becomes 0. When neither is 1, Q holds its value."
VSL:
REG Q @clk.pos PORT
  j=1 & k=1 -> ~Q
| j=1 & k=0 -> 1
| j=0 & k=1 -> 0
| * -> Q

Description: "Output out is the 8-bit input in sign-extended to 32 bits:
the lower 8 bits of out equal in, and the upper 24 bits all equal in's
sign bit (in[7])."
VSL:
out = {{24{in[7]}}, in}

Description: "A 4-input lookup table. Given a 4-bit input {a,b,c,d} (a is
the MSB), output out is 1 for input values 3, 5, 7, 8, 9, 11, 12, 13, 15,
and 14; it's 0 for values 0, 1, 6, and 10; and it's don't-care (1'bx) for
values 2 and 4."
VSL:
COMB out
  {a,b,c,d}=0 -> 0
| {a,b,c,d}=1 -> 0
| {a,b,c,d}=2 -> x
| {a,b,c,d}=3 -> 1
| {a,b,c,d}=4 -> x
| {a,b,c,d}=5 -> 1
| {a,b,c,d}=6 -> 0
| {a,b,c,d}=7 -> 1
| {a,b,c,d}=8 -> 1
| {a,b,c,d}=9 -> 1
| {a,b,c,d}=10 -> 0
| {a,b,c,d}=11 -> 1
| {a,b,c,d}=12 -> 1
| {a,b,c,d}=13 -> 1
| {a,b,c,d}=14 -> 1
| * -> 1

Description: "A 10-bit one-hot state vector 'state', with each bit of the
10-bit 'next_state' output computed by its own boolean formula: bit 0 of
next_state is 1 when in is 0 and any of state[4:0], state[7], state[8], or
state[9] is set; bit 1 of next_state is 1 when in is 1 and any of
state[0], state[8], or state[9] is set."
VSL:
next_state[0] = ~in & (state[4:0]|state[7]|state[8]|state[9])
next_state[1] = in & (state[0]|state[8]|state[9])

Now convert the given module description into VSL, following this exact
grammar and style.
"""


gir_agent = Agent(
    MODEL,
    name="GIR Agent",
    output_type=VSLOutput,
    model_settings={"temperature": 0},
    system_prompt=VSL_GRAMMAR_AND_EXAMPLES,
)