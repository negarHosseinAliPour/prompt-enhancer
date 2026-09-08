"""
vsl_core.py

VSL (Verilog Semantic Language) - a small notation for describing
register/FSM logic in a compact, unambiguous way instead of relying on
free-text JSON fields. Includes the parser, validator, Verilog renderer,
and the LLM agent that generates VSL from a natural-language prompt.
"""

import os
import pathlib
import re
from enum import Enum
from typing import Optional

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google_cloud import GoogleCloudProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from google import genai
import google.auth
import google.auth.transport.requests


_vertex_client = genai.Client(vertexai=True, location="global")

MODEL = GoogleModel(
    "gemini-3.1-pro-preview",
    provider=GoogleCloudProvider(client=_vertex_client),
)

_GCP_PROJECT_ID = "cs-poc-pjtbxc0qllmcraowi6w1wmi"
_GCP_REGION = "us-central1"


class _VertexTokenAuth(httpx.Auth):
    """Refreshes the Google OAuth access token on demand instead of baking
    a single short-lived token (~1h) into the client at import time.

    A long-running process (e.g. discover_grammar.py's multi-round,
    multi-hour discovery loop) that fetched the token once and reused it
    forever would start failing every request with a 401
    (ACCESS_TOKEN_TYPE_UNSUPPORTED / expired token) as soon as that single
    token expired -- exactly the crash seen after round 3 of a 1000-sample,
    8-round discovery run. This re-checks validity per request and only
    calls the network refresh when actually needed.
    """

    def __init__(self):
        self._creds, _ = google.auth.default()

    def auth_flow(self, request):
        if not self._creds.valid:
            self._creds.refresh(google.auth.transport.requests.Request())
        request.headers["Authorization"] = f"Bearer {self._creds.token}"
        yield request


MODEL_GPT_OSS = OpenAIChatModel(
    "openai/gpt-oss-120b-maas",
    provider=OpenAIProvider(
        base_url=f"https://{_GCP_REGION}-aiplatform.googleapis.com/v1/projects/{_GCP_PROJECT_ID}/locations/{_GCP_REGION}/endpoints/openapi",
        http_client=httpx.AsyncClient(auth=_VertexTokenAuth()),
    ),
)

if os.environ.get("VSL_MODEL") == "gpt-oss":
    MODEL = MODEL_GPT_OSS

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
    is_module_port: bool = Field(
        False,
        description="True if this signal's name literally appears as a port "
                    "in the fixed module interface (verified against the "
                    "real interface text via port_widths), regardless of "
                    "whether the VSL text used the PORT keyword. Unlike "
                    "is_port_declared (which only gets set for REG lines "
                    "that used PORT), this covers plain combinational "
                    "assignment targets too -- needed so the renderer never "
                    "emits a duplicate 'wire <name>;' for a signal that's "
                    "already declared by the interface itself."
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
    value_raw_verilog: Optional[str] = Field(
        None,
        description="For a comparison against a computed expression, e.g. "
                    "'cnt = ((NUM_DIV>>1)-1)' -- neither a plain constant "
                    "nor a plain signal reference. Passed through to the "
                    "renderer verbatim as the right-hand side; overrides "
                    "both 'value' and 'value_signal' when set."
    )


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
    target_is_plain_wire_port: bool = Field(
        False,
        description="True if target_signal is a module output port declared "
                    "WITHOUT 'reg' in the fixed interface (a plain wire). "
                    "Verilog forbids procedural assignment ('=') to a plain "
                    "wire, so the renderer must emit a continuous 'assign' "
                    "driven by a nested ternary/case expression instead of "
                    "an 'always @(*)' block for this CombBlock."
    )


class CircuitIR(BaseModel):
    module_name: str = "unknown"
    function: str = "parsed from VSL"
    signals: list[Signal] = Field(default_factory=list)
    combinational_ops: list[Operation] = Field(default_factory=list)
    comb_blocks: list[CombBlock] = Field(default_factory=list)
    register_updates: list[RegisterUpdate] = Field(default_factory=list)
    fsm_states: list[FSMStateEncoding] = Field(default_factory=list)


class VSLParseError(Exception):
    """Raised when VSL text doesn't match the grammar (syntax problem,
    as opposed to ValidationError which means a bad circuit)."""
    pass


class ValidationError(Exception):
    pass


# --- Parser: VSL text -> CircuitIR, no LLM involved ---------------------

_STATE_ENCODING: dict[str, int] = {}


_VERILOG_SIZED_LITERAL_RE = re.compile(r"^(\d+)'([bdhoBDHO])([0-9a-fA-F_xXzZ]+)$")
_VERILOG_BASE_TO_INT_BASE = {"b": 2, "d": 10, "h": 16, "o": 8}

# matches Verilog's don't-care literal: bare x/X, or a sized form like
# 1'bx, 8'bx, 4'hx. used both standalone and as one branch of a ternary
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


def _normalize_concat_part(part: str) -> Optional[str]:
    """Validates a single concatenation member and returns it in the form
    Verilog actually accepts, or None if it isn't a valid member at all.
    A replication can be written bare (5{a}) or wrapped in one extra brace
    pair ({5{a}}) when nested inside a bigger concat's braces -- but the
    extra wrapping brace pair is only valid VSL syntax, not valid Verilog,
    so it must be stripped here rather than passed through as-is (passing
    it through renders as an undefined bare identifier, e.g. Verilog sees
    '{5{a}}' where a plain '5{a}' or '{5{a}}' with correct nesting was
    needed, and silently drops the intended signal reference)."""
    part = part.strip()

    if _SIZED_CONST_RE.match(part):
        return part

    if re.match(r"^\w+(\[[^\]]+\])?$", part):
        return part

    
    while part.startswith("{") and part.endswith("}"):
        inner_stripped = part[1:-1].strip()
        
        inner_group_parts = _split_top_level_commas(inner_stripped)
        if len(inner_group_parts) > 1:
            normalized = [_normalize_concat_part(p.strip()) for p in inner_group_parts]
            if all(n is not None for n in normalized):
                return "{" + ", ".join(normalized) + "}"
            return None
        if inner_stripped == part:
            break
        part = inner_stripped

    m = _REPLICATION_RE.match(part)
    if m:
        count, inner = m.group(1), m.group(2).strip()
        # the replicated part can be a single thing, or itself a comma
        # list (an implicit concat) -- 5{a,b,c,d,e} means "repeat the
        # group {a,b,c,d,e} 5 times"
        inner_parts = _split_top_level_commas(inner)
        normalized_inner = [_normalize_concat_part(p.strip()) for p in inner_parts]
        if all(n is not None for n in normalized_inner):
            body = normalized_inner[0] if len(normalized_inner) == 1 else "{" + ", ".join(normalized_inner) + "}"
            return f"{count}{{{body}}}"
        return None

    return None


def _is_valid_concat_part(part: str) -> bool:
    return _normalize_concat_part(part) is not None

def _infer_part_select_width(hi_expr: str, lo_expr: str) -> Optional[int]:
    """Given the two sides of a bit-range like 'sel*4+3 : sel*4', checks
    whether hi_expr is lo_expr plus a constant offset (or lo_expr is
    hi_expr minus a constant), and if so returns the width of that range
    (offset + 1). Returns None if the two sides don't share a common base
    in this simple additive way -- e.g. two unrelated expressions."""
    hi_expr = hi_expr.strip()
    lo_expr = lo_expr.strip()

    if hi_expr == lo_expr:
        return 1

    m = re.match(r"^(.+?)\s*\+\s*(\d+)$", hi_expr)
    if m and m.group(1).strip() == lo_expr:
        return int(m.group(2)) + 1

    m = re.match(r"^(.+?)\s*-\s*(\d+)$", lo_expr)
    if m and m.group(1).strip() == hi_expr:
        return int(m.group(2)) + 1

    return None


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
            # A bare unsized digit (e.g. '0' or '1') used to pad a
            # concatenation, as opposed to a real signal name -- Verilog
            # requires every concatenation member to have a fixed, known
            # width, and an unsized literal like plain '0' doesn't have
            # one there (unlike everywhere else, where it's fine).
            # Render it as a properly-sized 1-bit literal instead.
            if re.match(r"^[01]$", part):
                rendered_parts.append(f"1'b{part}")
                continue

            normalized = _normalize_concat_part(part)
            if normalized is not None:
                rendered_parts.append(normalized)
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

    # bit-select on a parenthesized expression, e.g. (A * B)[7:0] or
    # (a+b)[3]. Every bit-select pattern below only matches when the part
    # before '[' is a bare signal name (\w+) -- Verilog itself is fine
    # indexing an arbitrary expression, but keeping that restriction here
    # keeps the rest of this function simple and unambiguous. Rather than
    # rejecting the expression case outright, reuse the same trick already
    # used a few lines up for a complex concatenation member: extract the
    # parenthesized expression into a fresh named COMB signal via
    # _parse_expression, then bit-select THAT signal instead, which the
    # patterns below already support.
    m = re.match(r"^\((.+)\)(\[.+\])$", token)
    if m and extra_ops is not None:
        inner_expr, bit_suffix = m.group(1).strip(), m.group(2)
        aux_name = _next_aux_signal()
        aux_op = _parse_expression(aux_name, inner_expr, extra_ops=extra_ops)
        extra_ops.append(aux_op)
        return _parse_operand(f"{aux_name}{bit_suffix}", extra_ops=extra_ops)

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

    # bit range with a computed (non-literal) bound, e.g.
    # in[sel*4+3:sel*4] -- Verilog can't express this as a plain [hi:lo]
    # range since both bounds must be constants there, so we detect the
    # common '<base>+<width-1> : <base>' shape and turn it into Verilog's
    # dynamic part-select, base +: width. Falls through to the generic
    # dynamic-index case below if the shape doesn't match this pattern.
    m = re.match(r"^(\w+)\[(.+):(.+)\]$", token)
    if m:
        sig, hi_expr, lo_expr = m.group(1), m.group(2).strip(), m.group(3).strip()
        if not (re.match(r"^\d+$", hi_expr) and re.match(r"^\d+$", lo_expr)):
            width = _infer_part_select_width(hi_expr, lo_expr)
            if width is not None and extra_ops is not None:
                aux_name = _next_aux_signal()
                aux_op = _parse_expression(aux_name, lo_expr, extra_ops=extra_ops)
                extra_ops.append(aux_op)
                return OperandRef(raw_verilog=f"{sig}[{aux_name} +: {width}]")

    # fixed bit index: foo[3]
    m = re.match(r"^(\w+)\[(\d+)\]$", token)
    if m:
        return OperandRef(
            signal=m.group(1),
            bit_index=int(m.group(2))
        )

    # dynamic/expression bit index:
    #   q[idx]
    #   q[{A,B,C}]
    #   q[a+b]
    #   q[state=D?1:0]
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
    (not inside parentheses, braces, or brackets).
    """

    paren_depth = 0
    brace_depth = 0
    bracket_depth = 0

    for i, ch in enumerate(s):
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
        elif paren_depth == 0 and brace_depth == 0 and bracket_depth == 0 and ch in seps:
            return s[:i], ch, s[i + 1:]

    return None

def _split_top_level_token(s: str, token: str) -> Optional[tuple[str, str, str]]:
    """Like _split_top_level, but splits on a full (possibly multi-char)
    token such as '<<' or '!=', only at paren/brace/bracket-depth 0."""
    depth = 0
    i = 0
    n = len(token)
    while i < len(s):
        ch = s[i]
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
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

    # unary NOT: ~<signal>, ~<signal>[bit], or ~(<expr>), as long as
    # there's no other top-level binary operator in the rest of the expression
    if expr.startswith("~"):
        inner = expr[1:].strip()
        if not any(_split_top_level_token(inner, tok) for tok in _BINARY_OP_TOKENS_BY_LENGTH) \
                and _split_top_level(inner, "?") is None:
            stripped_inner = _strip_outer_parens(inner)
            # stripping the parens can reveal a hidden binary expression
            # (e.g. ~(in1^in2)) -- that needs its own parse, not a plain operand
            if any(_split_top_level_token(stripped_inner, tok) for tok in _BINARY_OP_TOKENS_BY_LENGTH):
                aux_name = _next_aux_signal()
                aux_op = _parse_expression(aux_name, stripped_inner, extra_ops=extra_ops)
                if extra_ops is not None:
                    extra_ops.append(aux_op)
                return Operation(target=target, op=OpKind.NOT, operands=[OperandRef(signal=aux_name)])
            return Operation(target=target, op=OpKind.NOT, operands=[_parse_operand(stripped_inner)])

    # unary reduction: &<signal>, |<signal>, ^<signal> -- ANDs/ORs/XORs
    # all the bits of a signal into one bit. e.g. &in means "AND all 100
    # bits of in together". different from the binary form (a&b) because
    # the operator sits right at the start of the expression, nothing before it
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
        value_raw_verilog: Optional[str] = None
        try:
            value = _resolve_value(rhs)
        except VSLParseError:
            ref_signal, ref_bit_index, ref_bit_range = _parse_rhs_signal_ref(rhs)
            if ref_signal is not None:
                value_signal = ref_signal
                value_signal_bit_index = ref_bit_index
                value_signal_bit_range = ref_bit_range
                value = 0
            elif re.match(r"^[\w\s()+\-*/<>&|^~]+$", rhs) and re.search(r"[+\-*/<>&|^~]", rhs):
                # A computed expression (arithmetic/bitwise), not a plain
                # constant or signal -- e.g. 'cnt = ((NUM_DIV>>1)-1)' where
                # NUM_DIV is a module parameter. Pass it through verbatim
                # rather than rejecting it; the renderer emits it as-is.
                value_raw_verilog = rhs
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
                value_raw_verilog=value_raw_verilog,
            )

        m = re.match(r"^(\w+)\[(\d+):(\d+)\]$", lhs)
        if m:
            return SimpleCondition(
                signal=m.group(1), bit_range=(int(m.group(2)), int(m.group(3))),
                comparison=comparison, value=value, value_signal=value_signal,
                value_signal_bit_index=value_signal_bit_index,
                value_signal_bit_range=value_signal_bit_range,
                value_raw_verilog=value_raw_verilog,
            )
        m = re.match(r"^(\w+)\[(\d+)\]$", lhs)
        if m:
            return SimpleCondition(
                signal=m.group(1), bit_index=int(m.group(2)),
                comparison=comparison, value=value, value_signal=value_signal,
                value_signal_bit_index=value_signal_bit_index,
                value_signal_bit_range=value_signal_bit_range,
                value_raw_verilog=value_raw_verilog,
            )
        m = re.match(r"^\w+$", lhs)
        if m:
            return SimpleCondition(
                signal=lhs, comparison=comparison, value=value, value_signal=value_signal,
                value_signal_bit_index=value_signal_bit_index,
                value_signal_bit_range=value_signal_bit_range,
                value_raw_verilog=value_raw_verilog,
            )

    # No comparison operator found at all -- e.g. a bare 'out[7]' or 'busy'
    # used as its own boolean condition, the common Verilog/C convention
    # for "this bit/signal is truthy" (implicitly != 0), rather than an
    # explicit 'out[7]==1'. The loop above never matches this since there's
    # no comparison token to split on. Recognize the same signal/bit-index/
    # bit-range shapes as above and default the comparison to NEQ 0, rather
    # than rejecting a condition form that's completely unambiguous.
    m = re.match(r"^(\w+)\[(\d+):(\d+)\]$", term_text)
    if m:
        return SimpleCondition(
            signal=m.group(1), bit_range=(int(m.group(2)), int(m.group(3))),
            comparison=ComparisonKind.NEQ, value=0,
        )
    m = re.match(r"^(\w+)\[(\d+)\]$", term_text)
    if m:
        return SimpleCondition(
            signal=m.group(1), bit_index=int(m.group(2)),
            comparison=ComparisonKind.NEQ, value=0,
        )
    m = re.match(r"^\w+$", term_text)
    if m:
        return SimpleCondition(signal=term_text, comparison=ComparisonKind.NEQ, value=0)

    # Same bare-truthy shapes as above, but negated: '~wfull', '~busy[3]',
    # '~busy[7:0]' -- the common "this signal is false/empty" condition
    # (e.g. 'winc & ~wfull' meaning "write enabled and not full"). Mirrors
    # the NEQ-0 fallback just above but as EQ 0.
    if term_text.startswith("~"):
        negated = term_text[1:].strip()
        m = re.match(r"^(\w+)\[(\d+):(\d+)\]$", negated)
        if m:
            return SimpleCondition(
                signal=m.group(1), bit_range=(int(m.group(2)), int(m.group(3))),
                comparison=ComparisonKind.EQ, value=0,
            )
        m = re.match(r"^(\w+)\[(\d+)\]$", negated)
        if m:
            return SimpleCondition(
                signal=m.group(1), bit_index=int(m.group(2)),
                comparison=ComparisonKind.EQ, value=0,
            )
        m = re.match(r"^\w+$", negated)
        if m:
            return SimpleCondition(signal=negated, comparison=ComparisonKind.EQ, value=0)

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

_LEADING_FENCE_RE = re.compile(r"^(```|'''|~~~)")
_TRAILING_FENCE_RE = re.compile(r"(```|'''|~~~)$")


def _strip_fence_lines(text: str) -> str:
    """Strips leading/trailing 'fence' markers the model sometimes wraps
    its VSL output in (```, ```verilog, ''', ~~~, etc.), even though
    VSLOutput is a structured field and shouldn't need it. Handles both
    the fence on its own line (the common case) and a fence glued
    directly onto the first/last line of real content with no newline in
    between (e.g. "'''STATES: A=0..."), which the plain per-line check
    would otherwise miss and leave as unparseable garbage on the first
    token. Narrow by construction -- it only strips a recognized fence
    marker, so it can't eat real VSL content."""
    lines = text.strip("\n").splitlines()
    while lines and _FENCE_LINE_RE.match(lines[0].strip()):
        lines.pop(0)
    while lines and _FENCE_LINE_RE.match(lines[-1].strip()):
        lines.pop()
    if lines:
        first = lines[0]
        stripped_first = first.lstrip()
        leading_offset = len(first) - len(stripped_first)
        m = _LEADING_FENCE_RE.match(stripped_first)
        if m and len(stripped_first) > m.end():
            lines[0] = first[:leading_offset] + stripped_first[m.end():]
    if lines:
        last = lines[-1]
        stripped_last = last.rstrip()
        m = _TRAILING_FENCE_RE.search(stripped_last)
        if m and m.start() > 0:
            lines[-1] = stripped_last[:m.start()]
    return "\n".join(lines)


class PortInfo(BaseModel):
    """One parsed port from the fixed module interface."""
    name: str
    direction: str          # "input" | "output" | "inout"
    width: int = 1
    is_reg_typed: bool = False



_PORT_QUALIFIERS = {"wire", "reg", "logic", "var", "signed", "unsigned"}
_REG_TYPED_QUALIFIERS = {"reg", "logic", "var"}
_NOT_A_PORT_NAME = {"module", "endmodule", "parameter", "localparam", "input",
                    "output", "inout"} | _PORT_QUALIFIERS

_RANGE_RE = re.compile(r"^\[\s*(\d+)\s*:\s*(\d+)\s*\]")


def _split_port_segments(text: str) -> list[tuple[str, bool]]:
    """Splits an interface into (segment, resets_context) pairs, cutting at
    ',', ';', '(' and ')' -- but only at bracket depth 0, so a range like
    [WIDTH-1:0] stays intact. A ',' continues the previous declaration's
    direction/width context (Verilog's 'input [3:0] a, b' shares them);
    ';', '(' and ')' end it."""
    segments: list[tuple[str, bool]] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif depth == 0 and ch in ",;()":
            segments.append(("".join(current), ch != ","))
            current = []
        else:
            current.append(ch)
    segments.append(("".join(current), True))
    return segments


def parse_module_ports(module_interface: str) -> dict[str, PortInfo]:
    """Parses the fixed Verilog module interface into {port_name: PortInfo}.

    Handles both ANSI style ('module m(input wire [7:0] a, b, output reg q);')
    and the older non-ANSI style ('module m(a, q); input [7:0] a; output reg
    q;'), shared declarations across commas, and every common type qualifier.
    Ports whose range isn't a plain [int:int] (e.g. a parameterized
    [WIDTH-1:0]) are recorded at width 1 rather than dropped, so at least
    their direction and reg-ness are known."""
    ports: dict[str, PortInfo] = {}
    text = " ".join((module_interface or "").split())

    current_direction: Optional[str] = None
    current_width = 1
    current_is_reg = False

    for segment, resets_context in _split_port_segments(text):
        rest = segment.strip()
        if not rest:
            if resets_context:
                current_direction = None
            continue

        m = re.match(r"^(input|output|inout)\b\s*", rest)
        if m:
            current_direction = m.group(1)
            current_width = 1
            current_is_reg = False
            rest = rest[m.end():].strip()
        # otherwise this is either a continuation of the previous
        # declaration ('input [3:0] a, b' -- direction/width carry over) or
        # a bare name from a non-ANSI port list, which has no direction yet.

        if current_direction is None:
            continue

        # optional type qualifiers
        while True:
            m = re.match(r"^(\w+)\b\s*", rest)
            if m and m.group(1) in _PORT_QUALIFIERS:
                if m.group(1) in _REG_TYPED_QUALIFIERS:
                    current_is_reg = True
                rest = rest[m.end():].strip()
                continue
            break

        # optional [hi:lo] range
        m = _RANGE_RE.match(rest)
        if m:
            current_width = abs(int(m.group(1)) - int(m.group(2))) + 1
            rest = rest[m.end():].strip()
        elif rest.startswith("["):
            # non-literal range (e.g. [WIDTH-1:0]) -- skip it, keep width 1
            close = rest.find("]")
            if close != -1:
                rest = rest[close + 1:].strip()

        m = re.match(r"^(\w+)", rest)
        if m and m.group(1) not in _NOT_A_PORT_NAME and not m.group(1).isdigit():
            name = m.group(1)
            ports[name] = PortInfo(
                name=name,
                direction=current_direction,
                width=current_width,
                is_reg_typed=current_is_reg,
            )

        if resets_context:
            current_direction = None

    return ports


def _find_output_reg_names(module_interface: str) -> set[str]:
    """Every 'output reg/logic <name>' port -- these can't be driven with
    'assign', only from inside an always block, even for purely
    combinational logic."""
    return {p.name for p in parse_module_ports(module_interface).values()
            if p.direction == "output" and p.is_reg_typed}


def _find_plain_output_names(module_interface: str) -> set[str]:
    """Every output port that is NOT reg-typed -- i.e. a plain (implicit or
    explicit) wire. A CombBlock targeting one of these must render as a
    continuous 'assign', not an 'always @(*)' block, since Verilog forbids
    procedural assignment ('=') to a wire."""
    return {p.name for p in parse_module_ports(module_interface).values()
            if p.direction == "output" and not p.is_reg_typed}


def _find_port_widths(module_interface: str) -> dict[str, int]:
    """{port_name: width} for every port. A port with no [hi:lo] range is 1
    bit wide. This lets us correctly size signals that are only ever used as
    plain operands (e.g. inputs to a combinational expression) and never
    declared via a VSL REG line, which would otherwise default to width=1
    regardless of their real interface width."""
    return {name: p.width for name, p in parse_module_ports(module_interface).items()}


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
    plain_output_names = _find_plain_output_names(module_interface)
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
            signals_seen[name] = Signal(
                id=name, direction=PortDirection.INPUT, width=effective_width,
                is_register=is_register, is_module_port=name in port_widths,
            )
        elif is_register:
            # this is an authoritative REG declaration for this signal --
            # even if it was already registered earlier (e.g. used as an
            # operand in another REG's branch before we got here), the
            # declared width is the real one and should override the guess
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
            # not registered yet (e.g. a module port only referenced here,
            # inside a concat/replication, and nowhere else earlier) --
            # fall back to its declared interface width instead of
            # silently guessing 1, which corrupts any concat width sum
            if token in port_widths:
                return port_widths[token]
        return 1

    def _infer_op_width(op: Operation) -> int:
        """Infers an aux signal's width as the widest operand referenced
        in its Operation, so a ternary/binary-op result (e.g. picking
        between two 8-bit signals) isn't left at the default 1-bit width,
        which would silently truncate it in the rendered Verilog.

        A bit-select or bit-range on the operand (out_r[3], out_r[7:4])
        narrows the operand down to that many bits -- it must NOT be
        widened back to the parent signal's full width, or a reduction
        like out_r[3]^out_r[2] gets zero-extended to the parent's width
        before a following unary op (e.g. ~) inverts all those padding
        bits too, corrupting the result.
        """
        best = 1
        for operand in op.operands:
            if operand.raw_verilog is not None:
                best = max(best, _estimate_raw_verilog_width(operand.raw_verilog))
                continue
            if operand.bit_range is not None:
                hi, lo = operand.bit_range
                best = max(best, abs(hi - lo) + 1)
                continue
            if operand.bit_index is not None or operand.dynamic_bit_index is not None:
                best = max(best, 1)
                continue
            if not operand.signal:
                continue
            sig = signals_seen.get(operand.signal)
            if sig is not None:
                best = max(best, sig.width)
            elif operand.signal in port_widths:
                best = max(best, port_widths[operand.signal])
        return best


    # '#' has no meaning anywhere else in VSL (no delay syntax, no string
    # literals), so it's unambiguously a comment marker whether it starts
    # the line or trails after real content on it (e.g. "SP=0 -> 0   # full, hold").
    # Strip a trailing "# ..." before the existing full-comment-line filter,
    # so a trailing comment on a COMB branch doesn't get parsed as part of
    # the branch's result value.
    def _strip_trailing_comment(ln: str) -> str:
        idx = ln.find("#")
        return ln[:idx] if idx != -1 else ln

    lines = [
        _strip_trailing_comment(ln).rstrip()
        for ln in text.strip().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    lines = [ln for ln in lines if ln.strip()]

    # Join a statement that the model wrapped across multiple physical
    # lines because the expression got long, e.g.:
    #   COMB next_wfull
    #     * -> ((wptr[3] ^ rptr_sync1[3]) & (wptr[2] ^ rptr_sync1[2]) &
    #           ~(wptr[1] ^ rptr_sync1[1]) & ~(wptr[0] ^ rptr_sync1[0]))
    # VSL is otherwise one-statement-per-line, but nothing in the grammar
    # forbids this and it's a natural thing to write for a long boolean
    # chain. Detected purely by bracket balance -- '(', '{', '[' opened on
    # one line and not yet closed -- so a normal '| cond -> result' line
    # (always balanced on its own) is never affected.
    def _bracket_delta(ln: str) -> int:
        return sum(1 for ch in ln if ch in "([{") - sum(1 for ch in ln if ch in ")]}")

    joined_lines: list[str] = []
    pending = ""
    depth = 0
    for ln in lines:
        pending = f"{pending} {ln}".strip() if pending else ln
        depth += _bracket_delta(ln)
        if depth <= 0:
            joined_lines.append(pending)
            pending = ""
            depth = 0
    if pending:
        joined_lines.append(pending)
    lines = joined_lines

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("STATES:"):
            body = line[len("STATES:"):].strip()
            if body:
                for pair in body.split(","):
                    pair = pair.strip()
                    if not pair:
                        continue
                    if "=" not in pair:
                        raise VSLParseError(f"Cannot parse STATES entry '{pair}' (expected NAME=VALUE)")
                    name, val = pair.split("=", 1)
                    val = val.strip()
                    try:
                        _STATE_ENCODING[name.strip()] = _verilog_literal_to_int(val)
                    except ValueError:
                        _STATE_ENCODING[name.strip()] = int(val)
            i += 1
            continue

        if line.startswith("REG "):

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

            if width:
                width = int(width)
            elif port_flag and reg_name in port_widths:
                width = port_widths[reg_name]
            else:
                width = 1
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

            if comb_target not in port_widths:
                inferred_widths = [
                    w for w in (_infer_op_width(b.result_op) for b in comb_branches)
                    if w
                ]
                if inferred_widths:
                    signals_seen[comb_target].width = max(inferred_widths)

            ir.comb_blocks.append(CombBlock(
                target_signal=comb_target,
                branches=comb_branches,
                target_is_plain_wire_port=comb_target in plain_output_names,
            ))
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

            if (
                target_bit_index is None
                and target_bit_range is None
                and base_name not in port_widths
            ):
                signals_seen[base_name].width = _infer_op_width(op)
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
            # 'state, next' pattern: no branches here by design -- the
            # priority logic lives in the corresponding COMB block, checked
            # separately below.
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
        op.target for op in ir.combinational_ops if op.target
    }
    # A register's NEXT= target doesn't always need its own COMB block: it's
    # legitimate VSL for one register to directly chain off another
    # register's current value every edge (e.g. a 2-stage synchronizer:
    # REG sync1 NEXT=sync0 / REG sync0 NEXT=some_comb). The renderer already
    # handles this correctly -- it just emits 'reg <= next_signal;' verbatim,
    # which is valid Verilog whether next_signal is a COMB-driven wire or
    # another register's output. Only require an explicit COMB block when
    # NEXT= points at something that isn't itself a register target, since
    # that's the case that would otherwise reference an undefined wire.
    register_targets_declared = {ru2.target_register for ru2 in ir.register_updates}
    missing_comb_blocks = next_signals_expected - comb_targets_declared - register_targets_declared
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
    if term.value_raw_verilog is not None:
        rhs = term.value_raw_verilog
    elif term.value_signal is not None:
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


    port_reg_names = {ru.target_register for ru in ir.register_updates
                      if signals_by_id.get(ru.target_register) and signals_by_id[ru.target_register].is_port_declared
                      and not signals_by_id[ru.target_register].port_is_reg_typed}
    internal_name_for = {name: f"{name}_r" for name in port_reg_names}

    def _render_name(name: str) -> str:
        return internal_name_for.get(name, name)


    combinationally_driven = set()
    for op in ir.combinational_ops:
        if op.target_concat:
            combinationally_driven.update(op.target_concat)
        elif op.target:
            combinationally_driven.add(op.target)
    for cb in ir.comb_blocks:
        combinationally_driven.add(cb.target_signal)

    
    wires_to_declare = {
        s for s in combinationally_driven
        if not (signals_by_id.get(s) and signals_by_id[s].is_module_port)
        and s not in next_signals
    }

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
        
        driven_by_comb_block = any(cb.target_signal == next_sig for cb in ir.comb_blocks)
        next_sig_def = signals_by_id.get(next_sig)
        driven_by_always = bool(next_sig_def and next_sig_def.needs_always_block)
        state_reg = next(ru.target_register for ru in ir.register_updates if ru.next_signal == next_sig)
        state_sig_def = signals_by_id.get(state_reg)
        width_decl = f"[{state_sig_def.width - 1}:0] " if state_sig_def and state_sig_def.width > 1 else ""
        kind = "reg" if (driven_by_comb_block or driven_by_always) else "wire"
        lines.append(f"{kind} {width_decl}{next_sig};")
    always_driven_wires = set()
    for op in ir.combinational_ops:
        if op.target_concat:
            if any(signals_by_id.get(t) and signals_by_id[t].needs_always_block for t in op.target_concat):
                always_driven_wires.update(op.target_concat)
        elif op.target and signals_by_id.get(op.target) and signals_by_id[op.target].needs_always_block:
            always_driven_wires.add(op.target)
    for cb in ir.comb_blocks:
        if not cb.target_is_plain_wire_port:
            always_driven_wires.add(cb.target_signal)

    for wire_name in sorted(wires_to_declare):
        sig_def = signals_by_id.get(wire_name)
        width_decl = f"[{sig_def.width - 1}:0] " if sig_def and sig_def.width > 1 else ""
        kind = "reg" if wire_name in always_driven_wires else "wire"
        lines.append(f"{kind} {width_decl}{wire_name};")
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
        if cb.target_is_plain_wire_port:
            # A plain (non-reg) output port can't be assigned inside an
            # always block, so build the priority-branch logic as one
            # nested ternary expression and drive it with a continuous
            # assign instead. Branches are evaluated in the same priority
            # order as the always-block form: the last (default) branch
            # becomes the innermost ':' fallback.
            expr = _op_to_verilog_expr(cb.branches[-1].result_op, signals_by_id, internal_name_for)
            for branch in reversed(cb.branches[:-1]):
                cond_str = _condition_to_verilog(branch.condition, internal_name_for)
                val_str = _op_to_verilog_expr(branch.result_op, signals_by_id, internal_name_for)
                expr = f"({cond_str}) ? ({val_str}) : ({expr})"
            lines.append(f"assign {cb.target_signal} = {expr};")
            lines.append("")
            continue

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
            # If the COMB block driving next_signal already has a branch that
            # conditions on this register's own reset_signal, that COMB has
            # already fully computed the correct reset-case value -- forcing
            # a second, hardcoded 'if (reset) reg <= reset_value;' here would
            # silently override that computed value with the REG line's
            # literal reset_value whenever the two differ, producing wrong
            # hardware (e.g. a register whose real reset output is nonzero
            # but whose REG declaration used the common 'reset.sync->0'
            # boilerplate). So only add the extra sync-reset branch when the
            # feeding COMB block does NOT already branch on reset_signal.
            next_comb = next((cb for cb in ir.comb_blocks if cb.target_signal == ru.next_signal), None)
            comb_handles_reset = bool(
                is_sync_reset and next_comb and any(
                    branch.condition is not None
                    and any(term.signal == ru.reset_signal for term in branch.condition.terms)
                    for branch in next_comb.branches
                )
            )
            if is_sync_reset and not comb_handles_reset:
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

    def _op_key(op: "Operation"):
        """A comparable snapshot of an Operation's actual logic (what it
        computes and from what), ignoring aux-signal names that the parser
        invents fresh on every parse and would otherwise look like a
        'change' even when the expression is identical."""
        return (
            op.target, tuple(op.target_concat) if op.target_concat else None,
            op.target_bit_index, op.target_bit_range, op.op,
            tuple(
                (o.signal, o.bit_index, o.bit_range, o.dynamic_bit_index, o.raw_verilog)
                for o in op.operands
            ),
            op.const_value, op.const_is_x,
        )

    # combinational_ops: plain 'signal = expr' assignments outside a REG or
    # COMB block (e.g. 'sum = A + B'). This -- along with comb_blocks right
    # below -- was previously never compared at all, meaning any purely
    # combinational design (adders, multipliers, ALUs, comparators, muxes)
    # got NO structural feedback from a revision, no matter how much its
    # logic actually changed: only register_updates was ever diffed here,
    # so the revision loop's "structural_note" silently defaulted to
    # "No structural change detected" for every non-register edit.
    old_comb_ops = {op.target or tuple(op.target_concat or []): op for op in old.combinational_ops}
    new_comb_ops = {op.target or tuple(op.target_concat or []): op for op in new.combinational_ops}

    for key in set(old_comb_ops) | set(new_comb_ops):
        old_op = old_comb_ops.get(key)
        new_op = new_comb_ops.get(key)
        label = key if isinstance(key, str) else "{" + ", ".join(key) + "}"

        if old_op is None:
            changes.append(f"combinational signal '{label}': added")
        elif new_op is None:
            changes.append(f"combinational signal '{label}': removed")
        elif _op_key(old_op) != _op_key(new_op):
            changes.append(f"combinational signal '{label}': expression changed")

    # comb_blocks: priority/case-style combinational logic (e.g. an ALU's
    # opcode mux, or FSM next-state logic) -- same branch/condition shape
    # as register_updates.branches, just driving a wire instead of a reg.
    old_blocks = {cb.target_signal: cb for cb in old.comb_blocks}
    new_blocks = {cb.target_signal: cb for cb in new.comb_blocks}

    def _cond_key(cond):
        if cond is None:
            return None
        return tuple(
            (t.signal, t.bit_index, t.bit_range, t.comparison, t.value, t.value_signal)
            for t in cond.terms
        )

    for target in set(old_blocks) | set(new_blocks):
        old_cb = old_blocks.get(target)
        new_cb = new_blocks.get(target)

        if old_cb is None:
            changes.append(f"combinational block '{target}': added")
            continue
        if new_cb is None:
            changes.append(f"combinational block '{target}': removed")
            continue

        old_conditions = [_cond_key(b.condition) for b in old_cb.branches]
        new_conditions = [_cond_key(b.condition) for b in new_cb.branches]
        if old_conditions != new_conditions:
            changes.append(
                f"combinational block '{target}': branch/priority order changed "
                f"from {old_conditions} to {new_conditions}"
            )

        for i, (ob, nb) in enumerate(zip(old_cb.branches, new_cb.branches)):
            if _op_key(ob.result_op) != _op_key(nb.result_op):
                changes.append(f"combinational block '{target}' branch {i}: result expression changed")

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



_grammar_override = os.environ.get("VSL_GRAMMAR_FILE")
if _grammar_override:
    _GRAMMAR_FILE = pathlib.Path(_grammar_override)
else:
    _vsl_model_name = os.environ.get("VSL_MODEL")
    if _vsl_model_name and _vsl_model_name != "gemini":
        _GRAMMAR_FILE = pathlib.Path(__file__).parent / f"grammar_{_vsl_model_name}.txt"
    else:
        _GRAMMAR_FILE = pathlib.Path(__file__).parent / "grammar.txt"
try:
    VSL_GRAMMAR_AND_EXAMPLES = _GRAMMAR_FILE.read_text(encoding="utf-8")
except FileNotFoundError:
    raise RuntimeError(
        f"Could not find {_GRAMMAR_FILE}. This file is required -- it holds "
        "the VSL grammar produced by discover_grammar.py. Make sure grammar.txt "
        "is in the same directory as vsl_core.py."
    )


if os.environ.get("VSL_MODEL") == "gpt-oss":
    from pydantic_ai import NativeOutput
    _gir_output_type = NativeOutput(VSLOutput)
else:
    _gir_output_type = VSLOutput

gir_agent = Agent(
    MODEL,
    name="GIR Agent",
    output_type=_gir_output_type,
    model_settings={"temperature": 0, "max_tokens": 16384},
    system_prompt=VSL_GRAMMAR_AND_EXAMPLES,
)