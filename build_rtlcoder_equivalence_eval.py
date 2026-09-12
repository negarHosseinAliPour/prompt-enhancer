"""
Turns RTL-Coder_small (no testbenches, but a full reference Verilog module
per task in the 'output' field) into a genuinely GRADED eval/desc pair, in
the exact same {task_id, prompt, canonical_solution, test} / {task_id,
simple_description, detail_description} shape convert_v2.py produces for
v2 -- so it plugs straight into the existing main_vsl.py / compute_metrics.py
pipeline with zero pipeline changes.

How it works, per task:
  1. Parse the reference module's port list (handles both ANSI-style
     ports, e.g. `module m(input a, output reg [3:0] b);`, and old-style
     ports, e.g. `module m(a, b); input a; output reg [3:0] b;`).
  2. Rename the reference module RefModule, and re-emit its own header
     (same ports) as the TopModule header -- this is the fixed interface
     handed to the candidate, exactly as convert_v2.py does for v2.
  3. Generate a generic self-checking testbench: instantiate RefModule and
     TopModule side by side, drive identical randomized stimulus to every
     input port each cycle, and compare every output port. If a clock-like
     single-bit input is detected (name matches clk/clock, or is used after
     posedge/negedge in the reference body), the clock is toggled and
     stimulus/comparison happen once per clock period; otherwise (a purely
     combinational module) stimulus is applied and compared every #10 with
     no clock at all.
  4. Reports "Mismatches: %d in %d samples" -- the exact string
     convert_v2.py's run_sim() (and this pipeline's grading) already parses.

Tasks whose port list or reference body can't be parsed confidently are
skipped and counted, never silently miscompiled into a broken testbench.

Usage:
    python3 build_equivalence_eval.py \\
        --raw grammar_training_data/raw/rtl_coder_small.jsonl \\
        --out-prefix outputs/rtlcoder_equiv \\
        [--verify] [--limit N]

Writes <out-prefix>_eval.jsonl and <out-prefix>_desc.jsonl.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


CLOCK_NAME_RE = re.compile(r"^(clk|clock)[a-z0-9_]*$", re.IGNORECASE)


class ParseError(Exception):
    pass


def add_missing_reg(text: str) -> str:
    """Some reference modules declare an output as plain `output [w:0] name;`
    (implicit wire) but then drive it with a procedural (blocking or
    non-blocking) assignment inside an always block -- legal in some
    tools' lenient mode but rejected by iverilog ('not a valid l-value').
    This is a defect in the raw reference text, not something our
    generated testbench should fail on, so we patch the declaration
    in-place to add `reg` wherever the port is procedurally assigned."""
    always_blocks = re.findall(r"\balways\b.*?\bend\b", text, flags=re.DOTALL)
    assigned_names = set()
    for block in always_blocks:
        for am in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_$]*)\s*(\[[^\]]*\])?\s*<?=", block):
            assigned_names.add(am.group(1))

    def fix_decl(decl_match):
        direction, mods, width, names_blob = decl_match.groups()
        mods = mods or ""
        if direction != "output" or "reg" in mods:
            return decl_match.group(0)
        names = [n.strip() for n in names_blob.split(",")]
        if not any(n in assigned_names for n in names):
            return decl_match.group(0)
        w = width or ""
        return f"output reg {w}{names_blob};"

    decl_re = re.compile(
        r"\b(input|output|inout)\s+((?:reg\s+|wire\s+|signed\s+)*)(\[[^\]]*\]\s*)?([A-Za-z_][A-Za-z0-9_$,\s]*?)\s*;"
    )
    return decl_re.sub(fix_decl, text)


def _split_top_level_commas(text: str):
    """Splits on commas that are not inside () or []."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _find_matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    for j in range(open_idx, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    raise ParseError("unbalanced parens in module header")


def extract_module(text: str):
    """Returns (module_name, ports) where ports is a list of dicts:
    {name, direction, width} with width either None or a literal string
    like '[7:0]' (as written -- may reference localparams, kept verbatim)."""
    text = text.strip()
    m = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)", text)
    if not m:
        raise ParseError("no module declaration found")
    name = m.group(1)

    paren_open = text.index("(", m.end())
    paren_close = _find_matching_paren(text, paren_open)
    port_list_raw = text[paren_open + 1: paren_close]
    semi = text.index(";", paren_close)
    header_end = semi
    body_start = semi + 1
    end = text.rindex("endmodule")
    body = text[body_start:end]

    raw_ports = _split_top_level_commas(port_list_raw)
    if not raw_ports:
        raise ParseError("empty port list")

    ansi_dir_re = re.compile(
        r"^(input|output|inout)\s+(reg\s+|wire\s+|signed\s+)*(\[[^\]]*\]\s*)?([A-Za-z_][A-Za-z0-9_$]*)\s*(=.*)?$"
    )

    ports = []
    is_ansi = bool(ansi_dir_re.match(raw_ports[0]))

    if is_ansi:
        for tok in raw_ports:
            mm = ansi_dir_re.match(tok)
            if not mm:
                raise ParseError(f"could not parse ANSI port token: {tok!r}")
            direction, _mods, width, pname, _default = mm.groups()
            ports.append({"name": pname, "direction": direction,
                          "width": width.strip() if width else None})
    else:
        # non-ANSI: raw_ports are bare names in declared order; directions
        # and widths come from input/output/inout declarations in the body.
        bare_names = []
        for tok in raw_ports:
            mm = re.match(r"^([A-Za-z_][A-Za-z0-9_$]*)$", tok)
            if not mm:
                raise ParseError(f"non-ANSI port token not a bare name: {tok!r}")
            bare_names.append(mm.group(1))

        decl_re = re.compile(
            r"\b(input|output|inout)\s+(reg\s+|wire\s+|signed\s+)*(\[[^\]]*\]\s*)?([A-Za-z_][A-Za-z0-9_$,\s]*?)\s*;"
        )
        info = {}
        for dm in decl_re.finditer(body):
            direction, _mods, width, names_blob = dm.groups()
            width = width.strip() if width else None
            for nm in names_blob.split(","):
                nm = nm.strip()
                if nm:
                    info[nm] = {"direction": direction, "width": width}
        for nm in bare_names:
            if nm not in info:
                raise ParseError(f"no input/output/inout declaration found for port {nm!r}")
            ports.append({"name": nm, "direction": info[nm]["direction"],
                          "width": info[nm]["width"]})

    return name, ports, header_end, body_start, end


def detect_clock(ports, body):
    single_bit_inputs = [p for p in ports if p["direction"] == "input" and not p["width"]]
    # Prefer a name match.
    for p in single_bit_inputs:
        if CLOCK_NAME_RE.match(p["name"]):
            return p["name"]
    # Fall back to usage: any single-bit input used right after posedge/negedge.
    for p in single_bit_inputs:
        if re.search(r"\b(posedge|negedge)\s+" + re.escape(p["name"]) + r"\b", body):
            return p["name"]
    return None


def build_ref_and_top(name, ports, full_text, header_end, body_start, end):
    body = full_text[body_start:end].strip("\n")
    ref_body_renamed = re.sub(r"\b" + re.escape(name) + r"\b", "RefModule", full_text[:body_start]) \
        + body + "\nendmodule\n"
    # The RefModule text: header renamed + original body + endmodule.
    ref_header = re.sub(r"\b" + re.escape(name) + r"\b", "RefModule", full_text[:header_end + 1])
    ref_module_text = ref_header + "\n" + body + "\nendmodule\n"

    top_header = re.sub(r"\b" + re.escape(name) + r"\b", "TopModule", full_text[:header_end + 1])
    return top_header.strip() + "\n", ref_module_text


def width_bits(width):
    if not width:
        return 1
    m = re.match(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", width)
    if not m:
        return None  # parameterized/non-literal width -- can't size a $random mask safely
    hi, lo = int(m.group(1)), int(m.group(2))
    return abs(hi - lo) + 1


def gen_testbench(ports, clock_name, num_samples=200):
    inputs = [p for p in ports if p["direction"] == "input"]
    outputs = [p for p in ports if p["direction"] in ("output", "inout")]
    if not outputs:
        raise ParseError("no output ports -- nothing to compare")

    stim_inputs = [p for p in inputs if p["name"] != clock_name]
    for p in stim_inputs + outputs:
        if width_bits(p["width"]) is None:
            raise ParseError(f"port {p['name']!r} has a non-literal width {p['width']!r}")

    def decl(prefix, plist):
        lines = []
        for p in plist:
            w = f"{p['width']} " if p["width"] else ""
            lines.append(f"  {prefix} {w}{p['name']};")
        return "\n".join(lines)

    ref_ports_conn = ", ".join(f".{p['name']}({p['name']})" for p in ports)
    top_ports_conn = ", ".join(f".{p['name']}({p['name']})" for p in ports)

    stim_assigns = "\n".join(
        f"      {p['name']} = $random;" for p in stim_inputs
    )
    out_compare = " || ".join(f"(ref_{p['name']} !== top_{p['name']})" for p in outputs) \
        if False else None  # placeholder, real compare built below

    # Separate reg copies of each shared output aren't needed: outputs are
    # driven independently by each instance since we instantiate them with
    # distinct wire names (ref_<name> / top_<name>), while inputs are the
    # SAME regs fed to both instances (identical stimulus).
    ref_out_wires = "\n".join(
        f"  wire {p['width'] + ' ' if p['width'] else ''}ref_{p['name']};" for p in outputs
    )
    top_out_wires = "\n".join(
        f"  wire {p['width'] + ' ' if p['width'] else ''}top_{p['name']};" for p in outputs
    )
    ref_conn = ", ".join(
        [f".{p['name']}({p['name']})" for p in stim_inputs]
        + ([f".{clock_name}({clock_name})"] if clock_name else [])
        + [f".{p['name']}(ref_{p['name']})" for p in outputs]
    )
    top_conn = ", ".join(
        [f".{p['name']}({p['name']})" for p in stim_inputs]
        + ([f".{clock_name}({clock_name})"] if clock_name else [])
        + [f".{p['name']}(top_{p['name']})" for p in outputs]
    )

    mismatch_terms = " || ".join(f"(ref_{p['name']} !== top_{p['name']})" for p in outputs)

    input_reg_decl = "\n".join(
        f"  reg {p['width'] + ' ' if p['width'] else ''}{p['name']};" for p in stim_inputs
    )

    if clock_name:
        tb = f"""
module tb;
{input_reg_decl}
  reg {clock_name};
{ref_out_wires}
{top_out_wires}
  integer mismatches = 0;
  integer samples = 0;
  integer i;

  RefModule ref_dut ({ref_conn});
  TopModule top_dut ({top_conn});

  always #5 {clock_name} = ~{clock_name};

  initial begin
    {clock_name} = 0;
{stim_assigns}
    @(negedge {clock_name});
    for (i = 0; i < {num_samples}; i = i + 1) begin
{stim_assigns}
      @(negedge {clock_name});
      samples = samples + 1;
      if ({mismatch_terms}) mismatches = mismatches + 1;
    end
    $display("Mismatches: %0d in %0d samples", mismatches, samples);
    $finish;
  end
endmodule
"""
    else:
        tb = f"""
module tb;
{input_reg_decl}
{ref_out_wires}
{top_out_wires}
  integer mismatches = 0;
  integer samples = 0;
  integer i;

  RefModule ref_dut ({ref_conn});
  TopModule top_dut ({top_conn});

  initial begin
    for (i = 0; i < {num_samples}; i = i + 1) begin
{stim_assigns}
      #10;
      samples = samples + 1;
      if ({mismatch_terms}) mismatches = mismatches + 1;
    end
    $display("Mismatches: %0d in %0d samples", mismatches, samples);
    $finish;
  end
endmodule
"""
    return tb.strip() + "\n"


def run_sim(source: str, timeout: float = 30.0):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.sv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        cmd = f"iverilog -Wall -Wno-timescale -g2012 -s tb -o {d}/t.vvp {path} && vvp -n {d}/t.vvp"
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                               timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return "timeout", ""
    m = re.search(r"Mismatches: (\d+) in (\d+) samples", r.stdout)
    if m:
        return ("pass" if m.group(1) == "0" else f"fail {m.group(1)}/{m.group(2)}"), r.stdout
    if r.returncode != 0:
        return "compile_error", (r.stdout + r.stderr)
    return "no_result", (r.stdout + r.stderr)


def build_one(task_id, instruction, output_code):
    output_code = add_missing_reg(output_code)
    name, ports, header_end, body_start, end = extract_module(output_code)
    clock_name = detect_clock(ports, output_code[body_start:end])
    top_header, ref_module_text = build_ref_and_top(name, ports, output_code, header_end, body_start, end)
    testbench = gen_testbench(ports, clock_name)

    body_only = output_code[body_start:end].strip("\n")
    eval_record = {
        "task_id": task_id,
        "prompt": top_header,
        "canonical_solution": body_only + "\nendmodule\n",
        "test": testbench + "\n" + ref_module_text,
    }
    desc_record = {
        "task_id": task_id,
        "simple_description": instruction.strip().splitlines()[0] if instruction.strip() else "",
        "detail_description": instruction.strip(),
    }
    return eval_record, desc_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify", action="store_true",
                     help="Run every generated reference-vs-reference "
                          "testbench (RefModule playing both roles) to "
                          "confirm the testbench itself is sound before "
                          "trusting it to grade candidates.")
    args = ap.parse_args()

    eval_records, desc_records = [], []
    skipped = {}

    with open(args.raw, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if args.limit and idx >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task_id = f"rtlcoder_{idx}"
            try:
                eval_rec, desc_rec = build_one(task_id, rec.get("instruction", ""), rec.get("output", ""))
            except ParseError as e:
                skipped[task_id] = str(e)
                continue
            except Exception as e:
                skipped[task_id] = f"unexpected error: {e}"
                continue
            eval_records.append(eval_rec)
            desc_records.append(desc_rec)

    print(f"Parsed {len(eval_records)} / {len(eval_records) + len(skipped)} tasks")

    if args.verify:
        from concurrent.futures import ThreadPoolExecutor

        def check(rec):
            # Sanity check: candidate == reference should always show 0 mismatches.
            src = rec["test"] + "\n" + rec["prompt"] + "\n" + rec["canonical_solution"]
            src = src.replace("module TopModule", "module TopModule /*sanity*/")
            status, _out = run_sim(src)
            return rec["task_id"], status

        print("Verifying each generated testbench against its own reference (should all pass)...")
        with ThreadPoolExecutor(8) as ex:
            results = dict(ex.map(check, eval_records))
        bad = {k: v for k, v in results.items() if v != "pass"}
        print(f"  {len(eval_records) - len(bad)}/{len(eval_records)} self-checks passed")
        for tid, why in sorted(bad.items())[:30]:
            print(f"  BAD {tid}: {why}")
        eval_records = [r for r in eval_records if r["task_id"] not in bad]
        desc_records = [r for r in desc_records if r["task_id"] not in bad]
        for tid, why in bad.items():
            skipped[tid] = f"testbench self-check failed: {why}"

    for kind, records in (("eval", eval_records), ("desc", desc_records)):
        path = f"{args.out_prefix}_{kind}.jsonl"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(records)} records to {path}")

    print(f"\nSkipped {len(skipped)} of {idx + 1} raw tasks:")
    reasons = {}
    for why in skipped.values():
        key = why.split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    for k, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {c:5d}  {k}")


if __name__ == "__main__":
    main()