"""Convert Nellyw888/RTL-Coder_small into a lightweight, UNGRADED test pool
for discover_grammar.py's structural-validity test role only (no testbenches
exist for this dataset, so it can never serve as an execution-graded
reference-example source -- see build_verireason_problems.py for that role).

Each output record is just {task_id, description, module_interface}: no
reference_solution, no test, no tb_result. discover_grammar.py's _try_one()
only checks that a candidate grammar produces VSL that parses/validates/
renders -- it never runs a simulation for this role -- so this is all it
needs.

Usage:
    python3 build_rtlcoder_problems.py --raw raw/rtl_coder_small.jsonl --out converted/rtlcoder_small_problems.jsonl
"""
import argparse
import json
import re


def extract_module_header(verilog_text: str) -> str | None:
    m = re.search(r"\bmodule\s+\w+\s*", verilog_text)
    if not m:
        return None
    pos = m.end()
    if pos < len(verilog_text) and verilog_text[pos] == "#":
        popen = verilog_text.find("(", pos)
        if popen == -1:
            return None
        depth = 0
        pclose = None
        for i in range(popen, len(verilog_text)):
            if verilog_text[i] == "(":
                depth += 1
            elif verilog_text[i] == ")":
                depth -= 1
                if depth == 0:
                    pclose = i
                    break
        if pclose is None:
            return None
        pos = pclose + 1
        while pos < len(verilog_text) and verilog_text[pos] in " \t\r\n":
            pos += 1
    if pos >= len(verilog_text) or verilog_text[pos] != "(":
        return None
    depth = 0
    for i in range(pos, len(verilog_text)):
        if verilog_text[i] == "(":
            depth += 1
        elif verilog_text[i] == ")":
            depth -= 1
            if depth == 0:
                semi = verilog_text.find(";", i)
                if semi == -1:
                    return None
                return verilog_text[m.start():semi + 1] + "\n"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kept, dropped = 0, 0
    drop_reasons = {}
    with open(args.raw, encoding="utf-8") as f_in, open(args.out, "w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            if not line.strip():
                continue
            rec = json.loads(line)
            task_id = rec.get("task_id") or f"rtlcoder_small_{idx}"
            description = (rec.get("instruction") or "").strip()
            header = extract_module_header(rec.get("output") or "")
            if not description:
                drop_reasons[task_id] = "no instruction"
            elif not header:
                drop_reasons[task_id] = "no module header in output"
            else:
                problem = {
                    "task_id": task_id,
                    "description": description,
                    "module_interface": header,
                }
                f_out.write(json.dumps(problem) + "\n")
                kept += 1
                continue
            dropped += 1

    print(f"wrote {kept} problems to {args.out} ({dropped} dropped)")
    for tid, reason in list(drop_reasons.items())[:20]:
        print(f"  dropped {tid}: {reason}")


if __name__ == "__main__":
    main()
