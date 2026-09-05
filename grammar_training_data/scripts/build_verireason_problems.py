"""Convert Nellyw888/VeriReason-RTL-Coder_7b_reasoning_tb_simple into a
problem file for grading via grade_verireason.py.

Also extracts each task's module header ("module name(...);" or
"module name #(...)(...);", no body, no endmodule) from its reference
solution -- main_vsl.py's convention: the final candidate is
header + rendered_body + "endmodule", never a full module rendered on its
own.

Keeps tb_result (the dataset's precomputed correct simulation output) so
grading can diff the candidate's actual output against it directly, instead
of parsing each testbench's own $display wording -- those aren't consistent
across tasks (some say "ERROR: ...", others "ASSERTION FAILED: ...", etc).

Usage:
    python3 build_verireason_problems.py --raw raw/verireason_tb_simple.jsonl --out converted/verireason_problems.jsonl
"""
import argparse
import json
import re


def extract_verilog(output_field: str) -> str | None:
    # the "output" field often has a ```verilog snippet inside <think>...
    # </think> (an illustrative fragment from the reasoning) AND the real,
    # complete module inside <answer>...</answer> -- prefer the answer
    # block; fall back to the last fenced block in the text (the answer
    # normally comes after the reasoning) rather than the first.
    answer_match = re.search(r"<answer>(.*?)</answer>", output_field, re.DOTALL)
    search_text = answer_match.group(1) if answer_match else output_field
    blocks = re.findall(r"```(?:verilog)?\s*\n(.*?)```", search_text, re.DOTALL)
    if not blocks:
        return None
    return blocks[-1].strip()


def extract_tb_module_name(tb_text: str) -> str | None:
    m = re.search(r"\bmodule\s+(\w+)", tb_text)
    return m.group(1) if m else None


def extract_module_header(verilog_text: str) -> str | None:
    m = re.search(r"\bmodule\s+\w+\s*", verilog_text)
    if not m:
        return None
    pos = m.end()
    # optional parameter block: module name #( ... )
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
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)
            solution = extract_verilog(rec["output"])
            tb_name = extract_tb_module_name(rec["tb"])
            header = extract_module_header(solution) if solution else None
            tb_result = rec.get("tb_result", "").strip()
            if not solution:
                drop_reasons[rec["id"]] = "no code block in output"
            elif not tb_name:
                drop_reasons[rec["id"]] = "no module name in tb"
            elif not header:
                drop_reasons[rec["id"]] = "no module header in solution"
            elif not tb_result:
                drop_reasons[rec["id"]] = "no tb_result"
            else:
                problem = {
                    "task_id": rec["id"],
                    "description": rec["instruction"].strip(),
                    "reference_solution": solution,
                    "module_interface": header,
                    "test": rec["tb"],
                    "tb_top_module": tb_name,
                    "tb_result": rec["tb_result"],
                }
                f_out.write(json.dumps(problem) + "\n")
                kept += 1
                continue
            dropped += 1

    print(f"wrote {kept} problems to {args.out} ({dropped} dropped)")
    for tid, reason in drop_reasons.items():
        print(f"  dropped {tid}: {reason}")


if __name__ == "__main__":
    main()
