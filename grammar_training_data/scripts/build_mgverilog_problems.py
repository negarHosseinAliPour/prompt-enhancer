"""Convert GaTech-EIC/MG-Verilog into a lightweight, UNGRADED test pool for
discover_grammar.py's structural-validity test role only (no testbenches
exist for this dataset -- same role as RTL-Coder_small, see
build_rtlcoder_problems.py).

Each raw record's 'description' field is a dict with three chat-template-
wrapped variants (block_summary, high_level_global_summary,
detailed_global_summary), each embedding the module header inline after the
literal text "Module header:". We use detailed_global_summary (the most
complete natural-language description) and strip the surrounding
LLaMA-chat template boilerplate to recover just the description text and
the module header.

Usage:
    python3 build_mgverilog_problems.py --raw raw/mg_verilog_raw.jsonl --out converted/mgverilog_problems.jsonl
"""
import argparse
import json
import re

DESC_START = "unless otherwise stated."
HEADER_MARK = "Module header:"


def extract(desc_text: str) -> tuple[str | None, str | None]:
    if DESC_START not in desc_text or HEADER_MARK not in desc_text:
        return None, None
    after_start = desc_text.split(DESC_START, 1)[1]
    if HEADER_MARK not in after_start:
        return None, None
    description, rest = after_start.split(HEADER_MARK, 1)
    description = description.strip()

    rest = rest.strip()
    # rest looks like: "module foo (...);\n [/INST]\n" (or end of string)
    end_marker = "[/INST]"
    header = rest.split(end_marker, 1)[0].strip() if end_marker in rest else rest.strip()
    if not header.startswith("module"):
        m = re.search(r"\bmodule\b", header)
        if not m:
            return description or None, None
        header = header[m.start():]
    if not header.rstrip().endswith(";"):
        # module header should end at the first top-level ');' -- if the
        # split grabbed extra trailing junk, trim back to the last ');'
        idx = header.rfind(");")
        if idx == -1:
            return description or None, None
        header = header[: idx + 2]
    return description or None, header + "\n"


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
            task_id = f"mgverilog_{idx}"
            desc_blob = (rec.get("description") or {}).get("detailed_global_summary") or ""
            description, header = extract(desc_blob)
            if not description:
                drop_reasons[task_id] = "could not extract description"
            elif not header:
                drop_reasons[task_id] = "could not extract module header"
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
