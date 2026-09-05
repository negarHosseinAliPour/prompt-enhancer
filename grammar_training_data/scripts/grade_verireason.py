"""Grading adapter for VeriReason-style problems: dynamic tb top-module name
(no hardcoded "-s tb"), and correctness measured by diffing the candidate's
actual test_vectors.txt against the dataset's precomputed tb_result --
NOT by parsing the testbench's own $display wording, which is inconsistent
across tasks (some print "ERROR: ...", others "ASSERTION FAILED: ...", so a
keyword search under- or over-counts failures depending on the task).

Returns:
    {
        "score": float in [0, 1],
        "gradeable": bool,   # False = compile/tooling failure, not a code-quality signal
        "detail": str,
        "stdout": str,
        "stderr": str,
    }
"""
import os
import shutil
import subprocess
import tempfile


def _normalize_vector_line(line: str) -> str:
    # tb_result and test_vectors.txt both hold whitespace-separated fields
    # (hex or binary); collapse internal whitespace so formatting quirks
    # (extra spaces, trailing \r) don't count as mismatches.
    return " ".join(line.split())


def check_correctness_verireason(problem: dict, completion: str, timeout: float = 15.0) -> dict:
    tb_top = problem.get("tb_top_module")
    if not tb_top:
        return {"score": 0.0, "gradeable": False, "detail": "no tb_top_module on problem record", "stdout": "", "stderr": ""}

    expected_lines = [_normalize_vector_line(l) for l in problem["tb_result"].splitlines() if l.strip()]
    if not expected_lines:
        return {"score": 0.0, "gradeable": False, "detail": "no tb_result to compare against", "stdout": "", "stderr": ""}

    verilog_text = completion + "\n\n" + problem["test"]

    tmpdir = tempfile.mkdtemp(prefix="verireason_grade_")
    try:
        src_path = os.path.join(tmpdir, f"{problem['task_id']}.sv")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(verilog_text)

        compile_cmd = [
            "iverilog", "-Wall", "-Winfloop", "-Wno-timescale", "-g2012",
            "-s", tb_top, "-o", "test.vvp", f"{problem['task_id']}.sv",
        ]
        try:
            cp = subprocess.run(compile_cmd, cwd=tmpdir, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"score": 0.0, "gradeable": False, "detail": "failed: compile timeout", "stdout": "", "stderr": "timeout"}

        if cp.returncode != 0:
            return {"score": 0.0, "gradeable": False, "detail": "failed: compile error", "stdout": cp.stdout, "stderr": cp.stderr}

        try:
            rp = subprocess.run(["vvp", "-n", "test.vvp"], cwd=tmpdir, capture_output=True, text=True, timeout=timeout)
            out, err = rp.stdout, rp.stderr
        except subprocess.TimeoutExpired as e:
            out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="replace")
            return {"score": 0.0, "gradeable": False, "detail": "failed: simulation timeout", "stdout": out, "stderr": "timeout"}

        vectors_path = os.path.join(tmpdir, "test_vectors.txt")
        if not os.path.exists(vectors_path):
            return {"score": 0.0, "gradeable": False, "detail": "failed: no test_vectors.txt produced", "stdout": out, "stderr": err}

        with open(vectors_path, encoding="utf-8", errors="replace") as vf:
            actual_lines = [_normalize_vector_line(l) for l in vf if l.strip()]

        n_total = len(expected_lines)
        n_compare = min(len(actual_lines), n_total)
        n_mismatch = sum(1 for i in range(n_compare) if actual_lines[i] != expected_lines[i])
        n_mismatch += n_total - n_compare  # missing lines (sim aborted early) count as wrong

        score = max(0.0, min(1.0, 1.0 - (n_mismatch / n_total)))
        detail = f"{n_mismatch} mismatch(es) out of {n_total} test vectors (vs. tb_result) -> score {score:.4f}"
        return {"score": score, "gradeable": True, "detail": detail, "stdout": out, "stderr": err}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
