import json

for label, fname in [("Baseline", "outputs/baseline_history.jsonl"), ("Enhanced", "outputs/enhanced_vsl_history.jsonl")]:
    scores = {}
    with open(fname) as f:
        for line in f:
            d = json.loads(line)
            scores[d["task_id"]] = d["final_execution_score"]
    avg = sum(scores.values()) / len(scores)
    full = sum(1 for v in scores.values() if v >= 1.0)
    print(f"{label}: average_score={avg:.4f}, pass@1={full}/{len(scores)} ({100*full/len(scores):.1f}%)")