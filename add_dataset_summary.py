"""
Appends (or updates) one row in a master CSV summarizing a dataset run's
results: pass@1, resolve_rate, avg_score (continuous), self-healing
breakdown, and token/time metrics -- all from the pipeline's own history
file. Meant to be run once per dataset/run so the master CSV accumulates
a comparable row per dataset (baseline vs force-VSL, RTLLM, VerilogEval v1, etc).

Usage:
    python3 add_dataset_summary.py <history_file.jsonl> <dataset_label> [master_csv]

Example:
    python3 add_dataset_summary.py outputs/verilogeval_v2/baseline_history_verilogeval_v2_baseline_fixed_merged.jsonl VerilogEval_v2_Baseline
    python3 add_dataset_summary.py outputs/verilogeval_v2/enhanced_vsl_forcevsl_history_verilogeval_v2_forcevsl_v8_FINAL.jsonl VerilogEval_v2_ForceVSL_v8
"""
import json
import sys
import csv
import os

FIELDS = [
    'dataset', 'total_tasks',
    'pass_at_1_count', 'pass_at_1_pct', 'pass_at_1_avg_score_pct',
    'resolve_count', 'resolve_rate_pct', 'resolve_avg_score_pct',
    'improved_count', 'zero_to_full', 'zero_to_partial', 'partial_to_full', 'partial_to_partial',
    'still_zero_after_pipeline',
    'avg_model_calls', 'avg_total_tokens', 'total_tokens_all_tasks',
    'avg_llm_wall_time_s', 'avg_grading_wall_time_s', 'avg_total_wall_time_s',
    'avg_retries', 'avg_num_rounds',
]


def main():
    if len(sys.argv) < 3:
        print('Usage: python3 add_dataset_summary.py <history_file.jsonl> <dataset_label> [master_csv]')
        sys.exit(1)

    history_file = sys.argv[1]
    dataset_label = sys.argv[2]
    master_csv = sys.argv[3] if len(sys.argv) > 3 else 'outputs/vsl_master_summary.csv'

    history = {}
    with open(history_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            history[d['task_id']] = d

    total = len(history)

    pass1_scores = []
    resolve_scores = []
    improved = []
    same = []
    all_metrics = []

    for task_id, d in history.items():
        rounds = d.get('rounds', [])
        first_round = next((r for r in rounds if r.get('round') == -1), rounds[0] if rounds else None)
        p1 = first_round.get('execution_score', 0) if first_round else 0
        final = d.get('final_execution_score', 0)
        pass1_scores.append(p1)
        resolve_scores.append(final)
        if final > p1:
            improved.append((task_id, p1, final))
        else:
            same.append((task_id, p1, final))
        m = d.get('metrics')
        if m:
            all_metrics.append(m)

    pass1_count = sum(1 for s in pass1_scores if s >= 1.0)
    resolve_count = sum(1 for s in resolve_scores if s >= 1.0)

    zero_to_full = sum(1 for _, p1, f in improved if p1 == 0 and f >= 1.0)
    zero_to_partial = sum(1 for _, p1, f in improved if p1 == 0 and 0 < f < 1.0)
    partial_to_full = sum(1 for _, p1, f in improved if 0 < p1 < 1.0 and f >= 1.0)
    partial_to_partial = sum(1 for _, p1, f in improved if 0 < p1 < 1.0 and 0 < f < 1.0 and f > p1)
    still_zero = sum(1 for _, p1, f in same if f == 0)

    n_m = len(all_metrics)
    def avg(key):
        return round(sum(m.get(key, 0) for m in all_metrics) / n_m, 3) if n_m else 0

    row = {
        'dataset': dataset_label,
        'total_tasks': total,
        'pass_at_1_count': pass1_count,
        'pass_at_1_pct': round(pass1_count / total * 100, 2),
        'pass_at_1_avg_score_pct': round(sum(pass1_scores) / total * 100, 2),
        'resolve_count': resolve_count,
        'resolve_rate_pct': round(resolve_count / total * 100, 2),
        'resolve_avg_score_pct': round(sum(resolve_scores) / total * 100, 2),
        'improved_count': len(improved),
        'zero_to_full': zero_to_full,
        'zero_to_partial': zero_to_partial,
        'partial_to_full': partial_to_full,
        'partial_to_partial': partial_to_partial,
        'still_zero_after_pipeline': still_zero,
        'avg_model_calls': avg('model_calls'),
        'avg_total_tokens': avg('total_tokens'),
        'total_tokens_all_tasks': sum(m.get('total_tokens', 0) for m in all_metrics),
        'avg_llm_wall_time_s': avg('llm_wall_time_s'),
        'avg_grading_wall_time_s': avg('grading_wall_time_s'),
        'avg_total_wall_time_s': avg('total_wall_time_s'),
        'avg_retries': avg('retries'),
        'avg_num_rounds': avg('num_rounds'),
    }

    existing_rows = []
    if os.path.exists(master_csv):
        with open(master_csv, newline='') as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader if r['dataset'] != dataset_label]

    existing_rows.append(row)

    os.makedirs(os.path.dirname(master_csv), exist_ok=True)
    with open(master_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)

    print(json.dumps(row, indent=2))
    print('Updated {} (now {} row(s))'.format(master_csv, len(existing_rows)))


if __name__ == '__main__':
    main()
