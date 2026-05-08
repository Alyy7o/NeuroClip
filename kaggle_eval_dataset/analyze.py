import csv

with open('kaggle_eval_dataset/evaluation_results.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"{'ID':<8} {'Pred Range':<22} {'GT Range':<22} {'IoU':<8} {'Clips'}")
print("-" * 80)
for r in rows:
    ps = float(r['pred_start'])
    pe = float(r['pred_end'])
    gs = float(r['gt_start'])
    ge = float(r['gt_end'])
    if ps > 0 or pe > 0:
        pred = f"{ps:.1f}s - {pe:.1f}s"
    else:
        pred = "NO CLIPS RETURNED"
    gt = f"{gs:.1f}s - {ge:.1f}s"
    print(f"{r['id']:<8} {pred:<22} {gt:<22} {r['iou']:<8} {r['num_clips']}")

print(f"\n--- Summary ---")
print(f"Total queries: {len(rows)}")
print(f"Queries with clips: {sum(1 for r in rows if int(r['num_clips']) > 0)}")
print(f"Queries with NO clips: {sum(1 for r in rows if int(r['num_clips']) == 0)}")
has_sum = sum(1 for r in rows if r.get('topic_explanation', '').strip())
print(f"Queries with topic explanation: {has_sum}")
