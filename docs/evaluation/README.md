# NeuroClip Evaluation Data Pack

This folder contains a practical starter pack for evaluating summarization and compression modules.

## Files

- summarization_eval_pack.csv
  - 50 ready query cases for educational video summarization/retrieval testing.
  - Use this as your query benchmark sheet.

- compression_profiles.csv
  - Standard test profiles covering tiny, small, medium, large, and XL outputs.
  - Use to measure quality vs size trade-offs.

- generate_compression_test_set.ps1
  - PowerShell script to generate different-size test videos from your local input videos.

- educational_video_source_candidates.csv
  - Candidate educational sources/channels to collect legally usable evaluation videos.

## Recommended Dataset Size

- Summarization: 20-40 videos, 3-5 queries per video (at least 100 query-video pairs).
- Compression: 15-30 videos across different content styles:
  - talking head,
  - slide-heavy,
  - handwriting/whiteboard,
  - high motion,
  - mixed visuals.

## How to Prepare Summarization Ground Truth

1. Select your videos from the source candidates file.
2. For each selected video, map 3-5 rows from summarization_eval_pack.csv.
3. Annotate relevant segment times manually.
4. Save annotations in a sheet with columns:
   - video_id
   - query
   - gt_start
   - gt_end
   - relevance_label (1-3)

## How to Generate Compression Test Videos

Run from repo root:

```powershell
./docs/evaluation/generate_compression_test_set.ps1 -InputDir ./backend/uploads -OutputDir ./docs/evaluation/generated_compression_set
```

You can put any seed videos in InputDir. The script generates multiple durations/resolutions/bitrates for stress testing.

## Metrics to Report

### Summarization

- Precision@K, Recall@K, nDCG@K
- Temporal IoU overlap
- Boundary error (seconds)
- End-to-end query latency

### Compression

- Compression ratio
- Size reduction percent
- SSIM / PSNR / VMAF (if available)
- Processing time per minute of video

## Notes

- Keep the same test split for all model/config comparisons.
- Store all results in CSV for easy graph generation.
- If you need redistribution-safe media, prioritize open-license sources.
