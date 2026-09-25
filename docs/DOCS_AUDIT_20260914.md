# Documentation audit — 2026-09-14

Checks the eleven files in `docs/` and `README.md` against the code, the datasets
on disk, the retained evidence under `results/`, and the live rig inventories.
Every claim below was re-derived from those sources, not read back from the prose.
No document, label, dataset or setting was changed by this audit.

## What holds

The quantitative content of these documents is sound. The following were
recomputed and match exactly:

| Check | Result |
| --- | --- |
| Relative Markdown links in all twelve files | All resolve |
| Documented CLI flags against `argparse` (`run.py`, `label_video.py`, `hard_mining.py`, `evaluate_inputs.py`, `evaluate_models.py`, `compare_trials.py`, `analyze_glare.py`, `audit_dataset.py`, `repair_dataset.py`) | All exist, including `--modes native_color_clahe`, `--min-highlight`, `--confirmed-negatives`, `--verify-images` |
| Three checkpoint SHA-256 digests in EVALUATION.md | Byte-for-byte match with `weights/` |
| Dataset counts in AUDIT_20260914.md | Match disk exactly: color 2,466 / 831 / 815 (1,686 / 690 / 258 positive), mono 6,012 / 856 / 257 (2,910 / 452 / 229) |
| Derived totals: 11,237 audited images; 9,876 image/checkpoint pairs; 704 + 111 = 815; 168 + 536 = 704 | Consistent |
| 15 precision/recall/F1 triples in EVALUATION.md and TRACKING_SEGMENTATION.md | All recompute from their own TP/FP/FN |
| 12 precision/recall/F1 values in the six-checkpoint table | Match `results/audit_20260914/statistics.json` to the printed digit |
| Tile-NMS ablation (FP 515 → 237, TP 365 → 357; band 218 → 170, 353 → 351) | Matches `tile_nms_ablation.json` and the ROI table |
| Review-queue sizes in DATASET_REVIEW_UI.md (108 / 455 / 492, 246 pairs, 5,016 frames) | Match the live `ReviewStore` backend exactly |
| Source-dataset counts in AUDIT.md (4,524 / 492; 1,269 / 66 positive) | Match `source_dataset_audit.json` |

The 120-test Python suite and the Node viewer regression pass.

## Findings

### D1 — The name-collision queue holds different images, and the documents do not say so

AUDIT.md calls this P1 "Recording identifiers collide". DATASET_REVIEW_UI.md
calls the 492 frames "совпадения имён, а не доказанные дубликаты изображения".
MANUAL_DATA_REVIEW_20260914.md says "Shared names do not establish identical
pixels". All three hedge. The pixels settle it:

| Measurement over all 246 shared stems | Value |
| --- | ---: |
| Pairs with identical file bytes | 0 |
| Pairs with identical decoded pixels | 0 |
| Pairs with pixel MAE below 1/255 | 0 |
| Pixel MAE, minimum / median / maximum | 38.20 / 47.11 / 51.68 |

For scale, the genuine duplicate recording in DATASET_REPAIR.md measured MAE
0.000. A median of 47/255 is not a near-duplicate, a re-encode or an adjacent
frame — these are two different recordings that both carry the generic
`ShipCam0_00000…00245` stem. The train side holds 246 background labels and no
positives; the val side holds 18 positives.

Effect: 492 frames sit in a visual review queue, 192 of them already decided,
for a question no amount of looking at the images can answer. The defect is a
naming one — a generic recording key reused by two captures — and it is fixed by
recovering provenance, not by re-inspecting pixels. The documents should state
the measurement, retitle the queue as provenance-only, and drop the side-by-side
image comparison from that queue's instructions. This does not clear the
recording-identity work; it removes the leakage hypothesis from it, and with it
the P1 rating that sits next to the real leakage found in DATASET_REPAIR.md.

### D2 — README names the wrong Python

README.md:30 instructs `python3.11 -m venv venv`. The project venv is Python
3.12.11, and AUDIT_20260914.md:143 records 3.12.11 as the runtime that produced
every measurement in it. Following the README builds an interpreter no reported
result was measured on.

### D3 — Eight of eleven documents are untracked

`git status` for `docs/`:

| State | Files |
| --- | --- |
| Tracked | AUDIT.md, EVALUATION.md, GLARE_REVIEW.md, hydrorl-proposer-input.patch |
| Untracked | AUDIT_20260914.md, DATASET_REPAIR.md, DATASET_REVIEW_UI.md, FOREGROUND_BACKGROUND_PLAN.md, GLARE_PLAN.md, MANUAL_DATA_REVIEW_20260914.md, RIG_STATISTICS.md, TRACKING_SEGMENTATION.md |

README.md is tracked and links to all eight. A fresh clone gets the current
README with eight broken links, including the two the README presents first as
the latest measured review and the training-provenance record.

### D4 — The documented language rule is not what the project does

README.md states "User-facing text, comments and documentation are in English";
AUDIT.md's P3 disposition claims "English UI/messages/notebooks/docs". Actual
Russian-language content:

| File | Cyrillic lines |
| --- | ---: |
| `docs/DATASET_REVIEW_UI.md` | 53 (the whole document) |
| `viewer/dataset_review.html` | 32 |
| `viewer/dataset_review.js` | 21 |
| `viewer/dataset_review.py` | 1 (`'name': 'С рига'`, line 142) |
| `viewer/control_panel.html` | 1 (the link into the review UI, line 401) |

The newest interface and its guide are Russian while the rest is English. Either
claim or practice has to move; the audit does not assume which. Note that
`viewer/dataset_review.py` mixes both within one module — line 142 is Russian,
the export `limitations` string on line 209 is English.

### D5 — GLARE_PLAN.md's trial matrix describes settings that are already live

The plan states the cameras allow "500–65,487 microseconds … with exposure
compensation 0 EV", calls 0.0 EV the "Current baseline" (trial A), proposes
−0.5 EV as trial B, and offers `exposure_max_us=10000` as a later well-lit trial.
The 2026-09-14 inventories read:

| Camera | exposure_compensation | exposure_max_us |
| --- | ---: | ---: |
| Rig 0 cam0 | −0.5 EV | 65,487 |
| Rig 0 cam1 | 0.0 EV | 65,487 |
| Rig 1 cam0 | 0.0 EV | 10,000 |
| Rig 1 cam1 | −0.5 EV | 10,000 |

So trial B is already the live setting on two of four cameras, the 10 ms cap is
already in force on both rig-1 cameras, and the "allow 500–65,487 µs" sentence
holds for rig 0 only. Someone executing the plan as written would record a
"baseline" that is not the baseline on half the cameras.

The same table carries a second point neither GLARE_PLAN.md nor RIG_STATISTICS.md
makes: the four cameras sit in two different exposure states, split across both
rigs. Any comparison between rig 0 and rig 1 recordings, including the example
`compare_trials.py` invocation in RIG_STATISTICS.md that pairs
`20260911-144611/ShipCam0` with `20260911-144607/ShipCam1`, is confounded by an
exposure-cap difference of 6.5x before any scene difference is considered.

### D6 — AUDIT_20260914.md's test count is stale by the same day

It reports "**110 Python tests pass**". The suite now runs 120: the dataset
review UI shipped later on 2026-09-14 (DATASET_REVIEW_UI.md 12:47 against
AUDIT_20260914.md 11:31) and brought `tests/test_dataset_review.py` with it.
The audit document is the one README points to as the latest measured review,
so its verification section is the one a reader checks against.

### D7 — README's layout table has drifted

Not listed anywhere in README: `evaluate_models.py`, `scripts/report_statistics.py`,
`scripts/rig_inventory.py`. RIG_STATISTICS.md makes the first two the documented
route to the statistics AUDIT_20260914.md reports, and the third is what produces
the rig inventories quoted in D5 — so the README omits the tooling behind the
numbers it leads with. The `scripts/` row ("rig mount, training retry loop,
batch processing on the rig") and the `docs/` row ("audit, evaluation and glare
reports") both predate the rig-statistics and planning documents.

### D8 — The rig table reports one camera per rig

AUDIT_20260914.md's "Configured camera exposure compensation" row gives
"cam0: −0.5 EV" for rig 0 and "cam1: −0.5 EV" for rig 1. Both are correct and
the headers name the camera, but the second camera of each rig is at 0.0 EV and
is not shown. A reader scanning the row sees two rigs that agree, where the four
cameras actually split two and two.

## Suggested order

1. Commit the eight untracked documents (D3) — until then every other correction
   lives only on this machine.
2. Correct README's Python version (D2) and add the three missing tools (D7).
3. Rewrite the name-collision queue's purpose in DATASET_REVIEW_UI.md, AUDIT.md
   and MANUAL_DATA_REVIEW_20260914.md around the measurement in D1, and stop
   asking for a visual duplicate check there.
4. Update GLARE_PLAN.md's baseline and trial matrix against the current rig
   settings, and record the cross-rig exposure split in RIG_STATISTICS.md (D5, D8).
5. Settle the language rule (D4) and refresh the test count (D6).

None of these are measurement errors. Every number these documents report is
reproducible from the retained evidence; what has drifted is setup instructions,
version control state, the framing of one review queue, and a camera
configuration the plans were written before.
