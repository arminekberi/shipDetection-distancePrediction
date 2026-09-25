"""Render the saved checkpoint comparison as Markdown, JSON and a static plot."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from evaluate_inputs import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    report = json.loads((root / 'models/summary.json').read_text())
    entries = report['results']
    models = list(dict.fromkeys(Path(e['weights']).stem for e in entries))
    if len(entries) != len(models) * 2:
        raise ValueError('both validation and test results are required for every model')
    groups = {}
    identities = {}
    for entry in entries:
        model, split = Path(entry['weights']).stem, entry['split']
        rows = json.loads((root / 'models' / f'{model}_{split}.json').read_text())
        identity = [(row['frame'], row['truth']) for row in rows]
        if split in identities and identity != identities[split]:
            raise ValueError(f'frame/label pairing changed for {model}/{split}')
        identities[split] = identity
        if split == 'val':
            selected = {'validation': rows}
        else:
            selected = {'newer_704': [row for row in rows if row['frame'].startswith('ShipCam0_20260910_')],
                        'historical_111': [row for row in rows if row['frame'].startswith('renkliTekneTekne_')]}
            if sum(map(len, selected.values())) != len(rows):
                raise ValueError('test membership changed; review provenance grouping before reporting')
        for name, subset in selected.items():
            metric = summarize(subset, .45)
            metric.update(frames=len(subset), predict_ms_p95=float(np.percentile(
                [row['predict_ms'] for row in subset], 95)))
            groups.setdefault(name, {})[model] = metric
    (root / 'statistics.json').write_text(json.dumps(groups, indent=2) + '\n')
    lines = ['# Boat detector statistics — 2026-09-14', '',
             f'{len(models)} checkpoints; {len(identities["val"])} validation images and '
             f'{len(identities["test"])} test-directory images per checkpoint. '
             'Same saved color images, CPU FP32, four threads, imgsz=960, confidence=0.45, IoU=0.5. '
             'Image decoding and model warmup are outside prediction timings.', '',
             'The test directory is separated below by provenance. The 111-frame historical '
             'recording appeared in color-model training. The 704 newer frames are seven recordings '
             'from 2026-09-10 (168 positive boxes, 536 labeled-negative frames). Historical exposure '
             'and label completeness are not fully verified for every checkpoint. These are '
             'diagnostic results, not certified independent field accuracy.', '']
    for name, metrics in groups.items():
        lines += [f'## {name}', '',
                  '| Model | TP | FP | FN | Precision | Recall | F1 | p50 ms | p95 ms |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for model, m in metrics.items():
            lines.append(f'| {model} | {m["tp"]} | {m["fp"]} | {m["fn"]} | '
                         f'{m["precision"]:.1%} | {m["recall"]:.1%} | {m["f1"]:.3f} | '
                         f'{m["median_predict_ms"]:.1f} | {m["predict_ms_p95"]:.1f} |')
        lines.append('')
    lines += ['## Newer recordings, model by model', '',
              'Counts are TP / FP / FN at the same fixed threshold. Empty scenes are retained.', '',
              '| Model | Recording | Frames | TP / FP / FN |', '| --- | --- | ---: | --- |']
    for entry in entries:
        if entry['split'] != 'test':
            continue
        for name, m in entry['by_recording'].items():
            if name.startswith('ShipCam0_20260910_'):
                lines.append(f'| {Path(entry["weights"]).stem} | {name} | {m["frames"]} | '
                             f'{m["tp"]} / {m["fp"]} / {m["fn"]} |')
    lines += ['', '## Interpretation', '',
              'Precision = TP/(TP+FP); recall = TP/(TP+FN). All positives in this dataset have '
              'one labeled box; missing secondary targets and inconsistent box conventions can '
              'change both metrics. Adjacent frames are correlated, so frame counts are not '
              'counts of independent experiments. Latency was measured on the local CPU during '
              'the audit and is not a Jetson benchmark. No weights were promoted or retrained.', '',
              'Full predictions, thresholds, per-recording scores and hashes: [models/summary.json](models/summary.json).',
              '', '![Precision and recall by split](statistics.png)', '']
    (root / 'statistics.md').write_text('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    labels = [m.removeprefix('boat_').removesuffix('_best') for m in models]
    for ax, group, title in zip(axes, ('validation', 'newer_704'),
                                ('Validation: 831 frames', 'Newer diagnostic set: 704 frames')):
        y = np.arange(len(models))
        ax.barh(y - .17, [groups[group][m]['precision'] * 100 for m in models], .3,
                color='#2474b5', label='Precision')
        ax.barh(y + .17, [groups[group][m]['recall'] * 100 for m in models], .3,
                color='#d77a24', label='Recall')
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 100)
        ax.set_xlabel('% at confidence 0.45 / IoU 0.5')
        ax.set_title(title)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    axes[1].legend(loc='lower right')
    fig.suptitle('Checkpoint ranking depends on the recording set', fontsize=15)
    fig.tight_layout()
    fig.savefig(root / 'statistics.png', dpi=160)
    plt.close(fig)
    print(root / 'statistics.md')


if __name__ == '__main__':
    main()
