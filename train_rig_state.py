"""Train a causal GRU baseline for target [relative x,y, world vx,vy] from bearings + own state.

No target telemetry or depth in X. Train-only normalization, validation-only
checkpoint selection, held-out test available only via the evaluate subcommand.
This experimental Gaussian baseline is not a deployed or calibrated TMA replacement.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from boatdet.supervision import FEATURE_NAMES, TARGET_NAMES, sha256, write_json


class StateModel(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.gru = nn.GRU(len(FEATURE_NAMES), hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2 * len(TARGET_NAMES))

    def forward(self, x):
        _, hidden = self.gru(x)
        mean, log_variance = self.head(hidden[-1]).chunk(2, dim=-1)
        return mean, log_variance.clamp(-8, 8)


def load_split(folder, split):
    folder = Path(folder)
    schema = json.loads((folder / 'schema.json').read_text())
    if schema['features'] != list(FEATURE_NAMES) or schema['targets'] != list(TARGET_NAMES):
        raise ValueError('Unknown feature/target schema')
    path = folder / f'{split}.npz'
    if sha256(path) != schema['files_sha256'][split]:
        raise ValueError(f'{split} data changed; regenerate sequences')
    with np.load(path, allow_pickle=False) as data:
        x, y, sessions = data['X'], data['y'], data['session']
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f'{split}: need nonempty finite sequences')
    return torch.from_numpy(x), torch.from_numpy(y), sessions


def nll(mean, log_variance, target):
    return .5 * (log_variance + (target-mean).square() * torch.exp(-log_variance)).mean(-1)


def evaluate_model(model, x, y, sessions, normalization, batch_size=256):
    model.eval()
    xm, xs, ym, ys = normalization
    means, variances, losses = [], [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            mean, logvar = model((x[start:start+batch_size]-xm)/xs)
            losses.extend(nll(mean, logvar, (y[start:start+batch_size]-ym)/ys).tolist())
            means.append(mean*ys+ym)
            variances.append(torch.exp(logvar)*ys.square())
    mean, variance = torch.cat(means), torch.cat(variances)
    residual = mean-y
    by_session = []
    for sid in sorted(set(sessions)):
        indices = np.flatnonzero(sessions == sid)
        r = residual[indices]
        by_session.append({'session': str(sid), 'samples': len(indices),
                           'position_mae_m': float(r[:,:2].norm(dim=1).mean()),
                           'velocity_mae_mps': float(r[:,2:].norm(dim=1).mean()),
                           'normalized_nll': float(np.asarray(losses)[indices].mean())})
    return {'samples': len(x), 'sessions': by_session,
            'macro_session_nll': float(np.mean([r['normalized_nll'] for r in by_session])),
            'position_mae_m': float(residual[:,:2].norm(dim=1).mean()),
            'velocity_mae_mps': float(residual[:,2:].norm(dim=1).mean()),
            'marginal_95pct_coverage': ((residual.abs() <= 1.96*variance.sqrt()).float().mean(0)).tolist()}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_x, train_y, train_sessions = load_split(args.dataset, 'train')
    val_x, val_y, val_sessions = load_split(args.dataset, 'val')
    if set(train_sessions) & set(val_sessions):
        raise ValueError('Train and val sessions overlap')
    # No test data is opened here.
    norm = (train_x.mean((0,1)), train_x.std((0,1), unbiased=False).clamp_min(1e-6),
            train_y.mean(0), train_y.std(0, unbiased=False).clamp_min(1e-6))
    model = StateModel(args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    counts = {sid: sum(train_sessions == sid) for sid in set(train_sessions)}
    sampler = WeightedRandomSampler([1/counts[sid] for sid in train_sessions], len(train_sessions), replacement=True)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, sampler=sampler)
    best, bad_epochs, history = float('inf'), 0, []
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for x, y in loader:
            optimizer.zero_grad()
            mean, logvar = model((x-norm[0])/norm[1])
            loss = nll(mean, logvar, (y-norm[2])/norm[3]).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        report = evaluate_model(model, val_x, val_y, val_sessions, norm)
        score = report['macro_session_nll']
        history.append({'epoch': epoch, 'train_nll': float(np.mean(losses)), 'validation': report})
        print(f'epoch {epoch}: train NLL={np.mean(losses):.4f}, val NLL={score:.4f}', flush=True)
        if score < best:
            best, bad_epochs = score, 0
            torch.save({'state_dict': model.state_dict(), 'normalization': norm, 'hidden': args.hidden,
                        'schema': json.loads((args.dataset/'schema.json').read_text()),
                        'provenance': json.loads((args.dataset/'provenance.json').read_text()),
                        'epoch': epoch, 'validation': report, 'seed': args.seed,
                        'deployment_ready': False}, args.out/'best.pt')
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    write_json(args.out/'training.json', {'history': history, 'best_validation_nll': best,
               'train_samples': len(train_x), 'validation_samples': len(val_x),
               'test_used': False, 'deployment_ready': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('dataset', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p = sub.add_parser('evaluate')
    p.add_argument('dataset', type=Path)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--split', choices=('val', 'test'), default='test')
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'train':
        if min(args.epochs,args.hidden,args.batch_size,args.patience) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
            parser.error('epochs, hidden, batch-size, patience and lr must be positive')
        train(args)
    else:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        schema = json.loads((args.dataset/'schema.json').read_text())
        if schema != saved['schema']:
            raise ValueError('Checkpoint belongs to a different prepared sequence dataset')
        x, y, sessions = load_split(args.dataset,args.split)
        model = StateModel(saved['hidden'])
        model.load_state_dict(saved['state_dict'])
        report = evaluate_model(model, x, y, sessions, saved['normalization'])
        report.update(split=args.split, checkpoint_sha256=sha256(args.checkpoint),
                      synthetic=saved['provenance']['synthetic'], deployment_ready=False)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
