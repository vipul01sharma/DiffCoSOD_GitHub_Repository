"""Runtime / memory benchmark for DiffCoSOD with raw per-run records.

Replaces scripts/benchmark_runtime.py. Differences from the original:
  * repeated, individually recorded measurements (not a single timed call)
  * mean / std / median / 5th-95th percentile, plus every raw timing in a CSV
  * peak GPU memory reset for each sampling-step setting
  * both "seconds per group" and "seconds per image" (a group is group_size images)
  * environment, git commit, checkpoint SHA-256, config and manifest hashes in the output
Usage:
  python scripts/benchmark_runtime.py --config configs/diffcosod.yaml \
      --checkpoint runs/diffcosod/best.pt --manifest manifests/cocacoca_test.json \
      --sampling-steps 250 500 750 1000 --repeats 30 --warmup 5 --output-dir runs/runtime
"""
import argparse, csv, hashlib, json, os, platform, statistics, subprocess, sys, time

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None

def percentile(xs, q):
    xs = sorted(xs); k = (len(xs) - 1) * q; lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

def summarize(xs):
    return {'n': len(xs), 'mean': statistics.fmean(xs), 'std': statistics.stdev(xs) if len(xs) > 1 else 0.0,
            'median': statistics.median(xs), 'p05': percentile(xs, 0.05), 'p95': percentile(xs, 0.95),
            'min': min(xs), 'max': max(xs)}

def main(a):
    import torch
    from torch.utils.data import DataLoader
    from diffcosod.utils import load_yaml, seed_everything
    from diffcosod.data import GroupDataset
    from diffcosod.models import DiffCoSOD

    cfg = load_yaml(a.config); seed_everything(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffCoSOD(cfg).to(device)
    ck = torch.load(a.checkpoint, map_location=device); model.load_state_dict(ck['model']); model.eval()
    ds = GroupDataset(a.manifest, cfg['image_size'], cfg['group_size'], False, cfg['seed'])
    batch = next(iter(DataLoader(ds, batch_size=1, num_workers=0)))['images'].to(device)
    n_images = batch.shape[0] * batch.shape[1]          # images denoised together in one predict() call
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cuda = device.type == 'cuda'

    env = {
        'python': sys.version.split()[0], 'platform': platform.platform(), 'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version() if cuda else None,
        'device': torch.cuda.get_device_name(0) if cuda else platform.processor() or 'cpu',
        'nvidia_smi': run(['nvidia-smi', '--query-gpu=name,driver_version,clocks.max.sm,power.limit', '--format=csv,noheader']) if cuda else None,
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32 if cuda else None,
        'git_commit': run(['git', 'rev-parse', 'HEAD']), 'git_dirty': bool(run(['git', 'status', '--porcelain'])),
        'checkpoint_sha256': sha256(a.checkpoint), 'config_sha256': sha256(a.config), 'manifest_sha256': sha256(a.manifest),
        'input_shape': list(batch.shape), 'images_per_call': n_images, 'trainable_params_M': n_params / 1e6,
        'what_is_timed': 'model.predict(group): encoders + SFG + correlation + GADG + full reverse diffusion; excludes data loading',
        'repeats': a.repeats, 'warmup': a.warmup, 'seed': cfg['seed'],
    }
    os.makedirs(a.output_dir, exist_ok=True)
    raw_rows, summary = [], []
    for steps in a.sampling_steps:
        if cuda:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(a.warmup):
            with torch.no_grad(): model.predict(batch, steps)
        times = []
        for r in range(a.repeats):
            if cuda:
                torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
                with torch.no_grad(): model.predict(batch, steps)
                e.record(); torch.cuda.synchronize(); sec = s.elapsed_time(e) / 1000
            else:
                t0 = time.perf_counter()
                with torch.no_grad(): model.predict(batch, steps)
                sec = time.perf_counter() - t0
            times.append(sec); raw_rows.append({'timesteps': steps, 'run': r, 'seconds_per_call': sec})
        mem = torch.cuda.max_memory_allocated() / 1024 ** 3 if cuda else None
        st = summarize(times)
        summary.append({'timesteps': steps, 'seconds_per_group': st, 'seconds_per_image_median': st['median'] / n_images,
                        'images_per_second_median': n_images / st['median'], 'peak_memory_gb': mem,
                        'seconds_per_step_median': st['median'] / steps})
        print(f"steps={steps}: median {st['median']:.3f}s/group ({st['median']/n_images:.3f}s/img), "
              f"IQR-ish p05-p95 {st['p05']:.3f}-{st['p95']:.3f}, peak mem {mem}")
    json.dump({'environment': env, 'results': summary}, open(os.path.join(a.output_dir, 'runtime.json'), 'w'), indent=2)
    with open(os.path.join(a.output_dir, 'runtime_raw.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['timesteps', 'run', 'seconds_per_call']); w.writeheader(); w.writerows(raw_rows)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--checkpoint', required=True); ap.add_argument('--manifest', required=True)
    ap.add_argument('--output-dir', default='runs/runtime'); ap.add_argument('--sampling-steps', type=int, nargs='+', default=[250, 500, 750, 1000])
    ap.add_argument('--repeats', type=int, default=30); ap.add_argument('--warmup', type=int, default=5)
    main(ap.parse_args())
