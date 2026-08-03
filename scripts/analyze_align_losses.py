"""
analyze_align_losses.py — summarize template-alignment loss distributions.

Run the pipeline/eval on a subset first with loss logging enabled:

    # Windows PowerShell
    $env:ALIGN_LOSS_LOG = "S:\omr\eval_out\align_losses.csv"
    # then run your eval on the subset

Then:
    python scripts/analyze_align_losses.py S:\omr\eval_out\align_losses.csv
    python scripts/analyze_align_losses.py S:\omr\eval_out\align_losses.csv --plot losses.png

Prints per-class percentiles + the current threshold, and (with --plot)
saves per-class loss histograms with the current threshold (red) and the
p95 suggestion (green) marked, so the valley between genuine matches and
the outlier tail is visible.  pip install matplotlib for plotting.
"""
import sys
import csv
from collections import defaultdict


def load(path):
    rows = defaultdict(list)
    cur_thr = {}
    with open(path, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            rows[r['class']].append(float(r['loss']))
            cur_thr[r['class']] = float(r['threshold'])
    return rows, cur_thr


def pct(v, p):
    v = sorted(v)
    if not v:
        return float('nan')
    k = (len(v) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def summarize(rows, cur_thr):
    hdr = (f"{'class':22} {'n':>5} {'median':>7} {'p90':>7} {'p95':>7} "
           f"{'p99':>7} {'max':>7} {'cur_thr':>8} {'%>thr':>6}")
    print(hdr)
    print('-' * len(hdr))
    for cls in sorted(rows):
        v = rows[cls]
        thr = cur_thr[cls]
        over = 100.0 * sum(1 for x in v if x > thr) / len(v)
        print(f"{cls:22} {len(v):5d} {pct(v,50):7.3f} {pct(v,90):7.3f} "
              f"{pct(v,95):7.3f} {pct(v,99):7.3f} {max(v):7.3f} "
              f"{thr:8.2f} {over:6.1f}")
    print("\nSuggested loss_thresh (p95 of observed losses):")
    for cls in sorted(rows):
        print(f"  '{cls}': loss_thresh {pct(rows[cls],95):.2f}   "
              f"(current {cur_thr[cls]:.2f})")


def plot(rows, cur_thr, out_png):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot "
              "(pip install matplotlib)")
        return
    import math as _m
    classes = sorted(rows)
    n = len(classes)
    cols = 3
    rows_n = _m.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4, rows_n * 2.6))
    axes = axes.flatten() if n > 1 else [axes]
    for i, cls in enumerate(classes):
        ax = axes[i]
        v = rows[cls]
        ax.hist(v, bins=30, color='#3b6ea5', alpha=0.85)
        thr = cur_thr[cls]
        p95 = pct(v, 95)
        ax.axvline(thr, color='red', lw=1.2, label=f'cur {thr:.2f}')
        ax.axvline(p95, color='green', lw=1.2, ls='--', label=f'p95 {p95:.2f}')
        ax.set_title(f'{cls} (n={len(v)})', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    for j in range(n, len(axes)):
        axes[j].axis('off')
    fig.suptitle('Template-alignment loss distributions', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130)
    print(f'saved plot -> {out_png}')


def main(argv):
    if len(argv) < 2:
        print('usage: python scripts/analyze_align_losses.py '
              '<align_losses.csv> [--plot out.png]')
        return 1
    path = argv[1]
    rows, cur_thr = load(path)
    if not rows:
        print('no rows found in', path)
        return 1
    summarize(rows, cur_thr)
    if '--plot' in argv:
        idx = argv.index('--plot')
        out_png = argv[idx + 1] if idx + 1 < len(argv) else 'align_losses.png'
        plot(rows, cur_thr, out_png)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
