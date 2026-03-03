"""
Plot training curves (loss, accuracy, mIoU) from saved metrics.

Usage:
    python examples/nailseg/plot_curves.py --metrics_path <path_to_metrics.json>

    # Compare multiple runs
    python examples/nailseg/plot_curves.py \
        --metrics_path log/nailseg/pointnet/metrics.json \
                       log/nailseg/pointnet++/metrics.json \
                       log/nailseg/pointnext-s/metrics.json \
        --labels PointNet PointNet++ PointNeXt-S

    # Specify output directory
    python examples/nailseg/plot_curves.py --metrics_path <path> --output_dir ./plots
"""
import argparse
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_metrics(metrics_path):
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def plot_single_run(metrics, output_dir, run_name='training'):
    """Plot training curves for a single run."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = metrics['epochs']

    # 1. Loss curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Training Loss Curve', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150)
    plt.close(fig)

    # 2. Accuracy curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics['train_oa'], 'b-', linewidth=2, label='Train OA')
    ax.plot(epochs, metrics['val_oa'], 'r-', linewidth=2, label='Val OA')
    if 'train_macc' in metrics:
        ax.plot(epochs, metrics['train_macc'], 'b--', linewidth=1.5, label='Train mAcc')
    if 'val_macc' in metrics:
        ax.plot(epochs, metrics['val_macc'], 'r--', linewidth=1.5, label='Val mAcc')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Accuracy Curve', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'accuracy_curve.png'), dpi=150)
    plt.close(fig)

    # 3. mIoU curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics['train_miou'], 'b-', linewidth=2, label='Train mIoU')
    ax.plot(epochs, metrics['val_miou'], 'r-', linewidth=2, label='Val mIoU')
    if 'best_val_miou' in metrics:
        ax.axhline(y=metrics['best_val_miou'], color='g', linestyle='--',
                   linewidth=1.5, label=f'Best Val mIoU: {metrics["best_val_miou"]:.2f}')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('mIoU (%)', fontsize=14)
    ax.set_title('mIoU Curve', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'miou_curve.png'), dpi=150)
    plt.close(fig)

    # 4. Combined plot (all metrics in one figure)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    axes[0].plot(epochs, metrics['train_loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, metrics['train_oa'], 'b-', linewidth=2, label='Train OA')
    axes[1].plot(epochs, metrics['val_oa'], 'r-', linewidth=2, label='Val OA')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, metrics['train_miou'], 'b-', linewidth=2, label='Train mIoU')
    axes[2].plot(epochs, metrics['val_miou'], 'r-', linewidth=2, label='Val mIoU')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('mIoU (%)', fontsize=12)
    axes[2].set_title('mIoU', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'{run_name} Training Curves', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'all_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'Plots saved to {output_dir}/')


def plot_comparison(metrics_list, labels, output_dir):
    """Plot comparison curves for multiple runs."""
    os.makedirs(output_dir, exist_ok=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f']

    # 1. Loss comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        ax.plot(metrics['epochs'], metrics['train_loss'],
                color=colors[i % len(colors)], linewidth=2, label=label)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Training Loss Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss_comparison.png'), dpi=150)
    plt.close(fig)

    # 2. Val Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        ax.plot(metrics['epochs'], metrics['val_oa'],
                color=colors[i % len(colors)], linewidth=2, label=label)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Validation Accuracy Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=150)
    plt.close(fig)

    # 3. Val mIoU comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        ax.plot(metrics['epochs'], metrics['val_miou'],
                color=colors[i % len(colors)], linewidth=2, label=label)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('mIoU (%)', fontsize=14)
    ax.set_title('Validation mIoU Comparison', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'miou_comparison.png'), dpi=150)
    plt.close(fig)

    # 4. Combined comparison
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        c = colors[i % len(colors)]
        axes[0].plot(metrics['epochs'], metrics['train_loss'], color=c,
                     linewidth=2, label=label)
        axes[1].plot(metrics['epochs'], metrics['val_oa'], color=c,
                     linewidth=2, label=label)
        axes[2].plot(metrics['epochs'], metrics['val_miou'], color=c,
                     linewidth=2, label=label)

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Val Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('mIoU (%)', fontsize=12)
    axes[2].set_title('Val mIoU', fontsize=14)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Model Comparison', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'all_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'Comparison plots saved to {output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot NailSeg training curves')
    parser.add_argument('--metrics_path', type=str, nargs='+', required=True,
                        help='Path(s) to metrics.json file(s)')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='Labels for each run (used in comparison mode)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots (default: same as metrics dir)')
    args = parser.parse_args()

    if len(args.metrics_path) == 1:
        # Single run
        metrics = load_metrics(args.metrics_path[0])
        output_dir = args.output_dir or os.path.dirname(args.metrics_path[0])
        run_name = os.path.basename(os.path.dirname(args.metrics_path[0]))
        plot_single_run(metrics, output_dir, run_name)
    else:
        # Multiple runs comparison
        metrics_list = [load_metrics(p) for p in args.metrics_path]
        labels = args.labels or [os.path.basename(os.path.dirname(p))
                                 for p in args.metrics_path]
        output_dir = args.output_dir or 'plots'
        plot_comparison(metrics_list, labels, output_dir)
