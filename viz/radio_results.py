"""
Generates comparative bar charts from TensorBoard logs for RADIO experiments.
Saves both summary (one bar per model) and detailed (fold-level) plots.
"""

import os
import re
import shutil
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

LOGS_DIR = "../RADIO/runs"
OUTPUT_DIR = "final_plots"
METRIC_TAG = 'CV/Accuracy/test'

EXPERIMENT_GROUPS = {
    "I_RAVEN to RADIO1 TL": r"^RADIO1_(.+)_TRANSFER_RAVEN$",
    "RADIO2 to RADIO1 TL": r"^RADIO1_(.+)_TRANSFER$",
    "RADIO1 STL": r"^RADIO1_(.+)$",
    "I_RAVEN to RADIO2 TL": r"^RADIO2_(.+)_TRANSFER_RAVEN$",
    "RADIO1 to RADIO2 TL": r"^RADIO2_(.+)_TRANSFER$",
    "RADIO2 STL": r"^RADIO2_(.+)$",
}
MODEL_ORDER = ["MSRGNN", "SCAR", "WReN", "DRNet", "MRNet", "MXGNet"]
PROPOSED_MODEL = "MSRGNN"

HERO_COLOR = '#007ACC'
SUMMARY_BAR_COLOR = '#B0C4DE'

BASELINE_SORT_METHOD = "performance" # performance or alphabetical

USE_GLOBAL_Y_AXIS = False

BAR_WIDTH_SUMMARY = 0.6
BAR_WIDTH_DETAIL = 0.5
FIG_SIZE_SUMMARY = (12, 7)
FIG_SIZE_DETAIL_BASE = (6, 5) # Size per subplot
COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, 10))
DETAIL_COLORS = {'fold': '#87CEEB', 'avg': '#F08080'} # SkyBlue, LightCoral

def extract_final_scalar_value(log_dir, tag):
    """Finds the newest event file and extracts the last scalar value."""
    try:
        event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith('events.out.tfevents.')]
        if not event_files: return None
        newest_event_file = max(event_files, key=os.path.getmtime)
        event_acc = event_accumulator.EventAccumulator(newest_event_file, size_guidance={'scalars': 0})
        event_acc.Reload()
        if tag not in event_acc.Tags()['scalars']: return None
        scalar_events = event_acc.Scalars(tag)
        return scalar_events[-1].value if scalar_events else None
    except Exception: return None


def process_experiment_folds(experiment_path, metric_tag):
    """Returns a list of all final fold scores for an experiment."""
    fold_dirs = [os.path.join(experiment_path, d) for d in os.listdir(experiment_path) if d.startswith('fold_')]
    scores = [s for d in fold_dirs if (s := extract_final_scalar_value(d, metric_tag)) is not None]
    return scores if scores else None


def plot_summary_comparison_chart(group_title, results, output_path, metric_name):
    """Generates the high-level summary plot (one bar per model average)."""
    model_names = list(results.keys())
    means = [res['mean'] for res in results.values()]
    stds = [res['std'] for res in results.values()]

    bar_colors = [HERO_COLOR if name == PROPOSED_MODEL else SUMMARY_BAR_COLOR for name in model_names]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=FIG_SIZE_SUMMARY)
    bars = ax.bar(model_names, means, yerr=stds, color=bar_colors, capsize=4, zorder=3, width=BAR_WIDTH_SUMMARY)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    
    ax.set_title(group_title, fontsize=18, pad=20, weight='bold')
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_ylim(bottom=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', color='grey', alpha=0.5)
    ax.xaxis.grid(False)
    ax.tick_params(bottom=False, left=False)
    plt.xticks(rotation=10, ha="right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Summary plot saved to '{output_path}'")
    plt.close(fig)


def plot_detailed_fold_chart(group_title, results, output_path, metric_name):
    """Generates the detailed plot with a subplot for each model showing fold scores."""
    n_models = len(results)
    if n_models == 0: return
    
    ncols = min(n_models, 3)
    nrows = math.ceil(n_models / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_SIZE_DETAIL_BASE[0] * ncols, FIG_SIZE_DETAIL_BASE[1] * nrows), squeeze=False)
    axes = axes.flatten()
    
    for i, (model_name, data) in enumerate(results.items()):
        ax = axes[i]
        fold_scores = data['scores']
        n_folds = len(fold_scores)
        
        labels = [f'Fold {j}' for j in range(n_folds)] + ['Average']
        values = fold_scores + [data['mean']]
        errors = [0] * n_folds + [data['std']]
        colors = [DETAIL_COLORS['fold']] * n_folds + [DETAIL_COLORS['avg']]

        bars = ax.bar(labels, values, yerr=errors, color=colors, capsize=4, zorder=3, width=BAR_WIDTH_DETAIL)
        ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=8)
        
        ax.set_title(model_name, fontsize=14, weight='bold')

        if i % ncols == 0:
            ax.set_ylabel(metric_name, fontsize=12)

        if USE_GLOBAL_Y_AXIS:
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(bottom=min(values) * 0.9 if min(values) > 0 else -0.1)

        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', color='grey', alpha=0.5)
        ax.xaxis.grid(False)
        ax.tick_params(bottom=False, left=False, labelsize=9)
        ax.set_xticklabels(labels, rotation=45, ha='right')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    fig.suptitle(group_title, fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Detailed plot saved to '{output_path}'")
    plt.close(fig)


def main():
    if not os.path.isdir(LOGS_DIR):
        print(f"Error: Log directory '{LOGS_DIR}' not found."); return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_experiments = os.listdir(LOGS_DIR)
    grouped_experiments = {group_name: {} for group_name in EXPERIMENT_GROUPS.keys()}

    for exp_name in all_experiments:
        if not os.path.isdir(os.path.join(LOGS_DIR, exp_name)): continue
        for group_name, pattern in EXPERIMENT_GROUPS.items():
            match = re.match(pattern, exp_name)
            if match:
                model_name = match.group(1).replace('_', ' ')
                grouped_experiments[group_name][model_name] = os.path.join(LOGS_DIR, exp_name)
                break
    
    metric_name = METRIC_TAG.split('/')[-1].replace('_', ' ').capitalize()
    
    for group_name, experiments in grouped_experiments.items():
        if not experiments: continue
            
        print(f"\n--- Processing Group: {group_name} ---")
        results = {}
        for model_name, path in sorted(experiments.items()):
            scores = process_experiment_folds(path, METRIC_TAG)
            if scores:
                results[model_name] = {'scores': scores, 'mean': np.mean(scores), 'std': np.std(scores)}
                print(f"  - {model_name:<20}: Found {len(scores)} folds. Mean={results[model_name]['mean']:.4f}")
            else:
                print(f"  - {model_name:<20}: [Failed to process data]")

        if results:
            ordered_results = {}
            
            if PROPOSED_MODEL in results:
                ordered_results[PROPOSED_MODEL] = results[PROPOSED_MODEL]
            
            baseline_results = {model: data for model, data in results.items() if model != PROPOSED_MODEL}
            
            if BASELINE_SORT_METHOD == 'performance':
                # Sort by the 'mean' value in the nested dictionary, from highest to lowest.
                sorted_baselines = sorted(baseline_results.items(), key=lambda item: item[1]['mean'], reverse=True)
            else: # Default to alphabetical sorting if 'performance' is not specified.
                sorted_baselines = sorted(baseline_results.items())
                
            for model_name, data in sorted_baselines:
                ordered_results[model_name] = data

            filename_base = group_name.replace(' ', '_').replace('->', 'to').lower()
            
            # Generate both plots
            # plot_summary_comparison_chart(group_name, ordered_results, os.path.join(OUTPUT_DIR, f"{filename_base}_summary.png"), metric_name)
            # plot_detailed_fold_chart(group_name, ordered_results, os.path.join(OUTPUT_DIR, f"{filename_base}_details.png"), metric_name)

            plot_summary_comparison_chart(group_name, ordered_results, os.path.join(OUTPUT_DIR, f"{filename_base}_summary.png"), "Test accuracy")
            plot_detailed_fold_chart(group_name, ordered_results, os.path.join(OUTPUT_DIR, f"{filename_base}_details.png"), "Test accuracy")

if __name__ == "__main__":
    main()