
from __future__ import annotations
import argparse
import os
from typing import Iterable, List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)

# Configure matplotlib for larger, bold fonts
plt.rcParams.update({
    'font.size': 14,          # Base font size
    'font.weight': 'bold',    # Make all fonts bold
    'axes.titlesize': 18,     # Title font size
    'axes.titleweight': 'bold',
    'axes.labelsize': 16,     # Axis label font size  
    'axes.labelweight': 'bold',
    'legend.fontsize': 10,    # Legend font size
    'legend.title_fontsize': 12,
    'xtick.labelsize': 13,    # X-axis tick label size
    'ytick.labelsize': 13,    # Y-axis tick label size
    'figure.titlesize': 20,   # Figure title size
    'figure.titleweight': 'bold'
})

# ---------------------------- I/O ----------------------------

def load_any(path: str) -> pd.DataFrame:
    """Load CSV or Parquet into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        sep = "," if ext == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    elif ext in (".parquet", ".pq"):
        # Requires pyarrow or fastparquet
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext} ({path})")

def load_many(paths: Iterable[str], how: str = "concat") -> pd.DataFrame:
    """
    Load multiple files and combine. Currently supports 'concat' (row-wise).
    """
    dfs = [load_any(p) for p in paths]
    if not dfs:
        raise ValueError("No input files provided.")
    if how == "concat":
        # Align columns (outer join) to avoid dropping info
        return pd.concat(dfs, ignore_index=True, sort=False)
    else:
        raise ValueError(f"Unsupported combine mode: {how}")

# ---------------------- Routing + Baselines ----------------------

def expected_random_baseline(df: pd.DataFrame, deferal_rate: float, model_name: str, direct_source: str = 'p_true', gpt_effort: str = 'high'
) -> Tuple[float, float, float, float]:
    """
    Expected performance if each example defers to CoT with probability r (Bernoulli routing).
    Returns (accuracy, avg_tokens, std_tokens, cot_proportion) where cot_proportion == r.
    """
    r = float(deferal_rate)
    if direct_source not in ('p_true', 'verbalised', 'margin', 'perplexity'):
        raise ValueError("direct_source must be one of {'p_true','verbalised','margin','perplexity'}")

    if direct_source == 'p_true':
        d_corr = df['direct_answer_correct_p_true'].astype(float)
        d_len  = df['direct_answer_len_p_true'].astype(float)
    elif direct_source == 'verbalised':
        d_corr = df['direct_answer_correct_verbalised'].astype(float)
        d_len  = df['direct_answer_len_verbalised'].astype(float)
    else:  # 'margin' or 'perplexity'
        d_corr = df['direct_answer_correct_margin_pe'].astype(float)
        d_len  = df['direct_answer_len_margin_pe'].astype(float)

    c_corr, c_len = _select_cot_columns(df, model_name, gpt_effort)

    acc        = ((1 - r) * d_corr + r * c_corr).mean()
    avg_tokens = ((1 - r) * d_len  + r * c_len ).mean()
    e_l2       = ((1 - r) * (d_len**2) + r * (c_len**2)).mean()
    std_tokens = float(np.sqrt(max(e_l2 - avg_tokens**2, 0.0)))
    return float(acc), float(avg_tokens), std_tokens, r

def _select_columns(df: pd.DataFrame, confidence_type: str):
    """Return (conf, d_corr, d_len) arrays for a given confidence type."""
    if confidence_type == 'p_true':
        conf = df['p_true_confidence'].to_numpy(float)
        d_corr = df['direct_answer_correct_p_true'].to_numpy(float)
        d_len  = df['direct_answer_len_p_true'].to_numpy(float)
    elif confidence_type == 'verbalised':
        conf = df['verbalised_confidence'].to_numpy(float)
        d_corr = df['direct_answer_correct_verbalised'].to_numpy(float)
        d_len  = df['direct_answer_len_verbalised'].to_numpy(float)
    elif confidence_type == 'margin':
        conf = df['margin_confidence'].to_numpy(float)
        d_corr = df['direct_answer_correct_margin_pe'].to_numpy(float)
        d_len  = df['direct_answer_len_margin_pe'].to_numpy(float)
    elif confidence_type == 'perplexity':
        conf = df['perplexity_confidence'].to_numpy(float)
        d_corr = df['direct_answer_correct_margin_pe'].to_numpy(float)
        d_len  = df['direct_answer_len_margin_pe'].to_numpy(float)
    elif confidence_type == 'random':
        conf = np.random.rand(len(df)).astype(float)  # unused for expected baseline
        d_corr = df['direct_answer_correct_p_true'].to_numpy(float)
        d_len  = df['direct_answer_len_p_true'].to_numpy(float)
    else:
        raise ValueError(f"Unknown confidence type: {confidence_type}")
    return conf, d_corr, d_len

def _select_cot_columns(df: pd.DataFrame, model_name: str, gpt_effort: str = 'high'):
    """Return (c_corr, c_len) arrays for a given model_name."""
    if "qwen" in model_name.lower():
        c_corr = df['cot_answer_correct'].to_numpy(float)
        c_len  = df['cot_answer_len'].to_numpy(float)
    elif "gpt-oss" in model_name.lower():
        if gpt_effort == 'low':
            c_corr = df['low_cot_answer_correct'].to_numpy(float)
            c_len  = df['low_cot_answer_len'].to_numpy(float)
        elif gpt_effort == 'high':
            c_corr = df['high_cot_answer_correct'].to_numpy(float)
            c_len  = df['high_cot_answer_len'].to_numpy(float)
        elif gpt_effort == 'medium':
            c_corr = df['medium_cot_answer_correct'].to_numpy(float)
            c_len  = df['medium_cot_answer_len'].to_numpy(float)
        else:
            raise ValueError(f"Unknown gpt_effort: {gpt_effort}")
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return c_corr, c_len


def normalise_confidences(conf: np.ndarray) -> np.ndarray:
    """Normalise confidence scores to [0,1] range."""
    conf_min, conf_max = np.min(conf), np.max(conf)
    if conf_max == conf_min:
        print("Warning: All confidence scores are identical.")
        return np.full_like(conf, 0.5)  # All equal -> middle value
    
    return (conf - conf_min) / (conf_max - conf_min)


def simulate_online_confidence_gated_cot_fast(
    df: pd.DataFrame, confidence_type: str, percentiles: Iterable[float], model_name: str, run_id: int = 0, gpt_effort: str = 'high'
) -> List[Tuple[float, float, float, float, float]]:
    """Online percentile routing simulation. Returns (p, accuracy, avg_tokens, std_tokens, cot_proportion)."""
    # Always randomize dataset order for online routing with different seed per run
    df = df.sample(frac=1.0, random_state=42 + run_id).reset_index(drop=True)
    
    conf, d_corr, d_len = _select_columns(df, confidence_type)
    c_corr, c_len = _select_cot_columns(df, model_name, gpt_effort)

    percentiles = np.asarray(list(percentiles), dtype=float)
    results = []
    warmup_period = 20  # Handle case where df has fewer than 20 samples

    for p in tqdm(percentiles, desc=f"Processing percentiles ({confidence_type})", leave=False):
        thresholds = np.empty(len(conf), dtype=float)
        use_cot = np.zeros(len(conf), dtype=bool)
        
        # Running percentile computation with warm-up period
        seen = []
        for i in range(len(conf)):
            v = conf[i]
            
            # Always use direct for first warmup_period samples
            if i < warmup_period:
                use_cot[i] = False
                thresholds[i] = np.nan  # No threshold computed yet
                if not np.isnan(v):
                    seen.append(v)
            else:
                # From warmup_period onwards, use threshold-based routing
                if not np.isnan(v):
                    seen.append(v)
                
                if seen:
                    thresholds[i] = np.nanpercentile(seen, p)
                    
                    # Apply routing rule
                    if confidence_type in ['p_true', 'verbalised', 'margin', 'random']:
                        use_cot[i] = conf[i] <= thresholds[i]
                    else:  # perplexity
                        use_cot[i] = conf[i] >= thresholds[i]
                        
                    # Handle NaN confidences - default to direct
                    if np.isnan(conf[i]):
                        use_cot[i] = False
                else:
                    thresholds[i] = np.nan
                    use_cot[i] = False

        # Calculate final performance
        final_corr = np.where(use_cot, c_corr, d_corr)
        final_len  = np.where(use_cot, c_len, d_len)

        acc = float(final_corr.mean())
        avg_tokens = float(final_len.mean())
        std_tokens = float(final_len.std(ddof=0))
        cot_prop = float(use_cot.mean())

        results.append((p, acc, avg_tokens, std_tokens, cot_prop))

    return results

def simulate_online_confidence_gated_cot_multiple_runs(
    df: pd.DataFrame, confidence_type: str, percentiles: Iterable[float], 
    model_name: str, num_runs: int = 5, gpt_effort: str = 'high'
) -> List[Tuple[float, float, float, float, float, float, float]]:
    """
    Run online routing multiple times with different random seeds.
    Returns list of (p, mean_acc, std_acc, mean_tokens, std_tokens, mean_cot_prop, std_cot_prop)
    """
    percentiles = list(percentiles)
    all_results = []
    
    for run in range(num_runs):
        run_results = simulate_online_confidence_gated_cot_fast(df, confidence_type, percentiles, model_name, run_id=run, gpt_effort=gpt_effort)
        all_results.append(run_results)
    
    # Aggregate across runs
    aggregated = []
    for p_idx, p in enumerate(percentiles):
        accs = [all_results[run][p_idx][1] for run in range(num_runs)]
        tokens = [all_results[run][p_idx][2] for run in range(num_runs)]
        cot_props = [all_results[run][p_idx][4] for run in range(num_runs)]
        
        aggregated.append((
            p,
            np.mean(accs), np.std(accs),
            np.mean(tokens), np.std(tokens),
            np.mean(cot_props), np.std(cot_props)
        ))
    
    return aggregated

def simulate_offline_confidence_gated_cot(
    df: pd.DataFrame, confidence_type: str, percentiles: Iterable[float], model_name: str, threshold_type: str='absolute', gpt_effort: str = 'high'
) -> List[Tuple[float, float, float, float, float]]:
    """
    Offline percentile routing using fixed thresholds from full distribution.
    No confidence normalization - uses raw confidence values.
    Returns list of (p, accuracy, avg_tokens, std_tokens, cot_proportion)
    """
    conf, d_corr, d_len = _select_columns(df, confidence_type)
    c_corr, c_len = _select_cot_columns(df, model_name, gpt_effort)

    percentiles = np.asarray(list(percentiles), dtype=float)
    results = []
    valid_conf = ~np.isnan(conf)

    if threshold_type == 'percentile':
        # NaN-safe percentiles
        thresholds = [np.nanpercentile(conf, p) for p in percentiles]
    elif threshold_type == 'absolute':
        conf_min = np.nanmin(conf); conf_max = np.nanmax(conf)
        if not np.isfinite(conf_min) or conf_max == conf_min:
            print("Warning: confidences degenerate or all NaN; falling back to 0.5.")
            conf_norm = np.full_like(conf, 0.5, dtype=float)
        else:
            conf_norm = (conf - conf_min) / (conf_max - conf_min)
        conf = conf_norm
        thresholds = np.linspace(0, 1, len(percentiles))
    else:
        raise ValueError(f"Unknown threshold_type: {threshold_type}")
    
    for threshold in thresholds:
        # Calculate fixed threshold from full distribution (raw confidences)
        
        # Apply routing rule
        if confidence_type in ['p_true', 'verbalised', 'margin', 'random']:
            use_cot = conf <= threshold
        else:  # perplexity: higher is more uncertain
            use_cot = conf >= threshold

        use_cot = np.where(valid_conf, use_cot, False)
        # Calculate performance
        final_corr = np.where(use_cot, c_corr, d_corr)
        final_len = np.where(use_cot, c_len, d_len)

        acc = float(final_corr.mean())
        avg_tokens = float(final_len.mean())
        std_tokens = float(final_len.std(ddof=0))
        cot_prop = float(use_cot.mean())

        results.append((threshold, acc, avg_tokens, std_tokens, cot_prop))

    return results

def get_oracle_performance(df: pd.DataFrame, confidence_type: str, model_name: str, gpt_effort: str = 'high'):
    """Direct if correct; else CoT. Return (acc, avg_tokens, std_tokens, cot_prop)."""
    _, d_corr, d_len = _select_columns(df, confidence_type)
    c_corr, c_len = _select_cot_columns(df, model_name, gpt_effort)
    final_corr = np.where(d_corr == 1, d_corr, c_corr).astype(float)
    final_len  = np.where(d_corr == 1, d_len, c_len).astype(float)
    cot_prop = float(np.mean((d_corr == 0).astype(float)))
    return float(final_corr.mean()), float(final_len.mean()), float(final_len.std(ddof=0)), cot_prop

# ---------------------------- Plotting ----------------------------

_PALETTES = {
    'p_true': "#08519c",           # dark blue
    'verbalised': "#6baed6",       # light blue
    'margin': "#31a354",           # green
    'perplexity': "#fd8d3c",# orange
    'random': "#756bb1",           # purple
}
_MARKERS = {'p_true':'o','verbalised':'s','margin':'^','perplexity':'D','random':'P'}

def _apply_autoscale(ax, xs: List[float], ys: List[float]):
    # Per-plot autoscaling with small padding
    if not xs or not ys:
        ax.relim(); ax.autoscale_view(); return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = 0.02 * (x_max - x_min if x_max > x_min else 1.0)
    pad_y = 0.02 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)

def plot_accuracy_vs_routing(
    df: pd.DataFrame, model_name: str, percentiles: Iterable[float], outpath: str, title: str,
    confidence_types=('p_true', 'verbalised', 'margin', 'perplexity', 'random'),
    online: bool = False,
    threshold_type: str = 'absolute',
    num_runs: int = 1,
    gpt_effort: str = 'high'
):
    per_type = {}
    all_x, all_y = [], []
    for ctype in tqdm(confidence_types, desc="Processing confidence types", unit="type"):
        series = []
        if ctype == 'random':
            for p in percentiles:
                acc, avg_len, std_len, cot_prop = expected_random_baseline(df, p/100.0, model_name, direct_source='p_true', gpt_effort=gpt_effort)
                series.append({'x': cot_prop, 'y': acc, 'p': p, 'x_err': 0, 'y_err': 0})
        else:
            # Switch between online and offline routing
            if online:
                if num_runs > 1:
                    res = simulate_online_confidence_gated_cot_multiple_runs(df, ctype, percentiles, model_name, num_runs, gpt_effort)
                    for p, mean_acc, std_acc, mean_tokens, std_tokens, mean_cot_prop, std_cot_prop in res:
                        series.append({'x': mean_cot_prop, 'y': mean_acc, 'p': p, 'x_err': std_cot_prop, 'y_err': std_acc})
                else:
                    res = simulate_online_confidence_gated_cot_fast(df, ctype, percentiles, model_name, run_id=0, gpt_effort=gpt_effort)
                    for p, acc, avg_tokens, std_tokens, cot_prop in res:
                        series.append({'x': cot_prop, 'y': acc, 'p': p, 'x_err': 0, 'y_err': 0})
            else:
                res = simulate_offline_confidence_gated_cot(df, ctype, percentiles, model_name, threshold_type=threshold_type, gpt_effort=gpt_effort)
                for p, acc, avg_tokens, std_tokens, cot_prop in res:
                    series.append({'x': cot_prop, 'y': acc, 'p': p, 'x_err': 0, 'y_err': 0})
        per_type[ctype] = sorted(series, key=lambda d: d['x'])
        all_x += [d['x'] for d in series]; all_y += [d['y'] for d in series]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for ctype in confidence_types:
        series = per_type[ctype]
        x = np.array([d['x'] for d in series], float)
        y = np.array([d['y'] for d in series], float)
        x_err = np.array([d['x_err'] for d in series], float)
        y_err = np.array([d['y_err'] for d in series], float)
        color = _PALETTES.get(ctype, "#333333")
        marker = _MARKERS.get(ctype, 'o')
        ax.plot(x, y, linestyle='-', linewidth=1.5, alpha=0.6, color=color)
        ax.scatter(x, y, s=36, color=color, edgecolors='white', linewidths=0.6, marker=marker, label=ctype)
        
        # Add error bars if we have multiple runs
        if num_runs > 1 and online:
            ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='none', color=color, alpha=0.5, capsize=2)

    # Single Oracle star (verbalised only)
    ox, oy = [], []
    if 'verbalised' in confidence_types:
        acc_o, avg_tokens_o, std_tokens_o, cot_prop_o = get_oracle_performance(df, 'verbalised', model_name, gpt_effort)
        color = _PALETTES.get('verbalised', "#6baed6")
        ax.scatter([cot_prop_o], [acc_o], marker='*', s=140, color=color, edgecolors='black', zorder=5, label="Oracle")
        ox.append(cot_prop_o); oy.append(acc_o)

    _apply_autoscale(ax, all_x + ox, all_y + oy)
    ax.set_xlabel("CoT usage rate", fontsize=16, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=16, fontweight='bold')
    
    # Update title to reflect routing mode and reasoning effort
    routing_mode = "Online" if online else "Offline"
    # Use provided title (dataset name) and append routing mode
    ax.set_title(f"{title} ({routing_mode})", fontsize=18, fontweight='bold', pad=20)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add annotation for online routing
    if online:
        ax.text(0.02, 0.98, "Online routing", 
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Make tick labels bold and larger
    ax.tick_params(axis='both', which='major', labelsize=13)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    legend = ax.legend(uniq.values(), uniq.keys(), title="Method", ncol=2, 
                      fontsize=10, title_fontsize=12)
    legend.get_title().set_fontweight('bold')
    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight('bold')

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

    fig.tight_layout(pad=0.1)
    fig.savefig(outpath, dpi=200, bbox_inches="tight", format='pdf', pad_inches=0.0)
    print(f"[Saved] {outpath}")

def plot_accuracy_vs_tokens(
    df: pd.DataFrame, model_name: str, percentiles: Iterable[float], outpath: str, title: str,
    confidence_types=('p_true', 'verbalised', 'margin', 'perplexity', 'random'),
    online: bool = False,
    threshold_type: str = 'percentile',
    num_runs: int = 1,
    gpt_effort: str = 'high'
):
    per_type = {}
    all_x, all_y = [], []
    for ctype in tqdm(confidence_types, desc="Processing confidence types", unit="type"):
        series = []
        if ctype == 'random':
            for p in percentiles:
                acc, avg_len, std_len, cot_prop = expected_random_baseline(df, p/100.0, model_name, direct_source='p_true', gpt_effort=gpt_effort)
                series.append({'x': avg_len, 'y': acc, 'p': p, 'x_err': 0, 'y_err': 0})
        else:
            # Switch between online and offline routing
            if online:
                if num_runs > 1:
                    res = simulate_online_confidence_gated_cot_multiple_runs(df, ctype, percentiles, model_name, num_runs, gpt_effort)
                    for p, mean_acc, std_acc, mean_tokens, std_tokens, mean_cot_prop, std_cot_prop in res:
                        series.append({'x': mean_tokens, 'y': mean_acc, 'p': p, 'x_err': std_tokens, 'y_err': std_acc})
                else:
                    res = simulate_online_confidence_gated_cot_fast(df, ctype, percentiles, model_name, run_id=0, gpt_effort=gpt_effort)
                    for p, acc, avg_tokens, std_tokens, cot_prop in res:
                        series.append({'x': avg_tokens, 'y': acc, 'p': p, 'x_err': 0, 'y_err': 0})
            else:
                res = simulate_offline_confidence_gated_cot(df, ctype, percentiles, model_name, threshold_type=threshold_type, gpt_effort=gpt_effort)
                for p, acc, avg_tokens, std_tokens, cot_prop in res:
                    series.append({'x': avg_tokens, 'y': acc, 'p': p, 'x_err': 0, 'y_err': 0})
        per_type[ctype] = sorted(series, key=lambda d: d['x'])
        all_x += [d['x'] for d in series]; all_y += [d['y'] for d in series]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for ctype in confidence_types:
        series = per_type[ctype]
        x = np.array([d['x'] for d in series], float)
        y = np.array([d['y'] for d in series], float)
        x_err = np.array([d['x_err'] for d in series], float)
        y_err = np.array([d['y_err'] for d in series], float)
        color = _PALETTES.get(ctype, "#333333")
        marker = _MARKERS.get(ctype, 'o')
        ax.plot(x, y, linestyle='-', linewidth=1.5, alpha=0.6, color=color)
        ax.scatter(x, y, s=36, color=color, edgecolors='white', linewidths=0.6, marker=marker, label=ctype)
        
        # Add error bars if we have multiple runs
        if num_runs > 1 and online:
            ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='none', color=color, alpha=0.5, capsize=2)

    # Single Oracle star (verbalised only)
    ox, oy = [], []
    if 'verbalised' in confidence_types:
        acc_o, avg_tokens_o, std_tokens_o, cot_prop_o = get_oracle_performance(df, 'verbalised', model_name, gpt_effort)
        color = _PALETTES.get('verbalised', "#6baed6")
        ax.scatter([avg_tokens_o], [acc_o], marker='*', s=140, color=color, edgecolors='black', zorder=5, label="Oracle")
        ox.append(avg_tokens_o); oy.append(acc_o)

    _apply_autoscale(ax, all_x + ox, all_y + oy)
    ax.set_xlabel("Generated tokens (Avgerage)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=16, fontweight='bold')
    
    # Update title to reflect routing mode and reasoning effort
    routing_mode = "Online" if online else "Offline"
    # Use provided title (dataset name) and append routing mode
    ax.set_title(f"{title} ({routing_mode})", fontsize=18, fontweight='bold', pad=20)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add annotation for online routing
    if online:
        ax.text(0.02, 0.98, "Online routing\n(randomized order, 20-sample warmup)", 
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Make tick labels bold and larger
    ax.tick_params(axis='both', which='major', labelsize=13)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    legend = ax.legend(uniq.values(), uniq.keys(), title="Method", ncol=2, 
                      fontsize=10, title_fontsize=12)
    legend.get_title().set_fontweight('bold')
    # Make legend text bold
    for text in legend.get_texts():
        text.set_fontweight('bold')

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

    fig.tight_layout(pad=0.1)
    fig.savefig(outpath, dpi=200, bbox_inches="tight", format='pdf', pad_inches=0.0)
    print(f"[Saved] {outpath}")

# ---------------------------- CLI ----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Generate routing trade-off plots with percentile-gated CoT.")
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more CSV/Parquet files.")
    ap.add_argument("--combine", default="concat", choices=["concat"], help="How to combine multiple inputs.")
    ap.add_argument("--model-name", default="qwen", help="String to decide which CoT columns to use ('qwen' or 'gpt-oss').")
    ap.add_argument("--outdir", default="figs", help="Where to save figures.")
    ap.add_argument("--title", default="Dataset", help="Title prefix for figures.")
    ap.add_argument("--types", nargs="+", default=['p_true','verbalised','margin','perplexity','random'],
                    help="Confidence types to plot.")
    ap.add_argument("--threshold-type", default="absolute", choices=["absolute", "percentile"],
                    help="Threshold type for routing (default: absolute)")
    ap.add_argument("--online", action="store_true", 
                    help="Use online percentile routing (default: offline/batch)")
    ap.add_argument("--random-seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
    ap.add_argument("--num-runs", type=int, default=1,
                    help="Number of runs for online routing (default: 1)")
    ap.add_argument("--gpt-effort", default="high", choices=["low", "medium", "high"],
                    help="Reasoning effort for GPT-OSS models (default: high)")
    return ap.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    df = load_many(args.inputs, how=args.combine)

    percs = np.arange(0, 101, 5).tolist() # Default 0-100 by 5
    model_dir = os.path.join(args.outdir, args.model_name.lower().replace("-", "_"), args.threshold_type)
    os.makedirs(model_dir, exist_ok=True)

    # Add routing mode to output filenames
    mode_suffix = "_online" if args.online else "_offline"
    randomize_suffix = "_randomized" if args.online else ""
    runs_suffix = f"_{args.num_runs}runs" if args.online and args.num_runs > 1 else ""
    dataset_clean = args.title.lower().replace(" ", "_").replace("-", "_")

    # Keep dataset filename and combined flag
    is_combined = len(args.inputs) > 1

    # Use provided title (dataset name) for figure titles
    plot_title = args.title
    filename_prefix = "combined" if is_combined else dataset_clean

    # Output files in model subdirectory
    out1 = os.path.join(model_dir, f"{filename_prefix}_routing_ratio_vs_accuracy{mode_suffix}{randomize_suffix}{runs_suffix}_{args.threshold_type}.pdf")
    out2 = os.path.join(model_dir, f"{filename_prefix}_tokens_vs_accuracy{mode_suffix}{randomize_suffix}{runs_suffix}_{args.threshold_type}.pdf")

    plot_accuracy_vs_routing(
        df=df,
        model_name=args.model_name,
        percentiles=percs,
        outpath=out1,
        title=plot_title,
        confidence_types=tuple(args.types),
        online=args.online,
        threshold_type=args.threshold_type,
        num_runs=args.num_runs,
        gpt_effort=args.gpt_effort
    )

    plot_accuracy_vs_tokens(
        df=df,
        model_name=args.model_name,
        percentiles=percs,
        outpath=out2,
        title=plot_title,
        confidence_types=tuple(args.types),
        online=args.online,
        threshold_type=args.threshold_type,
        num_runs=args.num_runs,
        gpt_effort=args.gpt_effort
    )
    
    if args.online:
        if args.num_runs > 1:
            print(f"[Info] Dataset randomized with {args.num_runs} runs, seed {args.random_seed}")
        else:
            print(f"[Info] Dataset randomized with seed {args.random_seed}")

if __name__ == "__main__":
    main()
