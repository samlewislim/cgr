import argparse
import csv
import os
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd


# ---------------------------- I/O ----------------------------

def load_any(path: str) -> pd.DataFrame:
    """Load CSV or Parquet into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        sep = "," if ext == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    elif ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext} ({path})")


def load_many(paths: Iterable[str], how: str = "concat") -> pd.DataFrame:
    """Load multiple files and combine. Currently supports 'concat' (row-wise)."""
    dfs = [load_any(p) for p in paths]
    if not dfs:
        raise ValueError("No input files provided.")
    if how == "concat":
        return pd.concat(dfs, ignore_index=True, sort=False)
    else:
        raise ValueError(f"Unsupported combine mode: {how}")


# ---------------------- Column selection helpers ----------------------

def _select_columns(df: pd.DataFrame, confidence_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (conf, d_corr, d_len) arrays for a given confidence type."""
    if confidence_type == 'p_true':
        conf = df['p_true_confidence'].to_numpy(dtype=float)
        d_corr = df['direct_answer_correct_p_true'].to_numpy(dtype=float)
        d_len  = df['direct_answer_len_p_true'].to_numpy(dtype=float)
    elif confidence_type == 'verbalised':
        conf = df['verbalised_confidence'].to_numpy(dtype=float)
        d_corr = df['direct_answer_correct_verbalised'].to_numpy(dtype=float)
        d_len  = df['direct_answer_len_verbalised'].to_numpy(dtype=float)
    elif confidence_type == 'margin':
        conf = df['margin_confidence'].to_numpy(dtype=float)
        d_corr = df['direct_answer_correct_margin_pe'].to_numpy(dtype=float)
        d_len  = df['direct_answer_len_margin_pe'].to_numpy(dtype=float)
    elif confidence_type == 'perplexity':
        conf = df['perplexity_confidence'].to_numpy(dtype=float)
        d_corr = df['direct_answer_correct_margin_pe'].to_numpy(dtype=float)
        d_len  = df['direct_answer_len_margin_pe'].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown confidence type: {confidence_type}")
    return conf, d_corr, d_len


def _select_cot_columns(df: pd.DataFrame, model_name: str, gpt_effort: str = 'medium') -> Tuple[np.ndarray, np.ndarray]:
    """Return (c_corr, c_len) arrays for a given model_name."""
    mn = model_name.lower()
    if "qwen" in mn:
        c_corr = df['cot_answer_correct'].to_numpy(dtype=float)
        c_len  = df['cot_answer_len'].to_numpy(dtype=float)
    elif "gpt-oss" in mn:
        if gpt_effort == 'low':
            c_corr = df['low_cot_answer_correct'].to_numpy(dtype=float)
            c_len  = df['low_cot_answer_len'].to_numpy(dtype=float)
        elif gpt_effort == 'high':
            c_corr = df['high_cot_answer_correct'].to_numpy(dtype=float)
            c_len  = df['high_cot_answer_len'].to_numpy(dtype=float)
        else:
            c_corr = df['medium_cot_answer_correct'].to_numpy(dtype=float)
            c_len  = df['medium_cot_answer_len'].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return c_corr, c_len


# ---------------------- Metrics helpers ----------------------

def _metrics_from_mask(use_cot: np.ndarray,
                       d_corr: np.ndarray, d_len: np.ndarray,
                       c_corr: np.ndarray, c_len: np.ndarray) -> Tuple[float, float, float, float]:
    """Given a boolean mask 'use_cot', compute accuracy, avg tokens, std tokens, and CoT rate."""
    final_corr = np.where(use_cot, c_corr, d_corr)
    final_len  = np.where(use_cot, c_len, d_len)
    acc = float(np.nanmean(final_corr))
    avg_tokens = float(np.nanmean(final_len))
    std_tokens = float(np.nanstd(final_len))
    cot_rate = float(np.nanmean(use_cot.astype(float)))
    return acc, avg_tokens, std_tokens, cot_rate


def _apply_gate(conf: np.ndarray, threshold: float, ctype: str) -> np.ndarray:
    """Routing rule: defer to CoT depending on confidence vs threshold."""
    if ctype in ('p_true', 'verbalised', 'margin'):
        mask = conf <= threshold
    elif ctype == 'perplexity':
        mask = conf >= threshold
    else:
        raise ValueError(f"Unknown confidence type: {ctype}")
    mask = np.where(np.isnan(conf), False, mask)
    return mask


def expected_random_baseline(d_corr: np.ndarray, d_len: np.ndarray,
                             c_corr: np.ndarray, c_len: np.ndarray,
                             r: float) -> Tuple[float, float]:
    """Closed-form expected accuracy/tokens when deferring to CoT with probability r."""
    r = float(r)
    acc = float((1 - r) * np.nanmean(d_corr) + r * np.nanmean(c_corr))
    tokens = float((1 - r) * np.nanmean(d_len)  + r * np.nanmean(c_len))
    return acc, tokens


# ---------------------- Pareto + selection ----------------------

def build_pareto_frontier(points: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """Return the Pareto frontier where no point has higher acc and <= tokens."""
    pts = sorted(points, key=lambda t: (t[1], -t[0]))  # by tokens asc, acc desc
    frontier: List[Tuple[float, float, float]] = []
    best_acc = -np.inf
    for acc, tok, thr in pts:
        if acc > best_acc + 1e-12:  # strictly higher acc
            frontier.append((acc, tok, thr))
            best_acc = acc
    return frontier


def pick_accuracy_preserving(frontier: List[Tuple[float, float, float]],
                             acc_all_cot_cal: float,
                             epsilon: float) -> Tuple[float, float, float]:
    """Pick lowest-token point with acc >= acc_all_cot_cal - epsilon, else best acc."""
    eligible = [(acc, tok, thr) for (acc, tok, thr) in frontier if acc >= acc_all_cot_cal - epsilon]
    if eligible:
        acc, tok, thr = min(eligible, key=lambda t: (t[1], -t[0]))  # min tokens, then max acc
        return acc, tok, thr
    # fallback: highest accuracy, then min tokens among those
    best_acc = max(frontier, key=lambda t: t[0])[0]
    candidates = [(acc, tok, thr) for (acc, tok, thr) in frontier if abs(acc - best_acc) < 1e-12]
    acc, tok, thr = min(candidates, key=lambda t: t[1])
    return acc, tok, thr


# ---------------------- Monte Carlo CV ----------------------

def evaluate_confidence_type_monte_carlo(
    df: pd.DataFrame,
    model_name: str,
    confidence_type: str,
    seed: int,
    percentiles: List[float],
    epsilon: float,
    calibration_frac: float,
    repeats: int,
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    conf_all, d_corr_all, d_len_all = _select_columns(df, confidence_type)
    c_corr_all, c_len_all = _select_cot_columns(df, model_name)

    n = len(df)
    cal_size = max(1, int(round(calibration_frac * n)))
    if cal_size >= n:
        raise ValueError("Calibration fraction too large (no test data left).")

    acc_calib, tok_calib, rate_calib = [], [], []
    thresholds_selected = []  # Track selected thresholds
    acc_rand, tok_rand = [], []
    acc_all_cot, tok_all_cot = [], []
    acc_all_direct, tok_all_direct = [], []
    acc_oracle, tok_oracle = [], []
    rate_oracle = []
    # Track per-repeat category percentages (in percent, 0-100)
    pct_good_cot_necessary_and_used = []
    pct_good_direct_sufficient = []
    pct_bad_should_have_used_cot = []
    pct_bad_wasted_tokens_on_cot = []
    pct_bad_unavoidable_failure = []
    for r in range(repeats):
        idx = np.arange(n)
        rng.shuffle(idx)
        cal_idx = idx[:cal_size]
        test_idx = idx[cal_size:]

        conf_cal, conf_test = conf_all[cal_idx], conf_all[test_idx]
        d_corr_cal, d_corr_test = d_corr_all[cal_idx], d_corr_all[test_idx]
        d_len_cal, d_len_test = d_len_all[cal_idx], d_len_all[test_idx]
        c_corr_cal, c_corr_test = c_corr_all[cal_idx], c_corr_all[test_idx]
        c_len_cal, c_len_test = c_len_all[cal_idx], c_len_all[test_idx]

        # sweep fixed percentiles
        cand_points: List[Tuple[float, float, float]] = []
        acc_all_cot_on_cal = float(np.nanmean(c_corr_cal))

        if np.all(np.isnan(conf_cal)) or (np.nanmin(conf_cal) == np.nanmax(conf_cal)):
            use_cot_test = np.ones_like(conf_test, dtype=bool)
            acc_t, tok_t, _, cot_rate_t = _metrics_from_mask(use_cot_test, d_corr_test, d_len_test, c_corr_test, c_len_test)
            thr_sel = np.nan  # No meaningful threshold for this case
        else:
            for p in percentiles:
                thr = float(np.nanpercentile(conf_cal, p))
                use_cot_cal = _apply_gate(conf_cal, thr, confidence_type)
                acc_c, tok_c, _, _ = _metrics_from_mask(use_cot_cal, d_corr_cal, d_len_cal, c_corr_cal, c_len_cal)
                cand_points.append((acc_c, tok_c, thr))

            frontier = build_pareto_frontier(cand_points)
            _, _, thr_sel = pick_accuracy_preserving(frontier, acc_all_cot_on_cal, epsilon)

            use_cot_test = _apply_gate(conf_test, thr_sel, confidence_type)
            acc_t, tok_t, _, cot_rate_t = _metrics_from_mask(use_cot_test, d_corr_test, d_len_test, c_corr_test, c_len_test)

        thresholds_selected.append(thr_sel)  # Track the selected threshold

        # rate-matched random
        acc_r, tok_r = expected_random_baseline(d_corr_test, d_len_test, c_corr_test, c_len_test, cot_rate_t)

        # baselines
        acc_cot = float(np.nanmean(c_corr_test)); tok_cot = float(np.nanmean(c_len_test))
        acc_dir = float(np.nanmean(d_corr_test)); tok_dir = float(np.nanmean(d_len_test))
        oracle_corr = np.where(d_corr_test == 1.0, d_corr_test, c_corr_test)
        oracle_len  = np.where(d_corr_test == 1.0, d_len_test,  c_len_test)
        acc_orc = float(np.nanmean(oracle_corr)); tok_orc = float(np.nanmean(oracle_len))
        cot_rate_orc = float(np.mean(d_corr_test == 0.0))  #

        # collect
        acc_calib.append(acc_t); tok_calib.append(tok_t); rate_calib.append(cot_rate_t)
        # categorize routing decisions on the test set and collect percentages
        test_n = len(use_cot_test)
        if test_n > 0:
            # ensure boolean arrays: treat only exact 1.0 as correct, NaN -> False
            d_bool = (d_corr_test == 1.0)
            c_bool = (c_corr_test == 1.0)
            u_bool = (use_cot_test.astype(bool))

            good_cot_necessary_and_used = int(np.sum((~d_bool) & c_bool & u_bool))
            good_direct_sufficient = int(np.sum(d_bool & (~u_bool)))
            bad_should_have_used_cot = int(np.sum((~d_bool) & c_bool & (~u_bool)))
            bad_wasted_tokens_on_cot = int(np.sum(d_bool & u_bool))
            bad_unavoidable_failure = int(np.sum((~d_bool) & (~c_bool)))

            pct_good_cot_necessary_and_used.append(100.0 * good_cot_necessary_and_used / test_n)
            pct_good_direct_sufficient.append(100.0 * good_direct_sufficient / test_n)
            pct_bad_should_have_used_cot.append(100.0 * bad_should_have_used_cot / test_n)
            pct_bad_wasted_tokens_on_cot.append(100.0 * bad_wasted_tokens_on_cot / test_n)
            pct_bad_unavoidable_failure.append(100.0 * bad_unavoidable_failure / test_n)
        else:
            pct_good_cot_necessary_and_used.append(0.0)
            pct_good_direct_sufficient.append(0.0)
            pct_bad_should_have_used_cot.append(0.0)
            pct_bad_wasted_tokens_on_cot.append(0.0)
            pct_bad_unavoidable_failure.append(0.0)

        acc_rand.append(acc_r); tok_rand.append(tok_r)
        acc_all_cot.append(acc_cot); tok_all_cot.append(tok_cot)
        acc_all_direct.append(acc_dir); tok_all_direct.append(tok_dir)
        acc_oracle.append(acc_orc); tok_oracle.append(tok_orc)
        rate_oracle.append(cot_rate_orc)  


    def m(x): return float(np.nanmean(x))
    def s(x): return float(np.nanstd(x))

    return {
        "confidence_type": confidence_type,
        "calibration_threshold_acc_mean": m(acc_calib),
        "calibration_threshold_acc_std":  s(acc_calib),
        "calibration_threshold_tokens_mean": m(tok_calib),
        "calibration_threshold_tokens_std":  s(tok_calib),
        "calibration_threshold_cot_rate_mean": m(rate_calib),
        "calibration_threshold_cot_rate_std": s(rate_calib), 
        "calibration_threshold_threshold_mean": m(thresholds_selected),  
        "calibration_threshold_threshold_std": s(thresholds_selected),   
        "random_rate_matched_acc_mean": m(acc_rand),
        "random_rate_matched_tokens_mean": m(tok_rand),
        "random_rate_matched_cot_rate_mean": m(rate_calib),                   
        "all_cot_acc_mean": m(acc_all_cot),
        "all_cot_tokens_mean": m(tok_all_cot),
        "all_cot_cot_rate_mean": 1.0,                                         
        "all_direct_acc_mean": m(acc_all_direct),
        "all_direct_tokens_mean": m(tok_all_direct),
        "all_direct_cot_rate_mean": 0.0,                                      
        "oracle_acc_mean": m(acc_oracle),
        "oracle_tokens_mean": m(tok_oracle),
        "oracle_cot_rate_mean": m(rate_oracle),                                
        # Category percentages (means and stds)
        "pct_good_cot_necessary_and_used_mean": m(pct_good_cot_necessary_and_used),
        "pct_good_cot_necessary_and_used_std": s(pct_good_cot_necessary_and_used),
        "pct_good_direct_sufficient_mean": m(pct_good_direct_sufficient),
        "pct_good_direct_sufficient_std": s(pct_good_direct_sufficient),
        "pct_bad_should_have_used_cot_mean": m(pct_bad_should_have_used_cot),
        "pct_bad_should_have_used_cot_std": s(pct_bad_should_have_used_cot),
        "pct_bad_wasted_tokens_on_cot_mean": m(pct_bad_wasted_tokens_on_cot),
        "pct_bad_wasted_tokens_on_cot_std": s(pct_bad_wasted_tokens_on_cot),
        "pct_bad_unavoidable_failure_mean": m(pct_bad_unavoidable_failure),
        "pct_bad_unavoidable_failure_std": s(pct_bad_unavoidable_failure),
    }

# ---------------------- CLI & orchestration ----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Monte Carlo CV calibration & Pareto thresholding for CoT routing.")
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more CSV/Parquet files; concatenated as one pool.")
    ap.add_argument("--combine", default="concat", choices=["concat"], help="How to combine multiple inputs.")
    ap.add_argument("--model-name", required=True, help="Which model columns to use: e.g., 'qwen3-32b', 'gpt-oss'.")
    ap.add_argument("--confidence-types", nargs="+",
                    default=['p_true', 'verbalised', 'margin', 'perplexity'],
                    help="Which confidence signals to evaluate.")
    ap.add_argument("--percentile-step", type=int, default=5,
                    help="Percentile step size for calibration sweep (default: 5 => 0,5,10,...,100)")
    ap.add_argument("--epsilon", type=float, default=0.0,
                    help="Accuracy tolerance vs. All-CoT on calibration (default: 0.0)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--mc-repeats", type=int, default=5,
                    help="Number of Monte Carlo repeats (default: 5). Set to 0 to disable.")
    ap.add_argument("--calibration-frac", type=float, default=0.15,
                    help="Fraction of data used for calibration in Monte Carlo CV (default: 0.15)")
    ap.add_argument("--out", default="metrics_summary.csv", help="Output CSV for aggregated results")
    ap.add_argument("--select-best", action="store_true",
                    help="Also output a single 'Calibration Threshold' row choosing the best confidence type.")
    return ap.parse_args()


def main():
    args = parse_args()
    df = load_many(args.inputs, how=args.combine)

    percentiles = list(np.arange(0, 101, args.percentile_step))

    rows: List[Dict[str, float]] = []
    per_type_summary: Dict[str, Dict[str, float]] = {}

    for ctype in args.confidence_types:
        res = evaluate_confidence_type_monte_carlo(
            df=df,
            model_name=args.model_name,
            confidence_type=ctype,
            seed=args.seed,
            percentiles=percentiles,
            epsilon=args.epsilon,
            calibration_frac=args.calibration_frac,
            repeats=args.mc_repeats,
        )
        per_type_summary[ctype] = res
        
        # Calculate token savings vs all-CoT
        tokens_saved = res["all_cot_tokens_mean"] - res["calibration_threshold_tokens_mean"]
        
        rows.append({
            "Method": f"Calibration Threshold ({ctype})",
            "Acc_mean": res["calibration_threshold_acc_mean"],
            "Tokens_mean": res["calibration_threshold_tokens_mean"],
            "CotRate_mean": res["calibration_threshold_cot_rate_mean"],
            "Threshold_mean": res["calibration_threshold_threshold_mean"],
            "Threshold_std": res["calibration_threshold_threshold_std"],
            "Tokens_saved_vs_all_cot": tokens_saved,
            "Std_Acc": res["calibration_threshold_acc_std"],
            "Std_Tokens": res["calibration_threshold_tokens_std"],
            "Std_CotRate": res["calibration_threshold_cot_rate_std"],  # NEW
            # Category percentages
            "pct_good_cot_necessary_and_used_mean": res.get("pct_good_cot_necessary_and_used_mean", 0.0),
            "pct_good_cot_necessary_and_used_std": res.get("pct_good_cot_necessary_and_used_std", 0.0),
            "pct_good_direct_sufficient_mean": res.get("pct_good_direct_sufficient_mean", 0.0),
            "pct_good_direct_sufficient_std": res.get("pct_good_direct_sufficient_std", 0.0),
            "pct_bad_should_have_used_cot_mean": res.get("pct_bad_should_have_used_cot_mean", 0.0),
            "pct_bad_should_have_used_cot_std": res.get("pct_bad_should_have_used_cot_std", 0.0),
            "pct_bad_wasted_tokens_on_cot_mean": res.get("pct_bad_wasted_tokens_on_cot_mean", 0.0),
            "pct_bad_wasted_tokens_on_cot_std": res.get("pct_bad_wasted_tokens_on_cot_std", 0.0),
            "pct_bad_unavoidable_failure_mean": res.get("pct_bad_unavoidable_failure_mean", 0.0),
            "pct_bad_unavoidable_failure_std": res.get("pct_bad_unavoidable_failure_std", 0.0),
        })

    # Baselines: mean across types for stability
    def mean_over_types(key: str) -> float:
        return float(np.nanmean([per_type_summary[ct][key] for ct in per_type_summary]))

    all_cot_tokens = mean_over_types("all_cot_tokens_mean")
    
    rows_baselines = [
        {"Method": "All CoT",
        "Acc_mean": mean_over_types("all_cot_acc_mean"),
        "Tokens_mean": all_cot_tokens,
        "CotRate_mean": mean_over_types("all_cot_cot_rate_mean"),
        "Tokens_saved_vs_all_cot": 0.0,  # Reference point
        "Threshold_mean": "",  # N/A for baselines
        "Threshold_std": "",
        "Std_CotRate": ""},  # N/A for baselines
        {"Method": "All Direct",
        "Acc_mean": mean_over_types("all_direct_acc_mean"),
        "Tokens_mean": mean_over_types("all_direct_tokens_mean"),
        "CotRate_mean": mean_over_types("all_direct_cot_rate_mean"),
        "Tokens_saved_vs_all_cot": all_cot_tokens - mean_over_types("all_direct_tokens_mean"),
        "Threshold_mean": "",  # N/A for baselines
        "Threshold_std": "",
        "Std_CotRate": ""},  # N/A for baselines
        {"Method": "Random Routing (rate-matched)",
        "Acc_mean": mean_over_types("random_rate_matched_acc_mean"),
        "Tokens_mean": mean_over_types("random_rate_matched_tokens_mean"),
        "CotRate_mean": mean_over_types("random_rate_matched_cot_rate_mean"),
        "Tokens_saved_vs_all_cot": all_cot_tokens - mean_over_types("random_rate_matched_tokens_mean"),
        "Threshold_mean": "",  # N/A for baselines
        "Threshold_std": "",
        "Std_CotRate": ""},  # N/A for baselines
        {"Method": "Oracle",
        "Acc_mean": mean_over_types("oracle_acc_mean"),
        "Tokens_mean": mean_over_types("oracle_tokens_mean"),
        "CotRate_mean": mean_over_types("oracle_cot_rate_mean"),
        "Tokens_saved_vs_all_cot": all_cot_tokens - mean_over_types("oracle_tokens_mean"),
        "Threshold_mean": "",  # N/A for baselines
        "Threshold_std": "",
        "Std_CotRate": ""},  # N/A for baselines
    ]

    # Optionally pick best confidence type
    best_row = None
    if args.select_best:
        acc_all_cot = mean_over_types("all_cot_acc_mean")
        eligible = []
        for ct, res in per_type_summary.items():
            acc = res["calibration_threshold_acc_mean"]
            tok = res["calibration_threshold_tokens_mean"]
            if acc >= acc_all_cot - args.epsilon:
                eligible.append((tok, -acc, ct))  # min tokens, then max acc
        if not eligible:
            cand = sorted([( -res["calibration_threshold_acc_mean"], res["calibration_threshold_tokens_mean"], ct)
                           for ct, res in per_type_summary.items()])[0]
            ct_best = cand[2]
        else:
            ct_best = sorted(eligible)[0][2]

        resb = per_type_summary[ct_best]
        tokens_saved_best = resb["all_cot_tokens_mean"] - resb["calibration_threshold_tokens_mean"]
        best_row = {"Method": f"Calibration Threshold (best={ct_best})",
                    "Acc_mean": resb["calibration_threshold_acc_mean"],
                    "Tokens_mean": resb["calibration_threshold_tokens_mean"],
                    "Threshold_mean": resb["calibration_threshold_threshold_mean"],
                    "Threshold_std": resb["calibration_threshold_threshold_std"],
                    "Tokens_saved_vs_all_cot": tokens_saved_best,
                    "Std_Acc": resb["calibration_threshold_acc_std"],
                    "Std_Tokens": resb["calibration_threshold_tokens_std"],
                    "Std_CotRate": resb["calibration_threshold_cot_rate_std"],
                    # Category percentages for best
                    "pct_good_cot_necessary_and_used_mean": resb.get("pct_good_cot_necessary_and_used_mean", 0.0),
                    "pct_good_cot_necessary_and_used_std": resb.get("pct_good_cot_necessary_and_used_std", 0.0),
                    "pct_good_direct_sufficient_mean": resb.get("pct_good_direct_sufficient_mean", 0.0),
                    "pct_good_direct_sufficient_std": resb.get("pct_good_direct_sufficient_std", 0.0),
                    "pct_bad_should_have_used_cot_mean": resb.get("pct_bad_should_have_used_cot_mean", 0.0),
                    "pct_bad_should_have_used_cot_std": resb.get("pct_bad_should_have_used_cot_std", 0.0),
                    "pct_bad_wasted_tokens_on_cot_mean": resb.get("pct_bad_wasted_tokens_on_cot_mean", 0.0),
                    "pct_bad_wasted_tokens_on_cot_std": resb.get("pct_bad_wasted_tokens_on_cot_std", 0.0),
                    "pct_bad_unavoidable_failure_mean": resb.get("pct_bad_unavoidable_failure_mean", 0.0),
                    "pct_bad_unavoidable_failure_std": resb.get("pct_bad_unavoidable_failure_std", 0.0)}

    # Write CSV
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Add category percentage columns to fieldnames
    fieldnames = [
        "Method", "Acc_mean", "Tokens_mean", "CotRate_mean", "Threshold_mean", "Threshold_std", "Tokens_saved_vs_all_cot",
        "Std_Acc", "Std_Tokens", "Std_CotRate",
        "pct_good_cot_necessary_and_used_mean", "pct_good_cot_necessary_and_used_std",
        "pct_good_direct_sufficient_mean", "pct_good_direct_sufficient_std",
        "pct_bad_should_have_used_cot_mean", "pct_bad_should_have_used_cot_std",
        "pct_bad_wasted_tokens_on_cot_mean", "pct_bad_wasted_tokens_on_cot_std",
        "pct_bad_unavoidable_failure_mean", "pct_bad_unavoidable_failure_std",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_baselines:
            r.setdefault("Std_Acc", "")
            r.setdefault("Std_Tokens", "")
            r.setdefault("Std_CotRate", "")  # NEW
            # ensure category fields exist for baselines
            r.setdefault("pct_good_cot_necessary_and_used_mean", "")
            r.setdefault("pct_good_cot_necessary_and_used_std", "")
            r.setdefault("pct_good_direct_sufficient_mean", "")
            r.setdefault("pct_good_direct_sufficient_std", "")
            r.setdefault("pct_bad_should_have_used_cot_mean", "")
            r.setdefault("pct_bad_should_have_used_cot_std", "")
            r.setdefault("pct_bad_wasted_tokens_on_cot_mean", "")
            r.setdefault("pct_bad_wasted_tokens_on_cot_std", "")
            r.setdefault("pct_bad_unavoidable_failure_mean", "")
            r.setdefault("pct_bad_unavoidable_failure_std", "")
            writer.writerow(r)
        for r in rows:
            writer.writerow(r)
        if best_row is not None:
            writer.writerow(best_row)

    # Console pretty print
    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else str(x)

    print("\n=== Aggregated Results (means across repeats) ===")
    print(f"Model: {args.model_name} | Monte Carlo repeats={args.mc_repeats} | seed={args.seed} | epsilon={args.epsilon}")
    print(f"Inputs: {len(args.inputs)} files (concatenated)")
    print(f"Percentiles: {percentiles[:3]} ... {percentiles[-3:]}\n")

    for r in rows_baselines + rows + ([best_row] if best_row else []):
        if not r:
            continue
        method_name = f"{r['Method']:<45}"
        acc_info = f"Acc={fmt(r['Acc_mean'])}"
        tokens_info = f"Tokens={fmt(r['Tokens_mean'])}"
        cot_info = f"CoT={fmt(r.get('CotRate_mean',''))}"
        tokens_saved_info = f"Saved={fmt(r.get('Tokens_saved_vs_all_cot', ''))}"
        
        # Add threshold info if available
        threshold_info = ""
        if r.get('Threshold_mean', '') != "" and not isinstance(r.get('Threshold_mean'), str):
            threshold_info = f"Thr={fmt(r['Threshold_mean'])}"
            if r.get('Threshold_std', '') != "" and not isinstance(r.get('Threshold_std'), str):
                threshold_info += f"±{fmt(r['Threshold_std'])}"
        
        print(f"{method_name} {acc_info}  {tokens_info}  {cot_info}  {tokens_saved_info}", end="")
        if threshold_info:
            print(f"  {threshold_info}", end="")
        if r.get("Std_Acc", "") != "":
            std_info = f"(±{fmt(r['Std_Acc'])} acc, ±{fmt(r['Std_Tokens'])} tok"
            if r.get("Std_CotRate", "") != "":
                std_info += f", ±{fmt(r['Std_CotRate'])} cot"
            std_info += ")"
            print(f"  {std_info}")
        else:
            print("")


    print(f"\n[Saved] {out_path}")

if __name__ == "__main__":
    main()
