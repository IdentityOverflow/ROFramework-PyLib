"""
plot_logs.py — Plot brain training logs.

Usage:
    python plot_logs.py                     # reads logs/ relative to this script
    python plot_logs.py logs/Alice-729.log  # specific files
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

LOGS_DIR = Path(__file__).parent / "logs"
COLORS   = ["#7eb8f7", "#f4a55a", "#7ecb7e", "#d97ef5"]

K_COLS = [
    "K_food_max_proximity",
    "K_danger_max_proximity",
    "K_life",
    "K_satiation_norm",
    "K_valence_norm",
]
K_LABELS = ["food prox", "danger prox", "life", "satiation", "valence"]

EXTRA_K_COLS = [
    ("K_sat_eat",       "K(satiation → eat)  — hunger drives eating?"),
    ("K_food_eat_cond", "K(food prox → eat | food visible)  — react when food present?"),
    ("K_food_eat_lag2", "K(food prox[t] → eat[t+2])  — delayed eat response?"),
    ("K_fwd_autocorr",  "Motor persistence: ρ(fwd[t], fwd[t+1])"),
    ("K_turn_autocorr", "Motor persistence: ρ(turn[t], turn[t+1])"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_logs(paths: list[Path]) -> list[tuple[str, pd.DataFrame]]:
    result = []
    for p in paths:
        df = pd.read_csv(p, parse_dates=["timestamp"])
        for col in ["mean_reward", "w_norm", "eat_count", "mean_fwd", "mean_turn"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        result.append((p.stem, df))
    result.sort(key=lambda x: x[0])
    return result


def smooth(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window, min_periods=1, center=True).mean()


def _style(fig, axes):
    fig.patch.set_facecolor("#1a1a2e")
    for ax in np.array(axes).flat:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#cccccc")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("#e0e0e0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")


def _legend(ax):
    ax.legend(facecolor="#0f3460", labelcolor="#cccccc", framealpha=0.8)


# ── Post-hoc metrics computable from existing aggregated logs ─────────────────

def _motor_consistency(df: pd.DataFrame) -> pd.Series:
    """Rolling std of mean_fwd — low = consistent spiral, high = chaotic."""
    return df["mean_fwd"].rolling(30, min_periods=5).std().fillna(0)


def _fwd_autocorr_posthoc(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """Lag-1 autocorrelation of mean_fwd in a rolling window.
    Measures motor persistence at the 300-step-window timescale."""
    vals = df["mean_fwd"].values
    result = np.full(len(vals), np.nan)
    for i in range(window, len(vals)):
        chunk = vals[i - window:i]
        if np.std(chunk) > 1e-4:
            result[i] = np.corrcoef(chunk[:-1], chunk[1:])[0, 1]
    return pd.Series(result, index=df.index)


# ── Plot ──────────────────────────────────────────────────────────────────────

def _multiline(ax, logs, col_fn, ylabel, title, ylim=None, hline=None):
    """Plot one smoothed line per brain."""
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    if hline is not None:
        ax.axhline(hline, color="#888888", lw=0.6, ls="--", label=f"strong ({hline})")
    for (label, df), color in zip(logs, COLORS):
        series = col_fn(df)
        if series is not None:
            ax.plot(df["step"], smooth(series), label=label, color=color, lw=1.5)
    ax.set_ylabel(ylabel)
    _legend(ax)


def _wall_clock_axis(ax, df0):
    """Add a wall-clock x-axis on top of ax using the first brain's timestamps."""
    ax_time = ax.twiny()
    ax_time.set_xlim(ax.get_xlim())
    idx = np.linspace(0, len(df0) - 1, 6, dtype=int)
    ax_time.set_xticks(df0["step"].iloc[idx].values)
    ax_time.set_xticklabels(df0["timestamp"].dt.strftime("%H:%M").iloc[idx].values, fontsize=8)
    ax_time.tick_params(colors="#aaaaaa")
    ax_time.set_xlabel("wall clock (first brain)", color="#aaaaaa", fontsize=8)


def _numeric_col(df, col):
    """Return smoothable series for col, or None if absent."""
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce").fillna(0)


def _print_summary(logs):
    print("\n── Summary (last 10% of run) ──────────────────────────────────────")
    for label, df in logs:
        tail = df.iloc[max(0, len(df) - len(df) // 10):]
        print(f"  {label:<15}  reward={tail['mean_reward'].mean():+.3f}"
              f"  eats={tail['eat_count'].mean():.1f}/win"
              f"  fwd_std={df['mean_fwd'].std():.3f}"
              f"  turn_std={df['mean_turn'].std():.3f}"
              f"  |W|={df['w_norm'].iloc[-1]:.1f}")


def plot(logs: list[tuple[str, pd.DataFrame]], out_path: Path) -> None:
    has_extra = any(col in df.columns for _, df in logs for col, _ in EXTRA_K_COLS)

    n_rows = 3 + len(K_COLS) + 2 + (len(EXTRA_K_COLS) if has_extra else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows))
    _style(fig, axes)

    row = 0

    _multiline(axes[row], logs,
               lambda df: df["mean_reward"],
               "reward", "Mean reward (per 300-step window, smoothed)")
    _wall_clock_axis(axes[row], logs[0][1])
    row += 1

    _multiline(axes[row], logs,
               lambda df: df["eat_count"].astype(float),
               "eats / window", "Eat count per 300-step window (smoothed)")
    row += 1

    _multiline(axes[row], logs,
               lambda df: df["w_norm"],
               "|W_out|", "|W_out| norm over training")
    row += 1

    for col, klabel in zip(K_COLS, K_LABELS):
        _multiline(axes[row], logs,
                   lambda df, c=col: _numeric_col(df, c),
                   "ρ", f"K({klabel}) — best action ρ (smoothed)",
                   ylim=(0, 1), hline=0.7)
        row += 1

    _multiline(axes[row], logs,
               _motor_consistency,
               "std(mean_fwd)", "Motor variability: rolling std(mean_fwd)  — low=spiral, high=chaotic")
    row += 1

    # Action portrait scatter
    ax = axes[row]; row += 1
    ax.set_title("Action space portrait: mean_fwd vs mean_turn (each dot = 300-step window)")
    ax.axhline(0, color="#444466", lw=0.5, ls="--")
    ax.axvline(0, color="#444466", lw=0.5, ls="--")
    for (label, df), color in zip(logs, COLORS):
        ax.scatter(df["mean_fwd"], df["mean_turn"], c=color, alpha=0.25, s=8, label=label)
    ax.set_xlabel("mean_fwd", color="#cccccc")
    ax.set_ylabel("mean_turn", color="#cccccc")
    _legend(ax)

    if has_extra:
        for col, title in EXTRA_K_COLS:
            hline = None if "autocorr" in col else 0.7
            _multiline(axes[row], logs,
                       lambda df, c=col: _numeric_col(df, c),
                       "ρ", title, ylim=(0, 1), hline=hline)
            row += 1

    axes[-1].set_xlabel("step", color="#cccccc")
    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    _print_summary(logs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = sorted(LOGS_DIR.glob("*.log"))
        if not paths:
            print(f"No .log files found in {LOGS_DIR}")
            sys.exit(1)

    print(f"Loading {len(paths)} log(s):")
    for p in paths:
        print(f"  {p.name}")

    logs    = load_logs(paths)
    out     = LOGS_DIR / "training_curves.png"
    plot(logs, out)


if __name__ == "__main__":
    main()
