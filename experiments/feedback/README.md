# Feedback / fractal track

One axis — loop gain — read on three closed, untwisted loops. Motivated by
analog video feedback (camera pointed at its own monitor), which
`docs/ro_framework.md` §5.3 uses as its example of closure without
self-indexing. Results and interpretation: [findings.md](findings.md).

| # | script | loop | gain knob | geometry measure |
|---|---|---|---|---|
| F1 | `01_seed_spatial_scale.py` | Seed ring lattice | Rule 2a on/off, weight scale | spatial correlation C(d) along the ring; `measure_scale_distribution` (retired) |
| F2 | `02_video_rig.py` | 2-D image, rotate+zoom+blur, clip or contrast decider | g (brightness), up-down sweep | box-counting dimension of the lit mask, hysteresis, twin-arm T/U twist probe |
| F3 | `03_reservoir_gain_sweep.py` | ESN; Seed frozen / self-tuned | spectral radius; weight scale / Rule 2a | correlation dimension D2, participation ratio, saturation, Lyapunov λ |
| F3b | `04_tuned_scan.py` | Seed around its self-tuned weights | α·W* | same as F3, plus drive-only and zero-coupling baselines |
| F4 | `07_embodied_twist.py` + `../embodied/twist_mount.py` | embodied Seed-Hemi brain with the creditable channel, both loops closed | none / live / stale / blind / off, 15k headless steps | certification every 5k; eats, deaths, valence; D2 of the final activation window |
| F5 | `08_richness.py` | tape Seed (#F3c) and embodied mount (#F4), live arms | — | v1 `richness()` vector (degenerate here) and the v2 magnitude: Δm vs probe-noise jitter under coupling × δ, as SNR and bits |
| F3c | `06_seed_creditable_twist.py` | Seed as a core-lib Observer with a creditable (world-targeted) twist channel, certified by `twist_assessment`/`is_conscious` | α·W* under live / stale / blind / off channel | same as F3; live ≡ blind, off shifts the knee |

`fractal_tools.py` holds the estimators (box counting, spectral slope,
Grassberger–Procaccia D2, participation ratio). `05_summary_figure.py`
draws the cross-substrate gain curves from the saved JSON results.

```bash
P=/home/q/miniforge3/envs/ro-framework/bin/python
$P experiments/feedback/01_seed_spatial_scale.py --nodes 128 --steps 8000 --burn 2000 --seeds 3   # ~4 min
$P experiments/feedback/02_video_rig.py --n-gain 16 --per-gain 150 --seeds 2                       # ~3 min
$P experiments/feedback/02_video_rig.py --contrast 3 --pivot 0.2 --save experiments/feedback/results_02_video_rig_c3p02.json
$P experiments/feedback/03_reservoir_gain_sweep.py --seeds 3                                        # ~7 min
$P experiments/feedback/04_tuned_scan.py --seeds 3                                                  # ~8 min
$P experiments/feedback/06_seed_creditable_twist.py --seeds 3 --workers 12                           # ~11 min
$P experiments/feedback/07_embodied_twist.py --seeds 10 --workers 15                                 # ~20 min
$P experiments/feedback/08_richness.py --seeds 3                                                     # ~5 min
$P experiments/feedback/05_summary_figure.py
```
