"""
data/generate_synthetic.py — Generate the four synthetic fab/telemetry CSVs.

Run from the repo root:
    python data/generate_synthetic.py

Two hidden patterns are baked in:
  1. Via-type × anneal-temperature interaction:
       tungsten vias show a ~30 % resistance uplift above 420 °C anneal temp;
       copper vias are unaffected. (ANOVA will surface this in Milestone 4.)
  2. Humidity → yield:
       when cleanroom humidity exceeds 47 %, continuity-failure rate roughly
       doubles. (Correlation / logistic regression in Milestone 4.)
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

OUT = pathlib.Path(__file__).parent   # data/

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_LOTS      = 8
WAFERS_PER  = 5
DIES_PER    = 20
VIAS_PER    = 10          # vias measured per die

VIA_TYPES   = ["copper", "tungsten"]
LAYERS      = ["M1", "M2", "M3", "M4"]

# Cleanroom environmental parameters
ENV_STEPS   = 2_000       # one reading every ~90 s over ~50 h
HUMID_HIGH_LOTS = {"LOT004", "LOT006"}   # lots processed during humid excursion


# ---------------------------------------------------------------------------
# 1. process_log.csv
# ---------------------------------------------------------------------------
def make_process_log() -> pd.DataFrame:
    records = []
    base_time = pd.Timestamp("2024-03-01 06:00")

    for lot_num in range(1, N_LOTS + 1):
        lot_id = f"LOT{lot_num:03d}"
        for w in range(1, WAFERS_PER + 1):
            wafer_id = f"{lot_id}-W{w:02d}"
            anneal_temp = float(RNG.normal(410, 15))          # °C, ~N(410,15)
            records.append({
                "lot_id":           lot_id,
                "wafer_id":         wafer_id,
                "deposition_temp_c": round(float(RNG.normal(320, 8)), 2),
                "etch_time_s":       round(float(RNG.normal(90, 5)), 2),
                "anneal_temp_c":     round(anneal_temp, 2),
                "pressure_torr":     round(float(RNG.normal(0.005, 0.0005)), 6),
                "operator_id":       f"OP{RNG.integers(1, 5):02d}",
                "process_timestamp": (base_time + pd.Timedelta(hours=lot_num * 6 + w)).isoformat(),
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. environmental_log.csv
# ---------------------------------------------------------------------------
def make_env_log() -> pd.DataFrame:
    base_time = pd.Timestamp("2024-03-01 06:00")
    timestamps = [base_time + pd.Timedelta(seconds=90 * i) for i in range(ENV_STEPS)]

    # Base humidity ~38 %, with excursion mid-run
    humidity = RNG.normal(38, 3, ENV_STEPS)
    excursion_start = ENV_STEPS // 3
    excursion_end   = 2 * ENV_STEPS // 3
    humidity[excursion_start:excursion_end] += RNG.normal(12, 2, excursion_end - excursion_start)
    humidity = np.clip(humidity, 20, 80)

    return pd.DataFrame({
        "env_timestamp":    [t.isoformat() for t in timestamps],
        "temperature_c":    np.round(RNG.normal(21.5, 0.3, ENV_STEPS), 2),
        "humidity_pct":     np.round(humidity, 2),
        "particle_count_m3": RNG.integers(50, 500, ENV_STEPS).astype(int),
        "station_id":       RNG.choice(["STA1", "STA2", "STA3"], ENV_STEPS),
    })


# ---------------------------------------------------------------------------
# 3. design_manifest.csv  (one row per via_type × layer combination)
# ---------------------------------------------------------------------------
_NOMINAL = {
    ("copper",   "M1"): 0.8,  ("copper",   "M2"): 1.1,
    ("copper",   "M3"): 1.4,  ("copper",   "M4"): 1.7,
    ("tungsten", "M1"): 1.0,  ("tungsten", "M2"): 1.4,
    ("tungsten", "M3"): 1.8,  ("tungsten", "M4"): 2.1,
}
_PITCH  = {"M1": 0.18, "M2": 0.25, "M3": 0.36, "M4": 0.50}
_ASPECT = {"M1": 5.0,  "M2": 4.0,  "M3": 3.5,  "M4": 3.0}

def make_design_manifest() -> pd.DataFrame:
    rows = []
    for vt in VIA_TYPES:
        for layer in LAYERS:
            rows.append({
                "via_type":             vt,
                "layer":                layer,
                "pitch_um":             _PITCH[layer],
                "aspect_ratio":         _ASPECT[layer],
                "metal_stack":          f"{vt.upper()}-{layer}",
                "nominal_resistance_ohm": _NOMINAL[(vt, layer)],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. interconnect_measurements.csv   ← hidden patterns live here
# ---------------------------------------------------------------------------
def make_measurements(process_log: pd.DataFrame, env_log: pd.DataFrame) -> pd.DataFrame:
    # Pre-build lookup: wafer_id → anneal_temp_c
    anneal = process_log.set_index("wafer_id")["anneal_temp_c"].to_dict()
    lot_of  = process_log.set_index("wafer_id")["lot_id"].to_dict()

    # Humidity series for time-based lookup (simplified: use mean over excursion window)
    humid_vals = env_log["humidity_pct"].values
    n_env      = len(humid_vals)

    records = []
    meas_id = 0
    base_time = pd.Timestamp("2024-03-01 07:00")

    for lot_num in range(1, N_LOTS + 1):
        lot_id = f"LOT{lot_num:03d}"
        for w in range(1, WAFERS_PER + 1):
            wafer_id = f"{lot_id}-W{w:02d}"
            at = anneal[wafer_id]

            # Ambient humidity for this wafer (pull from env log slice)
            env_start = int(((lot_num - 1) * WAFERS_PER + (w - 1)) / (N_LOTS * WAFERS_PER) * n_env)
            env_end   = env_start + 100
            mean_humid = float(np.mean(humid_vals[env_start:min(env_end, n_env)]))

            for die in range(1, DIES_PER + 1):
                for via_num in range(1, VIAS_PER + 1):
                    via_type = RNG.choice(VIA_TYPES)
                    layer    = RNG.choice(LAYERS)
                    nominal  = _NOMINAL[(via_type, layer)]

                    # --- HIDDEN PATTERN 1: tungsten × high anneal temp ---
                    temp_factor = 1.0
                    if via_type == "tungsten" and at > 420:
                        temp_factor = 1.0 + 0.30 * ((at - 420) / 30)  # up to ~30 % uplift

                    resistance = float(RNG.normal(nominal * temp_factor, nominal * 0.05))
                    resistance = max(0.01, round(resistance, 4))

                    # --- HIDDEN PATTERN 2: high humidity → more failures ---
                    fail_prob = 0.04 if mean_humid <= 47 else 0.09
                    continuity_pass = bool(RNG.random() > fail_prob)

                    meas_time = base_time + pd.Timedelta(minutes=meas_id * 0.5)
                    records.append({
                        "meas_id":         meas_id,
                        "lot_id":          lot_id,
                        "wafer_id":        wafer_id,
                        "die_id":          die,
                        "via_id":          via_num,
                        "via_type":        via_type,
                        "layer":           layer,
                        "resistance_ohm":  resistance,
                        "continuity_pass": continuity_pass,
                        "mean_humidity_pct": round(mean_humid, 2),  # denorm for easy analysis
                        "anneal_temp_c":   at,                       # denorm for easy analysis
                        "meas_timestamp":  meas_time.isoformat(),
                    })
                    meas_id += 1

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating synthetic fab data...")

    process  = make_process_log()
    env      = make_env_log()
    design   = make_design_manifest()
    meas     = make_measurements(process, env)

    process.to_csv(OUT / "process_log.csv",                   index=False)
    env.to_csv(    OUT / "environmental_log.csv",             index=False)
    design.to_csv( OUT / "design_manifest.csv",               index=False)
    meas.to_csv(   OUT / "interconnect_measurements.csv",     index=False)

    print(f"  process_log.csv            {len(process):>6,} rows")
    print(f"  environmental_log.csv      {len(env):>6,} rows")
    print(f"  design_manifest.csv        {len(design):>6,} rows")
    print(f"  interconnect_measurements  {len(meas):>6,} rows")
    print("Done.")
