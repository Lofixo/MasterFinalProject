import argparse
import csv
import numpy as np
import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from build_data_pairs import build_segy_picks_pairs
from config import PICK_TYPES
from processing_segy import read_segy


def normalize_trace_to_unit_maxabs(trace: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    trace = trace.astype(np.float32, copy=False)
    m = float(np.max(np.abs(trace))) if trace.size else 0.0
    if m <= eps:
        return np.zeros_like(trace, dtype=np.float32)
    out = trace / m
    # Guard tiny numerical overshoots
    return np.clip(out, -1.0, 1.0)


def pad_or_truncate_trace(trace: np.ndarray, target_samples: int) -> np.ndarray:
    current_samples = trace.shape[0]
    if current_samples == target_samples:
        return trace
    if current_samples > target_samples:
        return trace[:target_samples]

    padded = np.zeros(target_samples, dtype=trace.dtype)
    padded[:current_samples] = trace
    return padded


def get_sampling_interval_from_time(time: np.ndarray) -> float:
    if time.size < 2:
        raise ValueError("SEG-Y time axis must contain at least 2 samples to infer dt")

    dt = float(time[1] - time[0])
    if dt <= 0.0:
        raise ValueError(f"Invalid non-positive SEG-Y sampling interval: {dt}")
    return dt


def resample_trace_to_target_dt(trace: np.ndarray, source_dt: float, target_dt: float) -> np.ndarray:
    if trace.size == 0:
        return np.zeros(0, dtype=np.float32)

    if target_dt <= 0.0:
        raise ValueError(f"target_dt must be positive, got {target_dt}")

    if np.isclose(source_dt, target_dt, rtol=0.0, atol=1e-12):
        return trace.astype(np.float32, copy=False)

    source_time = np.arange(trace.size, dtype=np.float64) * source_dt
    duration = source_time[-1]
    n_target = int(round(duration / target_dt)) + 1
    target_time = np.arange(n_target, dtype=np.float64) * target_dt

    # Ensure interpolation remains inside source support.
    if target_time.size and target_time[-1] > duration:
        target_time[-1] = duration

    resampled = np.interp(target_time, source_time, trace.astype(np.float64, copy=False))
    return resampled.astype(np.float32, copy=False)


def iter_picks(picks_file: Path, survey: str, reduction_velocity: float):
    """
    Yields (pick_offset, picked_time_reduced, ray_type_str) for recognized picks.

    Picks parsing is tolerant:
    - uses col0=offset, col1=time, col3=phase (as in your existing code)
    - ignores lines with <4 columns
    """
    phase_map = PICK_TYPES.get(survey.lower(), {}).copy()

    # If file endswith phases.dat, swap meanings of RX and RF
    if picks_file.name.lower().endswith("phases.dat"):
        for k, v in list(phase_map.items()):
            if v == "RX":
                phase_map[k] = "RF"
            elif v == "RF":
                phase_map[k] = "RX"

    with picks_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            cols = line.strip().split()
            if len(cols) < 4:
                continue

            try:
                pick_offset = float(cols[0])
                pick_time_raw = float(cols[1])
                phase = int(float(cols[3]))
            except ValueError:
                continue

            ray_type = phase_map.get(phase)
            if ray_type not in ("Water", "RX", "RF"):
                continue

            picked_time_reduced = pick_time_raw - abs(pick_offset) / reduction_velocity
            yield pick_offset, picked_time_reduced, ray_type


def find_trace_index_for_offset(trace_offsets: np.ndarray, pick_offset: float):
    idx = int(np.argmin(np.abs(trace_offsets - pick_offset)))
    return idx, float(trace_offsets[idx])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a wide CSV database from SEG-Y files and their corresponding picks files. Each row corresponds to a single pick and includes the trace samples as columns."
    )
    parser.add_argument(
        "--data_directory",
        required=True,
        help="Path to data folder (e.g., ./data)"
    )
    parser.add_argument(
        "--output_directory",
        required=True,
        help="Path to output folder (e.g., ./database)",
    )
    parser.add_argument(
        "--reduction_velocity",
        type=float,
        default=8.0,
        help="Reduction velocity used for reduced time (default: 8.0)",
    )
    parser.add_argument(
        "--time_shift",
        type=float,
        default=0.01,
        help="Time shift between samples in seconds (default: 0.01s). Used to shift traces to a common time reference if needed.",
    )
    args = parser.parse_args()
    target_dt = float(args.time_shift)

    data_dir = Path(args.data_directory).resolve()
    output_path = Path(args.output_directory).resolve()
    if output_path.suffix.lower() == ".csv":
        out_csv = output_path
        out_dir = out_csv.parent
    else:
        out_dir = output_path
        out_csv = out_dir / "database.csv"

    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_segy_picks_pairs(data_dir)
    if not pairs:
        raise SystemExit(f"No SEG-Y + picks pairs found in {data_dir}")

    # Build a base database with a common sampling interval (target_dt).
    # If source SEG-Y files have different dt, all traces are resampled to target_dt.
    n_samples = 0
    source_dts = []
    for _, _, segy_file, _ in pairs:
        wiggles, _, time = read_segy(str(segy_file))
        source_dt = get_sampling_interval_from_time(time)
        source_dts.append(source_dt)

        duration = (int(wiggles.shape[1]) - 1) * source_dt
        resampled_samples = int(round(duration / target_dt)) + 1
        n_samples = max(n_samples, resampled_samples)

    unique_dts = sorted({round(dt, 9) for dt in source_dts})
    if len(unique_dts) == 1:
        print(f"[INFO] All SEG-Y files share the same sampling interval: dt={unique_dts[0]:.9f} s")
    else:
        print("[WARN] SEG-Y files have different sampling intervals. Resampling to a common target dt.")
        print(f"[WARN] Source dt values (s): {unique_dts}")

    print(f"[INFO] Target sampling interval: dt={target_dt:.9f} s")
    print(f"[INFO] Database sample columns: {n_samples}")

    header = ["REGION", "OBS", "OBS_DEPTH", "RAY_TYPE", "OFFSET"]
    header += [f"sample{i}" for i in range(n_samples)]
    header += ["PICKED_TIME"]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        rows_written = 0

        for region, obs, segy_file, picks_file in pairs:
            wiggles, offsets, time = read_segy(str(segy_file))
            source_dt = get_sampling_interval_from_time(time)

            for pick_offset, picked_time, ray_type in iter_picks(picks_file, region, args.reduction_velocity):
                trace_idx, matched_offset = find_trace_index_for_offset(offsets, pick_offset)

                trace = wiggles[trace_idx, :]
                trace_resampled = resample_trace_to_target_dt(trace, source_dt, target_dt)
                trace_norm = normalize_trace_to_unit_maxabs(trace_resampled)
                trace_fixed = pad_or_truncate_trace(trace_norm, n_samples)

                row = [region, obs, float("nan"), ray_type, pick_offset]
                row += trace_fixed.tolist()
                row += [picked_time]
                writer.writerow(row)
                rows_written += 1

    print(f"[DONE] Wrote {rows_written} rows to {out_csv}")