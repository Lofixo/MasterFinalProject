# Main script for processing seismic data from SEG-Y files and picks.
# Batch mode: given a data root, generate one plot per SEG-Y + picks pair
# where both filenames start with the same number.

import argparse
import numpy as np
import re
import shutil
from config import REAL_TIME_SURVEYS
from pathlib import Path
from plotting import compute_plot_limits, plot_section
from processing_picks import read_picks, sort_picks, interpolate_picks
from processing_segy import read_segy, scale_samples


def leading_number(path: Path):
    matches = re.match(r"^(\d+)", path.name)
    return matches.group(1) if matches else None


def build_segy_picks_pairs(data_directory: Path):
    pairs = []

    for survey_directory in sorted([path for path in data_directory.iterdir() if path.is_dir()]):
        segy_directory = survey_directory / "segy"
        picks_directory = survey_directory / "picks"

        if not segy_directory.exists() or not picks_directory.exists():
            continue

        segy_files = [path for path in segy_directory.iterdir() if path.is_file()]
        picks_files = [path for path in picks_directory.iterdir() if path.is_file()]

        # Gulf of Lions file conditions
        if survey_directory.name.lower() == "gulf_of_lions":
            segy_files = [p for p in segy_files if p.name.endswith(".chan1.segy")]

        # Iberia file conditions
        if survey_directory.name.lower() == "iberia":
            picks_files = [p for p in picks_files if not p.name.endswith(".HYD.tx.in")]

        segy_map = {}
        for segy_file in segy_files:
            key = leading_number(segy_file)
            if key is not None:
                segy_map.setdefault(key, []).append(segy_file)

        picks_map = {}
        for picks_file in picks_files:
            key = leading_number(picks_file)
            if key is not None:
                picks_map.setdefault(key, []).append(picks_file)

        for key in sorted(set(segy_map.keys()) & set(picks_map.keys()), key=int):
            for segy_file in sorted(segy_map[key]):
                for picks_file in sorted(picks_map[key]):
                    pairs.append((survey_directory.name, key, segy_file, picks_file))

    return pairs


def process_pair(segy_file: Path, picks_file: Path, output_file: Path, title: str, reduction_velocity: float, survey: str):
    samples, offsets, time = read_segy(str(segy_file))

    # For real-time surveys, apply reduction velocity to the sample time axis.
    # We shift each trace's time origin by subtracting |offset|/RV, which is
    # equivalent to rolling each trace up by that amount of samples.
    if survey.lower() in REAL_TIME_SURVEYS:
        time_difference = time[1] - time[0]
        shifts = (np.abs(offsets) / reduction_velocity / time_difference).astype(int)
        shifted_samples = np.zeros_like(samples)
        for i, shift in enumerate(shifts):
            if shift < samples.shape[1]:
                shifted_samples[i, : samples.shape[1] - shift] = samples[i, shift:]
        samples = shifted_samples

    scaled_samples = scale_samples(samples, offsets)

    picks_data = read_picks(str(picks_file), reduction_velocity, survey)
    if picks_data is None:
        print(f"[INFO]: No valid picks in file: {picks_file}")
        return False

    refraction_offsets, refractions_times = sort_picks(picks_data["refractions_offset"], picks_data["refractions_time"])
    reflection_offsets, reflections_times = sort_picks(picks_data["reflections_offset"], picks_data["reflections_time"])
    water_offsets, water_times = sort_picks(picks_data["water_offset"], picks_data["water_time"])

    interpolated_refraction_offsets, interpolated_refraction_times = interpolate_picks(refraction_offsets, refractions_times, offsets)
    interpolated_reflection_offsets, interpolated_reflection_times = interpolate_picks(reflection_offsets, reflections_times, offsets)
    interpolated_water_offsets, interpolated_water_times = interpolate_picks(water_offsets, water_times, offsets)

    interpolated_data = {
        "refractions_offset": interpolated_refraction_offsets,
        "refractions_time": interpolated_refraction_times,
        "reflections_offset": interpolated_reflection_offsets,
        "reflections_time": interpolated_reflection_times,
        "water_offset": interpolated_water_offsets,
        "water_time": interpolated_water_times,
    }

    time_min, time_max, offset_min, offset_max = compute_plot_limits(
        offsets,
        interpolated_data["refractions_offset"],
        interpolated_data["refractions_time"],
        interpolated_data["reflections_offset"],
        interpolated_data["reflections_time"],
    )

    plot_section(offsets, time, scaled_samples, str(output_file), title, reduction_velocity, time_min, time_max, offset_min, offset_max, picks_data, interpolated_data)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate seismic plots for SEG-Y + picks pairs matched by leading filename number."
    )
    parser.add_argument(
        "--data_directory",
        required=True,
        help="Path to data folder (e.g., ./data)",
    )
    parser.add_argument(
        "--output_directory",
        required=True,
        help="Path to output folder (e.g., ./outputs)",
    )
    parser.add_argument(
        "--reduction_velocity",
        type=float,
        default=8.0,
        help="Reduction velocity used for reduced time (default: 8.0)",
    )

    # Argument parsing and setup
    args = parser.parse_args()
    data_directory = Path(args.data_directory).resolve()
    output_directory = Path(args.output_directory).resolve()
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Build list of SEGY + picks pairs to process
    pairs = build_segy_picks_pairs(data_directory)
    if not pairs:
        print(f"[INFO]: No matching SEG-Y + picks pairs found in: {data_directory}")
        raise SystemExit(0)

    # Process each pair and generate plots
    plots_generated = 0
    for survey, key, segy_file, picks_file in pairs:
        survey_out = output_directory / survey
        survey_out.mkdir(parents=True, exist_ok=True)
        out_name = f"{segy_file.stem}__{picks_file.stem}.png"
        output_file = survey_out / out_name

        print(f"[RUN ] {segy_file.name} + {picks_file.name} -> {output_file}")
        title = f"{survey} | {key}".strip()
        if process_pair(segy_file, picks_file, output_file, title, args.reduction_velocity, survey):
            plots_generated += 1

    print(f"[DONE] Generated {plots_generated}/{len(pairs)} plot(s) in: {output_directory}")