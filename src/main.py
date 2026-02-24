# Main script for processing seismic data from SEG-Y files and picks.
# Batch mode: given a data root, generate one plot per SEG-Y + picks pair
# where both filenames start with the same number.

import argparse
import re
from pathlib import Path

from processing_segy import read_segy, scale_wiggles
from processing_picks import read_picks, sort_picks, interpolate_picks
from plotting import compute_plot_limits, plot_section


def leading_number(path: Path):
    m = re.match(r"^(\d+)", path.name)
    return m.group(1) if m else None


def build_pairs(data_root: Path):
    """
    Expected layout:
      data_root/<survey>/segy/*
      data_root/<survey>/picks/*
    """
    pairs = []

    for survey_dir in sorted([d for d in data_root.iterdir() if d.is_dir()]):
        segy_dir = survey_dir / "segy"
        picks_dir = survey_dir / "picks"

        if not segy_dir.exists() or not picks_dir.exists():
            continue

        segy_files = [p for p in segy_dir.iterdir() if p.is_file()]
        picks_files = [p for p in picks_dir.iterdir() if p.is_file()]

        segy_map = {}
        for s in segy_files:
            key = leading_number(s)
            if key is not None:
                segy_map.setdefault(key, []).append(s)

        picks_map = {}
        for p in picks_files:
            key = leading_number(p)
            if key is not None:
                picks_map.setdefault(key, []).append(p)

        for key in sorted(set(segy_map.keys()) & set(picks_map.keys()), key=int):
            for segy_file in sorted(segy_map[key]):
                for picks_file in sorted(picks_map[key]):
                    pairs.append((survey_dir.name, key, segy_file, picks_file))

    return pairs


def process_pair(segy_file: Path, picks_file: Path, output_file: Path, title: str, rv: float):
    wiggles, offsets, time = read_segy(str(segy_file))
    scaled_wiggles = scale_wiggles(wiggles, offsets)

    picks_data = read_picks(str(picks_file), rv)
    if picks_data is None:
        print(f"[SKIP] No valid picks in file: {picks_file}")
        return False

    ro, rt = sort_picks(
        picks_data["refractions_offset"],
        picks_data["refractions_time"]
    )
    rlo, rlt = sort_picks(
        picks_data["reflections_offset"],
        picks_data["reflections_time"]
    )

    iro, irt = interpolate_picks(ro, rt, offsets)
    irlo, irlt = interpolate_picks(rlo, rlt, offsets)

    interpolated_data = {
        "refractions_offset": iro,
        "refractions_time": irt,
        "reflections_offset": irlo,
        "reflections_time": irlt,
    }

    time_min, time_max, offset_min, offset_max = compute_plot_limits(
        offsets,
        interpolated_data["refractions_offset"],
        interpolated_data["refractions_time"],
        interpolated_data["reflections_offset"],
        interpolated_data["reflections_time"],
    )

    plot_section(
        offsets,
        time,
        scaled_wiggles,
        str(output_file),
        title,
        rv,
        time_min,
        time_max,
        offset_min,
        offset_max,
        picks_data,
        interpolated_data,
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate seismic plots for SEG-Y + picks pairs matched by leading filename number."
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to data folder (e.g., ./data)",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Path to output folder (default: ./outputs)",
    )
    parser.add_argument(
        "--reduction_vel",
        type=float,
        default=1.0,
        help="Reduction velocity used for reduced time (default: 1.0)",
    )
    parser.add_argument(
        "--title_prefix",
        default="",
        help="Optional prefix added to every plot title",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(data_root)

    if not pairs:
        print(f"[INFO] No matching SEG-Y + picks pairs found in: {data_root}")
        raise SystemExit(0)

    ok = 0
    for survey, key, segy_file, picks_file in pairs:
        # Keep outputs organized per survey
        survey_out = output_dir / survey
        survey_out.mkdir(parents=True, exist_ok=True)

        out_name = f"{key}__{segy_file.stem}__{picks_file.stem}.png"
        output_file = survey_out / out_name

        title = f"{args.title_prefix} {survey} | {key}".strip()

        print(f"[RUN ] {segy_file.name} + {picks_file.name} -> {output_file}")
        if process_pair(segy_file, picks_file, output_file, title, args.reduction_vel):
            ok += 1

    print(f"[DONE] Generated {ok}/{len(pairs)} plot(s) in: {output_dir}")