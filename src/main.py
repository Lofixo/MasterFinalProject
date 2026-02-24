# Main script for processing seismic data from SEG-Y files, reading picks, and plotting seismic sections with optional picks and interpolated data.

import sys
import getopt
from processing_segy import read_segy, scale_wiggles
from processing_picks import read_picks, sort_picks, interpolate_picks
from plotting import compute_plot_limits, plot_section


if __name__ == "__main__":

    filename = sys.argv[1]

    opts, _ = getopt.getopt(
        sys.argv[2:],
        "",
        ["picks=", "reduction_vel=", "output=", "title="]
    )

    opts = {u: v for u, v in opts}

    picks_file = opts.get("--picks", None)
    RV = float(opts.get("--reduction_vel", 1))
    output_file = str(opts["--output"])
    title = str(opts["--title"])

    wiggles, offsets, time = read_segy(filename)
    scaled_wiggles = scale_wiggles(wiggles, offsets)

    interpolated_data = {
        "refractions_offset": [],
        "refractions_time": [],
        "reflections_offset": [],
        "reflections_time": [],
    }

    picks_data = None

    if picks_file is not None:
        picks_data = read_picks(picks_file, RV)

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
        output_file,
        title,
        RV,
        time_min,
        time_max,
        offset_min,
        offset_max,
        picks_data,
        interpolated_data
    )