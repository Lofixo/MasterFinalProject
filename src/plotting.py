# Plotting functions for seismic sections, including computing plot limits and plotting seismic sections with picks and interpolated data

import matplotlib.pyplot as plt
import numpy as np

def compute_plot_limits(offsets, interpolated_refractions_offset, interpolated_refractions_time, interpolated_reflections_offset, interpolated_reflections_time):
    time_min = 0
    time_max = 25

    all_times = []

    if len(interpolated_refractions_time) > 0:
        all_times.append(np.max(interpolated_refractions_time))

    if len(interpolated_reflections_time) > 0:
        all_times.append(np.max(interpolated_reflections_time))

    if len(all_times) > 0:
        time_max = max(all_times) + 5

    offset_min = np.min(offsets)
    offset_max = np.max(offsets)

    all_pick_offsets = []

    if len(interpolated_refractions_offset) > 0:
        all_pick_offsets.append(np.min(interpolated_refractions_offset))
        all_pick_offsets.append(np.max(interpolated_refractions_offset))

    if len(interpolated_reflections_offset) > 0:
        all_pick_offsets.append(np.min(interpolated_reflections_offset))
        all_pick_offsets.append(np.max(interpolated_reflections_offset))

    if len(all_pick_offsets) > 0:
        pick_min = min(all_pick_offsets)
        pick_max = max(all_pick_offsets)

        offset_min = max(offset_min, pick_min - 2)
        offset_max = min(offset_max, pick_max + 2)

    return time_min, time_max, offset_min, offset_max


def plot_section(offsets, time, scaled_wiggles, output_file, title, RV, time_min, time_max, offset_min, offset_max, picks_data=None, interpolated_data=None):
    fig, ax = plt.subplots()

    # Plot wiggles
    for idx, offset in enumerate(offsets):
        if offset < offset_min or offset > offset_max:
            continue

        offset_line = np.zeros(time.size) + offset
        trace = scaled_wiggles[idx, :] + offset
        ax.fill_betweenx(y=time, x2=offset_line, x1=trace, where=(trace > offset_line), color="k", lw=0)

    # Plot picks if available
    if picks_data is not None:
        # Water picks
        ax.plot(picks_data["water_offset"], picks_data["water_time"], ".", color="magenta", markersize=3, label="Water (original)")
        ax.plot(interpolated_data["water_offset"], interpolated_data["water_time"], ".", color="red", markersize=2, label="Water (interpolated)")

        # Refraction picks
        ax.plot(picks_data["refractions_offset"], picks_data["refractions_time"], ".", color="cyan", markersize=3, label="Refractions (original)")
        ax.plot(interpolated_data["refractions_offset"], interpolated_data["refractions_time"], ".", color="blue", markersize=2, label="Refractions (interpolated)")

        # Reflection picks
        ax.plot(picks_data["reflections_offset"], picks_data["reflections_time"], ".", color="orange", markersize=3, label="Reflections (original)")
        ax.plot(interpolated_data["reflections_offset"], interpolated_data["reflections_time"], ".", color="yellow", markersize=2, label="Reflections (interpolated)")

        ax.legend(loc="lower right")

    plt.xlim((offset_min, offset_max))
    plt.ylim((time_min, time_max))
    ax.invert_yaxis()

    ax.set_title(title)
    ax.set_xlabel("Distance [km]")
    ax.set_ylabel(f"Time - Distance / {RV} [sec]")

    plt.savefig(output_file, dpi=1200)
    plt.close()
