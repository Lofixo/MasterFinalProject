import segyio
import numpy as np
import matplotlib.pyplot as plt
import getopt
import sys

QUANTILE_CLIP = 0.8
PHASE_COLORS = {
    1: "tab:blue",
    2: "lime",
    3: "forestgreen",
    4: "turquoise",
    5: "yellow",
    6: "tab:orange",
    7: "purple",
    8: "pink",
    9: "fuchsia", 
    10: "tab:red"
}

if __name__ == "__main__":
    ##################
    # region Arguments
    ##################

    filename = sys.argv[1]
    opts, _ = getopt.getopt(
        sys.argv[2:], 
        "", 
        ["picks=", "reduction_vel=", "offset_min=", "offset_max=", "time_min=", "time_max=", "output=", "title="]
    )
    opts = {u:v for u, v in opts}
    picks_file = opts.get("--picks", None)
    offset_min = float(opts["--offset_min"])
    offset_max = float(opts["--offset_max"])
    time_min = float(opts["--time_min"])
    time_max = float(opts["--time_max"])
    RV = float(opts.get("--reduction_vel", 1))
    output_file = str(opts["--output"])
    title = str(opts["--title"])

    # endregion

    ########################
    # region Data processing
    ########################

    # Read trace file
    with segyio.open(filename, strict=False) as f:
        traces = f.trace
        wiggles = np.stack([traces[i] for i in range(traces.length)])
        offsets = f.attributes(segyio.tracefield.keys["offset"])[:] / 1000  # From m to Km
        time_increments = segyio.tools.dt(f) / 1e6                          # From microseconds to seconds 

    offset_size, time_size = wiggles.shape
    time = np.arange(time_size) * time_increments

    scaled_wiggles = wiggles.copy()
    clip = np.nanquantile(wiggles[wiggles>0], q=QUANTILE_CLIP)
    scaled_wiggles[scaled_wiggles > clip] = clip
    scaled_wiggles[scaled_wiggles < -clip] = -clip
    scaled_wiggles = 1.2 * scaled_wiggles * (offsets[1] - offsets[0]) / np.nanmax(scaled_wiggles)

    # Read picks file if provided
    if picks_file is not None:
        valid_picks = []
        with open(picks_file, "r") as f:
            for line in f:
                cols = line.strip().split()
                if len(cols) == 5:
                    valid_picks.append(cols)

        picks_array = np.array(valid_picks, dtype=float)
        picks_offset = picks_array[:, 0]    # Offset in km
        picks_time = picks_array[:, 1]      # Real time in s
        pick_type = picks_array[:, 4]       # Pick type (0 for refraction, 1 for reflection)

        # Convert real time to reduced time using absolute offset
        picks_reduced_time = picks_time - abs(picks_offset) / RV

        # Separate refraction and reflection picks
        refraction_mask = pick_type == 0
        refractions_offset = picks_offset[refraction_mask]
        refractions_reduced_time = picks_reduced_time[refraction_mask]

        reflection_mask = pick_type == 1
        reflections_offset = picks_offset[reflection_mask]
        reflections_reduced_time = picks_reduced_time[reflection_mask]

        # Sort picks by offset for interpolation and keep minimum reduced time for duplicate offsets
        def sort_picks(offset_array, time_array):
            # Sort by offset first
            idx = np.argsort(offset_array)
            offsets_sorted = offset_array[idx]
            times_sorted = time_array[idx]

            # TODO: Should this be done?
            # Find unique offsets
            unique_offsets = []
            unique_times = []
            i = 0

            while i < len(offsets_sorted):
                current_offset = offsets_sorted[i]

                # Find all entries with this offset and keep minimum reduced time
                same_mask = np.isclose(offsets_sorted, current_offset, atol=1e-6)
                same_times = times_sorted[same_mask]
                min_time = np.min(same_times)

                unique_offsets.append(current_offset)
                unique_times.append(min_time)
                i += np.sum(same_mask)

            return np.array(unique_offsets), np.array(unique_times)
        
        sorted_refractions_offset, sorted_refractions_reduced_time = sort_picks(refractions_offset, refractions_reduced_time)
        sorted_reflections_offset, sorted_reflections_reduced_time = sort_picks(reflections_offset, reflections_reduced_time)

        # Interpolate picks to get reduced time for all traces
        def interpolate_picks(offset_array, time_array):
            interpolated_offsets = []
            interpolated_times = []

            for i in range(len(offset_array) - 1):
                initial_offset, final_offset = offset_array[i], offset_array[i+1]
                initial_time, final_time = time_array[i], time_array[i+1]

                # Traces contained between the two picks
                if i < len(offset_array) - 2:
                    mask = (offsets >= initial_offset) & (offsets < final_offset)
                else:
                    mask = (offsets >= initial_offset) & (offsets <= final_offset)

                # Linear interpolation of reduced time for traces between the two picks
                offset_between = offsets[mask]
                time_between = initial_time + (offset_between - initial_offset) * (final_time - initial_time) / (final_offset - initial_offset)

                interpolated_offsets.extend(offset_between)
                interpolated_times.extend(time_between)

            return np.array(interpolated_offsets), np.array(interpolated_times)
        
        interpolated_refractions_offset, interpolated_refractions_reduced_time = interpolate_picks(sorted_refractions_offset, sorted_refractions_reduced_time)
        interpolated_reflections_offset, interpolated_reflections_reduced_time = interpolate_picks(sorted_reflections_offset, sorted_reflections_reduced_time)

    # endregion

    #################
    # region Plotting
    #################

    # Plot wiggles
    fig, ax = plt.subplots()
    for idx, offset in enumerate(offsets):
        if offset < offset_min or offset > offset_max:
            continue
        offset_line = np.zeros(time.size) + offset
        trace = scaled_wiggles[idx, :] + offset
        ax.fill_betweenx(y=time, x2=offset_line, x1=trace, where=(trace > offset_line), color="k", lw=0)

    # Plot picks if provided
    if picks_file is not None:
        # --- ORIGINAL PICKS ---
        ax.plot(refractions_offset, refractions_reduced_time, ".", color="red", markersize=3, label="Refraction picks (original)")
        ax.plot(reflections_offset, reflections_reduced_time, ".", color="orange", markersize=3, label="Reflection picks (original)")

        # --- INTERPOLATED POINTS ---
        ax.plot(interpolated_refractions_offset, interpolated_refractions_reduced_time, ".", color="blue", markersize=2, label="Refraction (interpolated)")
        ax.plot(interpolated_reflections_offset, interpolated_reflections_reduced_time, ".", color="green", markersize=2, label="Reflection (interpolated)")
        
        ax.legend(loc="lower right")

    # Set axes limits and invert y-axis
    plt.xlim((offset_min, offset_max))
    plt.ylim((time_min, time_max))
    ax.invert_yaxis()

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Distance [km]")
    ax.set_ylabel(f"Time - Distance / {RV} [sec]")

    # Save figure
    plt.savefig(output_file, dpi=1200)
    plt.clf()

    # endregion
