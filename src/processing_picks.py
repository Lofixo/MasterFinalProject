# Picks processing functions, including reading picks from a file, sorting picks by offset, and interpolating picks for trace offsets

import numpy as np

def read_picks(picks_file, RV):
    valid_picks = []

    with open(picks_file, "r") as f:
        for line in f:
            cols = line.strip().split()
            if len(cols) == 5:
                valid_picks.append(cols)

    if len(valid_picks) == 0:
        return None

    picks_array = np.array(valid_picks, dtype=float)

    picks_offset = picks_array[:, 0]
    picks_time = picks_array[:, 1]
    pick_type = picks_array[:, 4]

    reduced_time = picks_time - abs(picks_offset) / RV

    ref_mask = pick_type == 0
    refl_mask = pick_type == 1

    return {
        "refractions_offset": picks_offset[ref_mask],
        "refractions_time": reduced_time[ref_mask],
        "reflections_offset": picks_offset[refl_mask],
        "reflections_time": reduced_time[refl_mask],
    }


def sort_picks(offset_array, time_array):
    idx = np.argsort(offset_array)
    offsets_sorted = offset_array[idx]
    times_sorted = time_array[idx]

    unique_offsets = []
    unique_times = []

    i = 0
    while i < len(offsets_sorted):
        current_offset = offsets_sorted[i]

        same_mask = np.isclose(offsets_sorted, current_offset, atol=1e-6)
        min_time = np.min(times_sorted[same_mask])

        unique_offsets.append(current_offset)
        unique_times.append(min_time)

        i += np.sum(same_mask)

    return np.array(unique_offsets), np.array(unique_times)


def interpolate_picks(offset_array, time_array, trace_offsets):
    interpolated_offsets = []
    interpolated_times = []

    for i in range(len(offset_array) - 1):
        o1, o2 = offset_array[i], offset_array[i + 1]
        t1, t2 = time_array[i], time_array[i + 1]

        if i < len(offset_array) - 2:
            mask = (trace_offsets >= o1) & (trace_offsets < o2)
        else:
            mask = (trace_offsets >= o1) & (trace_offsets <= o2)

        offset_between = trace_offsets[mask]

        time_between = t1 + (offset_between - o1) * (t2 - t1) / (o2 - o1)

        interpolated_offsets.extend(offset_between)
        interpolated_times.extend(time_between)

    return np.array(interpolated_offsets), np.array(interpolated_times)
