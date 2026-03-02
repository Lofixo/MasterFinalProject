# SEGY file processing functions, including reading SEGY files and scaling wiggles for visualization

import numpy as np
import segyio
from config import QUANTILE_CLIP

def read_segy(filename):
    with segyio.open(filename, strict=False) as f:
        traces = f.trace
        wiggles = np.stack([traces[i] for i in range(traces.length)])
        offsets = f.attributes(segyio.tracefield.keys["offset"])[:] / 1000
        time_increments = segyio.tools.dt(f) / 1e6

    offset_size, time_size = wiggles.shape
    time = np.arange(time_size) * time_increments

    return wiggles, offsets, time


def scale_wiggles(wiggles, offsets):
    scaled = wiggles.copy()

    clip = np.nanquantile(wiggles[wiggles > 0], q=QUANTILE_CLIP)
    scaled[scaled > clip] = clip
    scaled[scaled < -clip] = -clip

    scaled = 1.2 * scaled * (offsets[1] - offsets[0]) / np.nanmax(scaled)

    return scaled