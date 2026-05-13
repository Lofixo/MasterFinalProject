# SEGY file processing functions, including reading SEGY files and scaling samples for visualization

import numpy as np
import segyio

def read_segy(filename):
    with segyio.open(filename, strict=False) as f:
        traces = f.trace
        samples = np.stack([traces[i] for i in range(traces.length)])
        offsets = f.attributes(segyio.tracefield.keys["offset"])[:] / 1000
        time_increments = segyio.tools.dt(f) / 1e6

    offset_size, time_size = samples.shape
    time = np.arange(time_size) * time_increments

    return samples, offsets, time


def scale_samples(samples, offsets):
    scaled = samples.copy()

    clip = np.nanquantile(samples[samples > 0], q=0.8)
    scaled[scaled > clip] = clip
    scaled[scaled < -clip] = -clip

    scaled = 1.2 * scaled * (offsets[1] - offsets[0]) / np.nanmax(scaled)

    return scaled
