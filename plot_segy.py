import segyio
import numpy as np
import matplotlib.pyplot as plt
import getopt
import sys

QUANTILE_CLIP = 0.8
PHASE_COLORS = {
    1: "tab:blue",
    2: "lime",     #RX OBS for the TOB
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
    # Read command line arguments
    filename = sys.argv[1]
    opts, _ = getopt.getopt(
        sys.argv[2:], 
        "", 
        ["picks=", "reduction_vel=", "xmin=", "xmax=", "tmin=", "tmax=", "output=", "title="]
    )
    opts = {u:v for u, v in opts}
    picks_file = opts.get("--picks", None)
    xmin = float(opts["--xmin"])
    xmax = float(opts["--xmax"])
    tmin = float(opts["--tmin"])
    tmax = float(opts["--tmax"])
    RV = float(opts.get("--reduction_vel", 1))
    output_file = str(opts["--output"])
    title = str(opts["--title"])

    # Read .segy file
    with segyio.open(filename, strict=False) as f:
        traces = f.trace
        wiggles = np.stack([traces[i] for i in range(traces.length)])
        x = f.attributes(segyio.tracefield.keys["offset"])[:] / 1000 # From m to Km
        increment_dt = segyio.tools.dt(f) / 1e6 # From microseconds to seconds 

    x_size, t_size = wiggles.shape
    time = np.arange(t_size) * increment_dt

    scaled_wiggles = wiggles.copy()
    clip = np.nanquantile(wiggles[wiggles>0], q=QUANTILE_CLIP)
    scaled_wiggles[scaled_wiggles > clip] = clip
    scaled_wiggles[scaled_wiggles < -clip] = -clip
    scaled_wiggles = 1.2 * scaled_wiggles * (x[1] - x[0]) / np.nanmax(scaled_wiggles)

    # Read picks file if provided
    if picks_file is not None:
        picks = np.loadtxt(picks_file, usecols=(0, 1))
        x_picks = picks[:, 0] # Offset in km
        t_real = picks[:, 1] # Real time in s

        # Convert real time to reduced time
        t_reduced = t_real - abs(x_picks) / RV

    # Read picks file if provided
    if picks_file is not None:
        valid_lines = []
        with open(picks_file, "r") as f:
            for line in f:
                cols = line.strip().split()
                if len(cols) == 5:
                    valid_lines.append(cols)

        picks_array = np.array(valid_lines, dtype=float)

        x_picks = picks_array[:, 0]  # Offset in km
        t_real = picks_array[:, 1]   # Real time in s

        # Convert real time to reduced time using absolute offset
        t_reduced = t_real - abs(x_picks) / RV


    # Plot wiggles
    nrows = 1
    fig, ax = plt.subplots(nrows=nrows, sharex=True, sharey=True)
    axes = [ax]
    for ax in axes:
        for i, offset in enumerate(x):
            if offset < xmin or offset > xmax:
                continue
            offset_line = np.zeros(time.size) + offset
            trace = scaled_wiggles[i, :] + offset
            ax.fill_betweenx(y=time, x2=offset_line, x1=trace, where=(trace > offset_line), color="k", lw=0)
        ax.set_ylabel(f"Time - Dist. / {RV} [sec]")
    axes[-1].set_xlabel(f"Distance [km]")
    axes[0].set_title(title)

    # Plot picks if provided
    if picks_file is not None:
        ax.plot(x_picks, t_reduced, ".", markersize=3, color="red", label="Picks")

    plt.ylim((tmin, tmax))
    ax.invert_yaxis()
    plt.xlim((xmin, xmax))
    plt.tight_layout()
    plt.savefig(output_file, dpi=1200)
    plt.clf()
