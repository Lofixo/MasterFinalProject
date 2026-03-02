# Surveys where SEGY data is in real time. For these, the reduction velocity will be applied to the wiggle time axis too.
REAL_TIME_SURVEYS = {"tagus"}

# Quantile clipping value for outlier removal in travel time residuals
QUANTILE_CLIP = 0.8

# Pick type mappings per survey
# Types: "Water", "RX" (reflection), "RF" (refraction), None (ignore)
PICK_TYPES = {
    "tagus": {
        -1: None,
        0:  None,
        1:  "Water",
        2:  "RX",
        3:  "RX",
        4:  "RF",
        5:  None,
        6:  None,
        7:  "RX",
        9:  "RF",
        10: "RF",
        11: "RF",
    },
    "iberia": {
        -1: None,
        0:  None,
        1:  "Water",
        2:  "RX",
        3:  "RX",
        4:  "RF",
        5:  "RX",
        6:  "RF",
        7:  None,
        9:  None,
        10: None,
    },
    "tyrrhenian": {
        -1: None,
        0:  None,
        1:  "Water",
        2:  "RX",
        3:  "RF",
        4:  "RX",
        5:  "RX",
    },
}

# Interpolation threshold, if the distance between two consecutive picks exceeds this multiplier times the average inter-pick spacing, skip that gap
INTERPOLATION_THRESHOLD_MULTIPLIER = 5
