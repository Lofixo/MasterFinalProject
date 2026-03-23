# Pick type mappings per survey
# Types: "Water", "RX" (refraction), "RF" (reflection), None (ignore)
PICK_TYPES = {
    "iberia": {
        -1: None,
        0: None,
        1: "Water",
        2: "RX",
        3: "RX",
        4: "RF",
        5: "RX",
        6: "RF",
        7: None,
        9: None,
        10: None,
    },
    # 9, 10, 11 are RF, but we won't be treating them as they were for testing purposes
    "tagus": {
        -1: None,
        0: None,
        1: "Water",
        2: "RX",
        3: "RX",
        4: "RF",
        5: None,
        6: None,
        7: "RX",
        9: None,
        10: None,
        11: None,
    },
    "tyrrhenian": {
        -1: None,
        0: None,
        1: "Water",
        2: "RX",
        3: "RF",
        4: "RX",
        5: "RX",
    },
}