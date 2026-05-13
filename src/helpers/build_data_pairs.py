import re
from pathlib import Path

def leading_number(path: Path):
    m = re.match(r"^(\d+)", path.name)
    return m.group(1) if m else None


def build_segy_picks_pairs(data_directory: Path):
    pairs = []
    for survey_dir in sorted([p for p in data_directory.iterdir() if p.is_dir()]):
        segy_dir = survey_dir / "segy"
        picks_dir = survey_dir / "picks"
        if not segy_dir.exists() or not picks_dir.exists():
            continue

        segy_files = [p for p in segy_dir.iterdir() if p.is_file()]
        picks_files = [p for p in picks_dir.iterdir() if p.is_file()]

        # Keep your existing special-case rules
        if survey_dir.name.lower() == "gulf_of_lions":
            segy_files = [p for p in segy_files if p.name.endswith(".chan1.segy")]
        if survey_dir.name.lower() == "iberia":
            picks_files = [p for p in picks_files if not p.name.endswith(".HYD.tx.in")]

        segy_map = {}
        for f in segy_files:
            k = leading_number(f)
            if k is not None:
                segy_map.setdefault(k, []).append(f)

        picks_map = {}
        for f in picks_files:
            k = leading_number(f)
            if k is not None:
                picks_map.setdefault(k, []).append(f)

        for k in sorted(set(segy_map.keys()) & set(picks_map.keys()), key=int):
            for segy_f in sorted(segy_map[k]):
                for picks_f in sorted(picks_map[k]):
                    pairs.append((survey_dir.name, k, segy_f, picks_f))

    return pairs
