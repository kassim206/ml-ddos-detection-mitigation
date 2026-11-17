#!/usr/bin/env python3

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.exists():
        raise RuntimeError("Refusing to overwrite mitigation state")

    state = {
        "active": False,
        "activated_at_utc": None,
        "attack_probability": None,
        "window_index": None,
        "initialized_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    temporary = args.output.with_suffix(".tmp")

    with temporary.open("x", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
