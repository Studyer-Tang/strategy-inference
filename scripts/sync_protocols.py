"""Sync protocols used by the installed CLI, excluding repository-only pilots."""

import argparse
from pathlib import Path

PACKAGED_PROTOCOLS = (
    "protocol.json",
    "calibration-protocol.json",
    "tail-diagnostic-protocol.json",
    "parametric-replay-protocol.json",
    "parameter-uncertainty-protocol.json",
    "joint-uncertainty-protocol.json",
    "multistep-protocol.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / "src" / "strategy_inference" / "protocols"
    for name in PACKAGED_PROTOCOLS:
        source = root / "experiments" / name
        destination = target / source.name
        if args.check:
            if not destination.is_file() or destination.read_bytes() != source.read_bytes():
                raise SystemExit(f"Protocol resource differs: {source.name}")
        else:
            target.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    print("Protocol resources match." if args.check else "Protocol resources synchronized.")


if __name__ == "__main__":
    main()
