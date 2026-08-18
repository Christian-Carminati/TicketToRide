"""Headless replay inspector entrypoint."""

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or step through game replay")
    parser.add_argument("--replay-file", type=str, required=False, help="Path to replay json file")
    args = parser.parse_args()

    if not args.replay_file or not os.path.exists(args.replay_file):
        print("No replay file specified or file does not exist. (Phase 5 will stream replays to Web UI)")
        return

    with open(args.replay_file, "r", encoding="utf-8") as f:
        replay_data = json.load(f)
    print(f"Loaded replay: {len(replay_data.get('steps', []))} steps recorded.")


if __name__ == "__main__":
    main()
