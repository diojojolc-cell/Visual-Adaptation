"""Display the existing training status files without loading model weights."""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--interval', type=float, default=5)
    args = parser.parse_args()
    while True:
        print('\033[2J\033[H', end='')
        for path in sorted(args.output.glob('status_rank*.json')):
            status = json.loads(path.read_text())
            age = time.time() - status.get('heartbeat', 0)
            print(f"rank {status['rank']} | PID {status['pid']} | {status['device']} | "
                  f"{status['state']} / {status['stage']} | heartbeat {age:.0f}s ago")
            print(f"u{status.get('update', 0)}/{status.get('total', '?')} | "
                  f"loss {status.get('loss', 0):.4f} | LR {status.get('lr', 0):.3e} | "
                  f"batch {status.get('batch_seconds', 0):.1f}s | "
                  f"CPU wait {status.get('cpu_wait_seconds', 0):.1f}s | "
                  f"ETA {status.get('eta_seconds', 0)/3600:.2f}h")
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
