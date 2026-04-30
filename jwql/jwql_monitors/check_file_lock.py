#!/usr/bin/env python3
"""
check_file_lock.py - Validates a lock file and removes it if the owning process is no longer running.

Usage: python3 check_file_lock.py <lock_file_path>
"""

import argparse
import logging
import os
import re

from jwql.utils import monitor_utils


def parse_pid(lock_file: str) -> int | None:
    """Read and parse the PID from a lock file.
    Expects a line in the format: 'Process Id = <pid>'
    """
    try:
        with open(lock_file, "r") as f:
            content = f.read().strip()

        match = re.search(r"Process\s+Id\s*=\s*(\d+)", content, re.IGNORECASE)
        if not match:
            logging.warning(f"Could not parse PID from lock file: {lock_file!r} (content: {content!r})")
            return None

        return int(match.group(1))

    except OSError as e:
        logging.error(f"Failed to read lock file {lock_file!r}: {e}")
        return None


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is currently running.
    Uses os.kill(pid, 0) — sends no signal, but raises if the process doesn't exist.
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        # No process with this PID
        return False
    except PermissionError:
        # Process exists but we don't have permission to signal it
        return True


def process_lock_file(lock_file: str) -> None:
    if not os.path.isfile(lock_file):
        logging.warning(f"Lock file not found (may have already been removed): {lock_file!r}")
        return

    logging.info(f"Checking lock file: {lock_file!r}")

    pid = parse_pid(lock_file)
    if pid is None:
        logging.error(f"Skipping {lock_file!r} — unable to determine PID.")
        return

    logging.info(f"Lock file {lock_file!r} contains PID {pid}")

    if is_process_running(pid):
        logging.info(f"Process {pid} is still running. Lock file retained: {lock_file!r}")
    else:
        try:
            os.remove(lock_file)
            logging.info(f"Process {pid} is not running. Stale lock file deleted: {lock_file!r}")
        except OSError as e:
            logging.error(f"Failed to delete lock file {lock_file!r}: {e}")


def main() -> None:
    module = os.path.basename(__file__).strip('.py')
    start_time, log_file = monitor_utils.initialize_instrument_monitor(module)

    parser = argparse.ArgumentParser(
        description="Check a lock file and remove it if the owning process is no longer running."
    )
    parser.add_argument("lock_file", help="Path to the lock file to check")
    args = parser.parse_args()

    process_lock_file(args.lock_file)


if __name__ == "__main__":
    main()
