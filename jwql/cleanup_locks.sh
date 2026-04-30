#!/usr/bin/env bash
# cleanup_locks.sh - Find all .lock files in a directory tree and remove stale ones.
#
# Usage: ./cleanup_locks.sh [search_path]
#
# Defaults:
#   search_path : directory containing this script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_PATH="${1:-$SCRIPT_DIR}"
CHECK_SCRIPT="$SCRIPT_DIR/jwql_monitors/check_file_lock.py"

# ── Validation ────────────────────────────────────────────────────────────────

if [[ ! -f "$CHECK_SCRIPT" ]]; then
    echo "ERROR: check_file_lock.py not found at $CHECK_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$SEARCH_PATH" ]]; then
    echo "ERROR: Search path does not exist: $SEARCH_PATH" >&2
    exit 1
fi

# ── Main ──────────────────────────────────────────────────────────────────────

echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Starting lock file cleanup in: $SEARCH_PATH"

LOCK_COUNT=0

# -mtime +0 matches files last modified more than 24 hours ago.
# Remove that flag entirely if you want to check ALL lock files regardless of age.
while IFS= read -r -d '' lock_file; do
    python3 "$CHECK_SCRIPT" "$lock_file"
    (( LOCK_COUNT++ )) || true
done < <(find "$SEARCH_PATH" -type f -name "*.lock" -mtime +0 -print0)

if (( LOCK_COUNT == 0 )); then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] No lock files found (older than 24h)."
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] Processed $LOCK_COUNT lock file(s)."
fi
