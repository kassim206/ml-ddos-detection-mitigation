#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EXP010="runs/experiment_010_realtime_detection_mitigation"
BASE="$EXP010/deployment_model"
ACQ="$BASE/acquisition"
PLAN="$ACQ/dataset_b_run_plan.json"
LOGBASE="testbed/logs/experiment_010"

LOGICAL_RUN_ID="${1:-}"

if [ -z "$LOGICAL_RUN_ID" ]; then
    echo "Usage: $0 <logical_run_id>" >&2
    echo "Example: $0 B01" >&2
    exit 2
fi

# Resolve class and partition exclusively from frozen run plan.
resolved="$(
python - "$PLAN" "$LOGICAL_RUN_ID" <<'PY'
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
wanted = sys.argv[2]

plan = json.loads(plan_path.read_text())
matches = []

def walk(obj):
    if isinstance(obj, dict):
        if obj.get("logical_run_id") == wanted:
            matches.append(obj)
        for value in obj.values():
            walk(value)
    elif isinstance(obj, list):
        for value in obj:
            walk(value)

walk(plan)

if len(matches) != 1:
    print(
        f"ERROR: expected exactly one run-plan entry for "
        f"{wanted}; found {len(matches)}",
        file=sys.stderr,
    )
    raise SystemExit(1)

entry = matches[0]

klass = entry.get("class")
partition = entry.get("partition")

if klass not in {"Benign", "Attack"}:
    print(f"ERROR: invalid class in run plan: {klass!r}", file=sys.stderr)
    raise SystemExit(1)

if partition not in {"train", "validation", "test"}:
    print(
        f"ERROR: invalid partition in run plan: {partition!r}",
        file=sys.stderr,
    )
    raise SystemExit(1)

kind = "benign" if klass == "Benign" else "attack"

print(f"{kind}|{klass}|{partition}")
PY
)"

IFS='|' read -r RUN_KIND DATASET_CLASS PARTITION <<< "$resolved"

# Refuse accidental recollection of an already initialized Dataset-B logical ID.
if grep -Rqs \
    "^LOGICAL_RUN_ID=${LOGICAL_RUN_ID}$" \
    "$ACQ/runs"/*/run_context.env 2>/dev/null
then
    echo "ERROR: logical run already initialized: $LOGICAL_RUN_ID" >&2
    exit 1
fi

# Staging must be clean before creating a run.
existing="$(
    find "$LOGBASE" \
        -type f \
        -not -name '.gitkeep' \
        -print 2>/dev/null || true
)"

if [ -n "$existing" ]; then
    echo "ERROR: staging is not empty:" >&2
    printf '%s\n' "$existing" >&2
    exit 1
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="exp010_dataset_b_${LOGICAL_RUN_ID}_${RUN_KIND}_${timestamp}"
RUN_DIR="$ACQ/runs/$RUN_ID"

if [ -e "$RUN_DIR" ]; then
    echo "ERROR: run directory already exists: $RUN_DIR" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"

printf '%s\n' "$RUN_ID" > "$RUN_DIR/run_id.txt"
printf '%s\n' "$LOGICAL_RUN_ID" > "$RUN_DIR/logical_run_id.txt"

cat > "$RUN_DIR/run_context.env" <<EOF
EXP010_RUN_ID=$RUN_ID
RUN_ID=$RUN_ID
LOGICAL_RUN_ID=$LOGICAL_RUN_ID
RUN_KIND=$RUN_KIND
DATASET_CLASS=$DATASET_CLASS
PARTITION=$PARTITION
RUN_DIR=$RUN_DIR
LOGBASE=$LOGBASE
EOF

cat > "$RUN_DIR/run_status.json" <<EOF
{
  "status": "INITIALIZED",
  "dataset": "Dataset B",
  "logical_run_id": "$LOGICAL_RUN_ID",
  "physical_run_id": "$RUN_ID",
  "class": "$DATASET_CLASS",
  "partition": "$PARTITION",
  "valid_dataset_run": false,
  "include_in_dataset_b": false,
  "traffic_workload_executed": false,
  "feature_extraction_permitted": false,
  "training_use_permitted": false
}
EOF

printf '%s\n' "$RUN_DIR" > "$ACQ/CURRENT_RUN"

echo "PASS: Dataset-B acquisition run initialized"
echo "LOGICAL_RUN_ID=$LOGICAL_RUN_ID"
echo "RUN_ID=$RUN_ID"
echo "RUN_KIND=$RUN_KIND"
echo "DATASET_CLASS=$DATASET_CLASS"
echo "PARTITION=$PARTITION"
echo "RUN_DIR=$RUN_DIR"
echo
echo "No workload has been executed."
echo "Restore with:"
echo "  source scripts/10_load_acquisition_run.sh"
