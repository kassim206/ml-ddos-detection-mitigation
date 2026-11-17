#!/usr/bin/env bash

_repo_root="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." &&
    pwd
)"

_current_file="$_repo_root/runs/experiment_010_realtime_detection_mitigation/deployment_model/acquisition/CURRENT_RUN"

if [ ! -s "$_current_file" ]; then
    echo "ERROR: no CURRENT_RUN exists" >&2
    return 1
fi

_run_dir_rel="$(cat "$_current_file")"
_run_dir="$_repo_root/$_run_dir_rel"

if [ ! -s "$_run_dir/run_context.env" ]; then
    echo "ERROR: run context missing: $_run_dir/run_context.env" >&2
    return 1
fi

set -a
# shellcheck disable=SC1090
source "$_run_dir/run_context.env"
set +a

# Keep repository-relative RUN_DIR for commands executed
# from the repository root.
cd "$_repo_root"

export EXP010_RUN_ID
export RUN_ID
export LOGICAL_RUN_ID
export RUN_KIND
export DATASET_CLASS
export PARTITION
export RUN_DIR
export LOGBASE

echo "PASS: acquisition context loaded"
echo "EXP010_RUN_ID=$EXP010_RUN_ID"
echo "RUN_ID=$RUN_ID"
echo "LOGICAL_RUN_ID=$LOGICAL_RUN_ID"
echo "RUN_KIND=$RUN_KIND"
echo "DATASET_CLASS=$DATASET_CLASS"
echo "PARTITION=$PARTITION"
echo "RUN_DIR=$RUN_DIR"
echo "LOGBASE=$LOGBASE"
