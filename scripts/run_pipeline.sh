#!/usr/bin/env bash
# =============================================================================
#  run_pipeline.sh  —  Suite-level LIBERO data collection & conversion pipeline
#
#  Processes ALL tasks in a chosen suite, one by one, and outputs a single
#  LeRobot v2.1 dataset that covers the entire suite.
#
#  Resume support
#  ──────────────
#  Progress is recorded in:
#      <collected_dir>/<suite_name>/.pipeline_state
#  Each line: "<task_name>=done"
#  On re-run the script automatically skips completed tasks.
#  If an intermediate demo.hdf5 already has enough demos it is reused
#  instead of re-collecting.
#
#  Steps per task:
#    1. Collect human demonstrations  (collect_demonstration.py)
#    2. Replay & render to HDF5       (create_dataset.py)
#  Final step (once, after all tasks):
#    3. Convert whole suite to LeRobot v2.1  (convert_libero_to_lerobot.py)
#
#  Usage:
#    bash scripts/run_pipeline.sh --suite spatial [options]
#
#  Run  bash scripts/run_pipeline.sh --help  for the full option list.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BDDL_ROOT="${REPO_ROOT}/libero/libero/bddl_files"

# ── Defaults ──────────────────────────────────────────────────────────────────
SUITE=""
DEVICE="uarm"
COLLECTED_DIR="${REPO_ROOT}/collected_demos"
NUM_DEMOS=3
ROBOTS="Panda"
LEROBOT_OUT_ROOT="${REPO_ROOT}/datasets/lerobot"
CONDA_ENV="libero"
UARM_PORT="/dev/ttyUSB0"
UARM_OUTPUT_MAX="1.0"
TASK_DEMOS_FILE="${SCRIPT_DIR}/task_demos.jsonl"

# ── Suite name resolver ────────────────────────────────────────────────────────
resolve_suite() {
    case "$1" in
        spatial|libero_spatial)  echo "libero_spatial"  ;;
        goal|libero_goal)        echo "libero_goal"     ;;
        object|libero_object)    echo "libero_object"   ;;
        10|libero_10)            echo "libero_10"       ;;
        90|libero_90)            echo "libero_90"       ;;
        *) echo "" ;;
    esac
}

# ── Help ───────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: bash scripts/run_pipeline.sh --suite <name> [options]

Suite selection (required):
  --suite <spatial|goal|object|10|90>
                              The LIBERO task suite to process.
                              All .bddl tasks in that suite will be collected.

Collection options:
  --num-demonstrations <N>    Default demonstrations per task when not specified in
                              task_demos.jsonl (default: ${NUM_DEMOS}).
  --device <keyboard|spacemouse|uarm>
                              Teleoperation device (default: ${DEVICE}).
                              'uarm' reads from a physical U-ARM servo arm via
                              serial port; JOINT_POSITION controller is used.
  --uarm-port <path>          Serial port for the U-ARM device
                              (default: ${UARM_PORT}, only used when --device uarm).
  --uarm-output-max <float>   JOINT_POSITION output_max in rad/step (default: ${UARM_OUTPUT_MAX}).
                              Hard ceiling per control step; increase for faster arm motion.
                              E.g. 0.2 → ~11.5 deg/step max.
  --task-demos-file <path>    JSONL file with per-task demo counts
                              (default: scripts/task_demos.jsonl).
                              Each line: {"suite":"...","task":"...","num_demos":N}
  --directory <path>          Root dir for collected intermediate demos
                              (default: ./collected_demos).
  --robots <Panda|...>        Robot type for robosuite (default: ${ROBOTS}).

Output options:
  --lerobot-output <path>     Root dir for LeRobot datasets
                              (default: ./datasets/lerobot).

Environment:
  --conda-env <name>          Conda environment name (default: ${CONDA_ENV}).
                              Pass "" to use the currently active Python.

  -h, --help                  Show this message and exit.

Resume behaviour:
  Progress is saved in <directory>/<suite>/.pipeline_state after each task.
  Simply re-run the same command to resume after an interruption.
  To restart from scratch, delete that state file.

Examples:
  bash scripts/run_pipeline.sh --suite spatial --num-demonstrations 5
  bash scripts/run_pipeline.sh --suite goal --device spacemouse
  bash scripts/run_pipeline.sh --suite 10 --num-demonstrations 10 --device uarm
EOF
    exit 0
}

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)               SUITE="$2";                shift 2 ;;
        --num-demonstrations)  NUM_DEMOS="$2";             shift 2 ;;
        --device)              DEVICE="$2";                shift 2 ;;
        --uarm-port)           UARM_PORT="$2";             shift 2 ;;
        --uarm-output-max)     UARM_OUTPUT_MAX="$2";       shift 2 ;;
        --task-demos-file)     TASK_DEMOS_FILE="$2";       shift 2 ;;
        --directory)           COLLECTED_DIR="$2";         shift 2 ;;
        --robots)              ROBOTS="$2";                shift 2 ;;
        --lerobot-output)      LEROBOT_OUT_ROOT="$2";      shift 2 ;;
        --conda-env)           CONDA_ENV="$2";             shift 2 ;;
        -h|--help)             usage ;;
        *) echo "[error] Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Validate suite ─────────────────────────────────────────────────────────────
if [[ -z "${SUITE}" ]]; then
    echo "[error] --suite is required.  Choices: spatial | goal | object | 10 | 90" >&2
    exit 1
fi

SUITE_NAME="$(resolve_suite "${SUITE}")"
if [[ -z "${SUITE_NAME}" ]]; then
    echo "[error] Unknown suite '${SUITE}'.  Choices: spatial | goal | object | 10 | 90" >&2
    exit 1
fi

BDDL_DIR="${BDDL_ROOT}/${SUITE_NAME}"
if [[ ! -d "${BDDL_DIR}" ]]; then
    echo "[error] BDDL directory not found: ${BDDL_DIR}" >&2
    exit 1
fi

# Collect all .bddl task files, sorted alphabetically for determinism
mapfile -t BDDL_FILES < <(find "${BDDL_DIR}" -name "*.bddl" | sort)
TOTAL_TASKS=${#BDDL_FILES[@]}

if [[ ${TOTAL_TASKS} -eq 0 ]]; then
    echo "[error] No .bddl files found in ${BDDL_DIR}" >&2
    exit 1
fi

# ── Python runner ──────────────────────────────────────────────────────────────
if [[ -n "${CONDA_ENV}" ]]; then
    PYTHON="conda run --no-capture-output -n ${CONDA_ENV} python"
else
    PYTHON="python"
fi

# ── Paths ──────────────────────────────────────────────────────────────────────
SUITE_COLLECTED_DIR="${COLLECTED_DIR}/${SUITE_NAME}"
LIBERO_DATASETS="${REPO_ROOT}/datasets/datasets/${SUITE_NAME}"
LEROBOT_OUT="${LEROBOT_OUT_ROOT}/${SUITE_NAME}_lerobot"
STATE_FILE="${SUITE_COLLECTED_DIR}/.pipeline_state"

mkdir -p "${SUITE_COLLECTED_DIR}"

# ── Banner ─────────────────────────────────────────────────────────────────────
cat <<EOF

╔══════════════════════════════════════════════════════════════════╗
║              LIBERO Suite Pipeline                               ║
╠══════════════════════════════════════════════════════════════════╣
  Suite       : ${SUITE_NAME}  (${TOTAL_TASKS} tasks)
  Demos/task  : ${NUM_DEMOS} (default; overridden per task via task_demos.jsonl)
  Task config : ${TASK_DEMOS_FILE}
  Device      : ${DEVICE}
  Robots      : ${ROBOTS}
  Conda env   : ${CONDA_ENV:-"(current environment)"}
  Collected   : ${SUITE_COLLECTED_DIR}
  LIBERO HDF5 : ${LIBERO_DATASETS}
  LeRobot out : ${LEROBOT_OUT}
  State file  : ${STATE_FILE}
╚══════════════════════════════════════════════════════════════════╝

EOF

cd "${REPO_ROOT}"

# ── Helper: look up num_demos for a task from task_demos.jsonl ────────────────
#  Falls back to NUM_DEMOS if the file doesn't exist or the task isn't listed.
get_task_demos() {
    local suite_name="$1"
    local task_name="$2"
    local result=""
    if [[ -f "${TASK_DEMOS_FILE}" ]]; then
        result="$(python3 - "${TASK_DEMOS_FILE}" "${suite_name}" "${task_name}" <<'PYEOF' 2>/dev/null || true
import sys, json
fpath, suite, task = sys.argv[1], sys.argv[2], sys.argv[3]
with open(fpath) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("suite") == suite and obj.get("task") == task:
            print(obj.get("num_demos", ""))
            break
PYEOF
)"
    fi
    echo "${result:-${NUM_DEMOS}}"
}

# ── Helper: is task already done? ─────────────────────────────────────────────
task_is_done() {
    local task_name="$1"
    [[ -f "${STATE_FILE}" ]] && grep -qx "${task_name}=done" "${STATE_FILE}"
}

# ── Helper: mark task as done ─────────────────────────────────────────────────
mark_task_done() {
    local task_name="$1"
    echo "${task_name}=done" >> "${STATE_FILE}"
}

# ── Helper: count demo groups in an HDF5 file ─────────────────────────────────
count_demos_in_hdf5() {
    local hdf5="$1"
    ${PYTHON} - "${hdf5}" <<'PYEOF' 2>/dev/null || echo 0
import sys, h5py
try:
    with h5py.File(sys.argv[1], "r") as f:
        print(len([k for k in f["data"].keys() if k.startswith("demo")]))
except Exception:
    print(0)
PYEOF
}

# ── Helper: find usable intermediate demo.hdf5 for a task ─────────────────────
#  Searches <suite_collected_dir>/<task_name>/ for a demo.hdf5 with >= NUM_DEMOS.
find_ready_intermediate() {
    local task_name="$1"
    local task_dir="${SUITE_COLLECTED_DIR}/${task_name}"
    [[ -d "${task_dir}" ]] || return 0

    while IFS= read -r -d '' candidate; do
        local n
        n="$(count_demos_in_hdf5 "${candidate}")"
        if (( n >= NUM_DEMOS )); then
            echo "${candidate}"
            return 0
        fi
    done < <(find "${task_dir}" -name "demo.hdf5" -print0 2>/dev/null)
}

# ═══════════════════════════════════════════════════════════════════════════════
# Main loop — process each task
# ═══════════════════════════════════════════════════════════════════════════════
TASK_NUM=0
SKIPPED=0
PROCESSED=0

for BDDL_FILE in "${BDDL_FILES[@]}"; do
    TASK_NUM=$(( TASK_NUM + 1 ))
    TASK_NAME="$(basename "${BDDL_FILE}" .bddl)"
    TASK_NUM_DEMOS="$(get_task_demos "${SUITE_NAME}" "${TASK_NAME}")"
    LIBERO_STD_HDF5="${LIBERO_DATASETS}/${TASK_NAME}_demo.hdf5"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "  Task %d/%d: %s  [target: %s demos]\n" "${TASK_NUM}" "${TOTAL_TASKS}" "${TASK_NAME}" "${TASK_NUM_DEMOS}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── Resume check 1: recorded as done in state file ────────────────────────
    if task_is_done "${TASK_NAME}"; then
        echo "  [skip] Already completed (state file)."
        SKIPPED=$(( SKIPPED + 1 ))
        echo ""
        continue
    fi

    # ── Resume check 2: LIBERO-std HDF5 already has enough demos ──────────────
    if [[ -f "${LIBERO_STD_HDF5}" ]]; then
        EXISTING="$(count_demos_in_hdf5 "${LIBERO_STD_HDF5}")"
        if (( EXISTING >= TASK_NUM_DEMOS )); then
            echo "  [skip] LIBERO HDF5 already has ${EXISTING}/${TASK_NUM_DEMOS} demos."
            mark_task_done "${TASK_NAME}"
            SKIPPED=$(( SKIPPED + 1 ))
            echo ""
            continue
        else
            echo "  [info] LIBERO HDF5 has ${EXISTING}/${TASK_NUM_DEMOS} demos — will redo."
        fi
    fi

    # ── Per-task collection directory ─────────────────────────────────────────
    TASK_COLLECT_DIR="${SUITE_COLLECTED_DIR}/${TASK_NAME}"
    mkdir -p "${TASK_COLLECT_DIR}"

    # ── Step 1: Collect demonstrations ────────────────────────────────────────
    # ── Helper: find_ready_intermediate must also use TASK_NUM_DEMOS ──────────
    DEMO_HDF5=""
    TASK_COLLECT_DIR_TMP="${SUITE_COLLECTED_DIR}/${TASK_NAME}"
    if [[ -d "${TASK_COLLECT_DIR_TMP}" ]]; then
        while IFS= read -r -d '' candidate; do
            n="$(count_demos_in_hdf5 "${candidate}")"
            if (( n >= TASK_NUM_DEMOS )); then
                DEMO_HDF5="${candidate}"
                break
            fi
        done < <(find "${TASK_COLLECT_DIR_TMP}" -name "demo.hdf5" -print0 2>/dev/null)
    fi

    if [[ -n "${DEMO_HDF5}" ]]; then
        echo "  ▶ [1/2] Reusing existing intermediate ($(count_demos_in_hdf5 "${DEMO_HDF5}") demos):"
        echo "          ${DEMO_HDF5}"
    else
        echo "  ▶ [1/2] Collecting ${TASK_NUM_DEMOS} demonstration(s) ..."
        echo "          Press SPACE to start/stop recording, ESC to finish each demo."
        echo ""

        ${PYTHON} scripts/collect_demonstration.py \
            --bddl-file         "${BDDL_FILE}" \
            --device            "${DEVICE}" \
            --directory         "${TASK_COLLECT_DIR}" \
            --num-demonstration "${TASK_NUM_DEMOS}" \
            --robots            "${ROBOTS}" \
            $([[ "${DEVICE}" == "uarm" ]] && echo "--uarm-port ${UARM_PORT} --uarm-output-max ${UARM_OUTPUT_MAX}")

        # Locate the newly created demo.hdf5 (most recently modified)
        DEMO_HDF5="$(find "${TASK_COLLECT_DIR}" -name "demo.hdf5" \
                     -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"

        if [[ -z "${DEMO_HDF5}" || ! -f "${DEMO_HDF5}" ]]; then
            echo "  [error] Could not locate demo.hdf5 after collection — skipping task." >&2
            echo ""
            continue
        fi
        echo "  [info] demo file: ${DEMO_HDF5}"
    fi

    # ── Step 2: Create LIBERO-standard HDF5 (replay & render) ─────────────────
    echo ""
    echo "  ▶ [2/2] Replaying demos and rendering images ..."
    ${PYTHON} scripts/create_dataset.py \
        --demo-file      "${DEMO_HDF5}" \
        --use-camera-obs \
        --use-actions

    echo "  [info] LIBERO HDF5: ${LIBERO_STD_HDF5}"

    mark_task_done "${TASK_NAME}"
    PROCESSED=$(( PROCESSED + 1 ))
    echo ""
done

# ═══════════════════════════════════════════════════════════════════════════════
# Final step — convert the whole suite to LeRobot v2.1
# ═══════════════════════════════════════════════════════════════════════════════
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Final: Converting suite → LeRobot v2.1"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  input  : ${LIBERO_DATASETS}"
echo "  output : ${LEROBOT_OUT}"
echo ""

${PYTHON} scripts/convert_libero_to_lerobot.py \
    --input-dir  "${LIBERO_DATASETS}" \
    --output-dir "${LEROBOT_OUT}"

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
cat <<EOF

╔══════════════════════════════════════════════════════════════════╗
║  ✓  Pipeline complete!                                           ║
╠══════════════════════════════════════════════════════════════════╣
  Suite          : ${SUITE_NAME}
  Total tasks    : ${TOTAL_TASKS}
  Newly processed: ${PROCESSED}
  Skipped (done) : ${SKIPPED}
  LeRobot dataset: ${LEROBOT_OUT}
╚══════════════════════════════════════════════════════════════════╝
EOF
