#!/bin/bash
###############################################################################
# UAV Embodied Intelligence OS - One-Click Baylands Inspection Launcher
#
# Starts all required processes in tmux:
#   1. PX4 SITL (x500_depth, baylands world)
#   2. MicroXRCEAgent (PX4-ROS2 bridge)
#   3. RGB image bridge
#   4. Depth image bridge
#   5. Camera info bridge
#   6. Python inspection script
#
# Usage:
#   ./scripts/start_inspection.sh
#   ./scripts/start_inspection.sh --natural-language "巡检baylands的3个目标点"
#   ./scripts/start_inspection.sh --kill   # Kill existing session
#
# Prerequisites:
#   - PX4-Autopilot installed at ~/PX4-Autopilot
#   - ROS2 Humble environment sourced
#   - Aerostack2 installed
#   - MicroXRCEDDSAgent installed
###############################################################################

set -euo pipefail

# ---------- Configuration ----------
SESSION_NAME="uav_inspection"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_DIR="${PX4_DIR:-${HOME}/PX4-Autopilot}"
MISSION_YAML="${PROJECT_DIR}/configs/missions/baylands_3point_inspection.yaml"
OUTPUT_DIR="${PROJECT_DIR}/outputs"

# PX4 SITL environment
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-3}"
PX4_GZ_STANDALONE="${PX4_GZ_STANDALONE:-1}"
PX4_SYS_AUTOSTART="${PX4_SYS_AUTOSTART:-4001}"
PX4_GZ_MODEL_POSE="${PX4_GZ_MODEL_POSE:-0,0,0}"
PX4_SIM_MODEL="${PX4_SIM_MODEL:-gz_x500_depth}"
PX4_GZ_WORLD="${PX4_GZ_WORLD:-baylands}"

# Gazebo image bridge topics
GZ_RGB_TOPIC="/world/${PX4_GZ_WORLD}/model/${PX4_SIM_MODEL}/link/camera_link/sensor/camera/image"
GZ_DEPTH_TOPIC="/world/${PX4_GZ_WORLD}/model/${PX4_SIM_MODEL}/link/camera_link/sensor/depth_camera/depth_image"
GZ_INFO_TOPIC="/world/${PX4_GZ_WORLD}/model/${PX4_SIM_MODEL}/link/camera_link/sensor/camera/camera_info"

# Timing (seconds to wait between startups)
PX4_STARTUP_WAIT=15
BRIDGE_STARTUP_WAIT=5
AGENT_STARTUP_WAIT=5
FINAL_SETTLE_WAIT=10

# ---------- Parse arguments ----------
EXTRA_ARGS=""
for arg in "$@"; do
    if [ "$arg" = "--kill" ]; then
        echo "[INFO] Killing existing session '${SESSION_NAME}'..."
        tmux kill-session -t "${SESSION_NAME}" 2>/dev/null && echo "  Done." || echo "  No session found."
        exit 0
    fi
    EXTRA_ARGS="${EXTRA_ARGS} ${arg}"
done

# ---------- Validation ----------
if ! command -v tmux &>/dev/null; then
    echo "[ERROR] tmux is not installed. Install with: sudo apt install tmux"
    exit 1
fi

if [ ! -d "${PX4_DIR}" ]; then
    echo "[ERROR] PX4-Autopilot not found at ${PX4_DIR}"
    echo "  Set PX4_DIR environment variable to the correct path."
    exit 1
fi

if ! command -v MicroXRCEAgent &>/dev/null; then
    echo "[WARN] MicroXRCEAgent not found in PATH, will try MicroXRCEDDSAgent..."
    XRCE_CMD="MicroXRCEDDSAgent"
    if ! command -v "${XRCE_CMD}" &>/dev/null; then
        echo "[ERROR] Neither MicroXRCEAgent nor MicroXRCEDDSAgent found."
        exit 1
    fi
else
    XRCE_CMD="MicroXRCEAgent"
fi

# Kill existing session if running
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

echo "============================================================"
echo "  UAV Embodied Intelligence OS - Inspection Launcher"
echo "============================================================"
echo "  Project:   ${PROJECT_DIR}"
echo "  PX4:       ${PX4_DIR}"
echo "  World:     ${PX4_GZ_WORLD}"
echo "  Model:     ${PX4_SIM_MODEL}"
echo "  Domain ID: ${ROS_DOMAIN_ID}"
echo "============================================================"
echo ""

# ---------- Create tmux session ----------
echo "[1/6] Starting PX4 SITL..."
tmux new-session -d -s "${SESSION_NAME}" -n "px4" \
    "cd ${PX4_DIR} && \
     export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} && \
     export PX4_GZ_STANDALONE=${PX4_GZ_STANDALONE} && \
     export PX4_SYS_AUTOSTART=${PX4_SYS_AUTOSTART} && \
     export PX4_GZ_MODEL_POSE='${PX4_GZ_MODEL_POSE}' && \
     export PX4_SIM_MODEL=${PX4_SIM_MODEL} && \
     export PX4_GZ_WORLD=${PX4_GZ_WORLD} && \
     ./build/px4_sitl_default/bin/px4; exec bash"

echo "  Waiting ${PX4_STARTUP_WAIT}s for PX4 to initialize..."
sleep "${PX4_STARTUP_WAIT}"

echo "[2/6] Starting MicroXRCEAgent..."
tmux new-window -t "${SESSION_NAME}" -n "xrce" \
    "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} && \
     ${XRCE_CMD} udp4 -p 8888; exec bash"

echo "  Waiting ${AGENT_STARTUP_WAIT}s for agent..."
sleep "${AGENT_STARTUP_WAIT}"

echo "[3/6] Starting RGB image bridge..."
tmux new-window -t "${SESSION_NAME}" -n "rgb_bridge" \
    "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} && \
     ros2 run ros_gz_image image_bridge ${GZ_RGB_TOPIC}@sensor_msgs/msg/Image@gz.msgs.Image; exec bash"

sleep 2

echo "[4/6] Starting depth image bridge..."
tmux new-window -t "${SESSION_NAME}" -n "depth_bridge" \
    "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} && \
     ros2 run ros_gz_image image_bridge ${GZ_DEPTH_TOPIC}@sensor_msgs/msg/Image@gz.msgs.Image; exec bash"

sleep 2

echo "[5/6] Starting camera info bridge..."
tmux new-window -t "${SESSION_NAME}" -n "cam_info" \
    "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} && \
     ros2 run ros_gz_bridge parameter_bridge ${GZ_INFO_TOPIC}@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo; exec bash"

echo "  Waiting ${BRIDGE_STARTUP_WAIT}s for bridges to stabilize..."
sleep "${BRIDGE_STARTUP_WAIT}"

echo "[6/6] Starting inspection script..."
echo "  Waiting ${FINAL_SETTLE_WAIT}s for all systems to settle..."
sleep "${FINAL_SETTLE_WAIT}"

tmux new-window -t "${SESSION_NAME}" -n "inspection" \
    "cd ${PROJECT_DIR} && \
     export ROS_DOMAIN_ID=${ROS_DOMAIN_ID} && \
     python3 examples/run_campus_inspection.py \
         --mission ${MISSION_YAML} \
         --output-dir ${OUTPUT_DIR} \
         ${EXTRA_ARGS}; \
     echo ''; echo 'Inspection complete. Press Enter to close.'; read"

echo ""
echo "============================================================"
echo "  All processes started in tmux session: ${SESSION_NAME}"
echo "============================================================"
echo ""
echo "  Attach to session:     tmux attach -t ${SESSION_NAME}"
echo "  Switch windows:        Ctrl+B then 0-5"
echo "  Window layout:"
echo "    0: px4          - PX4 SITL"
echo "    1: xrce         - MicroXRCEAgent"
echo "    2: rgb_bridge   - RGB image bridge"
echo "    3: depth_bridge - Depth image bridge"
echo "    4: cam_info     - Camera info bridge"
echo "    5: inspection   - Python inspection script"
echo ""
echo "  Kill session:          tmux kill-session -t ${SESSION_NAME}"
echo "  Or:                    ./scripts/start_inspection.sh --kill"
echo ""
echo "  Output directory:      ${OUTPUT_DIR}"
echo "============================================================"

# Attach to the inspection window
tmux select-window -t "${SESSION_NAME}:inspection"
tmux attach -t "${SESSION_NAME}"
