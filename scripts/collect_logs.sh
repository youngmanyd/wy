#!/usr/bin/env bash
# =============================================================================
# collect_logs.sh — Collect CSV experiment results from telemetry-dashboard
# =============================================================================
set -euo pipefail

OUTPUT="${1:-experiment_results.csv}"
NAMESPACE="uav-edge"

HEADER="request_id,e2e_ms,rgb_branch_ms,ir_branch_ms"
HEADER+=",gateway_compute_ms,rgb_preproc_compute_ms,ir_preproc_compute_ms"
HEADER+=",rgb_detect_compute_ms,ir_detect_compute_ms,fusion_compute_ms"
HEADER+=",tracker_compute_ms,sa_compute_ms,decision_compute_ms,dashboard_compute_ms"
HEADER+=",net_gw_rgb,net_gw_ir,net_rgb_rgbd,net_ir_ird"
HEADER+=",net_rgbd_fusion,net_ird_fusion,net_fusion_tracker,net_fusion_sa"
HEADER+=",net_tracker_dm,net_sa_dm,net_dm_dashboard"

echo "$HEADER" > "$OUTPUT"

echo "Collecting CSV results from telemetry-dashboard pods..."
kubectl logs -n "$NAMESPACE" -l app=telemetry-dashboard --tail=-1 2>/dev/null | \
    grep "^CSV_RESULT:" | \
    sed 's/^CSV_RESULT://' >> "$OUTPUT"

LINES=$(wc -l < "$OUTPUT")
echo "Collected $((LINES - 1)) result rows -> $OUTPUT"
