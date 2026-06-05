#!/usr/bin/env bash
# =============================================================================
# UAV Edge — CSV Telemetry Log Collection Script
# Extracts CSV_RESULT: lines from telemetry-dashboard pods.
#
# Usage:
#   ./collect_logs.sh [--output experiment_results.csv] [--follow]
# =============================================================================
set -euo pipefail

NAMESPACE="${NAMESPACE:-uav-edge}"
OUTPUT="experiment_results.csv"
FOLLOW=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output|-o) OUTPUT="$2"; shift 2 ;;
        --follow|-f) FOLLOW=true; shift ;;
        --namespace|-n) NAMESPACE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

CSV_HEADER="request_id,e2e_ms,rgb_branch_ms,ir_branch_ms"
CSV_HEADER+=",gateway_compute_ms,rgb_preprocessor_compute_ms,ir_preprocessor_compute_ms"
CSV_HEADER+=",rgb_detector_compute_ms,ir_detector_compute_ms,feature_fusion_compute_ms"
CSV_HEADER+=",object_tracker_compute_ms,situation_awareness_compute_ms,decision_maker_compute_ms"
CSV_HEADER+=",telemetry_dashboard_compute_ms"
CSV_HEADER+=",net_gateway_rgb_preproc_ms,net_gateway_ir_preproc_ms"
CSV_HEADER+=",net_rgb_preproc_rgb_det_ms,net_ir_preproc_ir_det_ms"
CSV_HEADER+=",net_rgb_det_fusion_ms,net_ir_det_fusion_ms"
CSV_HEADER+=",net_fusion_tracker_ms,net_fusion_sa_ms"
CSV_HEADER+=",net_tracker_decision_ms,net_sa_decision_ms"
CSV_HEADER+=",net_decision_dashboard_ms"

if $FOLLOW; then
    echo "Live-following telemetry-dashboard logs... (Ctrl+C to stop)"
    echo "$CSV_HEADER"
    kubectl logs -l app=telemetry-dashboard -n "$NAMESPACE" -f --tail=0 2>/dev/null | \
        grep --line-buffered 'CSV_RESULT:' | sed 's/.*CSV_RESULT://'
else
    echo "$CSV_HEADER" > "$OUTPUT"
    kubectl logs -l app=telemetry-dashboard -n "$NAMESPACE" --tail=-1 2>/dev/null | \
        grep 'CSV_RESULT:' | sed 's/.*CSV_RESULT://' >> "$OUTPUT"
    LINES=$(wc -l < "$OUTPUT")
    echo "Collected $((LINES - 1)) results -> ${OUTPUT}"
fi
