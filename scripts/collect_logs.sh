#!/usr/bin/env bash
# =============================================================================
# UAV Edge — Experiment Results Collection Script
# Collects CSV_RESULT lines from MS-10 pods and saves to experiment_results.csv
#
# Usage:
#   ./collect_logs.sh [--namespace uav-edge] [--output ./experiment_results.csv]
#   ./collect_logs.sh --tail 5000     # Increase log tail lines
#   ./collect_logs.sh --follow        # Live mode (stream results in real-time)
# =============================================================================
set -euo pipefail

NAMESPACE="${NAMESPACE:-uav-edge}"
OUTPUT="${OUTPUT:-./experiment_results.csv}"
TAIL_LINES="${TAIL_LINES:-10000}"
FOLLOW_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --namespace) NAMESPACE="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --tail) TAIL_LINES="$2"; shift 2 ;;
        --follow) FOLLOW_MODE=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# CSV header matching the output format from MS-10
CSV_HEADER="request_id,e2e_ms,rgb_branch_ms,ir_branch_ms"
CSV_HEADER+=",ms1_compute_ms,ms2_compute_ms,ms3_compute_ms,ms4_compute_ms,ms5_compute_ms"
CSV_HEADER+=",ms6_compute_ms,ms7_compute_ms,ms8_compute_ms,ms9_compute_ms,ms10_compute_ms"
CSV_HEADER+=",net_ms1_ms2,net_ms1_ms3,net_ms2_ms4,net_ms3_ms5"
CSV_HEADER+=",net_ms4_ms6,net_ms5_ms6,net_ms6_ms7,net_ms6_ms8"
CSV_HEADER+=",net_ms7_ms9,net_ms8_ms9,net_ms9_ms10"

if $FOLLOW_MODE; then
    echo "============================================================"
    echo "  Live CSV Collection (Ctrl+C to stop)"
    echo "  Namespace: ${NAMESPACE}"
    echo "  Output:    ${OUTPUT}"
    echo "============================================================"

    # Write header
    echo "$CSV_HEADER" > "$OUTPUT"

    # Stream and filter
    kubectl -n "$NAMESPACE" logs -l app=ms-10 -f --tail=0 | \
        grep --line-buffered "^CSV_RESULT:" | \
        sed -u 's/^CSV_RESULT://' | \
        tee -a "$OUTPUT"
else
    echo "============================================================"
    echo "  CSV Result Collection (Batch Mode)"
    echo "  Namespace: ${NAMESPACE}"
    echo "  Tail:      ${TAIL_LINES} lines"
    echo "  Output:    ${OUTPUT}"
    echo "============================================================"

    # Write header
    echo "$CSV_HEADER" > "$OUTPUT"

    # Collect from all MS-10 pod replicas
    PODS=$(kubectl -n "$NAMESPACE" get pods -l app=ms-10 -o jsonpath='{.items[*].metadata.name}')

    if [ -z "$PODS" ]; then
        echo "ERROR: No ms-10 pods found in namespace ${NAMESPACE}"
        exit 1
    fi

    COUNT=0
    for POD in $PODS; do
        echo "  Collecting from pod: ${POD}..."
        LINES=$(kubectl -n "$NAMESPACE" logs "$POD" --tail="$TAIL_LINES" | \
            grep "^CSV_RESULT:" | \
            sed 's/^CSV_RESULT://')
        if [ -n "$LINES" ]; then
            echo "$LINES" >> "$OUTPUT"
            LINE_COUNT=$(echo "$LINES" | wc -l)
            COUNT=$((COUNT + LINE_COUNT))
        fi
    done

    echo ""
    echo "============================================================"
    echo "  Collection Complete"
    echo "  Total CSV rows: ${COUNT}"
    echo "  Output file:    ${OUTPUT}"
    echo "============================================================"
    echo ""
    echo "Preview (first 5 rows):"
    head -6 "$OUTPUT"
    echo "..."
    echo ""
    echo "Analyze with Python:"
    echo "  import pandas as pd"
    echo "  df = pd.read_csv('${OUTPUT}')"
    echo "  print(df.describe())"
fi
