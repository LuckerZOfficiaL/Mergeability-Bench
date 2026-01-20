#!/bin/bash

# Script to sequentially finetune all 20 tasks from the N20 benchmark
# with R2 (Moderate Update) and R3 (Gradient Magnitude) regularization enabled
#
# Usage: bash scripts/run_fts.sh
#
# The regularization settings are controlled in conf/train/finetune.yaml:
#   - enable_moderate_update: true
#   - lambda_moderate_update: 0.01
#   - enable_grad_magnitude: true
#   - lambda_grad_magnitude: 0.001
#
# Models will be saved to:
#   /home/ubuntu/thesis/MM/model-merging/models/ViT-B-16/{dataset}_moderate_update_grad_magnitude/model.pt

set -e  # Exit on error

# List of N20 datasets
datasets=(
    "Cars"
    "CIFAR10"
    "CIFAR100"
    "DTD"
    "EMNIST"
    "EuroSAT"
    "FashionMNIST"
    "FER2013"
    "Flowers102"
    "Food101"
    "GTSRB"
    "KMNIST"
    "MNIST"
    "OxfordIIITPet"
    "PCAM"
    "RenderedSST2"
    "RESISC45"
    "SUN397"
    "SVHN"
    "STL10"
)

# Log file
LOG_DIR="/home/ubuntu/thesis/MM/model-merging/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/finetune_all_n20_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Starting finetuning for all N20 tasks" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Total number of datasets
TOTAL=${#datasets[@]}
CURRENT=0

# Track successes and failures
SUCCESS_COUNT=0
FAILURE_COUNT=0
FAILED_DATASETS=()

# Finetune each dataset
for dataset in "${datasets[@]}"; do
    CURRENT=$((CURRENT + 1))

    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "[$CURRENT/$TOTAL] Finetuning: $dataset" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Start time: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Run finetuning
    if uv run python scripts/finetune.py dataset=$dataset 2>&1 | tee -a "$LOG_FILE"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "" | tee -a "$LOG_FILE"
        echo "✓ Successfully completed: $dataset" | tee -a "$LOG_FILE"
        echo "End time: $(date)" | tee -a "$LOG_FILE"
    else
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        FAILED_DATASETS+=("$dataset")
        echo "" | tee -a "$LOG_FILE"
        echo "✗ FAILED: $dataset" | tee -a "$LOG_FILE"
        echo "End time: $(date)" | tee -a "$LOG_FILE"

        # Continue with next dataset instead of exiting
        continue
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "Progress: $CURRENT/$TOTAL completed (Success: $SUCCESS_COUNT, Failed: $FAILURE_COUNT)" | tee -a "$LOG_FILE"
done

# Final summary
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "FINAL SUMMARY" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Total datasets: $TOTAL" | tee -a "$LOG_FILE"
echo "Successful: $SUCCESS_COUNT" | tee -a "$LOG_FILE"
echo "Failed: $FAILURE_COUNT" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $FAILURE_COUNT -gt 0 ]; then
    echo "Failed datasets:" | tee -a "$LOG_FILE"
    for failed in "${FAILED_DATASETS[@]}"; do
        echo "  - $failed" | tee -a "$LOG_FILE"
    done
    echo "" | tee -a "$LOG_FILE"
fi

echo "Completion time: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Exit with error if any dataset failed
if [ $FAILURE_COUNT -gt 0 ]; then
    exit 1
fi

echo ""
echo "All finetuning tasks completed successfully!"
