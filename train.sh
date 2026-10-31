#!/bin/bash
set -euo pipefail

dataset=""
gpu="0"
fusion_head="v2"
artifact_views="1"
use_npr="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            dataset="$2"
            shift 2
            ;;
        --gpu)
            gpu="$2"
            shift 2
            ;;
        --fusion_head)
            fusion_head="$2"
            shift 2
            ;;
        --artifact_views)
            artifact_views="$2"
            shift 2
            ;;
        --npr)
            use_npr="1"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$dataset" ]]; then
    echo "Missing required argument: --dataset"
    exit 1
fi

if [[ "$dataset" != "sd-v1_4" ]]; then
    fusion_head="v1"
    use_npr="0"
fi

# Artifact branch training
python main.py \
    --phase train \
    --gpu "$gpu" \
    --mode branch \
    --branch artifact \
    --train_dataset "$dataset" \
    --epochs 10

# Semantic branch training (feature interpolation is only applied for sd-v1_4)
semantic_extra=""
if [[ "$dataset" == "sd-v1_4" ]]; then
    semantic_extra="--feat_interp"
fi

python main.py \
    --phase train \
    --gpu "$gpu" \
    --mode branch \
    --branch semantic \
    --train_dataset "$dataset" \
    --epochs 10 \
    $semantic_extra

# Optional NPR branch used by fusion v3.
npr_args=""
if [[ "$use_npr" == "1" || "$fusion_head" == "v3" || "$fusion_head" == "v3_npr" ]]; then
    python main.py \
        --phase train \
        --gpu "$gpu" \
        --mode branch \
        --branch npr \
        --train_dataset "$dataset" \
        --epochs 10
    npr_args="--npr"
fi

# Adaptive feature fusion.
python main.py \
    --phase train \
    --gpu "$gpu" \
    --mode fusion \
    --train_dataset "$dataset" \
    --epochs 30 \
    --fusion_head "$fusion_head" \
    --artifact_views "$artifact_views" \
    $npr_args
