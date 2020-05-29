#!/usr/bin/env bash

if [[ $# != 2 ]]; then
    echo 'Usage: bash run_baselines.sh /path/to/dataset test_sequence_int'
    exit
fi

export folder_dir=$1
export sequence_id=$2

echo "Extracting superpoint"
python extract_superpoint.py --dataset-path $folder_dir --test-sequence $sequence_id

echo "Extracting d2-net"
python extract_d2-net.py --dataset-path $folder_dir --test-sequence $sequence_id

echo "Extracting r2d2"
python extract_r2d2.py --dataset-path $folder_dir --test-sequence $sequence_id
