#!/usr/bin/env bash

if [[ $# != 3 ]]; then
    echo 'Usage: bash run_baselines.sh /path/to/dataset /path/to/output/folder test_sequence_int'
    exit
fi

export folder_dir=$1
export output_dir=$2
export sequence_id=$3

echo "Extracting superpoint"
python extract_superpoint.py --cuda --dataset-path $folder_dir --output-path $output_dir --test-sequence $sequence_id

echo "Extracting d2_net"
python extract_d2_net.py --dataset-path $folder_dir --output-path $output_dir --test-sequence $sequence_id

echo "Extracting r2d2"
python extract_r2d2.py --dataset-path $folder_dir --output-path $output_dir --test-sequence $sequence_id

echo "Extracting superglue"
python extract_superglue.py --dataset-path $folder_dir --output-path $output_dir --test-sequence $sequence_id