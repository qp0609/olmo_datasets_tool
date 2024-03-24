#!/usr/bin/env bash
nohup python compress_jsonl_to_zstd_dataset.py $1 -o $2 --workers $3 > nohup.out 2>&1 &
echo 'start ... ...'
