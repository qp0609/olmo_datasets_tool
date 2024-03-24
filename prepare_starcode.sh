#!/usr/bin/env bash
nohup python prepare_starcodedata_dataset.py $1 -o $2 --workers 10 > nohup.out 2>&1 &
echo 'start ... ...'
