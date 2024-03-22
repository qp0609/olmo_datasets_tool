#!/usr/bin/env bash
nohup python prepare_memmap_dataset.py /root/olmo_datasets_tool/data/RedPajamaArXiv /root/olmo_datasets_tool/data/RedPajamaWikipedia -o /root/olmo_datasets_tool/olmo_npy --workers 10 --tokenizer allenai-eleuther-ai-gpt-neox-20b-pii-special.json > output.log 2>&1 &
echo 'start ... ...'
