#!/bin/bash
#
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate bfcl
bfcl generate --model meta-llama/Llama-3-8B-Instruct-FC --test-category exec_custom01 --num-threads 1 --temperature 0.1
# --result-dir --allow-overwrite
bfcl evaluate --model meta-llama/Llama-3-8B-Instruct-FC --test-category exec_custom01
# --result-dir --score-dir
conda deactivate
