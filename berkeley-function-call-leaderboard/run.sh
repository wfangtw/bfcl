#!/bin/bash
#
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate bfcl
TEST_SET=exec_custom01
bfcl generate --model meta-llama/Llama-3-8B-Instruct-FC --test-category ${TEST_SET} --num-threads 1 --temperature 0.1
# --result-dir --allow-overwrite
bfcl evaluate --model meta-llama/Llama-3-8B-Instruct-FC --test-category ${TEST_SET}
# --result-dir --score-dir
conda deactivate
