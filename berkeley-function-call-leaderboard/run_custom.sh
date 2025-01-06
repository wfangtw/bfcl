#!/bin/bash
#
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate bfcl
#cd ${DATA_DIR}
bfcl generate \
  --model ${MODEL} \
  --test-category ${EXP_ID} \
  --num-threads ${N_THREADS} \
  --temperature ${TEMP} \
  --result-dir ${RESULT_DIR} --allow-overwrite

#meta-llama/Llama-3-8B-Instruct-FC
bfcl evaluate \
  --model ${MODEL} \
  --test-category ${EXP_ID} \
  --result-dir ${RESULT_DIR} \
  --score-dir ${SCORE_DIR}

conda deactivate
