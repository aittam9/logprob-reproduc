#!/bin/bash
#SBATCH --job-name=ewok_instruct
#SBATCH -e "/home/agnese.lombardi/progetti/logprob-reproduc-main/results_instruct/%x-%A.err"
#SBATCH -o "/home/agnese.lombardi/progetti/logprob-reproduc-main/results_instruct/%x-%A.out"
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --account=pr_neplab

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate progetto

cd /home/agnese.lombardi/progetti/logprob-reproduc-main/scripts

# EWoK isn't chunked like BLiMP -- get_probs_instruct.py --data ewok loops over every
# testsuite-*.csv under the EWoK output dir itself in one call, so this isn't an array
# job; it just runs each model in MODELS sequentially in a single job.
#
# get_probs_instruct.py only accepts instruction-tuned models (see utils/models.py
# INSTRUCT_MODELS) -- add more "-it" model names here to extend the sweep.
MODELS=(
    "Gemma-3-1B-it"
)

for MODEL in "${MODELS[@]}"; do
    OUT_FILE="/home/agnese.lombardi/progetti/logprob-reproduc-main/results_instruct/${MODEL}/ewok-ewok-core-1.0/ewok-ewok-core-1.0_scored.tsv"

    if [ -f "$OUT_FILE" ]; then
        echo "EWoK già completato per $MODEL ($OUT_FILE esiste), skip."
    else
        echo "=== Modello: $MODEL, Task: ewok ==="
        python get_probs_instruct.py --data ewok --model "$MODEL"
    fi
done
