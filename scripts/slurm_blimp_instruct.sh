#!/bin/bash
#SBATCH --job-name=blimp_instruct
#SBATCH -e "/home/agnese.lombardi/progetti/logprob-reproduc-main/results_instruct/%x-%A_%a.err"
#SBATCH -o "/home/agnese.lombardi/progetti/logprob-reproduc-main/results_instruct/%x-%A_%a.out"
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=35:00:00
#SBATCH --mem=16G
#SBATCH --account=pr_neplab
#SBATCH --array=0-66%5

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate progetto

cd /home/agnese.lombardi/progetti/logprob-reproduc-main/scripts

DATA_KEY="$SLURM_ARRAY_TASK_ID"

# get_probs_instruct.py only accepts instruction-tuned models (see utils/models.py
# INSTRUCT_MODELS) -- add more "-it" model names here to extend the sweep.
MODELS=(
    "Gemma-3-1B-it"
)

for MODEL in "${MODELS[@]}"; do
    OUT_FILE="/home/agnese.lombardi/progetti/logprob-reproduc-main/results_instruct/${MODEL}/${DATA_KEY}/${DATA_KEY}_scored.tsv"

    if [ -f "$OUT_FILE" ]; then
        echo "Task $DATA_KEY già completato per $MODEL ($OUT_FILE esiste), skip."
    else
        echo "=== Modello: $MODEL, Task: $DATA_KEY ==="
        python get_probs_instruct.py --data "$DATA_KEY" --model "$MODEL"
    fi
done
