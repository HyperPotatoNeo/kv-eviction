#!/bin/bash
#SBATCH --job-name=q-q-kvm
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-10
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=6:00:00
#SBATCH --requeue

# -------- Environment ------------------------------------------------ #
export HOME=/home/$USER
source ~/.bashrc
cd ~/compaction-release
conda activate compaction

MODEL=Qwen/Qwen3-4B
DATASET=quality
budget_path="head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json"

log_dir=logs/qa_evaluation/qwen-quality-kvmerger

# Array: name n_articles start_article compute_stats methods target_size query_config algorithm_config use_budget
configs=(
  # # ---- Baselines: full cache / no context ---- #
  "t0.99_orig          50  0 1 original,no_context 0.99 ss-plus-repeat summarize"
  #
  # ---- AM (highest attention keys) ---- #
  "t0.01_fastest             50  0 1 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.01 ss-plus-repeat best 1"
  "t0.02_fastest             50  0 1 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.02 ss-plus-repeat best 1"
  "t0.05_fastest             50  0 1 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.05 ss-plus-repeat best 1"
  "t0.1_fastest              50  0 1 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.1 ss-plus-repeat best 1"
  "t0.2_fastest              50  0 1 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.2 ss-plus-repeat best 1"
  #
  # ---- KVMerger (pure: Gaussian kernel merge, no beta) ---- #
  "t0.01_kvmerger             50  0 1 kvmerger 0.01 context-prefill kvmerger 0"
  "t0.02_kvmerger             50  0 1 kvmerger 0.02 context-prefill kvmerger 0"
  "t0.05_kvmerger             50  0 1 kvmerger 0.05 context-prefill kvmerger 0"
  "t0.1_kvmerger              50  0 1 kvmerger 0.1 context-prefill kvmerger 0"
  "t0.2_kvmerger              50  0 1 kvmerger 0.2 context-prefill kvmerger 0"
)

# Select configuration based on SLURM array task ID
config="${configs[$SLURM_ARRAY_TASK_ID]}"
read -r name n_articles start_article compute_stats methods target_size query_config algorithm_config use_budget <<< "$config"

model_flag=""
if [ -n "$MODEL" ]; then
  model_flag="--model-name $MODEL"
fi

dataset_flag=""
if [ -n "$DATASET" ]; then
  dataset_flag="--dataset-name $DATASET"
fi

perplexity_only_flag=""
if [ -n "$perplexity_only" ]; then
  perplexity_only_flag="--perplexity-only $perplexity_only"
fi

budget_flag=""
if [ -n "$budget_path" ] && [ "$use_budget" = "1" ]; then
  budget_flag="--precomputed-budget-path $budget_path --max-ratio-per-head 0.75"
fi

chunking_flag=""
batch_size_flag=""
if [ -n "$chunking" ]; then
  chunking_flag="--chunking $chunking"
  batch_size_flag="--batch-size 10"
fi

log_dir_flag=""
if [ -n "$log_dir" ]; then
  log_dir_flag="--log-dir $log_dir"
fi

methods_formatted=$(echo "$methods" | tr ',' ' ')

start_time=$(date +%s)
echo "Start time: $(date -d @$start_time)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u -m evaluation.run_qa_evaluation --name "$name" $dataset_flag --n-articles "$n_articles" --start-article "$start_article" --compute-stats "$compute_stats" --methods $methods_formatted --target-size "$target_size" --query-config "$query_config" --algorithm-config "$algorithm_config" $model_flag $perplexity_only_flag $budget_flag $chunking_flag $batch_size_flag $log_dir_flag

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
