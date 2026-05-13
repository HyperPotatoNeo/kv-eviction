#!/bin/bash
#SBATCH --job-name=q-qasper
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-74
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
DATASET=qasper
budget_path="head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json"

log_dir=logs/qa_evaluation/qwen-qasper

# Array: name n_articles start_article compute_stats methods target_size query_config algorithm_config use_budget
configs=(
  # ---- Baselines: full cache / no context ---- #
  "t0.99_summarize_a0   100   0 0 original,all 0.99 ss-plus-repeat summarize"
  "t0.99_summarize_a100 100 100 0 original,all 0.99 ss-plus-repeat summarize"
  "t0.99_summarize_a200 100 200 0 original,all 0.99 ss-plus-repeat summarize"
  "t0.99_orig_a0        100   0 0 original,no_context 0.99 ss-plus-repeat summarize"
  "t0.99_orig_a100      100 100 0 original,no_context 0.99 ss-plus-repeat summarize"
  "t0.99_orig_a200      100 200 0 original,no_context 0.99 ss-plus-repeat summarize"
  # #
  # # ---- AM (highest attention keys) ---- #
  "t0.01_fastest_a0     100   0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.01 ss-plus-repeat best 1"
  "t0.01_fastest_a100   100 100 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.01 ss-plus-repeat best 1"
  "t0.01_fastest_a200   100 200 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.01 ss-plus-repeat best 1"
  "t0.02_fastest_a0     100   0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.02 ss-plus-repeat best 1"
  "t0.02_fastest_a100   100 100 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.02 ss-plus-repeat best 1"
  "t0.02_fastest_a200   100 200 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.02 ss-plus-repeat best 1"
  "t0.05_fastest_a0     100   0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.05 ss-plus-repeat best 1"
  "t0.05_fastest_a100   100 100 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.05 ss-plus-repeat best 1"
  "t0.05_fastest_a200   100 200 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.05 ss-plus-repeat best 1"
  "t0.1_fastest_a0      100   0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.1 ss-plus-repeat best 1"
  "t0.1_fastest_a100    100 100 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.1 ss-plus-repeat best 1"
  "t0.1_fastest_a200    100 200 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.1 ss-plus-repeat best 1"
  "t0.2_fastest_a0      100   0 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.2 ss-plus-repeat best 1"
  "t0.2_fastest_a100    100 100 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.2 ss-plus-repeat best 1"
  "t0.2_fastest_a200    100 200 0 highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy 0.2 ss-plus-repeat best 1"
  #
  # ---- H2O ---- #
  "t0.01_h2o_a0         100   0 0 highest_attn_keys_mean_nobeta_direct 0.01 context-prefill kvzip-uniform"
  "t0.01_h2o_a100       100 100 0 highest_attn_keys_mean_nobeta_direct 0.01 context-prefill kvzip-uniform"
  "t0.01_h2o_a200       100 200 0 highest_attn_keys_mean_nobeta_direct 0.01 context-prefill kvzip-uniform"
  "t0.02_h2o_a0         100   0 0 highest_attn_keys_mean_nobeta_direct 0.02 context-prefill kvzip-uniform"
  "t0.02_h2o_a100       100 100 0 highest_attn_keys_mean_nobeta_direct 0.02 context-prefill kvzip-uniform"
  "t0.02_h2o_a200       100 200 0 highest_attn_keys_mean_nobeta_direct 0.02 context-prefill kvzip-uniform"
  "t0.05_h2o_a0         100   0 0 highest_attn_keys_mean_nobeta_direct 0.05 context-prefill kvzip-uniform"
  "t0.05_h2o_a100       100 100 0 highest_attn_keys_mean_nobeta_direct 0.05 context-prefill kvzip-uniform"
  "t0.05_h2o_a200       100 200 0 highest_attn_keys_mean_nobeta_direct 0.05 context-prefill kvzip-uniform"
  "t0.1_h2o_a0          100   0 0 highest_attn_keys_mean_nobeta_direct 0.1 context-prefill kvzip-uniform"
  "t0.1_h2o_a100        100 100 0 highest_attn_keys_mean_nobeta_direct 0.1 context-prefill kvzip-uniform"
  "t0.1_h2o_a200        100 200 0 highest_attn_keys_mean_nobeta_direct 0.1 context-prefill kvzip-uniform"
  "t0.2_h2o_a0          100   0 0 highest_attn_keys_mean_nobeta_direct 0.2 context-prefill kvzip-uniform"
  "t0.2_h2o_a100        100 100 0 highest_attn_keys_mean_nobeta_direct 0.2 context-prefill kvzip-uniform"
  "t0.2_h2o_a200        100 200 0 highest_attn_keys_mean_nobeta_direct 0.2 context-prefill kvzip-uniform"
  #
  # ---- KVZip ---- #
  "t0.01_kvzip_a0       100   0 0 global_highest_attn_keys_max_nobeta_direct 0.01 repeat kvzip"
  "t0.01_kvzip_a100     100 100 0 global_highest_attn_keys_max_nobeta_direct 0.01 repeat kvzip"
  "t0.01_kvzip_a200     100 200 0 global_highest_attn_keys_max_nobeta_direct 0.01 repeat kvzip"
  "t0.02_kvzip_a0       100   0 0 global_highest_attn_keys_max_nobeta_direct 0.02 repeat kvzip"
  "t0.02_kvzip_a100     100 100 0 global_highest_attn_keys_max_nobeta_direct 0.02 repeat kvzip"
  "t0.02_kvzip_a200     100 200 0 global_highest_attn_keys_max_nobeta_direct 0.02 repeat kvzip"
  "t0.05_kvzip_a0       100   0 0 global_highest_attn_keys_max_nobeta_direct 0.05 repeat kvzip"
  "t0.05_kvzip_a100     100 100 0 global_highest_attn_keys_max_nobeta_direct 0.05 repeat kvzip"
  "t0.05_kvzip_a200     100 200 0 global_highest_attn_keys_max_nobeta_direct 0.05 repeat kvzip"
  "t0.1_kvzip_a0        100   0 0 global_highest_attn_keys_max_nobeta_direct 0.1 repeat kvzip"
  "t0.1_kvzip_a100      100 100 0 global_highest_attn_keys_max_nobeta_direct 0.1 repeat kvzip"
  "t0.1_kvzip_a200      100 200 0 global_highest_attn_keys_max_nobeta_direct 0.1 repeat kvzip"
  "t0.2_kvzip_a0        100   0 0 global_highest_attn_keys_max_nobeta_direct 0.2 repeat kvzip"
  "t0.2_kvzip_a100      100 100 0 global_highest_attn_keys_max_nobeta_direct 0.2 repeat kvzip"
  "t0.2_kvzip_a200      100 200 0 global_highest_attn_keys_max_nobeta_direct 0.2 repeat kvzip"
  #
  # ---- KVMerger ---- #
  "t0.01_kvmerger_a0    100   0 0 kvmerger 0.01 context-prefill kvmerger 0"
  "t0.01_kvmerger_a100  100 100 0 kvmerger 0.01 context-prefill kvmerger 0"
  "t0.01_kvmerger_a200  100 200 0 kvmerger 0.01 context-prefill kvmerger 0"
  "t0.02_kvmerger_a0    100   0 0 kvmerger 0.02 context-prefill kvmerger 0"
  "t0.02_kvmerger_a100  100 100 0 kvmerger 0.02 context-prefill kvmerger 0"
  "t0.02_kvmerger_a200  100 200 0 kvmerger 0.02 context-prefill kvmerger 0"
  "t0.05_kvmerger_a0    100   0 0 kvmerger 0.05 context-prefill kvmerger 0"
  "t0.05_kvmerger_a100  100 100 0 kvmerger 0.05 context-prefill kvmerger 0"
  "t0.05_kvmerger_a200  100 200 0 kvmerger 0.05 context-prefill kvmerger 0"
  "t0.1_kvmerger_a0     100   0 0 kvmerger 0.1 context-prefill kvmerger 0"
  "t0.1_kvmerger_a100   100 100 0 kvmerger 0.1 context-prefill kvmerger 0"
  "t0.1_kvmerger_a200   100 200 0 kvmerger 0.1 context-prefill kvmerger 0"
  "t0.2_kvmerger_a0     100   0 0 kvmerger 0.2 context-prefill kvmerger 0"
  "t0.2_kvmerger_a100   100 100 0 kvmerger 0.2 context-prefill kvmerger 0"
  "t0.2_kvmerger_a200   100 200 0 kvmerger 0.2 context-prefill kvmerger 0"
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

budget_flag=""
if [ -n "$budget_path" ] && [ "$use_budget" = "1" ]; then
  budget_flag="--precomputed-budget-path $budget_path --max-ratio-per-head 0.75"
fi

chunking_flag="--chunking fixed --chunk-size 10000"
batch_size_flag="--batch-size 10"

log_dir_flag=""
if [ -n "$log_dir" ]; then
  log_dir_flag="--log-dir $log_dir"
fi

methods_formatted=$(echo "$methods" | tr ',' ' ')

start_time=$(date +%s)
echo "Start time: $(date -d @$start_time)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u -m evaluation.run_qa_evaluation --name "$name" $dataset_flag --n-articles "$n_articles" --start-article "$start_article" --compute-stats "$compute_stats" --compute-perplexity 0 --methods $methods_formatted --target-size "$target_size" --query-config "$query_config" --algorithm-config "$algorithm_config" $model_flag $budget_flag $chunking_flag $batch_size_flag $log_dir_flag

end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
