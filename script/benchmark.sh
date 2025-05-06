#!/bin/bash

# Set user
username_postfix=""

# Set device
gpu_id=1

# Set dataset parameters
dataset_name="ETTh1"
task="univariate"
input_window_size=48
output_window_size=24

# Set paths
result_path="$HOME/benchmark-$task$username"
#result_path="$HOME/benchmark$username/${dataset_name}_gpu${gpu_id}/"
pyfast_path="$HOME/workspace$username/pyFAST/"
python_interpreter="/usr/local/anaconda3/envs/torch26_env/bin/python"

# Set models
models=(
    ar rnn lstm gru ed
    cnn1d cnnrnn cnnrnnres lstnet
    nlinear dlinear
    transformer informer autoformer fedformer
    film triformer crossformer timesnet patchtst
    staeformer itransformer timesfm timer timexer timemixer
    coat tcoat codr
)

# Execute

echo "Step 1. check, if not exist then create: $result_path"
mkdir -p "$result_path"

echo "Step 2. change working directory to: $pyfast_path"
cd "$pyfast_path" || echo "not found $pyfast_path" || exit 1

echo "Step 3. run models and save logs to: $result_path"

#for model_name in "${models[@]}"; do
for index in "${!models[@]}"; do
  model_name=${models[$index]}
  index_str=$(printf "%02d" "$((index + 1))")

  model_log_filename="${dataset_name}_L${input_window_size}_H${output_window_size}_${index_str}_${model_name}.txt"

  echo "Validate $model_name, save $model_log_filename"

  $python_interpreter -m example.paper.benchmark \
    --device "cuda:$gpu_id" \
    --dataset_name "$dataset_name" \
    --task "$task" \
    --input_window_size "$input_window_size" \
    --output_window_size "$output_window_size" \
    --model_name "$model_name" \
    1> "$result_path/$model_log_filename"

done

echo "Done."
