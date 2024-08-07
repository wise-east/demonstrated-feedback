# run training jobs in parallel  (reference: https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop)

conda activate ditto

export HF_TOKEN=""

# benchmarks=("ccat50" "cmcc" "custom")
benchmarks=("cmcc")
# benchmarks=("ccat50")

task() {
    benchmark=$1
    author_key=$2
    # config num = 0 or 1 depending on author key % 2
    config_num=$((author_key % 2))

    rm -rf outputs/${benchmark}-mistral-7b-instruct-ditto_author${author_key}

    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file configs/multi_gpu_${config_num}.yaml \
        scripts/run_ditto.py configs/ditto-mistral-7b-instruct.yaml \
        --train_pkl=benchmarks/${benchmark}/processed/${benchmark}_train.pkl \
        --train_author_key=${author_key} \
        --output_dir=outputs/${benchmark}-mistral-7b-instruct-ditto_author${author_key} 

    python generate.py \
        --benchmark=$benchmark \
        --train_author_key=${author_key} \
        --method ditto

}

N=2 # set to number of gpus available / 4.

# loop through 0 to 9, including 9 
(
for benchmark in "${benchmarks[@]}"; do 
    for i in {0..4}; do 
        ((j=j%N)); ((j++==0)) && wait

        task $benchmark $i &

    done 
done
)
wait

echo "all done"