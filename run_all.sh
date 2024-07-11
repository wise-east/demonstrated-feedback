conda activate ditto

export HF_TOKEN=""

# benchmarks=("ccat50" "cmcc" "custom")
# benchmarks=("cmcc")
benchmarks=("ccat50")

rm -rf outputs/${benchmark}-mistral-7b-instruct-ditto

# loop through 0 to 9, including 9 
for benchmark in "${benchmarks[@]}"; do 
    for i in {0..9}; do 

        ACCELERATE_LOG_LEVEL=info accelerate launch \
            --config_file configs/multi_gpu_2.yaml \
            scripts/run_ditto.py configs/ditto-mistral-7b-instruct.yaml \
            --train_author_key=${i} \
            --output_dir=outputs/${benchmark}-mistral-7b-instruct-ditto_author${i} \
            --train_pkl=benchmarks/${benchmark}/processed/${benchmark}_train.pkl

        # save
        # python generate.py \
        #     --benchmark=$benchmark \
        #     --train_author_key=${i}


    done 
done