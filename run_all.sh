# run training jobs in parallel  (reference: https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop)

conda activate ditto

export HF_TOKEN=""

# benchmarks=("ccat50" "cmcc" "custom")
# benchmarks=("cmcc")
benchmarks=("ccat50" "cmcc" "speechpref")

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

# author_ids=(0 1 2 3 4 5 6 7 8 9)
# author ids from 0 to 9 using .. syntax
author_ids=($(seq 0 9))
(
for benchmark in "${benchmarks[@]}"; do 
    for i in "${author_ids[@]}"; do 

        # if benchmark is speechpref, only run author 0
        if [ "$benchmark" == "speechpref" ] && [ "$i" != "0" ]; then
            continue
        fi

        ((j=j%N)); ((j++==0)) && wait

        task $benchmark $i &

    done 
done
)
wait

echo "all done"
