# run training jobs in parallel  (reference: https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop)

conda activate ditto

export HF_TOKEN=""

# benchmarks=("ccat50" "cmcc" "custom")
benchmarks=("cmcc" "ccat50")
# benchmarks=("ccat50" "cmcc" "speechpref")
config_num=0

task() {
    benchmark=$1
    author_key=$2
    config_num=$3

    rm -rf outputs/${benchmark}-mistral-7b-instruct-ditto_author${author_key}

    # sleep 10

    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file configs/multi_gpu_${config_num}.yaml \
        scripts/run_ditto.py configs/ditto-mistral-7b-instruct.yaml \
        --train_pkl=benchmarks/${benchmark}/processed/${benchmark}_train.pkl \
        --train_author_key=${author_key} \
        --output_dir=outputs/${benchmark}-mistral-7b-instruct-ditto_author${author_key} 

    #python generate.py \
    #    --benchmark=$benchmark \
    #    --train_author_key=${author_key} \
    #    --method ditto

}

N=2 # set to number of gpus available / 4.
j=1 

# author_ids=(0 1 2 3 4 5 6 7 8 9)
# author ids from 0 to 9 using .. syntax

(
for benchmark in "${benchmarks[@]}"; do 

    if [ "$benchmark" == "cmcc" ]; then
        # author_ids=(2 3 5 6 7 8 11 13 15 17)
        author_ids=(11 13 15 17)
    elif [ "$benchmark" == "ccat50" ]; then
        author_ids=(32 28 20 10 27 12 15 38 23 30)
    elif [ "$benchmark" == "speechpref" ]; then
        author_ids=(0)
    fi

    for i in "${author_ids[@]}"; do 

        config_num=$((config_num % 2))

        echo "Running task $benchmark $i with config $config_num"
        task $benchmark $i $config_num &

        config_num=$((config_num + 1))

        ((j=j%N)); ((j++==0)) && wait

    done 
done
)
wait

echo "all done"
