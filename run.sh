# for running single author, single benchmark

conda activate ditto

export HF_TOKEN=""

benchmark="ccat50"
benchmark="speechpref" 
author_key=0

rm -rf outputs/${benchmark}-mistral-7b-instruct-ditto

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file configs/multi_gpu_2.yaml \
    scripts/run_ditto.py configs/ditto-mistral-7b-instruct.yaml \
    --train_pkl=benchmarks/${benchmark}/processed/${benchmark}_train.pkl \
    --train_author_key=${author_key} \
    --output_dir=outputs/${benchmark}-mistral-7b-instruct-ditto_author${author_key} 

python generate.py \
    --benchmark=$benchmark \
    --train_author_key=${author_key}

