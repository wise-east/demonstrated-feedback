# for running single author, single benchmark

conda activate ditto

export HF_TOKEN=""

# take benchmark and author key as commandline arguments
benchmark=$1 # one of [ccat50, cmcc, speechpref]
author_key=$2 # from 0 to 9 
method=$3 # one of [ditto, sft, zero, few]
device_id=$4 # from 0 to 7

# example: ./generate.sh ccat50 0

# ACCELERATE_LOG_LEVEL=info accelerate launch \
#     --config_file configs/generate.yaml \
#     generate.py \
#     --benchmark=$benchmark \
#     --train_author_key=${author_key}

CUDA_VISIBLE_DEVICES=$device_id python generate.py \
    --benchmark=$benchmark \
    --train_author_key=${author_key} \
    --method $method

