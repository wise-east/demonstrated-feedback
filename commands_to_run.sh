#!/bin/bash
source ~/.bashrc
conda activate ditto

# paste commands from `check_all_outputs.py`
commands_to_run=(
        "python generate.py --benchmark ccat50 --method ditto --train_author_key 27"
        "python generate.py --benchmark ccat50 --method ditto --train_author_key 12"
        "python generate.py --benchmark ccat50 --method sft --train_author_key 27"
        "python generate.py --benchmark ccat50 --method sft --train_author_key 12"
)

N=8 # cycle through gpu ids 0 to 7 and wait for every 8th command
device_id=0

# add CUDA_VISIBLE_DEVICES=$device_id to each command

for command in "${commands_to_run[@]}"; do
    ((device_id=device_id%N)); ((device_id++==0)) && wait
    CUDA_VISIBLE_DEVICES=$device_id $command &
done

wait

echo "all generations complete"