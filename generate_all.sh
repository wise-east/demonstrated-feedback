#!/bin/bash

# running all outputs with generate.sh 
# . generate.sh speechpref 0 ditto 1 -> generate for speechpref baseline for author 0 using model with ditto method on device 1
. ~/.bashrc
conda activate ditto

# methods=(ditto sft zero few)

authors=($(seq 0 9))
authors=(1 3 5 7 9)
benchmark=$1 # cmcc, ccat50, speechpref 
methods=$2

if $benchmark == "speechpref"; then
    authors=(0)
fi

N=8 # set to number of gpus available 
j=0

for method in "${methods[@]}"; do 
    for author in "${authors[@]}"; do 
        ((j=j%N)); ((j++==0)) && wait
        
        gpu_id=$(($j % $N))
        . generate.sh "$benchmark" "$author" "$method" "$gpu_id" &

    done; 
done

wait

echo "all generations complete"
