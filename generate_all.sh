#!/bin/bash

# running all outputs with generate.sh 
# . generate.sh speechpref 0 ditto 1 -> generate for speechpref baseline for author 0 using model with ditto method on device 1
. ~/.bashrc
conda activate ditto

# methods=(ditto sft zero few)

# author_ids=($(seq 0 9))
# author_ids=(1 3 5 7 9)
benchmark=$1 # cmcc, ccat50, speechpref 
methods=(ditto sft)

if [ "$benchmark" == "cmcc" ]; then
    author_ids=(2 3 5 6 7 8 11 13 15 17)
    # author_ids=(11 13 15 17)
elif [ "$benchmark" == "ccat50" ]; then
    author_ids=(32 28 20 10 27 12 15 38 23 30)
elif [ "$benchmark" == "speechpref" ]; then
    author_ids=(0)
fi

N=8 # set to number of gpus available 
j=0

for method in "${methods[@]}"; do 
    for author in "${author_ids[@]}"; do 
        ((j=j%N)); ((j++==0)) && wait
        
        gpu_id=$(($j % $N))
        . generate.sh "$benchmark" "$author" "$method" "$gpu_id" &

    done; 
done

wait

echo "all generations complete"
