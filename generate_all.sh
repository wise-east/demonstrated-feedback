# running all outputs with generate.sh 
# . generate.sh speechpref 0 ditto 1 -> generate for speechpref baseline for author 0 using model with ditto method on device 1


# methods=(ditto sft zero few)
methods=(ditto)

benchmark="cmcc" # cmcc, ccat50, speechpref 

authors=(0 1 2 3 4)

N=8 # set to number of gpus available 

for method in ${methods[@]}; do 
    for author in ${authors[@]}; do 
        ((j=j%N)); ((j++==0)) && wait
        
        gpu_id=$(($j % $N))
        . generate.sh $benchmark $author $method $gpu_id &

    done; 
done

wait

echo "all generations complete"
