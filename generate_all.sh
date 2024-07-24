# running all outputs with generate.sh 
# . generate.sh speechpref 0 ditto 1 -> generate for speechpref baseline for author 0 using model with ditto method on device 1


. generate.sh speechpref 0 ditto 0
. generate.sh speechpref 0 sft 1 
. generate.sh speechpref 0 zero 2
. generate.sh speechpref 0 few 3

methods=(ditto sft zero few)

for method in ${methods[@]}; do
    for author in {0..9}; do
        . generate.sh speechpref $author $method 0
    done    
done

methods=(ditto sft zero few); for method in ${methods[@]}; do for author in {0..9}; do . generate.sh ccat50 $author $method 0; done; done
methods=(ditto sft zero few); for method in ${methods[@]}; do for author in {0..9}; do . generate.sh cmcc $author $method 1; done; done