# usage: ./run_iota.sh <model> <method> (one of zeroshot, fewshot, iota, iota_no_exp) <dataset> (one of cmcc, ccat50, speechpref)

. ~/.bashrc
conda activate iota

# author_keys=$(seq 0 9)
model=$1
mode=$2
dataset=$3
author_keys=$4
additional_args=$5

# if there is a comma in the author_keys or it is a single value, then treat as a string of comma separated values
if [[ $author_keys == *","* ]]; then
    # process author_keys to array, where author_keys is a string of comma separated values
    IFS=',' read -r -a author_keys <<< "$author_keys"

# if there is a dash, then it is a range
elif [[ $author_keys == *"-"* ]]; then
    # process author_keys to array, where author_keys is a string of range
    IFS='-' read -r -a range <<< "$author_keys"
    author_keys=($(seq ${range[0]} ${range[1]}))

# if it is a single value, then it is an integer and convert to array
elif [[ $author_keys =~ ^[0-9]+$ ]]; then
    author_keys=($author_keys)

else
    echo "Invalid author keys"
    exit 1
fi

echo "Author keys: ${author_keys[@]}"


# Get the directory of the current script
script_dir=$(dirname "$(readlink -f "$0")")
echo "Current script directory: $script_dir"

if [ "$model" == "gpt-4o-0513" ]; then
    model_name_or_path=gpt-4o-2024-05-13
    script_name=config_openai.yaml

elif [ "$model" == "gpt-4o-0806" ]; then
    model_name_or_path=gpt-4o-2024-08-06
    script_name=config_openai.yaml

elif [ "$model" == "gpt-4o-mini" ]; then
    model_name_or_path=gpt-4o-mini-2024-07-18
    script_name=config_openai.yaml

elif [ "$model" == "gpt-3.5-turbo" ]; then
    model_name_or_path=gpt-3.5-turbo-0125
    script_name=config_openai.yaml
 
elif [ "$model" == "claude-3-sonnet" ]; then
    model_name_or_path=anthropic.claude-3-sonnet-20240229-v1:0
    script_name=config_bedrock.yaml

elif [ "$model" == "claude-3-haiku" ]; then
    model_name_or_path=anthropic.claude-3-haiku-20240307-v1:0
    script_name=config_bedrock.yaml

elif [ "$model" == "mistral7b" ]; then
    model_name_or_path=mistral.mistral-7b-instruct-v0:2
    script_name=config_bedrock.yaml

elif [ "$model" == "mixtral" ]; then 
    model_name_or_path=mistral.mixtral-8x7b-instruct-v0:1
    script_name=config_bedrock.yaml

else
    echo "Invalid model"
    exit 1
fi


# assert that both mode and dataset is provided
if [ -z "$mode" ] || [ -z "$dataset" ]; then
    echo "Usage: ./run_iota.sh <model> (one of gpt-4o, gpt-4o-mini, gpt-3.5-turbo, claude-sonnet, claude-haiku, mistral, mixtral) <mode> (one of zeroshot, fewshot, iota) <dataset> (one of cmcc, ccat50, speechpref)"
    exit 1
fi

run_iota(){

    dataset=$1
    mode=$2 
    additional_args=$3

    device_id=0

    for author_key in "${author_keys[@]}"
    do
        device_id=$((device_id%8))
        command="iota.run_iota $script_dir/../iota/configs/$script_name \
            --mode $mode \
            --model_name_or_path $model_name_or_path \
            --author_key $author_key \
            --dataset $dataset \
            --device_id $device_id \
            $additional_args &
        "

        # remove tabs and newlines from the command
        command=$(echo $command | tr -d '\n' | tr -d '\t')

        echo "Running: $command"
        eval $command

        echo "$mode with $additional_args for $dataset:$author_key is done"
        device_id=$((device_id+1))

    done
    wait
}

valid_modes=(zeroshot fewshot iota-naive iota-no-explanations opro cot)

if [[ ! " ${valid_modes[@]} " =~ " ${mode} " ]]; then
    echo "Invalid mode"

else
    run_iota "$dataset" "$mode" "$additional_args"

fi
