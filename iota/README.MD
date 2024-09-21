# Inference-only Task Alignment (IOTA) 

Updated name: Stepping Stones to Success: Low-Resource Personalized Alignment with
In-Context Trial and Error

[Project document: Quip](https://quip-amazon.com/uQVAAOHI5bQT/Justin-Cho-Summer-2024-Project-Doc-Training-free-Instruction-following-with-Demonstrated-Feedback-and-Preference-Explanations)

## Setup 
 
#### Create conda environment
```shell 
conda create -n iota python=3.10
conda activate iota
pip install -r requirements.txt
```

#### Set OpenAI API key
```shell 
export OPENAI_API_KEY=your-api-key
```

## Running zeroshot, few-shot, and IOTA

```shell
iota.run_iota iota/configs/config_openai.yml # openai models 
iota.run_iota iota/configs/config_mistral.yml # mistral 

```

You may add additional arguments to the command line to override the parameters in the config file. For example, to train the model with a different openai model: 

```shell
iota.run_iota iota/configs/config_openai.yml --model_name_or_path gpt-4o-mini-2024-07-18
```

These commands will run the training (if any) and generate predictions for the test set. The predictions and the models will be saved into individual directories in `output`, while the generated outputs will also be added to `all_results.jsonl`. The outputs stored here will be loaded for `iota.llm_eval`. 

`scritps/run_iota.sh` is a convenience script for running IOTA with some default configurations that are frequently used. 
`scripts/iota_commands.sh` contains the commands that use `run_iota.sh` to run the experiments in the paper.

## Running evaluation 

```shell 
iota.llm_eval --df1 <option 1 alias> --df2 <option 2 alias> -b <benchmark> -a <author_ids> [-e: generate explanations before making predictions]   
```

You can run `iota.llm_eval --help` to see the full list of options to use for df1 and df2 and arguments. The alias is configured based on the run config, and its logic can be found in `get_run_alias` in `iota/eval_utils.py`. 

## Configs

Most parameters in the configs file is self-explanatory or further explained in `IotaArguments`. Note that the `package_dir` parameter in the config file should be set to the directory of the IOTA codebase, relative to the home directory (i.e. output of `echo $HOME`). If this is not set, it will defaul


## Data 

The data used for training is in `benchmarks`. The basic format of the training data is: 
```python
data_test = {
    0: [ # author id, this is optional.
        {
            "prompt": "<input text>",
            "output": "<output text>"
        },
    ]
}
```

Refer to `proc_{benchmark}` script of each benchmark for further details on how the data is formatted for IOTA. 

## Test 

```shell
python iota/run_iota.py --test --small
```

This will limit the training set to two examples and show verbose logs that can be used to debug and understand the training procedure.

```shell
pytest -s
```

This will run the test suite for the IOTA codebase and show any print statements.


## Commands for results 

#### Human vs LLM benchmarking 

```shell
python tests/benchmark_llm_eval.py --model gpt -hu
python tests/benchmark_llm_eval.py --model claude -llm 
```

#### Human vs Human accuracy 
```shell
python tests/benchmark_llm_eval.py --model gpt -s 42 -nd 5 -ns 50 -e
```