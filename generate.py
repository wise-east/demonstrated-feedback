from peft import PeftConfig, PeftModel
import torch
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
import json
import pickle
import pdb
import pandas as pd
import argparse
from typing import List, Tuple
from loguru import logger

MISTRAL_CHAT_TEMPLATE = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

def form_few_shot_prompt(target_task:str, examples: List[Tuple[str,str]]) -> str: 

    system_prompt = "Perform the target task while using the following examples as stylistic guides."

    few_shot_example_texts_list = [] 
    for example in examples: 
        few_shot_example_texts_list.append(f"### Example\nTask: {example[0]}\nOutput:\n{example[1]}") 

    few_shot_examples_text = "\n\n".join(few_shot_example_texts_list)

    full_prompt = f"{system_prompt}\n\n{few_shot_examples_text}\n\nTarget Task: {target_task}"

    return full_prompt

def prepare_few_shot_prompt(prompt, train_data_path, author_key):
    with open(train_data_path, 'rb') as pickle_file:
        train_data = pickle.load(pickle_file)

    # form tuples with examples as (task, output)
    few_shot_examples = []
    for item in train_data[int(author_key)]: 
        few_shot_examples.append((item["prompt"], item["output"]))

    few_shot_prompt = form_few_shot_prompt(prompt, few_shot_examples)

    return few_shot_prompt

def is_problematic_output(output: str) -> bool:

    # strings that should not be in the output
    problematic_strings = [
        "[]",
        "[INST]",
        "[/INST]",
        "INST",
        "\INST"
    ]

    if any([string in output for string in problematic_strings]):
        return True

    # check for overly repetitive text 
    if len(set(output.split())) <= 0.3 * len(output.split()): 
        return True

    # check for text that doesn't make any sense 
    # TODO: tried with NLL computation vs author text but doesn't show clear distinction, gibberish text gets lower NLL. See tests/test_lm_probs.py in IOTA repo

    return False

def post_processing(fp): 
    """
    Rename the target task in the input to `target_task` for easier processing when comparing with IOTA results
    """

    with open(fp, "r") as f: 
        data = [json.loads(line) for line in f]

    for d in data: 
        if "Target Task: " in d['input']: 
            target_task = d['input'].split("Target Task: ")[1].strip()

            d['target_task'] = target_task
        else: 
            d['target_task'] = d['input']


    with open(fp, "w") as f: 
        for d in data: 
            json.dump(d, f)
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="gen script")

    # Add arguments
    parser.add_argument(
        "-b", "--benchmark", type=str, help="Name of benchmark dataset"
    )
    parser.add_argument(
        "-t", "--train_author_key", type=str, help="Author key in pkl file"
    )
    parser.add_argument(
        "-n", "--num_samples", type=int, default=5, required=False, help="Number of samples to generate"
    )
    parser.add_argument(
        "-m", "--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", required=False, help="Model ID to use"
    )
    parser.add_argument(
        "--method", type=str, default="ditto", required=False, help="Method to use. One of [sft, ditto, zero, few]"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5, required=False, help="Temperature for sampling"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024, required=False, help="Max number of tokens to generate"
    )

    # Execute the parse_args() method
    args = parser.parse_args()

    model_id = args.model_id
    
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
     
    if args.method in ["sft", "ditto"]: 
        if args.method == "ditto": 
            peft_model_path =  f"./outputs/{args.benchmark}-mistral-7b-instruct-ditto_author{args.train_author_key}/{args.method}"
        else: 
            peft_model_path =  f"./outputs/{args.benchmark}-mistral-7b-instruct-ditto_author{args.train_author_key}/checkpoint-40/{args.method}"

        base_model = PeftModel.from_pretrained(
            base_model, peft_model_path
        )
    elif args.method in ["zero", "few"]: 
        # nothing to do for zero-shot and few-shot 
        pass
    else: 
        raise NotImplementedError(f"Method `{args.method}` not implemented.")
    
    base_model.eval()

    generator = pipeline(
        "text-generation",
        model=base_model,
        device="cuda",
        tokenizer=tokenizer
    )
    
    generator.tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

    path = f"./benchmarks/{args.benchmark}/processed/{args.benchmark}_test.pkl"

    train_path = f"./benchmarks/{args.benchmark}/processed/{args.benchmark}_train.pkl"
    
    with open(path, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    
    spec_dataset = data[int(args.train_author_key)]    

    tasks = []
    reference_outputs = [] 
    outputs = [] 
    
    for item in spec_dataset:
        tasks.append([
            {
                "content": item["prompt"] if args.method != "few" else prepare_few_shot_prompt(item["prompt"], train_path, args.train_author_key),
                "role": "user"
            }
        ])

        reference_outputs.append({
            "response": item["output"],
            "input": item["prompt"],
            "model": "author",
            "method": "author",
            "benchmark": args.benchmark,
            "author_id": args.train_author_key
        })

    for task in tasks:

        task_outputs =[] 
        
        tried = 0 
        patience = 5 
        while len(task_outputs) < args.num_samples:
            outs = generator(
                task, 
                max_new_tokens=args.max_new_tokens, do_sample=True, 
                temperature=args.temperature,
                top_p=0.95,
                num_return_sequences=args.num_samples*2,
                return_full_text=False
            )
            
            for out in outs:
        
                if is_problematic_output(out["generated_text"]) and tried < patience:
                    logger.warning(f"Problematic output detected: \n>>{out['generated_text']}\n\n >>Skipping... (currently at {len(task_outputs)}/{args.num_samples} samples)")
                    continue

                task_outputs.append({
                    "response": out["generated_text"].strip(),
                    "input": task[0]["content"],
                    "model": model_id,
                    "method": args.method,
                    "benchmark": args.benchmark,
                    "author_id": args.train_author_key
                })

                if len(task_outputs) >= args.num_samples:
                    logger.info(f"Generated {args.num_samples} samples for task: {task[0]['content']}")
                    break

            tried += 1

        outputs += task_outputs
    
    if len(outputs) == args.num_samples * len(tasks): 
        logger.error(f"Generated {len(outputs)} samples instead of {args.num_samples * len(tasks)}")

    outputs += reference_outputs

    with open(f"./outputs/{args.benchmark}-mistral-7b-instruct-ditto_author{args.train_author_key}/generated.json", "w") as f:
        json.dump(outputs, f, indent=4)

    all_outputs_path = "./outputs/all_outputs.json"

    # append to all outputs with benchmark and author info 
    with open(all_outputs_path, "a") as f:
        for out in outputs:
            json.dump(out, f)
            f.write("\n")

    # remove duplicates using pd 
    df = pd.read_json(all_outputs_path, lines=True)
    df.drop_duplicates(inplace=True)

    # keep only the first X samples when grouped by input, model, method, benchamark, and author_id
    df = df.groupby(["input", "model", "method", "benchmark", "author_id"]).head(args.num_samples)  
    df.to_json(all_outputs_path, orient="records", lines=True)
    
    # post-processing to extract and rename target task
    post_processing(all_outputs_path)

if __name__ == "__main__":
    main()
    
