from peft import PeftConfig, PeftModel
import torch
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
import json
import pickle
import pdb
import pandas as pd
import argparse
from iota.prompts import form_few_shot_prompt

MISTRAL_CHAT_TEMPLATE = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

def prepare_few_shot_prompt(prompt, train_data_path, author_key):
    with open(train_data_path, 'rb') as pickle_file:
        train_data = pickle.load(pickle_file)

    # form tuples with examples as (task, output)
    few_shot_examples = []
    for item in train_data[int(author_key)]: 
        few_shot_examples.append((item["prompt"], item["output"]))

    few_shot_prompt = form_few_shot_prompt(prompt, few_shot_examples)

    return few_shot_prompt

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
        "-n", "--num_samples", type=int, default=3, required=False, help="Number of samples to generate"
    )
    parser.add_argument(
        "-m", "--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", required=False, help="Model ID to use"
    )
    parser.add_argument(
        "--method", type=str, default="ditto", required=False, help="Method to use. One of [sft, ditto, zero, few]"
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

    outputs = [] 

    for task in tasks:
        
        outs = generator(
            task, 
            max_new_tokens=1024, do_sample=True, 
            temperature=1,
            num_return_sequences=args.num_samples,
            return_full_text=False
        )

        
        for out in outs:
            print(f"SAMPLE:\n\t{out['generated_text']}", "\n\n")

            outputs.append({
                "response": out["generated_text"].strip(),
                "input": task[0]["content"],
                "model": model_id,
                "method": args.method,
                "benchmark": args.benchmark,
                "author_id": args.train_author_key
            })

    outputs += reference_outputs

    with open(f"./outputs/{args.benchmark}-mistral-7b-instruct-ditto_author{args.train_author_key}/generated.json", "w") as f:
        json.dump(outputs, f, indent=4)

    # append to all outputs with benchmark and author info 
    with open(f"./outputs/all_outputs.json", "a") as f:
        for out in outputs:
            json.dump(out, f)
            f.write("\n")

    # remove duplicates using pd 
    df = pd.read_json(f"./outputs/all_outputs.json", lines=True)
    df.drop_duplicates(inplace=True)
    df.to_json(f"./outputs/all_outputs.json", orient="records", lines=True)

if __name__ == "__main__":
    main()
    
