from flask import Flask, request, render_template, redirect, url_for, jsonify
import pandas as pd
import os
import markupsafe
from iota.eval_utils import load_dfs
from iota.utils import load_data, PACKAGE_DIR, OUTPUT_DIRECTORY
import json 
from iota.iota_trainer import IotaTrainer
from pathlib import Path 
from loguru import logger
import random 

app = Flask(__name__)
benchmarks = ["cmcc", "ccat50", "speechpref"]
benchmark_to_df_keys = {}
for benchmark in benchmarks:
    dfs = load_dfs(benchmark)
    benchmark_to_df_keys[benchmark] = list(dfs.keys())

benchmark_train_data = {
    "cmcc": load_data("cmcc", "train", "all"),
    "ccat50": load_data("ccat50", "train", "all"),
    "speechpref": load_data("speechpref", "train", "all")
}

def load_demos(benchmark, author_id): 

    random.seed(42)
    train_data = benchmark_train_data[benchmark][author_id]

    demos = random.sample(train_data, 5)

    return demos

benchmark_demos = {
    f"{benchmark}_{author_id}": load_demos(benchmark, author_id) for benchmark, author_data in benchmark_train_data.items() for author_id in author_data
}

def load_outputs2inputs(): 

    logger.info("Loading outputs2inputs")
    outputs_filepath = Path(PACKAGE_DIR) / OUTPUT_DIRECTORY / "all_results.jsonl"

    with open(outputs_filepath, "r") as f:
        outputs = [json.loads(line) for line in f]

    outputs2inputs = {}
    for output in outputs: 
        if output["generated_output"] in outputs2inputs:
            logger.error(f"Duplicate output found for {output}")
        outputs2inputs[output["generated_output"]] = output["input"]

    logger.info(f"Loaded {len(outputs2inputs)} outputs2inputs")

    return outputs2inputs

@app.route('/')
def index():

    logger.info(benchmark_to_df_keys)

    return render_template('index.html', benchmark_to_df_keys = benchmark_to_df_keys)

@app.route('/<benchmark>/<df1_name>/<df2_name>')
def get_author_id_list(benchmark, df1_name, df2_name):

    logger.info(f"Getting author id lists for {benchmark}, {df1_name}, {df2_name}")

    sorted_df_names = '_'.join(sorted([df1_name, df2_name]))
    eval_results_path = Path(PACKAGE_DIR) / OUTPUT_DIRECTORY / "eval_results" / f"head_to_head_results_gpt_{benchmark}_{sorted_df_names}.jsonl"

    if not os.path.exists(eval_results_path):
        error = f"Eval results path {eval_results_path} does not exist. This evaluation may not have been run yet."
        logger.error(error)
        return jsonify({"error": error})

    with open(eval_results_path, "r") as f:
        eval_results = [json.loads(line) for line in f]

    # compute per author win rate & total win rate
    author_win_rates = {}
    author_ids = list(set([result["author_id"] for result in eval_results]))

    total_df1_wins = 0
    total_df2_wins = 0
    for author_id in author_ids:
        author_results = [result for result in eval_results if result["author_id"] == author_id]

        df1_wins = 0 
        df2_wins = 0 
        for result in author_results: 

            if result["winner"] == "A":
                if result["method_a"] == df1_name:
                    df1_wins += 1
                else: 
                    df2_wins += 1
            else:
                if result["method_b"] == df1_name:
                    df1_wins += 1
                else: 
                    df2_wins += 1

        total_df1_wins += df1_wins
        total_df2_wins += df2_wins

        df1_win_rate = df1_wins / len(author_results)
        df2_win_rate = df2_wins / len(author_results)

        author_win_rates[author_id] = {
            df1_name: round(df1_win_rate * 100, 2),
            df2_name: round(df2_win_rate * 100,2)
        }
                        

    logger.info(author_win_rates)

    return jsonify({
        "author_ids": author_ids,
        "author_win_rates": author_win_rates,
        "total_df1_wins": total_df1_wins,
        "total_df1_win_rate": round(total_df1_wins / len(eval_results) * 100, 2),
        "total_df2_wins": total_df2_wins,
        "total_df2_win_rate": round(total_df2_wins / len(eval_results) * 100, 2)
    })

@app.route('/<benchmark>/<author_id>/<df1_name>/<df2_name>')
def get_samples(benchmark, author_id, df1_name, df2_name):

    logger.info(f"Getting sample for {benchmark}, {author_id}, {df1_name}, {df2_name}")

    sorted_df_names = '_'.join(sorted([df1_name, df2_name]))

    eval_results_path = Path(PACKAGE_DIR) / OUTPUT_DIRECTORY / "eval_results" / f"head_to_head_results_gpt_{benchmark}_{sorted_df_names}.jsonl"

    if not os.path.exists(eval_results_path):
        error = f"Eval results path {eval_results_path} does not exist. This evaluation may not have been run yet."
        logger.error(error)
        return jsonify({"error": error})

    with open(eval_results_path, "r") as f:
        eval_results = [json.loads(line) for line in f]

    author_results = [result for result in eval_results if result["author_id"] == int(author_id)]

    outputs2inputs = load_outputs2inputs()

    df1_name_input = ""
    df2_name_input = ""
    for result in author_results:
        result["input_a"] = outputs2inputs.get(result["text_a"], result["task"])
        result["input_b"] = outputs2inputs.get(result["text_b"], result["task"])

        if result["method_a"] == df1_name:
            if df1_name_input == "":
                df1_name_input = result["input_a"]
                df2_name_input = result["input_b"]

    # compute overall win rates for each df name 
    df1_wins = [] 
    df2_wins = [] 
    for result in author_results: 

        if result["winner"] == "A":
            if result["method_a"] == df1_name:
                df1_wins.append(1)
            else: 
                df2_wins.append(1)
        else:
            if result["method_b"] == df1_name:
                df1_wins.append(1)
            else: 
                df2_wins.append(1)

    # sort results such that df1 wins are shown first
    author_results = sorted(author_results, key=lambda x: (x["winner"] == "A" and x["method_a"] == df1_name) or (x["winner"] == "B" and x["method_b"] == df1_name), reverse=True)

    df1_win_rate = sum(df1_wins) / len(author_results)
    df2_win_rate = sum(df2_wins) / len(author_results)

    logger.info(f"Loaded eval results from {eval_results_path} for author {author_id}: {len(author_results)} samples")

    data = {
        "author_results": author_results, 
        "demos":  benchmark_demos[f"{benchmark}_{author_id}"],
        "last_index": len(author_results) - 1, 
        "df1_win_rate": round(df1_win_rate * 100, 2),
        "df2_win_rate": round(df2_win_rate * 100,2),
        "df1_name_input": df1_name_input,
        "df2_name_input": df2_name_input
    }

    return data 

if __name__ == '__main__':
    app.run(port=8198)