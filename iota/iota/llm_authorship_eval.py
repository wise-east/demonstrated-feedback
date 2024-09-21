"""
Compare outputs from different approaches on the same tasks and authors using gpt-4 eval

To see which setups can be compared, run iota.llm_eval -h 
Usage: iota.llm_eval --df1 gpt-4o-0806_zeroshot --df2 ditto -b cmcc -m gpt -e -a 0-9

For testing, use the -t flag to use the gpt-4o-mini model.

For batch prediction inputs, use the --make_batch_prediction_inputs flag. This will create a file with the batch prediction inputs that can be used to make predictions in batch.
e.g., iota.llm_eval --df1 gpt-4o-0806_zeroshot --df2 ditto -b cmcc -m gpt -e -a 0-9 -mb

then, to load the batch prediction outputs and update the all_eval_results, use the --load_batch_prediction_outputs flag
e.g., iota.llm_eval --df1 gpt-4o-0806_zeroshot --df2 ditto -b cmcc -m gpt -e -a 0-9 -lb

ditto outputs are within the demonstrated-feedback repo in outputs/all_outputs.json
all other outputs, including iota outputs, are within the GREASE-IOTA repo in out/all_results.jsonl
"""

import json
import os
from pathlib import Path
import pandas as pd
from iota.utils import load_data, extract_answer_and_explanation
from iota.eval_utils import HOME_DIR, load_dfs
from iota.batch_openai_utils import load_batch_results_with_batch_id, get_batch_status, submit_batch_openai_batch_predictions
from iota.generation_models import OpenAIModel, BedRockModel
import random
from tqdm import tqdm
from argparse import ArgumentParser
from typing import List, Dict, Any
from loguru import logger
import traceback
from scipy.stats import bootstrap
import numpy as np
from pprint import pprint
from iota.prompts import form_pairwise_eval_prompt
import tiktoken 
from collections import defaultdict

enc = tiktoken.encoding_for_model("gpt-4o")

cost_per_input_token = 2.5 / 1e6 # $2.5 per million tokens 
cost_per_output_token = 10 / 1e6 # $10 per million tokens
estimated_output_tokens = 384 # just rough estimate including explanations, could get better estimate from actual data. 


def estimate_per_sample_eval_cost(demos, text_a, text_b, generate_explanation:bool=True) -> float: 

    eval_prompt = form_pairwise_eval_prompt(demos, text_a=text_a, text_b=text_b, generate_explanation=generate_explanation)
    tokenized_prompt = enc.encode(eval_prompt)
    input_token_length = len(tokenized_prompt)

    estimated_cost = input_token_length * cost_per_input_token + estimated_output_tokens * cost_per_output_token

    return estimated_cost

def estimate_total_eval_cost(eval_data: pd.DataFrame, train_data: Dict[str, Any], yes:bool=False) -> float:

    author_ids = eval_data.author_id.unique()

    num_comparisons = 0 
    total_estimated_cost = 0
    for author_id in author_ids:

        # load demos if not preloaded into eval_data 
        demos = None 
        if "demos" not in eval_data.columns:
            random.seed(42)
            demos = random.sample(train_data[author_id], 5)
            demos = [d["output"] for d in demos]

        author_df = eval_data[eval_data["author_id"] == author_id]

        for idx, sample in author_df.iterrows():
            if "winner" in sample and sample["winner"] is not None:
                continue 
            # use preloaded demos if available
            if "demos" in sample: 
                demos = sample["demos"]
            num_comparisons += 1 
            text_a = sample["text_a"]
            text_b = sample["text_b"]
            estimated_cost = estimate_per_sample_eval_cost(demos, text_a, text_b, generate_explanation=True)

            total_estimated_cost += estimated_cost 

    if num_comparisons > 0:

        logger.info(f"Estimated cost for this eval: ${total_estimated_cost:.2f}")
        logger.info(f"Number of comparisons: {num_comparisons}")
        logger.info(f"Estimated cost per comparison: ${total_estimated_cost / num_comparisons:.2f}")
        logger.info(f"If batch prediction inputs are made, the cost will be reduced by 2x: ${total_estimated_cost / 2:.2f}")

        if yes:
            should_continue = True
        else:
            should_continue = input("Continue with this eval? (y/n): ")
            should_continue = should_continue.lower() == "y"
    else: 
        should_continue = True 

    return total_estimated_cost, should_continue

def prepare_eval_data(merged_df: pd.DataFrame, author_id: int, df1_name:str, df2_name:str, sample_size: int = -1): 
    """
    Prepare data for head to head evaluation from a merged df between two dataframes, where model responses that share the same task and author_id are merged.
    """

    author_df = merged_df[merged_df["author_id"] == author_id]

    if sample_size != -1:
        sample_size = min(sample_size, len(author_df))
        author_df = author_df.sample(sample_size)

    eval_data = []

    # form eval data
    for _, row in author_df.iterrows():
        df1_output = row[f"generated_output_{df1_name}"]
        df2_output = row[f"generated_output_{df2_name}"]

        eval_sample = {
            "text_a": df1_output,
            "text_b": df2_output,
            "method_a": df1_name,
            "method_b": df2_name,
            "task": row["task"],
            "author_id": row["author_id"],
            "winner": None,
            "explanation": None
        }

        eval_sample_reversed = {
            "text_a": df2_output,
            "text_b": df1_output,
            "method_a": df2_name,
            "method_b": df1_name,
            "task": row["task"],
            "author_id": row["author_id"],
            "winner": None,
            "explanation": None
        }


        eval_data.append(eval_sample) 
        eval_data.append(eval_sample_reversed)

    return eval_data


def parse_author_ids_arg(author_ids: str) -> List[int]:

    if author_ids is None:
        return None

    if "," in author_ids:
        author_ids = author_ids.split(",")
        author_ids = [int(author_id) for author_id in author_ids]
    elif "-" in author_ids:
        start, end = author_ids.split("-")
        author_ids = list(range(int(start), int(end) + 1))
    else:
        author_ids = [int(author_ids)]

    return author_ids

def main():

    # load dataframes
    cmcc_dfs_dict = load_dfs(benchmark="cmcc")
    ccat50_dfs_dict = load_dfs(benchmark="ccat50")
    speechpref_dfs_dict = load_dfs(benchmark="speechpref")

    keys = {
        "cmcc": sorted(cmcc_dfs_dict.keys()),
        "ccat50": sorted(ccat50_dfs_dict.keys()),
        "speechpref": sorted(speechpref_dfs_dict.keys()),
    }
    pprint(keys)

    parser = ArgumentParser()
    parser.add_argument("--df1", type=str, required=True, help=f"One of {keys}")
    parser.add_argument(
        "--df2", type=str, required=True, help=f"One of the same keys as above"
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        default="cmcc",
        help="One of {cmcc, ccat50, speechpref}",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="gpt",
        help="Model name. One of [gpt, claude]",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite of existing results",
    )
    parser.add_argument(
        "-a", 
        "--author_ids",
        type=str, 
        default=None,
        help="Author ids to evaluate, either single value, range, or comma separated list"
    )

    parser.add_argument(
        "-c",
        "--max_comps_per_author",
        type=int,
        default=40,
        help="Maximum number of comparisons per author",
    )

    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="use cheaper model for testing",
    )

    parser.add_argument(
        "-mb", 
        "--make_batch_prediction_inputs", 
        action="store_true",
        help="Make batch prediction inputs to reduce api costs"
    )

    parser.add_argument(
        "-lb", 
        "--load_batch_prediction_outputs", 
        action="store_true",
        help="Load batch prediction outputs to update all_eval_results"
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )

    parser.add_argument("-e", "--generate_explanations", action="store_true")
    args = parser.parse_args()

    author_ids = parse_author_ids_arg(args.author_ids)
    if author_ids is None:
        author_ids = sorted([int(author_id) for author_id in author_ids])

    max_comps_per_author = args.max_comps_per_author
    logger.info(f"Limited to {max_comps_per_author} comparisons per author, for author_ids: {author_ids}")

    # set model_name_or_path based on alias 
    if args.model_name == "gpt":
        model_name_or_path = "gpt-4o-2024-08-06"  # use latest model
        if args.test:
            model_name_or_path = "gpt-4o-mini-2024-07-18"
            args.model_name = "gpt-mini"
        model = OpenAIModel(model_name_or_path)
    elif args.model_name == "claude":
        model_name_or_path = "anthropic.claude-3-sonnet-20240229-v1:0"
        model = BedRockModel(model_name_or_path)

    dfs_dict = load_dfs(benchmark=args.benchmark)

    train_data = load_data(args.benchmark, "train", "all")
    test_data = load_data(args.benchmark, "test", "all")

    # get the two dataframes to compare 
    df1 = dfs_dict[args.df1]
    df2 = dfs_dict[args.df2]

    # merge the two dataframes on benchmark, task, and author_id
    merged_df = pd.merge(
        df1,
        df2,
        on=["task", "author_id", "benchmark"],
        suffixes=(f"_{args.df1}", f"_{args.df2}"),
        how="inner",
    )

    # get statistics of merged_df, showing expected comparisons per author and how many unique tasks are being evaluated
    # author_ids = merged_df["author_id"].unique()
    num_comps = 0
    for author_id in author_ids:
        author_df = merged_df[merged_df["author_id"] == author_id]
        print(
            f"Author {author_id} has {len(author_df)} comparisons ({len(df1[df1.author_id == author_id])} from {args.df1} and {len(df2[df2.author_id==author_id])} from {args.df2}), with {len(author_df['task'].unique())} unique tasks"
        )

        tasks = author_df["task"].unique()

        # check that the test_data author ids share the same task as the author ids in the dfs
        test_tasks = [d["prompt"] for d in test_data[author_id]]

        try:
            assert set(tasks) == set(
                test_tasks
            ), f"Author {author_id} tasks do not match test data tasks: {tasks} vs {test_tasks}"

        except Exception as e:
            print(f"Assertion Error: {e}")
            traceback.print_exc()
            breakpoint()

        num_comps += min(max_comps_per_author, len(author_df))

    # create eval directory and see if results already exist 
    sorted_df_names = sorted([str(args.df1), str(args.df2)])
    sorted_df_name_string = "_".join(sorted_df_names)

    # save comparison results
    results_dir = Path(HOME_DIR) / f"project/GREASE-IOTA/out/eval_results"
    os.makedirs(results_dir, exist_ok=True)

    results_path = (
        results_dir / f"head_to_head_results_{args.model_name}_{args.benchmark}_{sorted_df_name_string}.jsonl"
    )

    if results_path.exists() and not args.force:
        logger.info(f"Results already exist at: {results_path}. If you want to overwrite, use the -f flag")
        # return
    
        all_eval_results = pd.read_json(results_path, lines=True, orient="records")

    else: 
        logger.info("Preparing eval data")
        all_eval_results = pd.DataFrame()


    # prepare eval data
    for author_id in author_ids:
        # continue if we already have eval data prepared for this author 
        logger.info(f"Preparing eval data for author {author_id}")
        if len(all_eval_results) \
            and author_id in all_eval_results["author_id"].unique() \
                and len(all_eval_results[all_eval_results["author_id"] == author_id]) >= max_comps_per_author: 
            logger.info(f"Already have eval data for author {author_id}, skipping")
            continue 
            
        else: 
            if len(all_eval_results) and author_id in all_eval_results["author_id"].unique():
                n_existing_comps = len(all_eval_results[all_eval_results["author_id"] == author_id])
            else:
                n_existing_comps = 0
            n_comps_needed = max(max_comps_per_author - n_existing_comps, 0) 
            logger.info(f"Preparing {n_comps_needed} eval data for author {author_id} from scratch")
            eval_data = prepare_eval_data(merged_df, author_id, args.df1, args.df2)

            # sample eval data to maximum of MAX_COMPS_PER_AUTHOR
            random.seed(42)
            random.shuffle(eval_data)
            eval_data = eval_data[n_existing_comps:n_existing_comps + n_comps_needed]
            all_eval_results = pd.concat([all_eval_results, pd.DataFrame(eval_data)])

    estimated_eval_cost, should_continue = estimate_total_eval_cost(all_eval_results, train_data, args.yes)
    if not args.load_batch_prediction_outputs and not should_continue:
        logger.info("Exiting eval")
        return

    # make or load predictions 
    batch_prediction_inputs = []
    batch_prediction_inputs_path = results_dir / f"batch_prediction_inputs_{results_path.name}"

    for author_id in author_ids: 
    
        eval_data = all_eval_results[all_eval_results["author_id"] == author_id]
        random.seed(42)
        demos = random.sample(train_data[author_id], 5)
        demos = [d["output"] for d in demos]

        # evaluate eval data if winner is not already present. iterate through df
        for idx, sample in tqdm(eval_data.iterrows(), total=len(eval_data)):

            # make batch prediction inputs
            if args.make_batch_prediction_inputs:
                batch_input = model.prepare_batch_prediction_input(
                    sample["text_a"],
                    sample["text_b"],
                    demos,
                    generate_explanation=args.generate_explanations,
                    custom_id = results_path.name + f"_{author_id}_{idx}"
                )
                batch_prediction_inputs.append(batch_input)
                continue 

            if args.load_batch_prediction_outputs:
                continue

            if "winner" not in sample or sample["winner"] is None: 
                answer, explanation = model.eval_head_to_head(
                    sample["text_a"],
                    sample["text_b"],
                    demos,
                    generate_explanation=args.generate_explanations,
                )

                eval_data.loc[idx, "winner"] = answer
                eval_data.loc[idx, "explanation"] = str(explanation)

        # update saved data 
        # breakpoint()
        all_eval_results[all_eval_results["author_id"] == author_id] = eval_data
        all_eval_results.to_json(results_path, lines=True, orient="records")

    if args.load_batch_prediction_outputs:
        logger.info("Loading batch prediction outputs to update all_eval_results")
        # check that the file exists
        if not batch_prediction_inputs_path.exists():
            logger.error(f"Batch prediction inputs not found at: {batch_prediction_inputs_path}")
            logger.info("Use the --make_batch_prediction_inputs flag to create batch prediction inputs first")
            return

        # get batch id stroed in batch_prediction_inputs_path
        with open(batch_prediction_inputs_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]

        batch_id = data[-1].get("batch_id")
        if batch_id is None:
            logger.error(f"Batch id not found in {batch_prediction_inputs_path}")
            return

        # check status 
        status = get_batch_status(batch_id)
        logger.info(f"Batch status: {status.status} with {status.request_counts}")

        results = load_batch_results_with_batch_id(batch_id)

        for result in results: 
            response_text = result['response']['body']['choices'][0]['message']['content']
            answer, explanation = extract_answer_and_explanation(response_text, generate_explanation=args.generate_explanations)

            # parse the custom id to get author_id and idx
            custom_id = result["custom_id"]
            author_id = int(custom_id.split("_")[-2])
            idx = int(custom_id.split("_")[-1])
            try: 
                # keep original indices 
                original_indices = all_eval_results[all_eval_results["author_id"] == author_id].index

                # if any of the original indices >=20, no need to reset index 
                should_update_index = idx < 20

                eval_data = all_eval_results[all_eval_results["author_id"] == author_id]
                if should_update_index: 
                    eval_data= eval_data.reset_index(drop=True)
                eval_data.loc[idx, "winner"] = answer
                eval_data.loc[idx, "explanation"] = str(explanation)
            
                if should_update_index:
                    # update to original indices
                    eval_data = eval_data.set_index(original_indices)        
                all_eval_results[all_eval_results["author_id"] == author_id] = eval_data
            
            except Exception as e: 
                logger.error(f"Failed to update eval data for author {author_id} with idx {idx}")
                logger.error(e)
                breakpoint()
                continue 


        all_eval_results.to_json(results_path, lines=True, orient="records")

    if args.make_batch_prediction_inputs:

        # check that the file does not already exist
        if batch_prediction_inputs_path.exists():
            logger.error(f"Batch prediction inputs already exist at: {batch_prediction_inputs_path}. If submitted, use the --load_batch_prediction_outputs flag to load submitted batch results and update all_eval_results.")
            return

        with open(batch_prediction_inputs_path, "w") as f:
            for batch_input in batch_prediction_inputs:
                f.write(json.dumps(batch_input) + "\n")

        logger.info(f"Saved batch prediction inputs to: {batch_prediction_inputs_path}")
        submit_batch_openai_batch_predictions(batch_prediction_inputs_path)

        return


    total_win_counts = defaultdict(dict)

    # compute results 
    for author_id in author_ids:

        eval_data = all_eval_results[all_eval_results["author_id"] == author_id]
        win_counts = {args.df1: 0, args.df2: 0}

        for idx, sample in eval_data.iterrows():
            if sample["winner"] == "A":
                win_counts[sample["method_a"]] += 1
            elif sample["winner"] == "B":
                win_counts[sample["method_b"]] += 1

        print(f"Author {author_id} results: ", end="")
        print(win_counts)
        total_win_counts[author_id] = win_counts


    # print results for table 
    print(f"{args.df1} vs {args.df2} results: ({args.df1} wins over {args.df2})")
    for author_id in author_ids:

        win_counts = total_win_counts[author_id]
        # print iota win rate
        print(
            f"{win_counts[args.df1] / (win_counts[args.df1] + win_counts[args.df2]+ 1e-10) * 100:.2f}",
            end=", ",
        )

        df1_wins = [1] * win_counts[args.df1] + [0] * win_counts[args.df2]
        df1_wins = (df1_wins,)
        # bootstrap_ci = bootstrap(
        #     df1_wins,
        #     np.mean,
        #     confidence_level=0.95,
        #     random_state=1,
        #     method="percentile",
        # )
        # print(f"{args.df1} win rate CI: {bootstrap_ci.confidence_interval}, standard error: {bootstrap_ci.standard_error}")
        # print(f"+- {bootstrap_ci.standard_error*100:.2f}%")



    print(f"\n\nTotal results: ")
    df1_total_wins = 0 
    df2_total_wins = 0
    for author_id in author_ids:
        win_counts = total_win_counts[author_id]
        df1_total_wins += win_counts[args.df1]
        df2_total_wins += win_counts[args.df2]

    print(f"{args.df1} total win rate: {df1_total_wins / (df1_total_wins + df2_total_wins+1e-10) * 100:.2f}%")
    df1_wins = [1] * df1_total_wins + [0] * df2_total_wins
    df1_wins = (df1_wins,)
    bootstrap_ci = bootstrap(
        df1_wins,
        np.mean,
        confidence_level=0.95,
        random_state=1,
        method="percentile",
    )

    print(f"{args.df1} win rate CI: {bootstrap_ci.confidence_interval}, standard error: {bootstrap_ci.standard_error}")

    # TODO print out in a way that's easy for copy-pasting into a table


if __name__ == "__main__":
    main()
