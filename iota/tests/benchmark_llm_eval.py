# benchmark gpt4's performance on determining relative style consistency of two given texts compared to another text
# usage: python benchmark_llm_eval.py -ns 40 -nd 5 -e -s 42 -m prometheus
# for prometheus to do multi gpu: export VLLM_WORKER_MULTIPROC_METHOD=spawn ;  python benchmark_llm_eval.py -ns 40 -nd 3 -e -s 42 -m prometheus. reference: https://github.com/vllm-project/vllm/issues/6152

from iota.iota_trainer import IotaTrainer
from iota.utils import load_data, PACKAGE_DIR, DEFAULT_DATA_DIR, extract_answer_and_explanation
from iota.generation_models import OpenAIModel, BedRockModel, PrometheusModel
from iota.llm_authorship_eval import estimate_total_eval_cost
from iota.batch_openai_utils import get_batch_status, submit_batch_openai_batch_predictions, load_batch_results_with_batch_id
from iota.prompts import get_dummy_eval_prompt
from loguru import logger
import os
import yaml
from tqdm import tqdm
import pandas as pd
import json
from typing import List, Dict, Optional
import random
import traceback
import hashlib
from argparse import ArgumentParser
from scipy.stats import bootstrap
import numpy as np
from collections import defaultdict
from pathlib import Path
import math
from sklearn.feature_extraction.text import TfidfVectorizer

"""
data format for comparison: 

data = {
    "demo": "",
    "text_a": "", 
    "text_b": "",
    "answer": "A/B"
}
"""
def ask_should_continue(eval_samples, train_data): 
    if not eval_samples:
        return True

    # transform to df and rename target to author_id for estimate_total_eval_cost
    df_eval_samples = pd.DataFrame(eval_samples)
    df_eval_samples = df_eval_samples.rename(columns={"target": "author_id"})
    estimated_cost, should_continue = estimate_total_eval_cost(df_eval_samples, train_data)

    return should_continue

def compute_accuracy(eval_samples: List[Dict[str, str]]) -> float:
    # per author accuracy 
    accuracy_per_author = defaultdict(list)
    for sample in eval_samples:
        author_id = sample.get("target", sample["author_id"])
        is_correct = sample["winner"] == sample["answer"]
        accuracy_per_author[author_id].append(is_correct)
        if sample.get("note") is not None:
            if is_correct:
                logger.info(f"Correct prediction for {sample['author_id_a']} vs {sample['author_id_b']} on benchmark {sample['benchmark']} author id {sample['author_id']} with note: {sample['note']}.")
            else: 
                logger.info(f"Incorrect prediction for {sample['author_id_a']} vs {sample['author_id_b']} on benchmark {sample['benchmark']} author id {sample['author_id']} with note: {sample['note']}. Expected answer is {sample['answer']} but got {sample['winner']}.")

    # calculate accuracy
    for author_id in accuracy_per_author: 
        corrects = accuracy_per_author[author_id]
        correct = sum(corrects)
        accuracy = correct / len(corrects)
        logger.info(f"Author {author_id} accuracy: {accuracy*100}% ({correct}/{len(corrects)})")

    # calculate overall accuracy and confidence interval
    corrects = [correct for corrects in accuracy_per_author.values() for correct in corrects]
    correct = sum(corrects)
    accuracy = correct / len(corrects)
    logger.info(f"Overall accuracy: {accuracy*100:.2f}%")

    corrects = (corrects,)
    bootstrap_ci = bootstrap(
        corrects,
        np.mean,
        confidence_level=0.95,
        random_state=1,
        method="percentile",
    )

    logger.info(f"Overall confidence interval: {bootstrap_ci.confidence_interval}")

    return accuracy

def parse_batch_results(eval_samples, results, generate_explanations=False): 

    custom_id_to_answer = {} 
    if eval_samples[0].get("custom_id") is not None:
        for result in results: 
            response_text = result['response']['body']['choices'][0]['message']['content']
            answer, explanation = extract_answer_and_explanation(response_text, generate_explanation=generate_explanations)
            custom_id = result['custom_id']
            custom_id_to_answer[custom_id] = answer

        for eval_sample in eval_samples: 
            custom_id = eval_sample['custom_id']
            eval_sample['winner'] = custom_id_to_answer[custom_id]

    # results order should be the same as eval_samples based on earlier check that compares message content
    else: 
        for eval_sample, result in zip(eval_samples,results):
            response_text = result['response']['body']['choices'][0]['message']['content']
            answer, explanation = extract_answer_and_explanation(response_text, generate_explanation=generate_explanations)
            eval_sample['winner'] = answer

    compute_accuracy(eval_samples)


def sample_by_author(eval_data: List[Dict[str, str]], n_samples_per_author: int, seed: int =42) -> List[Dict[str, str]]:

    author_ids = sorted(list(set([sample["target"] for sample in eval_data])))
    sampled_data = []
    for author_id in author_ids:
        author_samples = [sample for sample in eval_data if sample["target"] == author_id]
        n_samples_per_author = min(n_samples_per_author, len(author_samples))
        random.seed(seed)
        sampled_data += random.sample(author_samples, n_samples_per_author)

        logger.info(f"Sampled {n_samples_per_author} samples for author {author_id}")
    
    logger.info(f"Total samples: {len(sampled_data)}")

    return sampled_data 

def swap_order(data: Dict[str, str]) -> Dict[str, str]:
    """Swap order of text_a and text_b, author_id_a and author_id_b, and answer. 
    """
    new_data = data.copy()
    new_data["text_a"] = data["text_b"]
    new_data["text_b"] = data["text_a"]
    new_data["author_id_a"] = data["author_id_b"]
    new_data["author_id_b"] = data["author_id_a"]
    new_data["answer"] = "B" if data["answer"] == "A" else "A"
    
    return new_data

def prepare_single_eval_samples(eval_samples): 
    # take the pairwise evaluation data and create single evaluation data
    single_eval_data = []
    for eval_sample in eval_samples: 
        correct_sample = {
            "demos": eval_sample["demos"],
            "text": eval_sample["text_a"] if eval_sample["answer"] == "A" else eval_sample["text_b"],
            "author_id": eval_sample["author_id_a"] if eval_sample["answer"] == "A" else eval_sample["author_id_b"],
            "winner": eval_sample["winner"],
            "answer": "yes"
        }

        incorrect_sample = {
            "demos": eval_sample["demos"],
            "text": eval_sample["text_a"] if eval_sample["answer"] == "B" else eval_sample["text_b"],
            "author_id": eval_sample["author_id_a"] if eval_sample["answer"] == "B" else eval_sample["author_id_b"],
            "winner": eval_sample["winner"],
            "answer": "no"
        }
        single_eval_data += [correct_sample, incorrect_sample]

    return single_eval_data

def prepare_ccat50_pairwise_eval_samples(df: pd.DataFrame, n_samples_per_author: int, n_demo_samples: int = 5):

    # author_ids = list(df["author_id"].unique())
    author_ids = range(0, 40)
    eval_samples = [] 

    corpus = list(df["response"])

    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(corpus)
    df["tfidf"] = list(X.toarray())


    # add tf-idf to df for sampling cases that are more challenging and mimicking the actual eval scenario of model outputs with similar content 
    for author in author_ids: 
        author_df = df[df["author_id"] == author]
        n_samples_per_output = math.ceil(n_samples_per_author / len(author_df)) 
        author_eval_data = [] 

        for i, row in author_df.iterrows(): 
            author_id_a = row["author_id"]
            prompt = row["prompt"]
            response_a = row["response"]

            tfidf_a = row["tfidf"]

            cosine_similarities = tfidf_a @ X.T
            cosine_similarities = cosine_similarities.flatten()
            # for indices that are the same as the author_id_a, set to 0
            cosine_similarities[author_df.index] = 0

            # get top n_samples_per_output indices
            top_indices = np.argsort(cosine_similarities)[::-1][:n_samples_per_output]
            other_samples = df.iloc[top_indices]

            # randomly select demo samples from other prompts for author_id_a and author_id_b
            author_a_demo_samples = df[
                (df["author_id"] == author_id_a) & (df["prompt"] != prompt)
            ].sample(n_demo_samples)
            author_a_demo_samples_responses = list(author_a_demo_samples["response"])

            for _, other_sample in other_samples.iterrows():
                response_b = other_sample["response"]
                author_id_b = other_sample["author_id"]

                author_a_classification_task = {
                    "demos": author_a_demo_samples_responses,
                    "text_a": response_a,
                    "text_b": response_b,
                    "answer": "A",
                    "author_id_a": author_id_a,
                    "author_id_b": author_id_b,
                    "prompt": prompt,
                    "target": author_id_a, # the author that we are testing the model on
                    "winner": None
                }

                author_eval_data.append(author_a_classification_task)

        author_eval_data = random.sample(author_eval_data, n_samples_per_author) 

        eval_samples += author_eval_data

    return eval_samples
    
def prepare_cmcc_pairwise_eval_samples(df: pd.DataFrame, n_demo_samples: int = 5): 
    """
    Prepare evaluation data for CMCC benchmark.
    The difference with CCAT50 is that different authors can share the same prompt, so we prioritize these shared prompts for evaluation
    because this would let us control for content similarity better. Also, this is the setup (comparing outputs for the same prompt) for how models will be compared to one another. 
    """

    # get number of data samples where rows grouped by prompts have more than one
    count_df = df.groupby("prompt").count()
    count_df = count_df[count_df["author_id"] > 1]["author_id"]

    # keep only these prompts in df
    df = df[df["prompt"].isin(count_df.index)]

    if len(df) == 0:
        logger.info(f"No shared prompts between authors for CMCC.")
        return []

    # for each count, compute nC2 and add to total number of common prompts
    n_total_common_prompts = 0
    for count in count_df:
        n_total_common_prompts += count * (count - 1) // 2
    logger.info(n_total_common_prompts)

    # split df by authors, join on shared prompts
    merged = []
    author_ids = list(df["author_id"].unique())
    # author_ids = range(0, 19)
    for author in author_ids:
        author_df = df[df["author_id"] == author]
        for other_author in author_ids[author_ids.index(author) + 1 :]:
            other_author_df = df[df["author_id"] == other_author]
            merged_df = author_df.join(
                other_author_df.set_index("prompt"),
                on="prompt",
                lsuffix="_a",
                rsuffix="_b",
                how="inner",
            )
            merged.append(merged_df)

    merged_df = pd.concat(merged)

    # for each row, randomly select another response from a different prompt for author_id_a and author_id_b to form two samples for each row for evaluation

    logger.info("Preparing evaluation data...")
    eval_samples = []
    for i, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        author_id_a = row["author_id_a"]
        author_id_b = row["author_id_b"]
        prompt = row["prompt"]
        response_a = row["response_a"]
        response_b = row["response_b"]

        # randomly select demo samples from other prompts for author_id_a and author_id_b
        author_a_demo_samples = df[
            (df["author_id"] == author_id_a) & (df["prompt"] != prompt)
        ].sample(n_demo_samples)
        author_b_demo_samples = df[
            (df["author_id"] == author_id_b) & (df["prompt"] != prompt)
        ].sample(n_demo_samples)

        author_a_demo_samples_responses = list(author_a_demo_samples["response"])
        author_b_demo_samples_responses = list(author_b_demo_samples["response"])

        author_a_classification_task = {
            "demos": author_a_demo_samples_responses,
            "text_a": response_a,
            "text_b": response_b,
            "answer": "A",
            "author_id_a": author_id_a,
            "author_id_b": author_id_b,
            "prompt": prompt,
            "target": author_id_a, # the author that we are testing the model on
            "winner": None
        }

        author_b_classification_task = {
            "demos": author_b_demo_samples_responses,
            "text_a": response_a,
            "text_b": response_b,
            "answer": "B",
            "author_id_a": author_id_a,
            "author_id_b": author_id_b,
            "prompt": prompt,
            "target": author_id_b, # the author that we are testing the model on
            "winner": None
        }

        eval_samples += [
            author_a_classification_task,
            author_b_classification_task
        ]

    return eval_samples

def prepare_eval_samples(
    train_data: List[Dict[str, List[Dict[str, str]]]],
    data_name: str,
    n_demo_samples: int = 5,
    n_samples_per_author: int = 50,
    method: str = "pairwise",
) -> List[Optional[Dict[str, str]]]:
    """train data format
    [{author_id: [{prompt: "", output: ""}]}]

    returns:
        [{demo: "", text_a: "", text_b: "", answer: "A/B"}]
    """

    # restructure data to have dataframe with columns author_id, prompt, response
    flattened_data = []
    for author in train_data:
        for sample in train_data[author]:
            flattened_data.append(
                {
                    "author_id": author,
                    "prompt": sample["prompt"],
                    "response": sample["output"],
                }
            )
    df = pd.DataFrame(flattened_data)

    # drop duplicates where author_id and prompt are the same
    df = df.drop_duplicates(subset=["author_id", "prompt"])

    # reindex
    df = df.reset_index(drop=True)

    if data_name == "cmcc":
        eval_samples = prepare_cmcc_pairwise_eval_samples(df, n_demo_samples=n_demo_samples)

    if data_name == "ccat50": 
        eval_samples = prepare_ccat50_pairwise_eval_samples(df, n_demo_samples=n_demo_samples, n_samples_per_author=n_samples_per_author)

    if method == "pairwise":
        pass 
    else: 
        eval_samples = prepare_single_eval_samples(eval_samples)

    return eval_samples


def prepare_human_vs_llm_data(
    model_type: str,
    model_name: str,
    train_data: List[Dict[str, List[Dict[str, str]]]],
    data_name: str,
    n_samples_per_author: int = 1,
    n_few_shot_examples: int = 3,
) -> List[Optional[Dict[str, str]]]:
    """train data format
    [{author_id: [{prompt: "", output: ""}]}]

    returns:
        [{demo: "", text_a: "", text_b: "", answer: "A/B"}]
    """

    # create dataset with sample human responses as demo and one human response and one gpt response as text_a and text_b

    # data of interest format: [{author_id: "", prompt: "", output: "", "examples": [{"prompt": "", "output": ""}]}]
    all_samples = []
    for author_id in train_data:
        author_texts = train_data[author_id]
        author_samples = [] 
        for idx, text in enumerate(author_texts):
            text = author_texts[idx]
            prompt = text["prompt"]
            response = text["output"]
            other_texts = author_texts[:idx] + author_texts[idx + 1 :]
            sampled_texts = random.sample(other_texts, n_few_shot_examples)

            author_samples.append(
                {
                    "author_id": author_id,
                    "prompt": prompt,
                    "output": response,
                    "examples": sampled_texts,
                }
            )
        
        author_samples = random.sample(author_samples, n_samples_per_author)
        all_samples += author_samples

    # generate gpt outputs with few-shot setup
    iota_trainer = IotaTrainer()
    iota_trainer.train_data = train_data[0]  # dummy

    iota_trainer.setup_generation_model(model_type, model_name)

    eval_datapath = f"aa_eval/aa_eval_{data_name}_human_vs_gpt.jsonl"

    if os.path.exists(eval_datapath):
        logger.info(f"Loading previously created eval samples from {eval_datapath}...")
        with open(eval_datapath, "r") as f:
            eval_data = [json.loads(line) for line in f]
    else:
        eval_data = []
        logger.info("Generating gpt responses for human vs gpt evaluation...")
        for sample in tqdm(all_samples):

            # generate gpt response
            gpt_response = iota_trainer._generate(sample["prompt"])
            demos = [example["output"] for example in sample["examples"]]

            eval_sample = {
                "demos": demos,
                "text_a": sample["output"],
                "text_b": gpt_response,
                "answer": "A",
                "author_id": sample["author_id"],
            }

            eval_data.append(eval_sample)

            eval_sample = {
                "demos": demos,
                "text_a": gpt_response,
                "text_b": sample["output"],
                "answer": "B",
                "author_id": sample["author_id"],
            }

            eval_data.append(eval_sample)

        with open(eval_datapath, "w") as f:
            for item in eval_data:
                f.write(json.dumps(item) + "\n")

    return eval_data


def prepare_model_vs_model_data(): 

    model_vs_model_samples_path = PACKAGE_DIR / "tests/model_vs_model_test_cases.json"

    with open(model_vs_model_samples_path, "r") as f:
        model_vs_model_samples = json.load(f)
        samples = model_vs_model_samples["eval_samples"]

    # make author_id int for consistency
    for sample in samples: 
        sample["author_id"] = int(sample["author_id"])

    return samples

def main():

    parser = ArgumentParser()
    parser.add_argument(
        "-ns",
        "--num_samples_per_author",
        type=int,
        default=50,
        help="Number of samples to evaluas per author.",
    )
    parser.add_argument(
        "-nd",
        "--num_demo_examples",
        type=int,
        default=5,
        help="Number of examples to show for each author.",
    )

    parser.add_argument(
        "-nt",
        "--num_trials",
        type=int,
        default=1,
        help="Number of trials to run for each sample. If set to >1, the most frequent answer will be chosen.",
    )

    parser.add_argument(
        "-f", "--force", action="store_true", help="Force re-evaluation of all samples."
    )
    parser.add_argument(
        "-e",
        "--generate_explanation",
        action="store_true",
        help="Generate explanations for each prediction.",
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "-hu",
        "--human_vs_gpt",
        action="store_true",
        help="Evaluate human vs gpt performance.",
    )
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        default="gpt-4o-2024-08-06",
        help="Model to use for evaluation: one of [gpt-4o-2024-08-06, gpt-4o-2024-05-13, gpt-4o-mini-2024-07-18, anthropic.claude-3-sonnet-20240229-v1:0, prometheus]",
    )
    parser.add_argument(
        "-b", 
        "--benchmark", 
        type=str,
        default="cmcc",
        help="Benchmark to evaluate on: one of [cmcc, ccat50]"
    )
    parser.add_argument(
        "-mb",
        "--make_batch_predictions", 
        action="store_true",
        help="Submit batch predictions for evaluation."
    )
    parser.add_argument(
        "-cn",
        "--custom_notes",
        type=str,
        default=None,
        help="Custom notes for the evaluation. Useful for small modifications with the same eval setup and tracking. This will be added to the eval samples file and the batch prediction input file."
    )
    parser.add_argument(
        "-mv",
        "--model_vs_model",
        action="store_true",
        help="Evaluate model vs model performance with manually collected incorrectly labeled samples from previous runs."
    )
    args = parser.parse_args()

    if args.human_vs_gpt: 
        task = "human vs gpt"
    elif args.model_vs_model:
        task = "model vs model"
    else:
        task = "human vs human"

    if not args.model_vs_model: 
        logger.info(f"Running evaluation to benchmark {task} performance of {args.model_name_or_path} on {args.benchmark} with {args.num_samples_per_author} samples per author and {args.num_demo_examples} demo examples while generating explanations: {args.generate_explanation}")

    

    # dummy prompt that just shows the template of the evaluatino prompt with some dummy data. used for hashing to keep track of eval results & also for logging purposes.
    dummy_prompt = get_dummy_eval_prompt(
        num_demo_examples=args.num_demo_examples,
        generate_explanation=args.generate_explanation,
    )

    logger.info(f"Eval prompt used:\n\n{dummy_prompt}")
    model_name_or_path = args.model_name_or_path

    # configure and initiate model
    if 'gpt' in model_name_or_path:
        model = OpenAIModel(model_name_or_path)

    elif 'claude' in model_name_or_path:
        model = BedRockModel(model_name_or_path)

    elif model_name_or_path == "prometheus":
        model = PrometheusModel("prometheus-eval/prometheus-bgb-8x7b-v2.0")
        # model = PrometheusModel("prometheus-eval/prometheus-7b-v2.0")
        model_name_or_path = model.model_name_or_path.split("/")[-1]

    logger.info(f"Evaluating with model: {model_name_or_path}")

    # load training data
    data_dir = DEFAULT_DATA_DIR

    # get hash of eval_prompt
    hash_input = dummy_prompt
    hash_object = hashlib.sha256(hash_input.encode())
    hash_hex = hash_object.hexdigest()

    benchmark = args.benchmark

    # load data
    train_data = load_data(
        benchmark, "train", author_key="all", data_dir=data_dir
    )

    if args.human_vs_gpt:
        ### human vs gpt eval. we want gpt outputs to never be preferred over the human one.
        model_name = "gpt-4o-2024-05-13"
        model_type = "openai"

        eval_samples = prepare_human_vs_llm_data(
            model_name=model_name,
            model_type=model_type,
            train_data=train_data,
            data_name=benchmark,
            n_samples_per_author=args.num_samples_per_author,
            n_few_shot_examples=args.num_demo_examples,
        )

        eval_data = {"eval_prompt": dummy_prompt, "eval_samples": eval_samples}

        save_path = Path(f"aa_eval/aa_eval_human_vs_gpt_results_with_{model_name_or_path}_on_{benchmark}_ns_pa{args.num_samples_per_author}_nd{args.num_demo_examples}_exp{args.generate_explanation}_seed{args.seed}_{hash_hex}.jsonl")
        if args.custom_notes is not None:
            save_path = save_path.parent / f"{save_path.stem}_{args.custom_notes}{save_path.suffix}"

    elif args.model_vs_model: 
        eval_samples = prepare_model_vs_model_data()
        save_path = Path(f"aa_eval/aa_eval_model_vs_model_results_with_{model_name_or_path}_exp{args.generate_explanation}_{hash_hex}.jsonl")
        if args.custom_notes is not None:
            save_path = save_path.parent / f"{save_path.stem}_{args.custom_notes}{save_path.suffix}"

        # load previously formed eval results if any. this may be partially completed
        if os.path.exists(save_path) and not args.force:
            logger.info(f"Loading previously created eval samples from {save_path}...")
            with open(save_path, "r") as f:
                eval_data = json.load(f)
                eval_samples = eval_data["eval_samples"]
        
    else:
        ### human vs human eval. we want the model to predict the correct author of the response.
        save_path = Path(f"aa_eval/aa_eval_results_with_{model_name_or_path}_on_{benchmark}_ns_pa{args.num_samples_per_author}_nd{args.num_demo_examples}_exp{args.generate_explanation}_seed{args.seed}_{hash_hex}.jsonl")
        if args.custom_notes is not None:
            save_path = save_path.parent / f"{save_path.stem}_{args.custom_notes}{save_path.suffix}"

        # load previously formed eval results if any. this may be partially completed
        if os.path.exists(save_path) and not args.force:
            logger.info(f"Loading previously created eval samples from {save_path}...")
            with open(save_path, "r") as f:
                eval_data = json.load(f)
                eval_samples = eval_data["eval_samples"]

        else:
            logger.info("Starting from scratch...")

            # form eval data
            eval_samples = prepare_eval_samples(
                train_data, benchmark, n_demo_samples=args.num_demo_examples, n_samples_per_author=args.num_samples_per_author
            )

            # sample args.num_samples_per_author samples per author 
            eval_samples = sample_by_author(eval_samples, args.num_samples_per_author, args.seed)

            # swap orders of text_a and text_b for half of the samples
            # eval_samples = [swap_order(sample) if i % 2 == 0 else sample for i, sample in enumerate(eval_samples)]

            with open(save_path, "w") as f:
                eval_data = {
                    "eval_prompt": dummy_prompt,
                    "eval_samples": eval_samples,
                }
                json.dump(eval_data, f, indent=4)

    batch_prediction_input_path = save_path.parent / f"batch_prediction_{save_path.name}"
    if args.custom_notes is not None:
        batch_prediction_input_path = batch_prediction_input_path.parent / f"{batch_prediction_input_path.stem}_{args.custom_notes}{batch_prediction_input_path.suffix}"
    
    if args.make_batch_predictions:

        logger.info("Preparing batch prediction inputs...")

        # save batch prediction inputs to file and submit batch predictions
        # create batch predictions 
        batch_prediction_inputs = [] 
        if os.path.exists(batch_prediction_input_path):
            logger.error(f"Batch prediction input file already exists: {batch_prediction_input_path}. Loading submitted results if any...")

            # get batch id, which is in the last row of the file
            with open(batch_prediction_input_path, "r") as f:
                batch_prediction_inputs = [json.loads(line) for line in f]

            # check that eval samples and batch prediction inputs match 
            for idx, eval_sample, batch_input in zip(range(len(eval_samples)), eval_samples, batch_prediction_inputs[:-1]):

                batch_input_from_sample = model.prepare_batch_prediction_input(eval_sample["text_a"], eval_sample["text_b"], eval_sample["demos"], generate_explanation=args.generate_explanation, custom_id = save_path.name + f"_{eval_sample['target']}_{idx}")

                if not batch_input_from_sample['body']['messages'][0]['content'].split("OPTION A")[-1] == batch_input['body']['messages'][0]['content'].split("OPTION A")[-1]: 
                    logger.error(f"Batch prediction input for sample {idx} does not match with the eval sample.")
                    breakpoint() 

            batch_id = batch_prediction_inputs[-1].get("batch_id") 
            if batch_id is None: 
                logger.error("Batch id not found in the last row of the file. Exiting...")
                return

            status = get_batch_status(batch_id)
            logger.info(f"Batch status: {status.status} with {status.request_counts}")

            # if batch status shows as complete, load results
            if status.output_file_id:
                results = load_batch_results_with_batch_id(batch_id)
                parse_batch_results(eval_samples, results, generate_explanations=args.generate_explanation)
            else: 
                logger.info(f"Batch not completed yet. Try later.")

        else: 
            should_continue = ask_should_continue(eval_samples, train_data)
            if not should_continue:
                logger.info("Decided not to continue with batch predictions. Exiting...")
                return

            for idx, sample in enumerate(eval_samples):
                custom_id = save_path.name + f"_{sample['target']}_{idx}"
                batch_input = model.prepare_batch_prediction_input(sample["text_a"], sample["text_b"], sample["demos"], generate_explanation=args.generate_explanation, custom_id = custom_id)
                batch_prediction_inputs.append(batch_input)
                sample['custom_id'] = custom_id # needed for mapping back to eval samples

            with open(batch_prediction_input_path, "w") as f:
                for item in batch_prediction_inputs:
                    f.write(json.dumps(item) + "\n")

            with open(save_path, "w") as f:
                eval_data = {
                    "eval_prompt": dummy_prompt,
                    "eval_samples": eval_samples,
                }
                json.dump(eval_data, f, indent=4)

            logger.info(f"Batch prediction inputs saved to {batch_prediction_input_path}")
            logger.info(f"Submitting batch predictions...")
            submit_batch_openai_batch_predictions(batch_prediction_input_path) 
        

    # otherwise do live predictions
    else: 
        if batch_prediction_input_path.exists():
            logger.info(f"Batch prediction input file found: {batch_prediction_input_path}. Rerun the command with -mb to load batch prediction attempt. Exiting...")
            return

        # eval_samples without predictions 
        incomplete_eval_samples = [sample for sample in eval_samples if "winner" not in sample or sample["winner"] is None]
        should_continue = ask_should_continue(incomplete_eval_samples, train_data)
        if not should_continue:
            logger.info("Decided not to continue with batch predictions. Exiting...")
            return

        # evaluate
        logger.info(f"Starting on demand evaluation with: {model_name_or_path}")

        # go through while loop to account for rate limits & failures 
        while not all("winner" in sample and sample["winner"] is not None for sample in eval_samples):
            # show how many samples still need predictions
            num_remaining_preds = len(
                [
                    sample
                    for sample in eval_samples
                    if "winner" not in sample
                ]
            )
            num_total_samples = len(eval_samples)
            logger.info(
                f"{num_remaining_preds}/{num_total_samples} samples need predictions..."
            )

            # if prometheus, do batch
            # if args.model == "prometheus":
            # this leads to GPU memory issues 
            if False:
                text_as = [sample["text_a"] for sample in eval_samples]
                text_bs = [sample["text_b"] for sample in eval_samples]
                demos = [sample["demos"] for sample in eval_samples]

                predictions, explanations = model.batch_eval_head_to_head(
                    text_as, text_bs, demos
                )
                for i, sample in enumerate(eval_samples):
                    sample["winner"] = predictions[i]
                    sample["explanation"] = explanations[i]

            # if other models, do one by one
            else:
                for sample in tqdm(eval_samples):
                    # skip if already evaluated. we know it's evaluated if sample["winner"] exists
                    if "winner" in sample and sample["winner"] is not None:
                        continue

                    try:
                        answers = []
                        for i in range(args.num_trials):
                            answer, explanation = model.eval_head_to_head(
                                text_a=sample["text_a"],
                                text_b=sample["text_b"],
                                demos=sample["demos"],
                                generate_explanation=args.generate_explanation,
                            )

                            answers.append(answer)

                        # choose the answer that is most frequent
                        answer = max(set(answers), key=answers.count)
                        sample["winner"] = answer
                        sample["explanation"] = explanation

                    except Exception as e:
                        error_trace = (
                            traceback.format_exc()
                        )  # Get the full traceback as a string
                        logger.error(
                            f"Error generating response: {e}\nFull trace: {error_trace}"
                        )
                        continue

            with open(save_path, "w") as f:
                eval_data = {
                    "eval_prompt": dummy_prompt,
                    "eval_samples": eval_samples,
                }
                json.dump(eval_data, f, indent=4)

        logger.info(f"Results saved to {save_path}")
        # calculate accuracy
        compute_accuracy(eval_samples)




if __name__ == "__main__":
    main()
