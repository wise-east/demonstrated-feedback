# purpose: evaluate the Style embedding or UAR model on the cmcc/ccat/speechpref datasets and get a sense of how close author embeddings are to each other for the same author across different datasets.
# usage: python emb_eval.py -m style -b cmcc

from iota.evaluation_models import StyleEmbeddingModel, UAREmbeddingModel
from iota.eval_utils import load_dfs, AUTHOR_IDS
from iota.utils import load_data
from argparse import ArgumentParser
from loguru import logger
from rouge import Rouge
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import List

from nltk.tokenize import sent_tokenize
import nltk 

# Ensure you have downloaded the necessary NLTK data files
nltk.download('punkt')


def main():

    parser = ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="style", help="One of {uar, style}"
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        default="cmcc",
        help="One of {cmcc, ccat50, speechpref}",
    )
    parser.add_argument(
        "-g",
        "--granularity", 
        type=str,
        default="all",
        help="Granularity to use for measuring similarity. One of {all, sample, sentence}"
    )

    args = parser.parse_args()

    if args.model == "uar":
        logger.info("Using UAR model")
        model = UAREmbeddingModel()
    elif args.model == "style":
        logger.info("Using Style model")
        model = StyleEmbeddingModel()

    if args.benchmark == "speechpref":
        from speechllm.models import create_reward_fn
        from speechllm.utils import format_prompt_response_for_reward_model

        speech_reward_model = create_reward_fn()

        def get_voice_suitability_score_single(user_instruction: str, response: str):
            samples = [
                format_prompt_response_for_reward_model(user_instruction, response)
            ]
            rewards = speech_reward_model(samples).tolist()
            return rewards[0]

    dataset = args.benchmark
    # dataset = "cmcc" # this being loaded from the config file instead of getting the argument caused a bug

    # load train, val, test data
    train_data = load_data(dataset, "train", "all")
    val_data = load_data(dataset, "val", "all")
    test_data = load_data(dataset, "test", "all")

    # load ditto and iota results
    dfs_dict = load_dfs(benchmark=args.benchmark)

    logger.info(f"Evaluating on {args.benchmark} dataset")

    # get each author's episode embeddings for train/va/test and check cosine similarity
    all_cosims = []
    all_author_embeddings = {}
    model_sim_results = {}
    # author_ids = sorted(list(train_data.keys()))
    # author_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    author_ids = AUTHOR_IDS[args.benchmark]

    rouge = Rouge()
    model_rouge_results = defaultdict(dict)

    emb_sim_results = []

    for author_id in author_ids:

        train_outputs = [sample["output"] for sample in train_data[author_id]]
        train_embeddings = model.get_corpus_embeddings(train_outputs)

        val_outputs = [sample["output"] for sample in val_data[author_id]]
        val_embeddings = model.get_corpus_embeddings(val_outputs)

        test_outputs = [sample["output"] for sample in test_data[author_id]]
        test_embeddings = model.get_corpus_embeddings(test_outputs)

        train_val_embeddings = model.get_corpus_embeddings(train_outputs + val_outputs)

        logger.info(
            f"Number of training samples for author {author_id}: {len(train_outputs)}"
        )
        logger.info(f"Number of val samples for author {author_id}: {len(val_outputs)}")
        logger.info(
            f"Number of test samples for author {author_id}: {len(test_outputs)}"
        )

        all_outputs = train_outputs + val_outputs + test_outputs
        all_embeddings = model.get_corpus_embeddings(all_outputs)

        # check cosine similarity between train/val and train/test
        train_val_sim = model.compute_similarity(train_embeddings, val_embeddings)
        train_test_sim = model.compute_similarity(train_embeddings, test_embeddings)
        val_test_sim = model.compute_similarity(val_embeddings, test_embeddings)

        avg = (train_val_sim + train_test_sim + val_test_sim) / 3

        print(
            f"Author {author_id} avg {args.model} cosim: {avg:.3f} | train/val: {train_val_sim:.3f} | train/test: {train_test_sim:.3f} | val/test: {val_test_sim:.3f}"
        )

        all_cosims.append(avg)

        all_author_embeddings[author_id] = {
            "train": train_embeddings,
            "val": val_embeddings,
            "train_val": train_val_embeddings,
            "test": test_embeddings,
            "all": all_embeddings,
        }

        for df_name, df in dfs_dict.items():

            # if "iota-naive" not in df_name: 
            #     continue  

            # if "mini" in df_name or "mistral" in df_name: 
            #     continue 

            per_author_df = df[df["author_id"] == author_id]
            output_column_name = "generated_output"

            if len(per_author_df):

                # if False: 
                rouge_references = []
                rouge_candidates = []

                for sample in test_data[author_id]:
                    prompt = sample["prompt"]
                    try:
                        model_responses = per_author_df[per_author_df["task"] == prompt][
                            output_column_name
                        ]
                        for model_response in model_responses:
                            rouge_references.append(sample["output"])
                            rouge_candidates.append(model_response)

                    except IndexError:
                        breakpoint()

                # skip rouge calculations for now 
                if False: 
                    if rouge_references and rouge_candidates:
                        scores = rouge.get_scores(
                            rouge_candidates, rouge_references, avg=True
                        )
                        model_rouge_results[df_name][author_id] = scores
                    else:
                        logger.warning(
                            f"No overlapping samples between test set and generated samples for author {author_id} in {df_name} df"
                        )

                model_outputs = per_author_df[output_column_name].tolist()

                if args.granularity == "all": 
                    model_output_embeddings = model.get_corpus_embeddings(model_outputs)

                    # check cosine similarity between train and ditto/iota outputs
                    train_sim = model.compute_similarity(
                        train_embeddings, model_output_embeddings
                    )

                    train_val_sim = model.compute_similarity(
                        train_val_embeddings, model_output_embeddings
                    )

                    val_sim = model.compute_similarity(val_embeddings, model_output_embeddings)
                    test_sim = model.compute_similarity(test_embeddings, model_output_embeddings)
                    all_sim = model.compute_similarity(all_embeddings, model_output_embeddings)

                elif args.granularity == "sample":

                    model_output_embeddings = [model.get_sample_embeddings(sample) for sample in model_outputs]

                    # for each sample, get the max similarity achieved with any of the train/val/test/train+val samples
                    per_sample_train_embeddings = [model.get_sample_embeddings(sample) for sample in train_outputs]
                    per_sample_val_embeddings = [model.get_sample_embeddings(sample) for sample in val_outputs]
                    per_sample_test_embeddings = [model.get_sample_embeddings(sample) for sample in test_outputs]
                    per_sample_train_val_embeddings = per_sample_train_embeddings + per_sample_val_embeddings
                    per_sample_all_embeddings = per_sample_train_val_embeddings + per_sample_test_embeddings

                    train_sims = [] 
                    val_sims = []
                    test_sims = []
                    train_val_sims = []
                    all_sims = [] 
                    for sample_idx, sample_embedding in enumerate(model_output_embeddings):
                        train_sims.append(model.compute_max_similarity(sample_embedding, per_sample_train_embeddings))
                        val_sims.append(model.compute_max_similarity(sample_embedding, per_sample_val_embeddings))
                        test_sims.append(model.compute_max_similarity(sample_embedding, per_sample_test_embeddings))
                        train_val_sims.append(model.compute_max_similarity(sample_embedding, per_sample_train_val_embeddings))
                        all_sims.append(model.compute_max_similarity(sample_embedding, per_sample_all_embeddings))
                    train_sim = np.mean(train_sims)
                    val_sim = np.mean(val_sims)
                    test_sim = np.mean(test_sims)
                    train_val_sim = np.mean(train_val_sims)
                    all_sim = np.mean(all_sims)

                elif args.granularity == "sentence":
                    # split outputs to sentences 
                    def get_sample_embeddings_with_tokenization(sample):
                        sentences = sent_tokenize(sample)
                        embeddings = [model.get_sample_embeddings(sentence) for sentence in sentences]
                        return embeddings


                    model_output_sentence_embeddings = [
                        get_sample_embeddings_with_tokenization(sample) for sample in model_outputs
                    ]

                    # for each sentence, get the max similarity achieved with any of the train/val/test/train+val samples
                    per_sentence_train_embeddings = [
                        get_sample_embeddings_with_tokenization(sample) for sample in train_outputs
                    ]
                    per_sentence_val_embeddings = [
                        get_sample_embeddings_with_tokenization(sample) for sample in val_outputs
                    ]
                    per_sentence_test_embeddings = [
                        get_sample_embeddings_with_tokenization(sample) for sample in test_outputs
                    ]
                    per_sentence_train_val_embeddings = per_sentence_train_embeddings + per_sentence_val_embeddings
                    per_sentence_all_embeddings = per_sentence_train_val_embeddings + per_sentence_test_embeddings


                    def get_mean_max_sentence_pair_similarities(per_sentence_data_embeddings, model_output_sentence_embeddings):

                        sample_sims = [] 
                        for data_sample in per_sentence_data_embeddings: 
                            data_sample_simps = [] 
                            # for each sentence in the train sample, get the max similarity with any of the output sentences
                            for data_sent_emb in data_sample: 
                                for output_sample_sent_embs in model_output_sentence_embeddings:                            
                                    data_sample_simps.append(model.compute_max_similarity(data_sent_emb, output_sample_sent_embs))
                            sample_sims.append(np.mean(data_sample_simps))
                            
                        return np.mean(sample_sims)

                    train_sim = get_mean_max_sentence_pair_similarities(per_sentence_train_embeddings, model_output_sentence_embeddings)
                    val_sim = get_mean_max_sentence_pair_similarities(per_sentence_val_embeddings, model_output_sentence_embeddings)
                    # test_sim = get_mean_max_sentence_pair_similarities(per_sentence_test_embeddings, model_output_sentence_embeddings)
                    test_sim = 0 
                    train_val_sim = get_mean_max_sentence_pair_similarities(per_sentence_train_val_embeddings, model_output_sentence_embeddings)
                    # all_sim = get_mean_max_sentence_pair_similarities(per_sentence_all_embeddings, model_output_sentence_embeddings)
                    all_sim = 0 




                if args.benchmark == "speechpref":
                    # get voice suitability scores
                    voice_suitability_scores = []
                    for idx, row in per_author_df.iterrows():
                        prompt = row['task']
                        output = row[output_column_name]
                        voice_suitability_score = get_voice_suitability_score_single(
                            prompt, output
                        )
                        voice_suitability_scores.append(voice_suitability_score)

                    logger.info(
                        f"\tAuthor {author_id} {df_name:>40} voice suitability scores: {sum(voice_suitability_scores) / len(voice_suitability_scores):.3f}"
                    )

                # logger.info(
                #     f"\tAuthor {author_id} {df_name:>40} sim:\t {all_sim:.3f} (all) \t| {train_sim:.3f} (train) \t| {val_sim:.3f} (val) \t| {train_val_sim:.3f} (train_val) | \t| {test_sim:.3f} (test) \t| (len: {len(per_author_df)})"
                # )                
                
                logger.info(
                    f"Author {author_id} {df_name:>40} sim:\t {train_sim:.3f} (train) \t| {val_sim:.3f} (val) \t| {train_val_sim:.3f} (train_val) | (len: {len(per_author_df)})"
                )

                sim_result = {
                    "author_id": author_id,
                    "name": df_name.split("epochs")[0] if "epochs" in df_name else df_name,
                    "epoch": int(df_name.split("epochs:")[1].split("_")[0]) if "epochs" in df_name else 0,
                    "train": train_sim,
                    "val": val_sim,
                    "train_val": train_val_sim,
                    "test": test_sim,
                    "all": all_sim,
                }

                emb_sim_results.append(sim_result)

                if df_name not in model_sim_results:
                    model_sim_results[df_name] = {}
                model_sim_results[df_name][author_id] = all_sim
                # model_sim_results[df_name][author_id] = test_sim

        if "author" not in model_sim_results:
            model_sim_results["author"] = {}

        if author_id not in model_sim_results["author"]:
            model_sim_results["author"][author_id] = avg

    print(
        f"Average {args.model} cosim among all authors: {sum(all_cosims) / len(all_cosims):.3f}"
    )

    print("\n\n")

    # what is the delta when compared with other authors?
    inter_author_sim_matrix = {}
    for author_id in author_ids:
        inter_author_sim_matrix[author_id] = {}
        for other_author_id in all_author_embeddings:
            # if author_id == other_author_id:
            #     continue
            all_sim = model.compute_similarity(
                all_author_embeddings[author_id]["all"],
                all_author_embeddings[other_author_id]["all"],
            )

            inter_author_sim_matrix[author_id][other_author_id] = all_sim

    # print author vs author results as matrix
    print("Author similarity matrix")
    print("ID\t" + "\t".join([str(author_id) for author_id in inter_author_sim_matrix] + ["Mean"]))
    for author_id in author_ids:
        print(author_id, end="\t")
        for other_author_id in inter_author_sim_matrix[author_id]:
            print(
                f"{inter_author_sim_matrix[author_id][other_author_id]:.3f}", end="\t"
            )

        # add the mean similarity of this author with all other authors
        mean_sim = sum(inter_author_sim_matrix[author_id].values()) / len(inter_author_sim_matrix[author_id])
        print(f"{mean_sim:.3f}")

    # print model results
    for df_name in model_sim_results:
        avg = sum(model_sim_results[df_name].values()) / len(model_sim_results[df_name])

        result_string = f"{df_name:>40} | {avg:.3f} |"
        for author_id in author_ids:
            if author_id in model_sim_results[df_name]:
                result_string += (
                    f" {author_id}: {model_sim_results[df_name][author_id]:.3f} |"
                )

        print(result_string)


    emb_sim_df = pd.DataFrame(emb_sim_results)
    # first group by name , then by author id, find the argmax of the epoch that gets the highest similarity for each group for ecah sim type 
    name_groups = emb_sim_df.groupby('name')
    for name, group in name_groups: 
        author_groups = group.groupby('author_id')
        
        for sim_type in ['train', 'val', 'train_val']: 
            print(f"{name} {sim_type} max sim epochs:\t", end="")
            for author_id in author_ids: 
                author_group = author_groups.get_group(author_id)
                max_epoch = author_group[author_group[sim_type] == author_group[sim_type].max()]['epoch'].values[0]
                max_sim = author_group[author_group[sim_type] == author_group[sim_type].max()][sim_type].values[0]
                # print(f"{name} {author_id} {sim_type} max sim: {max_sim:.3f} at epoch {max_epoch}")
                print(f"{author_id}:{max_epoch+1}", end=",")

            print()

    # print rouge results
    for df_name in model_rouge_results:

        breakpoint()

        avg = sum(model_rouge_results[df_name].values()) / len(
            model_rouge_results[df_name]
        )

        result_string = f"{df_name:>40} | {avg:.3f} |"
        for author_id in author_ids:
            if author_id in model_rouge_results[df_name]:
                result_string += (
                    f" {author_id}: {model_rouge_results[df_name][author_id]:.3f} |"
                )

        print(result_string)


if __name__ == "__main__":
    main()
