import scipy.stats as stats
import json
import pandas as pd
from pathlib import Path
from iota.utils import PACKAGE_DIR, OUTPUT_DIRECTORY
import os
from typing import Optional, Union, Dict, Any
from argparse import Namespace

HOME_DIR = os.environ["HOME"]

MODEL_NAME_ALIAS_DICT = {
    "gpt-4o-2024-05-13": "gpt-4o-0513",
    "gpt-4o-2024-08-06": "gpt-4o-0806",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "mistral.mistral-7b-instruct-v0:2": "mistral7b",
    "anthropic.claude-3-sonnet-20240229-v1:0": "claude-3-sonnet",
    "anthropic.claude-3-haiku-20240307-v1:0": "claude-3-haiku",
}

AUTHOR_IDS = {
    "ccat50": [32, 28, 20, 10, 27, 12, 15, 38, 23, 30],
    "cmcc": [2, 3, 5, 6, 7, 8, 11, 13, 15, 17],
    "speechpref": [0]
}


def update_df_columns(df: pd.DataFrame):
    """
    Fix certain column values for past runs that are incorrect or empty
    """
    # if "summarize_after_n_steps" is set to 1e-6 or none, set it to 1e6
    df["summarize_after_n_steps"] = df["summarize_after_n_steps"].apply(
        lambda x: 1e6 if x == 1e-6 or pd.isna(x) else x
    )

    # if "summarize_after_n_epochs" is set to 1e-6 or none, set it to 1e6
    df["summarize_after_n_epochs"] = df["summarize_after_n_epochs"].apply(
        lambda x: 1e6 if x == 1e-6 or pd.isna(x) else x
    )

    # drop models that are not needed
    models_to_drop = []
    df = df[~df["model_name_or_path"].isin(models_to_drop)]
    # df = add_run_alias_to_df(df)
    # group by run alias and set the first 150 to epoch 0
    groups = df.groupby("run_alias")
    for name, group in groups:
        # epoch 4 are actually epoch 0 outputs
        if "epochs:4" in name:
            df.loc[group.index, "num_train_epochs"] = 0
            # update their alias 
            first_150_group = add_run_alias_to_df(df.loc[group.index], num_epochs=0)
            df.loc[group.index, "run_alias"] = first_150_group["run_alias"]



    # if "mode" does not exist, follow the following logic to determine the mode
    for idx, row in df.iterrows():
        # if row["mode"] is not NaN, continue
        if pd.notna(row["mode"]):
            continue
        # if num_train_epochs is 0 and num_incontext_examples != 0 then mode is fewshot
        if (row["num_train_epochs"] == 0) & (row["num_incontext_examples"] != 0):
            df.loc[idx, "mode"] = "fewshot"
        # if num_train_epochs is 0 and num_incontext_examples == 0 then mode is zeroshot
        elif (row["num_train_epochs"] == 0) & (row["num_incontext_examples"] == 0):
            df.loc[idx, "mode"] = "zeroshot"
        # if num_train_epochs >= 2 then mode is iota 
        # TODO be more specific with the mode (iota-naive, iota-no-explanations, etc.) not important until we run into these cases
        # elif row["num_train_epochs"] >= 2:
        #     df.loc[idx, "mode"] = "iota"
        else:
            raise ValueError(f"Could not determine mode: {row}")

    # add date column if it does not exist
    for idx, row in df.iterrows():
        if pd.notna(row["date"]):
            continue

        df.loc[idx, "date"] = row["output_dir"].split("/")[-1]

    return df


def add_run_alias_to_df(df: pd.DataFrame, num_epochs=None) -> pd.DataFrame:
    """
    Add run_alias column to the dataframe based on the model_name_or_path, mode, num_incontext_examples, num_train_epochs, num_undesired_outputs.

    Args:
    df: pd.DataFrame

    Returns:
    pd.DataFrame
    """
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        run_alias = get_run_alias(row_dict, num_epochs)
        df.loc[idx, "run_alias"] = run_alias

    return df


def get_run_alias(args: Optional[Union[Namespace, Dict[str, Any]]] = None, epoch_num: int = None) -> str:

    # if args is dict, convert to namespace
    if isinstance(args, dict):
        args = Namespace(**args)

    model_alias = MODEL_NAME_ALIAS_DICT.get(
        args.model_name_or_path, args.model_name_or_path
    )

    if epoch_num is None:
        epoch_num = args.num_train_epochs

    if args.mode == "zeroshot":
        alias = f"{model_alias}_zeroshot"

    elif args.mode == "fewshot":
        alias = f"{model_alias}_fewshot_ice:{args.num_incontext_examples}"

    elif args.mode == "iota-naive":
        alias = f"{model_alias}_iota-naive_ice:{args.num_incontext_examples}_epochs:{epoch_num}_undesired:{args.num_undesired_outputs}"
    elif args.mode == "iota-full":
        alias = f"{model_alias}_iota-full_ice:{args.num_incontext_examples}_epochs:{epoch_num}_undesired:{args.num_undesired_outputs}"

    elif args.mode == "iota-no-explanations":
        alias = f"{model_alias}_iota-no-explanations_ice:{args.num_incontext_examples}_epochs:{epoch_num}_undesired:{args.num_undesired_outputs}"

    elif args.mode == "cot":
        alias = f"{model_alias}_cot_ice:{args.num_incontext_examples}"

    elif args.mode == "opro": 
        alias = f"{model_alias}_opro_ice:{args.num_incontext_examples}"

    else:
        raise NotImplementedError(f"Mode {args.mode} not implemented")

    return alias


def load_ditto_results(benchmark: str):

    ditto_path = (
        Path(HOME_DIR) / "project/demonstrated-feedback/outputs/all_outputs.json"
    )

    with open(ditto_path, "r") as f:
        ditto_data = [json.loads(line) for line in f]

    df = pd.DataFrame(ditto_data)

    # rename columns to match iota output results
    df = df.rename(columns={"target_task": "task", "response": "generated_output"})

    if benchmark != "all":
        df = df[df["benchmark"] == benchmark]

    ditto_df = df[df["method"] == "ditto"]
    sft_df = df[df["method"] == "sft"]
    author_df = df[df["method"] == "author"]

    # drop duplicates
    author_df = author_df.drop_duplicates(subset=["task", "author_id"])

    ditto_dfs = {"ditto": ditto_df, "sft": sft_df, "author": author_df}

    return ditto_dfs


def load_iota_results(benchmark: str):

    iota_path = PACKAGE_DIR / OUTPUT_DIRECTORY / "all_results.jsonl"
    with open(iota_path, "r") as f:
        all_iota_data = [json.loads(line) for line in f]

    all_iota_df = pd.DataFrame(all_iota_data)

    if benchmark != "all":
        all_iota_df = all_iota_df[all_iota_df["benchmark"] == benchmark]

    # set author_id to int(author_key)
    all_iota_df["author_id"] = all_iota_df["author_key"].apply(lambda x: int(x))

    # set run_alias
    # all_iota_df = add_run_alias_to_df(all_iota_df)
    all_iota_df = update_df_columns(all_iota_df)

    return all_iota_df


# load dataframes of interest
def load_dfs(benchmark: str = "cmcc"):

    ditto_dfs = load_ditto_results(benchmark)
    all_iota_df = load_iota_results(benchmark)

    dfs = ditto_dfs

    # load iota results
    for subset_name, df in all_iota_df.groupby("run_alias"):
        dfs[f"{subset_name}"] = df

    return dfs


def calculate_p_value_and_beta(n, accuracy, observed_win_rate, alpha=0.2):
    # Expected win rate under the null hypothesis (no advantage)
    null_win_rate = 0.5

    # Adjust the observed win rate considering the classifier's accuracy
    adjusted_win_rate = 0.5 + (observed_win_rate - 0.5) / accuracy

    # Calculate the adjusted variance
    adjusted_variance = (
        adjusted_win_rate * (1 - adjusted_win_rate) / n * (1 / accuracy**2)
    )

    # Calculate the standard error (SE) using the adjusted variance
    se = adjusted_variance**0.5

    # Calculate z-score for the null hypothesis
    z_score_null = (adjusted_win_rate - null_win_rate) / se

    # Calculate p-value from the z-score (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score_null)))

    # Determine the critical z-value for the given alpha
    z_alpha = stats.norm.ppf(1 - alpha / 2)

    # Calculate the non-centrality parameter
    non_centrality = (adjusted_win_rate - null_win_rate) / se

    # Calculate beta (Type II error)
    # Find the z-value under H1 where the test statistic still falls within the acceptance region of H0
    z_beta = z_alpha - non_centrality
    beta = stats.norm.cdf(z_beta)

    return p_value, beta


if __name__ == "__main__":

    # Example usage:
    n_samples = 50  # number of samples
    classifier_accuracy = 0.87  # accuracy of the classifier
    observed_win_rate = 0.65  # observed win rate that Model A wins over Model B

    p_value, beta = calculate_p_value_and_beta(
        n_samples, classifier_accuracy, observed_win_rate
    )
    print(f"The calculated p-value is: {p_value}")
    print(f"The calculated beta (Type II error probability) is: {beta}")
