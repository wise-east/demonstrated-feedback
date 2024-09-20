import json 
import pandas as pd 

def get_command_to_run_from_alias(alias:str): 

    benchmark, method, author_id = alias.split("_")
    command = f"python generate.py --benchmark {benchmark} --method {method} --train_author_key {author_id}"

    return command

df = pd.read_json("all_outputs_post_filter.json", lines=True)
# for checking errors
# with open("all_outputs_post_filter.json", "r") as f:
#     data = [] 
#     for idx, line in enumerate(f):
#         try: 
#             data.append(json.loads(line))
#         except: 
#             print(f"Error at line {idx}")
#             print(line)


alias_of_interest = [
    "ccat50_author",
    "ccat50_ditto",
    "ccat50_sft",
    "cmcc_author", 
    "cmcc_ditto",
    "cmcc_sft",
    "speechpref_ditto",
    "speechpref_sft",
]

full_alias_of_interest = [] 
for alias in alias_of_interest:

    if "ccat50" in alias: 
        author_ids = [32, 28, 20, 10, 27, 12, 15, 38, 23, 30]
    elif "cmcc" in alias:
        author_ids = [2, 3, 5, 6, 7, 8, 11, 13, 15, 17]
    if "speech" in alias:
        author_ids = [0]
    for author_id in author_ids:
        full_alias_of_interest.append(alias + "_" + str(author_id))

n_target_samples = 5

df['run_alias'] = df["benchmark"] + "_" + df["method"] + "_" + df["author_id"].astype(str)

remaining_generations = 0 
commands_to_run = []
for full_alias in full_alias_of_interest:
    if full_alias not in df["run_alias"].unique():
        print(f"[-] Alias {full_alias} not found in outputs")
        remaining_generations += 1 
        if not "author" in full_alias:
            commands_to_run.append(get_command_to_run_from_alias(full_alias))
        continue

    # make sure we have all inputs of interest
    alias_df = df[df["run_alias"] == full_alias]
    unique_tasks = alias_df["input"].unique()
    test_size = max(3, len(unique_tasks))
    target_total = n_target_samples * test_size

    if "author" in full_alias:
        target_total = 3 

    if len(alias_df) < target_total:
        # print text in yellow 
        print(f"\033[93m[X] Alias {full_alias} has less than {target_total} samples ({len(alias_df)})\033[0m")
        remaining_generations += 1
        if not "author" in full_alias:
            commands_to_run.append(get_command_to_run_from_alias(full_alias))
        continue
    else: 
        # print text in green 
        print(f"\033[92m[V] Alias {full_alias} has enough samples ({len(alias_df)})\033[0m")


print(f"\n\nRemaining generations: {remaining_generations}/{len(full_alias_of_interest)}")
print("\n\nCommands to run:")
for command in commands_to_run:
    print(f'\t\"{command}\"')