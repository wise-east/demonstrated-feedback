from iota.iota_trainer import IotaTrainer, IOTADataset
from iota.utils import load_data, PACKAGE_DIR, extract_answer_and_explanation
from iota.llm_authorship_eval import estimate_per_sample_eval_cost
from iota.batch_openai_utils import load_batch_results_with_batch_id, get_batch_status, submit_batch_openai_batch_predictions
from iota.eval_utils import AUTHOR_IDS, get_run_alias
from pathlib import Path
import uuid
import json
from tqdm import tqdm
from loguru import logger
import click
import pandas as pd  
from collections import defaultdict

# python checkpoint_val_pairwise_eval.py --benchmark cmcc --model gpt
# python checkpoint_val_pairwise_eval.py --benchmark cmcc --model claude
# python checkpoint_val_pairwise_eval.py --benchmark ccat50 --model gpt
# python checkpoint_val_pairwise_eval.py --benchmark ccat50 --model claude


@click.command()
@click.option("--benchmark", type=str, help="one of cmcc,ccat50")
@click.option("--model", type=str, help="one of gpt or claude")
@click.option("--eval_mode", type=str, default="val_vs_author", help="one of [val_vs_author, val_vs_val]")
def main(benchmark, model, eval_mode): 
    out_dir = PACKAGE_DIR / "out" 
    method_name = "iota-naive"
    epochs = range(0, 4)

    batch_predictions_input_path = out_dir / "validation_per_epoch_eval" / f"batch_predictions_input_{model}_{benchmark}.jsonl"
    if eval_mode == "val_vs_val":
        batch_predictions_input_path = out_dir / "validation_per_epoch_eval" / f"batch_predictions_input_{model}_{benchmark}_val_vs_val.jsonl"

    if model == "gpt": 
        model = "gpt-4o-2024-08-06"
    elif model == "claude":
        model = "anthropic.claude-3-sonnet-20240229-v1:0"
    else: 
        raise ValueError("model must be one of gpt or claude")

    head_to_head_eval_batch_data = [] 
    all_eval_samples = [] 
    per_author_per_epoch_validation_outputs = [] 

    # create unique id for custom_id with uuid

    logger.info(f"Collecting eval data for benchmark: {benchmark}")
    author_ids = AUTHOR_IDS[benchmark]

    logger.info(f"Collecting eval data for model: {model}")

    for _author_id in author_ids: 
        logger.info(f"Collecting eval data for author_id: {_author_id}")
        val_data= load_data(benchmark, "val", _author_id)
        val_data = IOTADataset(val_data)
        train_data = load_data(benchmark, "train", _author_id)
        demos = [_data['output'] for _data in train_data][:5]

        # find latest model dir 
        model_dirs_parent = out_dir / model / benchmark / method_name / str(_author_id) 

        model_dirs = model_dirs_parent.glob("*")

        # among the model_dirs, choose the latest one with model.json inside it 
        for model_dir in model_dirs:
            if (model_dir / "model.json").exists():
                # check that all epochs are present
                for _epoch in epochs:
                    if not (model_dir / f"model_epoch{_epoch}.json").exists():
                        logger.info(f"Skipping {model_dir} because model_epoch{_epoch}.json is missing")
                        break
                break

        # continue 
        for _epoch in epochs:
            # load the model
            iota_trainer = IotaTrainer.from_checkpoint(model_dir, model_name=f"model_epoch{_epoch}.json")
            # set how many samples you want to create
            target_num_outputs_per_sample = 5  
            iota_trainer.args.num_return_sequences_val = target_num_outputs_per_sample
            iota_trainer.current_epoch = _epoch + 1
            iota_trainer.update_steps = (_epoch + 1) * len(iota_trainer.train_data)

            logger.info(f"Collecting eval data for epoch: {_epoch}")
            run_alias = get_run_alias(iota_trainer.args, epoch_num=_epoch)
            eval_save_path = out_dir/ "validation_per_epoch_eval"/ f"{run_alias}_{benchmark}_{_author_id}.json"
            if eval_mode == "val_vs_val":
    
                if eval_save_path.exists():
                    with open(eval_save_path, "r") as f:
                        eval_samples = json.load(f)

                    # set up eval pairs between different epochs
                    for _sample in eval_samples: 
                        if _sample["method_a"] == run_alias: 
                            val_output = _sample["text_a"]
                        else: 
                            val_output = _sample["text_b"]
                        if isinstance(val_output, list): 
                            val_output = val_output[0]
                        per_author_per_epoch_validation_output_sample = {
                            "author_id": _author_id, 
                            "epoch": _epoch, 
                            "method": run_alias, 
                            "task": _sample["task"],
                            "demos": _sample["demos"] if "demos" in _sample else _sample["demo"],
                            "val_output": val_output
                        }

                        per_author_per_epoch_validation_outputs.append(per_author_per_epoch_validation_output_sample)
                else: 
                    # if the eval file does not exist, exit and create it first 
                    logger.error(f"Eval file {eval_save_path} does not exist. Please run val_vs_author first")
                    return 
            continue 

            eval_save_path.parent.mkdir(parents=True, exist_ok=True)

            if eval_save_path.exists():
                with open(eval_save_path, "r") as f:
                    eval_samples = json.load(f)

                for _eval_sample in eval_samples: 
                    if "author_id" not in _eval_sample: 
                        _eval_sample["author_id"] = _author_id
                    if "epoch" not in _eval_sample: 
                        _eval_sample["epoch"] = _epoch

            else: 
                # should already be configured to the right model but double check just in case 
                assert iota_trainer.model.model_name_or_path == model

                # predict on the validation set
                val_preds = iota_trainer.eval(val_data, split="val", should_save=False)

                sim = val_preds['similarity']

                eval_samples = [] 
                # create head to head comparisons data 
                for _val_pred in val_preds['results']: 

                    eval_sample = {
                        "method_a": run_alias,
                        "method_b": "author",
                        "demos": demos,
                        "task": _val_pred['task'], 
                        "text_a": _val_pred['generated_output'],
                        "text_b": _val_pred['reference_output'],
                        "winner": None, 
                        "explanation": None, 
                        "similarity": sim, 
                        "custom_id": str(uuid.uuid4())
                    }

                    eval_sample_swapped = {
                        "method_a": "author",
                        "method_b": get_run_alias(iota_trainer.args, epoch_num=_epoch),
                        "demos": demos,
                        "task": _val_pred['task'], 
                        "text_a": _val_pred['reference_output'],
                        "text_b": _val_pred['generated_output'],
                        "winner": None, 
                        "explanation": None, 
                        "similarity": sim,
                        "custom_id": str(uuid.uuid4())
                    }

                    eval_samples += [eval_sample, eval_sample_swapped]

                with open(eval_save_path, "w") as f:
                    json.dump(eval_samples, f, indent=4)


            all_eval_samples += eval_samples

            for _eval_sample in eval_samples: 
                try: 

                    # correct typos or data structure inconsistencies
                    if isinstance(_eval_sample['text_a'], list): 
                        _eval_sample['text_a'] = _eval_sample['text_a'][0]
                    if isinstance(_eval_sample['text_b'], list): 
                        _eval_sample['text_b'] = _eval_sample['text_b'][0]

                    if "demo" in _eval_sample:
                        _eval_sample['demos'] = _eval_sample['demo']

                    batch_input = iota_trainer.model.prepare_batch_prediction_input(
                        text_a = _eval_sample['text_a'],
                        text_b = _eval_sample['text_b'],
                        demos = _eval_sample.get("demos", _eval_sample.get('demo')), 
                        generate_explanation=True, 
                        custom_id = _eval_sample['custom_id']
                    )
                except Exception as e: 
                    breakpoint() 


                # batch_input['model_name'] = model
                # batch_input['benchmark'] = benchmark
                # batch_input['author_id'] = _author_id
                # batch_input['epoch'] = _epoch

                head_to_head_eval_batch_data.append(batch_input)

            # save with corrections 
            with open(eval_save_path, "w") as f:
                json.dump(eval_samples, f, indent=4)
            # breakpoint() 


    if eval_mode == "val_vs_val":
        eval_samples = [] 
        # create val vs val eval samples
        df = pd.DataFrame(per_author_per_epoch_validation_outputs)
        # drop duplicates 
        df = df.drop_duplicates(subset=['author_id', 'method', 'task', 'val_output'])
        per_author_per_epoch_groups = df.groupby(['author_id', 'method', 'task'])


        for i, _epoch in enumerate(epochs): 
            for _epoch2 in epochs[i+1:]: 
                for _author_id in author_ids: 
                    for _task in per_author_per_epoch_groups.groups.keys(): 
                        _group = per_author_per_epoch_groups.get_group((_author_id, get_run_alias(iota_trainer.args, epoch_num=_epoch), _task[2]))
                        _group2 = per_author_per_epoch_groups.get_group((_author_id, get_run_alias(iota_trainer.args, epoch_num=_epoch2), _task[2]))

                        for idx, row in _group.iterrows(): 
                            for idx2, row2 in _group2.iterrows(): 
                                eval_sample = {
                                    "method_a": row['method'],
                                    "method_b": row2['method'],
                                    "demos": demos,
                                    "task": row['task'], 
                                    "text_a": row['val_output'],
                                    "text_b": row2['val_output'],
                                    "winner": None, 
                                    "explanation": None, 
                                    "similarity": 0.5,
                                    "custom_id": str(uuid.uuid4())
                                }

                                eval_sample_swapped = {
                                    "method_a": row2['method'],
                                    "method_b": row['method'],
                                    "demos": demos,
                                    "task": row2['task'], 
                                    "text_a": row2['val_output'],
                                    "text_b": row['val_output'],
                                    "winner": None, 
                                    "explanation": None, 
                                    "similarity": 0.5,
                                    "custom_id": str(uuid.uuid4())
                                }

                                eval_samples += [eval_sample, eval_sample_swapped]

        breakpoint() 




    if batch_predictions_input_path.exists():
        with open(batch_predictions_input_path, "r") as f:
            existing_batch_data = [json.loads(line) for line in f]
        batch_id = existing_batch_data[-1]['batch_id']

        status = get_batch_status(batch_id)
        logger.info(f"Batch status: {status.status} with {status.request_counts}")

        results = load_batch_results_with_batch_id(batch_id)

        custom_id2results = {} 
        # process results 
        for result in results: 
            response_text = result['response']['body']['choices'][0]['message']['content']
            answer, explanation = extract_answer_and_explanation(response_text, generate_explanation=True)
            custom_id2results[result['custom_id']] = {
                "answer": answer, 
                "explanation": explanation
            }

        for _eval_sample in all_eval_samples:
            if _eval_sample['custom_id'] in custom_id2results: 
                _eval_sample['winner'] = custom_id2results[_eval_sample['custom_id']]['answer']
                _eval_sample['explanation'] = custom_id2results[_eval_sample['custom_id']]['explanation']
            else: 
                logger.warning(f"Missing results for custom_id: {_eval_sample['custom_id']}")
        
        # compute per author per epoch accuracy and get argmax accuracy epoch 
        per_author_per_epoch_accuracy = []
        per_author_max_sim_epoch = {} 

        df = pd.DataFrame(all_eval_samples)
        for author_id in author_ids:
            author_id_max_sim = -1 
            for epoch in epochs:
                author_epoch_df = df[(df['author_id'] == author_id) & (df['epoch'] == epoch)]

                if author_epoch_df.similarity.iloc[0] > author_id_max_sim:
                    author_id_max_sim = author_epoch_df.similarity.iloc[0]
                    per_author_max_sim_epoch[author_id] = epoch

                win_counts = defaultdict(int) 
                for idx, _eval_sample in author_epoch_df.iterrows(): 
                    if _eval_sample['winner'] == "A": 
                        win_counts[_eval_sample['method_a']] += 1
                    elif _eval_sample['winner'] == "B":
                        win_counts[_eval_sample['method_b']] += 1

                total = len(author_epoch_df)

                per_author_per_epoch_accuracy.append({
                    "author_id": author_id, 
                    "epoch": epoch, 
                    "win_counts": win_counts, 
                    "total": total, 
                    "model_win_rate": win_counts[get_run_alias(iota_trainer.args, epoch_num=epoch)] / total
                })

        per_author_per_epoch_accuracy_df = pd.DataFrame(per_author_per_epoch_accuracy)
        # get epoch with highest accuracy for each author
        best_epoch_per_author = per_author_per_epoch_accuracy_df.groupby('author_id')['model_win_rate'].idxmax()

        best_epoch_value_per_author = per_author_per_epoch_accuracy_df.loc[best_epoch_per_author]

        print(f"Results for {model} on {benchmark} benchmark")

        print("max epochs for each author with val win rates vs author")
        for _author_id in author_ids:
            _best_epoch = best_epoch_value_per_author[best_epoch_value_per_author['author_id'] == _author_id]
            print(f"{_author_id}: {int(_best_epoch['epoch']) + 1}", end=", ")
        # for idx, row in best_epoch_value_per_author.iterrows():
        #     print(f"{row['author_id']}: {int(row['epoch']) + 1}", end=", ")

        print() 

        print("max sim epochs for each author when compared with val data")
        for _author_id in author_ids: 
            print(f"{_author_id}: {per_author_max_sim_epoch[_author_id] + 1}", end=", ")

        return 

    else: 
        total_expected_cost = 0

        for _eval_sample in all_eval_samples: 

            total_expected_cost += estimate_per_sample_eval_cost(
                _eval_sample.get('demo', _eval_sample.get('demos')), 
                _eval_sample['text_a'],
                _eval_sample['text_b'],
                generate_explanation=True
            )

        logger.info(f"Total expected cost for all eval samples: ${total_expected_cost/2:.2f}")

        should_continue = input("Do you want to continue? (y/n): ")
        should_continue = should_continue.lower() == "y"

        with open(batch_predictions_input_path, "w") as f:
            for _batch_input in head_to_head_eval_batch_data:
                f.write(json.dumps(_batch_input) + "\n")

        # submit the batch data 
        submit_batch_openai_batch_predictions(batch_predictions_input_path)

if __name__ == "__main__":
    main()