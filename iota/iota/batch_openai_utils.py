# usag: python batch_openai_cli.py --batch_id <batch_id> --check_status <true/false> --retrieve <true/false>

import click 
from openai import OpenAI
import json 
from typing import List, Dict, Any
from loguru import logger
client = OpenAI()

def submit_batch_openai_batch_predictions(batch_prediction_inputs_path): 

    from openai import OpenAI
    client = OpenAI()

    batch_input_file = client.files.create(
        file=open(batch_prediction_inputs_path, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    batch_obj = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"Batch prediction inputs for {batch_prediction_inputs_path.name}",
        }
    )

    # add batch id to batch input file 
    batch_id = batch_obj.id
    logger.info(f"Created batch with id: {batch_id}")
    with open(batch_prediction_inputs_path, "a+") as f:
        f.write(f'{{"batch_id": "{batch_id}"}}\n')

def load_batch_results_with_batch_id(batch_id: str) -> List[Dict[str, Any]]:
    status = get_batch_status(batch_id)
    results = load_batch_results(status.output_file_id)
    return results 

def load_batch_results(output_file_id: str) -> List[Dict[str, Any]]: 
    file_response = client.files.content(output_file_id)
    results = []
    for line in file_response.text.split("\n"):
        if line: 
            try: 
                results.append(json.loads(line))
            except Exception as e: 
                print(f"Failed to parse line: {line}")
                continue
    return results

def get_batch_status(batch_id: str) -> Any:
    status = client.batches.retrieve(batch_id)
    return status 

@click.command()
@click.option("-b", "--batch_id", type=str, help="Batch ID of the batch submitted")
@click.option("--check_status", type=bool, help="Check status of the batch", default=True)
@click.option("--retrieve", type=bool, help="Retrieve the batch results")
def batch_openai_cli(batch_id, check_status: bool, retrieve: bool):
    if check_status:
        status = get_batch_status(batch_id)
        print(status)
        output_file_id = status.output_file_id

        if retrieve:
            # retrieve the batch results
            results = load_batch_results(output_file_id)
            print(results)
    return

if __name__ == "__main__":
    batch_openai_cli()