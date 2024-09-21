import os
from typing import List, Dict, Any, NewType, Tuple, Optional
import pickle
from pathlib import Path
from transformers import HfArgumentParser
import sys
from loguru import logger
import dataclasses
from dataclasses import dataclass
import json


DataClassType = NewType("DataClassType", Any)
HOME_DIR = Path(os.environ["HOME"])
PACKAGE_DIR = Path(__file__).parent.parent.absolute()
DEFAULT_DATA_DIR = os.path.join(PACKAGE_DIR, "benchmarks")
OUTPUT_DIRECTORY = "out"

def get_git_hash() -> str:
    """Get the git hash of the current commit"""

    try:
        import subprocess

        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
    except:
        git_hash = "unknown"

    return git_hash


def load_data(
    dataset_name: str, split: str, author_key: int, data_dir: str = DEFAULT_DATA_DIR
) -> List[Dict[str, str]]:
    data_path = os.path.join(
        data_dir, f"{dataset_name}/processed/{dataset_name}_{split}.pkl"
    )

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data path {data_path} does not exist. Check if the config file has the correct datapath or whether the data exsits in the default data directory: <package_dir>/benchmarks."
        )

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    if author_key != "all":
        return data[author_key]
    else:
        return data


def set_absolute_path(path: str, package_dir: str) -> Path:
    """If path is relative, set it to absolute path using home_dir and package_dir"""

    if path.startswith("/"):
        return Path(path)
    else:
        return Path(os.environ["HOME"]) / package_dir / path


def process_other_args(other_args: List[str]) -> List[str]:

    processed_other_args = []
    idx = 0
    while idx < len(other_args):
        # if there's an = in the arg, we assume it's a key-value pair
        if "=" in other_args[idx]:
            processed_other_args.append(other_args[idx])
        # if not, we assume it's a key-value pair split by space
        elif "--" in other_args[idx]:
            if idx + 1 >= len(other_args) or "--" in other_args[idx + 1]:
                processed_other_args.append(f"{other_args[idx]}=true")
            else:
                processed_other_args.append(f"{other_args[idx]}={other_args[idx+1]}")
                idx += 1
        else:
            raise ValueError(
                f"Invalid command line argument value: {other_args[idx]}, not paired with a key"
            )

        idx += 1

    return processed_other_args


# from: https://github.com/huggingface/alignment-handbook/blob/main/src/alignment/configs.py
class H4ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(
        self, yaml_arg: str, other_args: Optional[List[str]] = None
    ) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'] or ['--arg val', '--arg2 val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {
            arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args
        }
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys

                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(
                            f"Duplicate argument provided: {arg}, may cause unexpected behavior"
                        )

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> DataClassType | Tuple[DataClassType]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            other_args = sys.argv[2:]
            processed_other_args = process_other_args(other_args)
            output = self.parse_yaml_and_args(
                os.path.abspath(sys.argv[1]), processed_other_args
            )
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output

def process_answer(answer_output: str) -> str:
    answer_output = answer_output.strip().upper()

    answer_output = answer_output.replace("OPTION", "").strip()

    return answer_output

def process_json_expected_response(response: str, key: str) -> str:
    """Process the response from the model as json and return the value of the key in the JSON response"""

    try:
        response = response.replace("json", "").replace("```", "").strip()
        response = json.loads(response)[key]
    except:

        try:
            logger.error(
                f"Failed to parse response: {response} for {key}. \n\nAttempting to parse..."
            )
            response = (
                response.split(f'"{key}":')[1].replace("{", "").replace("}", "").strip()
            )
            logger.info(f"Using attempted parsing of response for {key}: {response}")
        except:
            logger.error(
                f"Failed to parse response: {response}. Using it as it was given from the model."
            )

    return response


def extract_answer_and_explanation(model_outputs: str, generate_explanation:bool) -> Tuple[str, str]:
    """
    Extract the answer and explanation from the model outputs
    
    Args:
        model_outputs (str): model outputs as string
        generate_explanation (bool): whether to generate explanation
        
    Returns:
        Tuple[str, str]: answer, explanation
    """

    answer = process_json_expected_response(model_outputs, key="answer")
    answer = process_answer(answer)

    if generate_explanation:
        explanation = process_json_expected_response(model_outputs, key="explanation")
    else:
        explanation = ""

    return answer, explanation