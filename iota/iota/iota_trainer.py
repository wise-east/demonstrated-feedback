from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional, Any
from tqdm import tqdm
import pickle
from loguru import logger
import os
import json
import random
from collections import defaultdict
import torch
import yaml

# local imports
from .utils import (
    process_json_expected_response,
    set_absolute_path,
    load_data,
    get_git_hash,
    PACKAGE_DIR, 
    DEFAULT_DATA_DIR
)
from .eval_utils import get_run_alias
from .prompts import get_few_shot_writing_prompt, get_generate_cot_style_guides_prompt, get_cot_style_guide_based_writing_prompt, get_opro_writing_prompt, get_opro_optimization_prompt, get_generate_explanation_prompt
from .evaluation_models import StyleEmbeddingModel, UAREmbeddingModel
from .generation_models import (
    HuggingfaceModel,
    OpenAIModel,
    BedRockModel,
    UnifiedGenerationModel,
)

DESIRED_OUTPUT_ALIAS = "Your Writing"
UNDESIRED_OUTPUT_ALIAS = "Stylistically Inconsistent Writing"

@dataclass
class IotaArguments:

    model_type: Optional[str] = field(
        default="openai",
        metadata={"help": "Type of model to use: one of transformers, openai, claude"},
    )

    model_name_or_path: Optional[str] = field(
        default="gpt-4o-mini-2024-07-18",
        metadata={
            "help": "Model name to use. One of gpt-4o-mini-2024-07-18, gpt-4o-2024-05-13 for openai and any model path for huggingface models"
        },
    )

    exp_model_type: Optional[str] = field(
        default=None,
        metadata={"help": "same choices as model_type"},
    )

    exp_model_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "same choices as model_name_or_path"},
    )


    load_from_checkpoint: Optional[str] = field(
        default="",
        metadata={
            "help": "Load from a checkpoint. Provide the path to the checkpoint directory."
        },
    )

    output_dir: Optional[str] = field(
        default="out",
        metadata={
            "help": "Directory to save model checkpoints and eval/test results"
        },  # model checkpoints are just intermediate generated outputs & explanations
    )

    logging_dir: Optional[str] = field(
        default="logs", metadata={"help": "Directory to save logs"}
    )

    package_dir: Optional[str] = field(
        default=PACKAGE_DIR,
        metadata={"help": "Directory of the IOTA package"},
    )

    datapath: Optional[str] = field(
        default=DEFAULT_DATA_DIR,
        metadata={
            "help": "Directory of the datapath, uses relative path to package_dir if not starting with /"
        },
    )

    dataset: Optional[str] = field(
        default="speechpref",
        metadata={
            "help": "Name of the dataset to use: one of speechpref, ccat50, cmcc"
        },
    )

    author_key: Optional[int] = field(
        default=0, metadata={"help": "Author index to use for the dataset"}
    )

    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "Number of training epochs"}
    )

    train_size: Optional[int] = field(
        default=-1,
        metadata={"help": "Number of training examples to use. If -1, use all."},
    )

    num_incontext_examples: Optional[int] = field(
        default=3,
        metadata={"help": "Number of in-context examples to use for each task"},
    )

    incontext_selection_strategy: Optional[str] = field(
        default="random",
        metadata={
            "help": "Strategy for selecting in-context examples: one of random, sequential (first N)"
        },
    )

    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed for training"}
    )

    generate_explanations: Optional[bool] = field(
        default=True, metadata={"help": "Generate explanations for undesirable outputs"}
    )

    use_other_examples_for_explanation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use other examples as context for generating explanations for undesirable outputs"
        },
    )

    add_consistent_outputs: Optional[bool] = field(
        default=False,
        metadata={"help": "Add consistent outputs to the training data as reference outputs"},
    )

    num_undesired_outputs: Optional[int] = field(
        default=2,
        metadata={
            "help": "Number of undesirable outputs to show for each example. If -1, use all."
        },
    )

    max_tokens: Optional[int] = field(
        default=4096, metadata={"help": "Max tokens for model generation"}
    )

    temperature: Optional[float] = field(
        default=1, metadata={"help": "Temperature for model generation"}
    )

    top_k: Optional[int] = field(
        default=50, metadata={"help": "Top k for model generation"}
    )

    top_p: Optional[float] = field(
        default=0.95, metadata={"help": "Top p for model generation"}
    )

    num_return_sequences_val: Optional[int] = field(
        default=2,
        metadata={"help": "Number of sequences to generate at validation time"},
    )

    num_return_sequences_test: Optional[int] = field(
        default=5, metadata={"help": "Number of sequences to generate at test time"}
    )

    do_train: Optional[bool] = field(default=True, metadata={"help": "Run training"})

    do_eval: Optional[bool] = field(
        default=True, metadata={"help": "Run evaluation during training"}
    )

    eval_every_n_steps: Optional[int] = field(
        default=1e6, metadata={"help": "Run evaluation every n steps"}
    )

    early_stopping: Optional[bool] = field(
        default=True, metadata={"help": "Use early stopping"}
    )

    early_stopping_patience: Optional[int] = field(
        default=1e6, metadata={"help": "Early stopping patience"}
    )

    eval_metric: Optional[str] = field(
        default="style_embedding",
        metadata={
            "help": "Evaluation metric to use: cosine similarity with [style_embedding, uar] or speechpref_rm"
        },
    )

    embedding_model: Optional[str] = field(
        default="style",
        metadata={"help": "Embedding model to use: one of [style, uar]"},
    )

    summarize: Optional[bool] = field(
        default=True, metadata={"help": "Summarize explanations to deduce style rules"}
    )

    summary_logic: Optional[str] = field(
        default="steps",
        metadata={
            "help": "Summarization logic: one of [epochs, steps, length_threshold, patience]"
        },
    )

    summarize_after_n_steps: Optional[int] = field(
        default=1e6,
        metadata={"help": "Summarize after n steps if summary_logic is steps"},
    )

    summarize_after_n_epochs: Optional[int] = field(
        default=1e6,
        metadata={"help": "Summarize after n epochs if summary_logic is epochs"},
    )

    summarize_at_end: Optional[bool] = field(
        default=False, metadata={"help": "Summarize at the end of training"}
    )

    do_predict: Optional[bool] = field(
        default=True, metadata={"help": "Run prediction on test data"}
    )

    small: Optional[bool] = field(
        default=False, metadata={"help": "Use small model for testing"}
    )

    test: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Run in test mode to pause after each generated output for inspection"
        },
    )

    mode: Optional[str] = field(
        default="iota",
        metadata={
            "help": "Mode of operation: one of [iota-naive, iota-full, iota-no-explanations, fewshot, zeroshot]"
        },
    )

    device_id: Optional[int] = field(
        default=0,
        metadata={"help": "GPU Device id to use for loading embedding models"},
    )

    @classmethod
    def load_from_config(cls, config: Dict[str, Any]) -> "IotaArguments":
        cls_instance = cls()
        # keep only the keys that are in the dataclass
        config = {k: v for k, v in config.items() if k in asdict(cls_instance).keys()}

        return cls(**config)


class IOTAExample:

    def __init__(self, task: str, reference_outputs: List[str]):
        self.task = task
        self.reference_outputs = reference_outputs # more reference outputs can be added from the IOTA process if candidates are considered consistent with the style
        self.reference_output = reference_outputs[0] # keep the original 
        self.generated_outputs: List[Dict[str, str]] = []
        self.complete = False

    # make namespace items subscriptable
    def __getitem__(self, key):
        return getattr(self, key)

    def add_generated_output(self, generated_output: str, explanation: str = None):
        self.generated_outputs.append(
            {
                "output": generated_output,
                "explanation": explanation,
                "included_in_summary": False,
                "ignore": False,
            }
        )

    def add_reference_output(self, reference_output: str):
        self.reference_outputs.append(reference_output)

    def mark_complete(self): 
        self.complete = True

    def __str__(self):
        return f"Task: {self.task}\nReference output: {self.reference_outputs}\nGenerated outputs: {self.generated_outputs}"

    def __repr__(self):
        return self.__str__()

    def as_dict(self):
        return {
            "task": self.task,
            "reference_output": self.reference_outputs,
            "generated_outputs": self.generated_outputs,
            "complete": self.complete,
        }

    @classmethod
    def load_from_dict(cls, state: Dict[str, Any]):
        cls_instance = cls(state["task"], state["reference_output"])
        cls_instance.generated_outputs = state["generated_outputs"]
        return cls_instance


class IOTADataset:

    def __init__(self, data: List[Dict[str, str]]):

        self.samples = []
        if data is not None:
            for sample in data:
                self.samples.append(IOTAExample(sample["prompt"], [sample["output"]]))
        self.summarization = (
            ""  # for summarization of the explanations to deduce style rules
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def shuffle(self):
        random.shuffle(self.samples)

    def __str__(self):
        return "\n\n".join([str(sample) for sample in self.samples])

    def __repr__(self):
        return self.__str__()

    def as_dict(self):
        return {
            "samples": [sample.as_dict() for sample in self.samples],
            "summarization": self.summarization,
        }

    @classmethod
    def load_from_dict(cls, state: Dict[str, Any]):
        data = [IOTAExample.load_from_dict(sample) for sample in state["samples"]]
        cls_instance = cls([])
        cls_instance.samples = data
        cls_instance.summarization = state["summarization"]
        return cls_instance


class IotaTrainer:

    def __init__(self, args: IotaArguments = IotaArguments()) -> None:

        self.args = args
        self.setup_generation_model()
        self.setup_explanation_model() 
        self.embedding_model = (
            StyleEmbeddingModel(device_id=args.device_id)
            if self.args.embedding_model == "style"
            else UAREmbeddingModel(device_id=args.device_id)
        )
        self.setup_data()
        self.configure_mode()
        self.num_return_sequences = 1
        self.update_steps = 0
        self.current_epoch = 0
        self.patience = self.args.early_stopping_patience
        self.best_eval_score = -1
        self.guidance_summary = (
            ""  # for summarization of the explanations to deduce style rules
        )

        self.eval_results = defaultdict(list)
        self.steps_input_outputs = []

        if self.args.num_incontext_examples > len(self.train_data) - 1:
            logger.warning(
                f"Number of incontext examples ({self.args.num_incontext_examples}) is greater than the number of available in context examples ({len(self.train_data)}). Using all examples."
            )
            self.args.num_incontext_examples = len(self.train_data)

    @classmethod
    def from_checkpoint(cls, path: str, model_name:str="") -> "IotaTrainer":
        
        model_name = "model.json" if model_name == "" else model_name
        model_fname = os.path.join(path, model_name)
        config_fname = os.path.join(path, "config.yaml")
        eval_results_fname = os.path.join(path, "results.json")

        # load config
        if not os.path.exists(config_fname):
            logger.warning(
                f"Config file not found at {config_fname}. Loading default config."
            )
            args = IotaArguments()
        else:
            with open(config_fname, "r") as f:
                config = yaml.safe_load(f)

            args = IotaArguments.load_from_config(config)

        trainer = IotaTrainer(args=args)

        # load trained trajectory, if available
        if not os.path.exists(model_fname):
            assert (
                "iota" not in args.mode
            ), f"Model file not found at {model_fname} when expected for {args}"
        else:
            with open(model_fname, "r") as f:
                train_data = json.load(f)
                train_data = IOTADataset.load_from_dict(train_data)
            trainer.train_data = train_data
            logger.info(f"Loaded previous training data from {model_fname}")

        # load eval results
        if not os.path.exists(eval_results_fname):
            logger.warning(
                f"Eval results file not found at {eval_results_fname}. Loading empty results"
            )
            eval_results = defaultdict(list)
        else:
            with open(eval_results_fname, "r") as f:
                eval_results = json.load(f)
            # make eval_results a defaultdict
            eval_results = defaultdict(list, eval_results)

        trainer.eval_results = eval_results

        try:
            trainer.update_steps = eval_results["val"][-1]["step"]
            trainer.current_epoch = eval_results["val"][-1]["epoch"]
            trainer.best_eval_score = max(
                [result["similarity"] for result in eval_results["val"]]
            )
        except Exception as e:
            logger.warning(f"Could not load previous training state: {e}")

        # print stats
        logger.info(f"Loaded trainer from checkpoint at {path}")
        logger.info(f"Current epoch: {trainer.current_epoch}")
        logger.info(f"Current step: {trainer.update_steps}")
        logger.info(f"Best eval score: {trainer.best_eval_score:.4f}")

        return trainer

    def setup_data(self):
        """
        Setup the training, validation, and test data
        """

        ### Set datapath and load data
        data_dir = set_absolute_path(self.args.datapath, self.args.package_dir)

        train_data = load_data(
            self.args.dataset, "train", self.args.author_key, data_dir
        )
        val_data = load_data(self.args.dataset, "val", self.args.author_key, data_dir)
        test_data = load_data(self.args.dataset, "test", self.args.author_key, data_dir)

        if self.args.small:
            train_data = train_data[:3]
            val_data = val_data[:2]

            logger.info(
                f"Running in small mode. Using only {len(train_data)} training samples and {len(val_data)} validation sample."
            )


        if self.args.train_size != -1:
            train_data = train_data[: self.args.train_size]

        self.train_data = IOTADataset(train_data)
        self.val_data = IOTADataset(val_data)
        self.test_data = IOTADataset(test_data)

    def setup_explanation_model(self):
        """
        Setup the explanation model
        """
        if self.args.exp_model_type is None:
            self.args.exp_model_type = self.args.model_type
        if self.args.exp_model_name_or_path is None:
            self.args.exp_model_name_or_path = self.args.model_name_or_path

        if self.args.exp_model_type == "transformers":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = HuggingfaceModel(self.args.exp_model_name_or_path, device)

        elif self.args.exp_model_type == "openai":

            model = OpenAIModel(self.args.exp_model_name_or_path)

        elif self.args.exp_model_type == "bedrock":

            model = BedRockModel(self.args.exp_model_name_or_path)

        else:
            raise NotImplementedError(
                f"Model type {self.args.exp_model_type} not implemented"
            )

        self.exp_model = model

    def setup_generation_model(
        self, model_type: str = "", model_name_or_path: str = ""
    ) -> None:
        """
        Setup the generation model
        """

        model_type = self.args.model_type if model_type == "" else model_type
        model_name_or_path = (
            self.args.model_name_or_path
            if model_name_or_path == ""
            else model_name_or_path
        )

        #### setup model
        if model_type == "transformers":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = HuggingfaceModel(model_name_or_path, device)

        elif model_type == "openai":

            model = OpenAIModel(model_name_or_path)

        elif model_type == "bedrock":

            model = BedRockModel(model_name_or_path)

        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")

        self.model = model
        self.model_max_length = self.model.get_max_length()

    def configure_mode(self):

        # set to actual number of all few shot examples if set to -1
        if self.args.num_incontext_examples == -1:
            self.args.num_incontext_examples = len(self.train_data)
        # set to actual number of all training samples if set to -1
        if self.args.train_size == -1:
            self.args.train_size = len(self.train_data)

        if "iota" not in self.args.mode:
            self.args.num_train_epochs = 0
            self.args.do_train = False
            self.args.do_eval = False

        if self.args.mode == "zeroshot":
            # no need to have any examples
            self.args.num_incontext_examples = 0

        if self.args.mode == "fewshot":
            # no need to generate any undesirable outputs or explanations
            assert (
                self.args.num_incontext_examples > 0
                or self.args.num_incontext_examples == -1
            ), "Number of in-context examples must be greater than 0 or set to -1 for fewshot mode"

        if "iota" in self.args.mode:
            assert (
                self.args.num_train_epochs > 0
            ), "Number of training epochs must be greater than 0 for IOTA mode"
            assert (
                self.args.num_incontext_examples > 0
                or self.args.num_incontext_examples == -1
            ), "Number of in-context examples must be greater than 0 or set to -1 for IOTA mode"
            assert (
                self.args.num_undesired_outputs > 0
                or self.args.num_undesired_outputs == -1
            ), "Number of undesirable outputs must be greater than 0 or set to -1 for IOTA mode"

            if self.args.mode == "iota-naive":
                self.args.summarize = False
                # no patience
                self.args.early_stopping = False

            if self.args.mode == "iota-full":
                self.args.summarize = True
                self.args.summarize_at_end = True
                # patience
                self.args.early_stopping = True
                self.args.early_stopping_patience = 5
                # eval every 5 steps
                self.args.eval_every_n_steps = 5

            if "no-explanations" in self.args.mode:
                self.args.summarize = False
                self.args.early_stopping = False
                self.args.generate_explanations = False

        # CoT (nothing to do for CoT)

        # Train optimized prompt 
        if self.args.mode == "opro" and not self.args.test:
            self.args.early_stopping = True
            self.args.early_stopping_patience = 5
            self.opro_prompt = self.optimize_OPRO_prompt()

    def update_steps_input_outputs(self, item: Dict[str, Any]) -> None:
        """For tracking intermediate steps and outputs during training
        """

        item["step"] = self.update_steps
        item["epoch"] = self.current_epoch
        self.steps_input_outputs.append(item)

        # save steps_input_outputs to file
        with open(
            os.path.join(self.args.output_dir, "steps_input_outputs.jsonl"), "a+"
        ) as f:
            f.write(json.dumps(item) + "\n")

    def _generate(self, input_text: str, model=None) -> str:
        """
        Generate output for a given input text

        Args:
            input_text (str): input text

        Returns:
            str: generated output
        """

        if model is None:
            model = self.model

        messages = [{"role": "user", "content": input_text}]

        generated_output = model.generate(
            messages=messages,
            gen_kwargs={
                "max_length": self.args.max_tokens,
                "temperature": self.args.temperature,
                "top_k": self.args.top_k,
                "top_p": self.args.top_p,
                "do_sample": True,
            },
        )

        # remove ``` from the output 
        generated_output = generated_output.replace("```", "").strip()

        if self.args.test: 
            print(f"Input text: {input_text}")
            print(f"Generated output: {generated_output}")
            breakpoint()

        return generated_output

    def optimize_OPRO_prompt(self):
        # iteratively self refine prompt based on the generated output and results 

        initial_prompt = "Let's think step by step."

        learned_prompt = initial_prompt

        patience = self.args.early_stopping_patience
        best_score = -1e6 
        prompt_score_pairing = []
        num_iterations = 0 
        while True > 0:
        # use training set as fewshot demonstrations
        # generate outputs for validation set and score them 
            scores = [] 
            for example in self.val_data: 
                input_text = get_opro_writing_prompt(example.as_dict(), learned_prompt)
                generated_output = self._generate(input_text)
                output = process_json_expected_response(generated_output, key="response")
                thought = process_json_expected_response(generated_output, key="thought")

                example.add_generated_output(output)
                # score the output
                reference_embedding = self.embedding_model(example.reference_output)
                generated_embedding = self.embedding_model(generated_output)
                similarity = self.embedding_model.compute_similarity(reference_embedding.reshape(1, -1), generated_embedding.reshape(1, -1))
                scores.append(similarity)
            # calculate average score
            avg_score = sum(scores) / len(scores)
            logger.info(f"Average score: {avg_score:.3f} using prompt: {learned_prompt}")
            avg_score = round(avg_score, 3)

            prompt_score_pairing.append((learned_prompt, avg_score))
            # sort by score in ascending order 
            prompt_score_pairing = sorted(prompt_score_pairing, key=lambda x: x[1])

            # keep only the top 10 prompts
            prompt_score_pairing = prompt_score_pairing[-10:]

            # update best score and adjust patience accordingly 
            if avg_score > best_score:
                best_score = avg_score
                patience = self.args.early_stopping_patience
            else: 
                patience -= 1
                logger.info(f"Didn't improve on previous best score: {best_score} > {avg_score}. Patience: {patience}")
                if patience == 0:
                    break 

            # update prompt based on the score trajectories
            optimization_prompt = get_opro_optimization_prompt(prompt_score_pairing, self.train_data)

            # generate output for the optimization prompt
            new_prompt = self._generate(optimization_prompt)
            learned_prompt = new_prompt

            logger.info(f"Generated new prompt: {new_prompt}")

            num_iterations += 1
        
        # get the best prompt and set it to opro_prompt
        best_prompt = prompt_score_pairing[-1][0]
        self.opro_prompt = best_prompt
    
    def prepare_writing_generation_prompt(
        self,
        target_example: IOTAExample,
        examples: List[IOTAExample],
        mode: str,
        num_undesired_outputs: int = None,
        desirable_only: bool = False,
        shuffle_examples: bool = False,
        order_generated_outputs_recent_first: bool = False,
        print_input: bool = False,
    ) -> str:
        """
        Prepare input text for the model

        Args:
            task (str): target task
            examples (List[IOTAExample]): few shot examples
            num_undesired_outputs (int, optional): number of undesirable outputs to show for each example. Defaults to None.
            desirable_only (bool, optional): show only desirable outputs. Defaults to False.
            shuffle_examples (bool, optional): shuffle examples. Defaults to False.
            order_generated_outputs_recent_first (bool, optional): order generated outputs with recent ones first. Defaults to False.
            print_input (bool, optional): print input text. Defaults to False.

        Returns:
            str: input text
        """

        if mode == "zeroshot": 
            input_text = target_example.task

        if shuffle_examples:
            random.seed(self.args.seed)
            random.shuffle(examples)

        if mode == "fewshot":

            input_text = get_few_shot_writing_prompt(target_example, examples, mode, DESIRED_OUTPUT_ALIAS, UNDESIRED_OUTPUT_ALIAS)

        if "iota" in mode: 

            # select examples based on coverage of explanations 

            # select examples based on coverage of style (choose ones farthest from first selected)

            for example in examples: 
                # if included in summary or marked to be ignored, exclude 
                example.generated_outputs = [gen_output for gen_output in example.generated_outputs if not gen_output.get("included_in_summary", False) and not gen_output.get("ignore", False)]

                # sample only the most recent generated outputs if num_undesired_outputs is > 0 
                if num_undesired_outputs is None:  # default to training args
                    num_undesired_outputs = self.args.num_undesired_outputs
                if num_undesired_outputs == -1:  # use all examples if set to -1
                    num_undesired_outputs = len(example.generated_outputs)
                example.generated_outputs = example.generated_outputs[-num_undesired_outputs:]

                # order generated outputs with recent ones first
                if order_generated_outputs_recent_first:
                    example.generated_outputs = example.generated_outputs[::-1]

                # remove explanations from generated outputs if args.generate_explanations is False
                if not self.args.generate_explanations:
                    for gen_output in example.generated_outputs:
                        gen_output.pop("explanation", None)

            input_text = get_few_shot_writing_prompt(target_example, examples, mode, DESIRED_OUTPUT_ALIAS, UNDESIRED_OUTPUT_ALIAS)

        if mode == "cot": 
            cot_style_guide_prompt = get_generate_cot_style_guides_prompt(target_example, examples)
            cot_style_guide = self._generate(cot_style_guide_prompt)
            input_text = get_cot_style_guide_based_writing_prompt(target_example, examples, cot_style_guide)

        if mode == "opro": 
            input_text = get_opro_writing_prompt(target_example, self.opro_prompt)
            
        if print_input:
            run_alias = get_run_alias(self.args)
            logger.debug(f"Full prompt for {target_example.task} for run alias {run_alias}:\n\n{input_text}")

        return input_text


    def generate_explanation(self, task: str, reference_text: str, generated_text: str, examples: List[IOTAExample]) -> str:
        """
        Generate explanation for how the bad output should be edited to make it more stylistically consistent with the good output

        Args:
            task (str): task description
            reference_text (str): style-consistent output
            generated_text (str): candidate text that is potentially style-inconsistent 
            examples (List[IOTAExample]): few shot examples to further help determine style consistency

        Returns:
            str: explanation
        """

        generate_explanation_prompt, template = get_generate_explanation_prompt(task, reference_text, generated_text, examples, self.args.use_other_examples_for_explanation)

        response = self._generate(generate_explanation_prompt, model=self.exp_model)

        explanation = process_json_expected_response(response, key="explanation")
        is_consistent = process_json_expected_response(response, key="is_consistent")
        if is_consistent.lower() not in ["yes", "no"]:
            logger.error(f"Invalid response for is_consistent: {is_consistent}. Should be 'yes' or 'no'. Proceeding as 'no' for now.")
            is_consistent = "no"
        is_consistent = is_consistent.lower() == "yes"
        
        self.update_steps_input_outputs(
            {
                "type": "explanation",
                "input": generate_explanation_prompt,
                "output": explanation,
                "prompt_template": template,
                "task": task,
                "reference_output": reference_text,
                "generated_output": generated_text,
            }
        )

        return {
            "explanation": explanation,
            "is_consistent": is_consistent,
        }

    def summarize_explanations_as_rules(self, skip_included: bool = True) -> str:
        """
        Summarize rules from the generated outputs and explanations.
        Also consider the previous summary if available.

        Returns:
            str: extracted rules in JSON format
        """
        # collect all suggested edits from the generated outputs
        if skip_included:
            # skipping if already included in the summary
            suggested_edits = [
                data["explanation"]
                for example in self.train_data
                for data in example.generated_outputs
                if data.get("explanation") is not None
                and not data.get("included_in_summary")
            ]
        else:
            suggested_edits = [
                data["explanation"]
                for example in self.train_data
                for data in example.generated_outputs
                if data.get("explanation") is not None
            ]

        formatted_suggested_edits = [f"> {edit}" for edit in suggested_edits]
        suggested_edits_text = "\n".join(formatted_suggested_edits)

        # summarize the suggested edits into guidance
        if self.guidance_summary:
            summarization_prompt_template = "Create an updated style guide given the previous style guide and suggested edits into a nonredundant and comprehensive bulleted list. Be specific but concise, limiting each guide to a single sentence. If there are disagreements in the style guide and the suggested edits, prioritize the suggested edits.\n\n# Previous style guide:\n{guidance_summary}\n\n# Suggested edits:\n{suggested_edits_text}"
            summarization_prompt = summarization_prompt_template.format(
                guidance_summary=self.guidance_summary,
                suggested_edits_text=suggested_edits_text,
            )
        else:
            summarization_prompt_template = "Create a style guide by summarizing the following suggested edits into a nonredundant and comprehensive bulleted list. Be specific but concise, limiting each guide to a single sentence.\n\n# Suggested edits:\n{suggested_edits_text}"
            summarization_prompt = summarization_prompt_template.format(
                suggested_edits_text=suggested_edits_text
            )

        logger.debug(f"Summarization prompt:\n\n{summarization_prompt}")

        summary = self._generate(summarization_prompt)

        self.update_steps_input_outputs(
            {
                "type": "summarization",
                "input": summarization_prompt,
                "output": summary,
                "prompt_template": summarization_prompt_template,
                "suggested_edits": suggested_edits,
                "previous_guidance_summary": self.guidance_summary,
            }
        )

        logger.debug(
            f"Summarized guidance results\n\n# Suggested edits: {suggested_edits_text}\n\n#Summary: \n> {summary}"
        )
        self.guidance_summary = summary

        # mark the examples as being included in the summary
        for example in self.train_data:
            for generated_output in example.generated_outputs:
                if generated_output.get("explanation") is not None:
                    generated_output["included_in_summary"] = True

    def sample_few_shot_examples(self, few_shot_examples: List[IOTAExample], n_samples:int, sampling_strategy:str, seed:int=42) -> List[IOTAExample]:

        n_samples = min(n_samples, len(few_shot_examples))

        if sampling_strategy == "random":
            random.seed(seed)
            sampled_few_shot_examples = random.sample(few_shot_examples, n_samples)
        elif sampling_strategy == "sequential":
            sampled_few_shot_examples = few_shot_examples[:n_samples]

        return sampled_few_shot_examples

    def train_iota_step(self, target_example, few_shot_examples) -> None:

        # skip if already was able to generate a consistent output for the target example from previous iterations
        if target_example.complete:
            logger.info(f"Skipping target example {target_example.task}. Already complete.")
            return

        # generate an output given the target example and sampled fewshot examples
        input_text = self.prepare_writing_generation_prompt(
            target_example, few_shot_examples, mode = self.args.mode
        ) 

        # Handle cases where max length has been exceeded. Necessary for smaller models 
        # 1: try summarizing the guidance from fewshot examples
        # 2. truncate by removing the last undesirable outputs from the few shot examples
        # 3. truncate by reducing number of fewshot samples
        tried_summarizing = False
        tried_removing_older_undesirable_outputs = False
        while len(input_text) > self.model_max_length:
            if (
                not tried_summarizing
                and self.args.summary_logic == "length_threshold"
            ):
                logger.warning(
                    f"Input text for task {target_example.task} exceeds max length. Attempting to summarize the guidance."
                )
                tried_summarizing = True
                self.summarize_explanations_as_rules()

            elif not tried_removing_older_undesirable_outputs:
                logger.warning(
                    f"Input text for task {target_example.task} exceeds max length. Attempting to remove undesirable outputs."
                )
                tried_removing_older_undesirable_outputs = True
                # ignore all but the latest generated output
                for example in few_shot_examples:
                    if example.generated_outputs:
                        for gen_output in example.generated_outputs[:-1]:
                            gen_output["ignore"] = True
            else:
                logger.warning(
                    f"Input text for task {target_example.task} exceeds max length. Attempting to reduce number of few shot examples."
                )
                few_shot_examples = few_shot_examples[:-1]

            input_text = self.prepare_writing_generation_prompt(
                target_example,
                few_shot_examples,
                mode = self.args.mode,
            )

        output = self._generate(input_text)

        self.update_steps_input_outputs(
            {
                "type": "output",
                "prompt_template": input_text,
                "few_shot_examples": [ex.as_dict() for ex in few_shot_examples],
                "output": output,
                "task": target_example.task,
                "reference_output": target_example.reference_outputs,
            }
        )

        # generate an explanation for the generated output
        if self.args.generate_explanations:
            # generate explanation
            style_analysis = self.generate_explanation(
                task = target_example.task, 
                reference_text = target_example.reference_outputs[0], 
                generated_text = output, 
                examples = few_shot_examples
            )
            explanation = style_analysis["explanation"]
            is_consistent = style_analysis["is_consistent"]
        else:
            explanation = None
        # update IOTA example with output and explanation
        if explanation: 
            if not is_consistent:
                target_example.add_generated_output(output, explanation)
            else: 
                if self.args.add_consistent_outputs:
                    target_example.add_reference_output(output) 
                logger.info("Generated output is consistent with the style. Skipping further iterations for this sample.")
                target_example.mark_complete() 

        if self.args.test:
            logger.debug(f"Target task: {target_example.task}")
            logger.debug(f"Reference output: {target_example.reference_outputs[0]}")
            logger.debug(f"Generated output: {output}")

            newline_char = "\n"
            # number of new lines in each output
            logger.debug(
                f"Number of new lines in reference output: {target_example.reference_outputs[0].count(newline_char)}"
            )
            logger.debug(
                f"Number of new lines in generated output: {output.count(newline_char)}"
            )

            # number of words in each output
            logger.debug(
                f"Number of words in reference output: {len(target_example.reference_outputs[0].split())}"
            )
            logger.debug(
                f"Number of words in generated output: {len(output.split())}"
            )

            logger.debug(f"Explanation: {explanation}")

    def train_iota_epoch(self) -> bool:
        """
        Train IOTA model for one epoch

        Args:
            debug (bool, optional): print debug logs. Defaults to True.

        Returns:
            bool: end training flag
        """

        # end training if all training samples are marked complete 
        if all([example.complete for example in self.train_data]):
            return True

        logger.info(f"Training epoch: {self.current_epoch}")
        self.train_data.shuffle()

        for target_idx in tqdm(range(len(self.train_data))):
            
            # get target example to train on 
            target_example = self.train_data[target_idx]

            # load fewshot examples, which are all examples except the target example
            few_shot_examples = (
                self.train_data[:target_idx] + self.train_data[target_idx + 1 :]
            )

            # sample fewshot examples if number of incontext examples is less than the total number of examples
            few_shot_examples = self.sample_few_shot_examples(
                few_shot_examples, self.args.num_incontext_examples - 1, self.args.incontext_selection_strategy, self.args.seed
            )

            self.train_iota_step(target_example=target_example, few_shot_examples=few_shot_examples) 
            self.update_steps += 1

            # summarize explanations as rules
            if self.args.summarize and self.args.summary_logic == "steps":
                if self.update_steps % self.args.summarize_after_n_steps == 0:
                    self.summarize_explanations_as_rules()

            # evaluate on validation data (stepwise logic)
            if self.val_data and self.args.do_eval and self.update_steps % self.args.eval_every_n_steps == 0:

                logger.info(
                    f"Running evaluation for epoch {self.current_epoch} at step {self.update_steps}..."
                )
                self.eval(self.val_data)

                # check for early stopping 
                if self.patience == 0 and self.args.early_stopping:

                    if (
                        self.args.summarize
                        and self.args.summary_logic == "patience"
                    ):
                        logger.info(
                            "Early stopping triggered. Summarizing explanations and resetting patience."
                        )
                        self.summarize_explanations_as_rules()
                        self.patience = self.args.early_stopping_patience
                        self.args.summary_logic = (
                            None  # stop summarizing after patience
                        )
                    else:
                        logger.info("Early stopping triggered. Exiting training.")
                        end_training = True
                        return end_training



    def train(self) -> None:
        """
        Full training loop for IOTA model
        """
        if "iota" not in self.args.mode: 
            logger.info(f"No training required for {self.args.mode} mode.")
            return

        for epoch in tqdm(range(self.args.num_train_epochs)):
            self.current_epoch = epoch
            end_training = self.train_iota_epoch()

            if self.args.summarize and self.args.summary_logic == "epochs":
                if self.current_epoch % self.args.summarize_after_n_epochs == 0:
                    self.summarize_explanations_as_rules()

            if end_training:
                break

            # save model every epoch for analysis purposes 
            self.save_model(fname=f"model_epoch{self.current_epoch}.json")
            # predict outputs for test data
            if self.args.do_predict: 
                self.eval(self.test_data, split="test")

        # do final eval after training is complete, only if it wasn't stopped from early stopping, otherwise it would be redundant
        if self.val_data and self.args.do_eval and self.patience != 0:
            self.eval(self.val_data)

        # also do eval after summarization
        if (
            self.args.summarize
            and self.args.summarize_at_end
            and self.val_data
            and self.args.do_eval
        ):
            self.summarize_explanations_as_rules()
            self.eval(self.val_data, save_model_name="model_after_summarization.json")

        logger.debug(f"Training complete. Results saved to {self.args.output_dir}")

    def get_embedding_similarity(
        self, reference_data: List[IOTAExample], model_outputs: List[str]
    ) -> float:

        reference_outputs = [example.reference_outputs[0] for example in reference_data]
        reference_embeddings = self.embedding_model.get_corpus_embeddings(
            reference_outputs
        )
        model_embeddings = self.embedding_model.get_corpus_embeddings(model_outputs)

        sim = self.embedding_model.compute_similarity(
            reference_embeddings, model_embeddings
        )

        return float(sim)  # for serialization and saving to json

    def eval_on_test(self):
        if self.args.mode == "iota":
            logger.info("Loading best model from training for iota...")
            self = IotaTrainer.from_checkpoint(self.args.output_dir)

        logger.info("Running prediction on test data...")
        return self.eval(self.test_data, split="test")

    def eval(self, data, split="val", save_model_name="model.json", should_save:bool=True) -> None:
        """
        Evaluate IOTA model on given data

        Returns:
            Dict: evaluation results
        """

        eval_results = {
            "epoch": self.current_epoch,
            "step": self.update_steps,
            "results": self.predict(data, split),
            "git_hash": get_git_hash(),
        }

        model_outputs = [
            result["generated_output"] for result in eval_results["results"]
        ]

        sim = self.get_embedding_similarity(self.val_data, model_outputs)
        eval_results["similarity"] = sim

        logger.info(
            f"Epoch {self.current_epoch} | Step {self.update_steps} | Similarity score on {split} set: {sim:.4f}"
        )

        if split == "val" and should_save:
            if sim > self.best_eval_score:
                logger.info(
                    f"Similarity score improved from {self.best_eval_score:.4f} to {sim:.4f}"
                )
                self.best_eval_score = sim
                self.patience = self.args.early_stopping_patience
                logger.info("Saving best model...")
                self.save_model(fname=save_model_name)
            else:
                logger.info(
                    f"Similarity score did not improve. Current best score: {self.best_eval_score:.4f}"
                )
                self.patience -= 1
                logger.info(f"Patience remaining: {self.patience}")

        # store per epoch eval results
        try:
            self.eval_results[split].append(eval_results)
        except KeyError as e:
            import pdb

            pdb.set_trace()
            raise e

        if should_save: 
            self.save_results(self.eval_results, "results")

        git_hash = get_git_hash()
        if split == "test": 
            run_alias = get_run_alias(self.args, epoch_num=self.current_epoch)
            args_dict = asdict(self.args)
            args_dict = {k: str(v) for k, v in args_dict.items()}

            # append to all outputs
            all_results_path = PACKAGE_DIR / "out/all_results.jsonl"
            with open(all_results_path, "a") as f:
                for pred in eval_results["results"]:
                    pred.update(args_dict)
                    # add date
                    pred["date"] = self.args.date
                    pred["git_hash"] = git_hash
                    pred["run_alias"] = run_alias

                    f.write(json.dumps(pred) + "\n")


        return eval_results

    def predict(self, data: IOTADataset, split: str) -> None:
        """
        Generate outputs for a given dataset

        Args:
            data (IOTADataset): dataset to predict on

        Returns:
            List[Dict]: prediction results
        """

        # set number of return sequences for test set
        num_return_sequences = (
            self.args.num_return_sequences_test
            if split == "test"
            else self.args.num_return_sequences_val
        )

        predict_results = []
        for example in tqdm(data):

            few_shot_examples = self.train_data

            # for some reason, random.sample complains and doesn't treat few_shot_examples as a proper sequence.
            few_shot_examples = few_shot_examples[:]  

            few_shot_examples = self.sample_few_shot_examples(
                few_shot_examples, self.args.num_incontext_examples, self.args.incontext_selection_strategy, self.args.seed
            )

            input_text = self.prepare_writing_generation_prompt(
                example, few_shot_examples, mode = self.args.mode
            )

            for _ in tqdm(range(num_return_sequences), total=num_return_sequences):

                output = self._generate(input_text)

                predict_results.append(
                    {
                        "task": example.task,
                        "input": input_text,
                        "reference_output": (
                            example.reference_outputs
                            if getattr(example, "reference_output", None) is not None
                            else None
                        ),
                        "generated_output": output,
                        "method": self.args.mode,
                        "run_alias": get_run_alias(self.args, epoch_num=self.current_epoch),
                        "benchmark": self.args.dataset,
                        "author_id": self.args.author_key,
                    }
                )

        logger.info(
            f"Gathered {len(predict_results)} ({len(data)} x {self.args.num_return_sequences_test}) predictions."
        )

        return predict_results

    def save_results(self, results, fname: str) -> None:

        savepath = os.path.join(self.args.output_dir, f"{fname}.json")

        with open(savepath, "w") as f:
            json.dump(results, f, indent=4)

    def save_model(self, output_dir: str = None, fname: str = "model.json") -> None:
        if output_dir is None:
            output_dir = self.args.output_dir

        savepath = os.path.join(output_dir, fname)

        # save updated train data
        with open(savepath, "w") as f:
            json.dump(self.train_data.as_dict(), f, indent=4)
