import unittest
import yaml
import os
from iota.utils import load_data, PACKAGE_DIR, HOME_DIR, process_other_args
from iota.iota_trainer import IotaTrainer, IotaArguments, IOTADataset, DESIRED_OUTPUT_ALIAS, UNDESIRED_OUTPUT_ALIAS
from iota.generation_models import OpenAIModel
from iota.run_iota import run_iota
from iota.prompts import get_generate_explanation_prompt, get_few_shot_writing_prompt, get_cot_style_guide_based_writing_prompt, get_generate_cot_style_guides_prompt, get_opro_optimization_prompt, get_opro_writing_prompt
from scipy.stats import bootstrap
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import pickle


def find_significant_win_rate(
    total_count, baseline_win_rate=0.5, alpha=0.05, step=0.001
):
    current_win_rate = baseline_win_rate

    while True:
        win_count = int(current_win_rate * total_count)
        successes = np.array([win_count])
        trials = np.array([total_count])

        _, p_value = proportions_ztest(successes, trials, value=baseline_win_rate)

        if p_value < alpha:
            return current_win_rate

        current_win_rate += step


class TestIotaBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Get the directory of the current file (test_iota.py)
        # Construct the path to the config file relative to the current directory
        config_path = os.path.join(PACKAGE_DIR, "iota", "configs", "config_test.yaml")
        # Load the configuration
        with open(config_path, "r") as f:
            cls.config = yaml.safe_load(f)

        cls.train_data = load_data(
            cls.config["dataset"], "train", cls.config["author_key"]
        )

        # TODO replace directory with more up to date sample and include it into the repo
        cls.model_dir = os.path.join(
            PACKAGE_DIR, "out/gpt-4o-2024-05-13/cmcc/2024-08-08_19-34-15"
        )


class TestIotaUtils(TestIotaBase):

    def test_load_data(self):

        for split in ["train", "test"]:
            data = load_data(self.config["dataset"], split, self.config["author_key"])

            # each sample should contain "prompt" and "output" keys
            for sample in data:
                self.assertIn("prompt", sample)
                self.assertIn("output", sample)

    def test_find_significant_win_rate(self):

        # Parameters
        total_count = 20
        baseline_win_rate = 0.5
        alpha = 0.05

        # Find the required win rate
        required_win_rate = find_significant_win_rate(
            total_count, baseline_win_rate, alpha
        )

        print(f"Total games: {total_count}")
        print(f"Baseline win rate: {baseline_win_rate:.2f}")
        print(f"Significance level (alpha): {alpha:.2f}")
        print(
            f"Required win rate for statistical significance: {required_win_rate:.4f}"
        )

        # Calculate the minimum number of wins needed
        min_wins = int(required_win_rate * total_count)
        print(f"Minimum number of wins needed: {min_wins}")

        # Calculate the z-statistic and p-value for the required win rate
        successes = np.array([min_wins])
        trials = np.array([total_count])
        z_stat, p_value = proportions_ztest(successes, trials, value=baseline_win_rate)
        print(f"P-value: {p_value:.4f}")

    def test_bootstrap(self):

        win_rate = 0.57
        total_count = 400
        win_count = int(win_rate * total_count)
        lose_count = total_count - win_count
        data = [1] * win_count + [0] * lose_count
        data = (data,)
        bootstrap_ci = bootstrap(
            data, np.mean, confidence_level=0.95, random_state=1, method="percentile"
        )

        print(bootstrap_ci.confidence_interval)
        print(bootstrap_ci.standard_error)

    def test_stat_sig(self):
        # Observed win rate
        win_rate = 0.57
        total_count = 200
        win_count = int(win_rate * total_count)

        # Baseline win rate (e.g., 50% win rate)
        baseline_win_rate = 0.5

        # Define the number of successes (wins) and the number of trials (total samples)
        successes = np.array([win_count])
        trials = np.array([total_count])

        # Perform the one-sample proportion z-test
        z_stat, p_value = proportions_ztest(successes, trials, value=baseline_win_rate)

        # Print the results
        print(f"Z-statistic: {z_stat}")
        print(f"P-value: {p_value}")

        # Interpret the results
        alpha = 0.05  # significance level
        if p_value < alpha:
            print(
                "The observed win rate is statistically significant compared to the baseline."
            )
        else:
            print(
                "The observed win rate is not statistically significant compared to the baseline."
            )

    def test_process_other_args(self):

        test_cases = [
            {
                "other_args": ["--model_name_or_path", "gpt2"],
                "processed_args": ["--model_name_or_path=gpt2"],
            },
            {
                "other_args": ["--model_name_or_path", "gpt2", "--max_length=50"],
                "processed_args": ["--model_name_or_path=gpt2", "--max_length=50"],
            },
            {
                "other_args": ["--model_name_or_path", "gpt2", "50"],
                "processed_args": None,
            },
            {
                "other_args": ["--model_name_or_path", "gpt2", "--test"],
                "processed_args": ["--model_name_or_path=gpt2", "--test=true"],
            },
            {
                "other_args": [
                    "--model_name_or_path",
                    "gpt2",
                    "--test",
                    "--max_length=50",
                ],
                "processed_args": [
                    "--model_name_or_path=gpt2",
                    "--test=true",
                    "--max_length=50",
                ],
            },
        ]

        for test_case in test_cases:
            if test_case["processed_args"] is None:
                with self.assertRaises(ValueError):
                    process_other_args(test_case["other_args"])
                continue
            else:
                processed_args = process_other_args(test_case["other_args"])
                self.assertEqual(processed_args, test_case["processed_args"])

    def test_model_loading(self):
        trainer = IotaTrainer.from_checkpoint(self.model_dir)
        for train_sample in trainer.train_data:
            # make sure that train_sample.generated_outputs is not empty
            self.assertTrue(len(train_sample.generated_outputs) > 0)

        trainer.args.output_dir = self.model_dir
        trainer.save_model()

    def test_load_iota_args_from_config(self):
        config = self.config
        args = IotaArguments.load_from_config(config)

        # check that all keys shared by args and config have same value
        for key in args.__dict__.keys():
            if key in config:
                self.assertEqual(args.__dict__[key], config[key])

    def test_opro_optimization_template(self): 

        args = IotaArguments(
            mode="opro",
            dataset="cmcc",
            test=True,
        )

        trainer = IotaTrainer(args)
        sample = trainer.train_data[0]
        initial_prompt = "Let's think step by step"
        output = get_opro_writing_prompt(sample, initial_prompt)

        prompt_score_pairing = [("hi", 0.5), ("yo", 0.6)]
        optimization_prompt_output = get_opro_optimization_prompt(prompt_score_pairing, trainer.train_data)

    def test_opro_optimization(self): 

        args = IotaArguments(
            mode="opro",
            dataset="cmcc",
            test=True,
        )

        trainer = IotaTrainer(args)
        trainer.optimize_OPRO_prompt()

        return 

    def test_explanation_generation(self):

        for benchmark in ["cmcc", "ccat50"]: 

            args = IotaArguments(
                mode="iota-naive",
                dataset=benchmark,
                test=True,
                small=True
            )

            trainer = IotaTrainer(args)

            sample = trainer.train_data[0]
            few_shot_examples = trainer.train_data[1:]
            input_text = trainer.prepare_writing_generation_prompt(sample, few_shot_examples, args.mode)
            output = trainer._generate(input_text)

            style_analysis = trainer.generate_explanation(
                sample.task, sample.reference_outputs[0], output, examples=few_shot_examples
            )

    def test_iota_generate_jinja_template(self): 

        args = IotaArguments(
            mode="iota-naive",
            dataset="cmcc",
            test=True,
        )

        trainer = IotaTrainer(args)
        sample = trainer.train_data[0]
        few_shot_examples = trainer.train_data[1:]
        for example in few_shot_examples: 
            example.add_generated_output("hi", "yo")
            example.add_generated_output("hey", "yo")

        few_shot_writing_prompt = get_few_shot_writing_prompt(sample, few_shot_examples, "iota-naive", DESIRED_OUTPUT_ALIAS, UNDESIRED_OUTPUT_ALIAS)

        print(few_shot_writing_prompt)

    def test_cot_generate_jinja_template(self):
            
        args = IotaArguments(
            mode="cot",
            dataset="cmcc",
            test=True,
        )

        trainer = IotaTrainer(args)
        sample = trainer.train_data[0]
        few_shot_examples = trainer.train_data[1:]
        for example in few_shot_examples: 
            example.add_generated_output("hi", "yo")

        cot_style_guide_prompt = get_generate_cot_style_guides_prompt(sample, few_shot_examples)

        style_guide_writing_prompt = get_cot_style_guide_based_writing_prompt(sample, few_shot_examples, "<some dummy style guide>")

    def test_cot(self): 

        args = IotaArguments(
            mode="cot",
            dataset="cmcc",
            test=True,
        )

        trainer = IotaTrainer(args)
        sample = trainer.train_data[0]
        few_shot_examples = trainer.train_data[1:]

        cot_style_guide_prompt = get_generate_cot_style_guides_prompt(sample, few_shot_examples)
        style_guide = trainer._generate(cot_style_guide_prompt)

        style_guide_writing_prompt = get_cot_style_guide_based_writing_prompt(sample, few_shot_examples, style_guide)
        output = trainer._generate(style_guide_writing_prompt)

    def test_summarization(self):

        trainer = IotaTrainer.from_checkpoint(self.model_dir)

        trainer.summarize_explanations_as_rules(skip_included=True)

        print(trainer.guidance_summary)

        # check that all existing examples were marked as included in summary
        for example in trainer.train_data:
            for gen_output in example.generated_outputs:
                self.assertTrue(gen_output["included_in_summary"])

    # execute iota iota/configs/config_test.yaml
    def test_run_iota(self):
        # Placeholder for actual test
        self.assertTrue(True)
        return  # only run this test manually
        args = IotaArguments.load_from_config(self.config)
        run_iota(args)


    def test_data_equivalence_with_ditto(self): 

        for benchmark in ["cmcc", "ccat50"]: 

            # make sure that two files are equivalent 
            ditto_fp= f"{PACKAGE_DIR}/../demonstrated-feedback/benchmarks/{benchmark}/processed/{benchmark}_train.pkl"
            ours_fp= f"{PACKAGE_DIR}/benchmarks/{benchmark}/processed/{benchmark}_train.pkl"

            with open(ditto_fp, 'rb') as f:
                ditto_data = pickle.load(f)

            with open(ours_fp, 'rb') as f:
                our_data = pickle.load(f)

            for author_id in ditto_data: 
                for i in range(len(ditto_data[author_id])):
                    assert ditto_data[author_id][i] == our_data[author_id][i], f"{author_id}, {i}: \n\n{ditto_data[author_id][i] } \n\n != \n\n {our_data[author_id][i]}"

    
