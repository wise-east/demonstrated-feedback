from typing import List, Tuple, Dict, Any
from jinja2 import Environment, FileSystemLoader
from iota.utils import PACKAGE_DIR
import re

prompt_templates_dir = PACKAGE_DIR / "iota" / "prompt_templates"
jinja_env = Environment(loader=FileSystemLoader(prompt_templates_dir))

def clear_excess_newlines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)

def form_single_binary_prompt(
    demos: List[str], text: str, generate_explanation: bool = False
) -> str: 

    template = jinja_env.get_template("eval_single_binary_template.jinja")

    template_data = {
        "demos": demos,
        "text": text,
        "generate_explanation": generate_explanation
    }

    single_binary_eval_prompt = template.render(**template_data)
    return clear_excess_newlines(single_binary_eval_prompt)

def form_single_range_prompt(
    demos: List[str], text: str, generate_explanation: bool = False
) -> str: 

    template = jinja_env.get_template("eval_single_range_template.jinja")

    template_data = {
        "demos": demos,
        "text": text,
        "generate_explanation": generate_explanation
    }

    single_range_eval_prompt = template.render(**template_data)
    return clear_excess_newlines(single_range_eval_prompt)

def form_pairwise_eval_prompt(
    demos: List[str], text_a: str, text_b: str, generate_explanation: bool = True
) -> str:

    template = jinja_env.get_template("eval_pairwise_template.jinja")

    template_data = {
        "demos": demos,
        "text_a": text_a,
        "text_b": text_b,
        "generate_explanation": generate_explanation
    }

    pairwise_eval_prompt = template.render(**template_data)

    return clear_excess_newlines(pairwise_eval_prompt)


def get_dummy_eval_prompt(
    num_demo_examples: int, generate_explanation: bool = False
) -> str:

    if num_demo_examples == 1:
        demos = ["hi"]
    elif num_demo_examples > 1:
        demos = ["hi", "yo"]
    text_a = "hi"
    text_b = "hello"

    return form_pairwise_eval_prompt(demos, text_a, text_b, generate_explanation)


def get_few_shot_writing_prompt(target_example, examples, mode, desired_output_alias, undesired_output_alias=None) -> str: 

    if mode == "fewshot": 
        examples_data = {
            "examples": [
                {
                    "task": ex.task,
                    "reference_output": ex.reference_outputs[0]
                } for ex in examples
            ],
            "desired_output_alias": desired_output_alias,
            "target_task": target_example.task
        }

    elif "iota" in mode: 
        examples_data = {
            "examples": [
                {
                    "task": ex.task,
                    "reference_output": ex.reference_outputs[0],
                    "generated_outputs": ex.generated_outputs
                } for ex in examples
            ],
            "desired_output_alias": desired_output_alias,
            "undesired_output_alias": undesired_output_alias,
            "target_task": target_example.task
        }
    else: 
        raise ValueError(f"Mode {mode} not supported for few-shot writing prompts")

    template = jinja_env.get_template("iota_template.jinja")

    fewshot_writing_prompt = template.render(**examples_data)

    return clear_excess_newlines(fewshot_writing_prompt)

def get_generate_cot_style_guides_prompt(target_example, examples) -> str: 

    template_data = {
        "target_task": target_example["task"],
        "examples": examples
    }

    template = jinja_env.get_template("cot_generate_style_guide_template.jinja")

    cot_style_guides = template.render(**template_data)

    return clear_excess_newlines(cot_style_guides)

def get_cot_style_guide_based_writing_prompt(target_example: Dict[str, Any], examples: List[Dict[str, Any]], cot_style_guide:str) -> str: 

    template_data = {
        "target_task": target_example["task"],
        "examples": examples,
        "cot_style_guide": cot_style_guide
    }

    template = jinja_env.get_template("cot_style_guide_based_writing_template.jinja")

    cot_style_guide_based_writing = template.render(**template_data)

    return clear_excess_newlines(cot_style_guide_based_writing)


def get_opro_writing_prompt(target_example: Dict[str, Any], opro_prompt:str) -> str: 

    template_data = {
        "target_task": target_example["task"],
        "opro_prompt": opro_prompt
    }

    template = jinja_env.get_template("opro_writing_template.jinja")

    opro_writing = template.render(**template_data)

    return clear_excess_newlines(opro_writing)

def get_opro_optimization_prompt(prompt_score_pairing: List[Tuple[str, float]], examples: Dict[str, Any]) -> str: 

    template_data = {
        "prompt_score_pairing": prompt_score_pairing,
        "examples": examples
    }

    template = jinja_env.get_template("opro_optimization_template.jinja")

    opro_optimization = template.render(**template_data)

    return clear_excess_newlines(opro_optimization)


def get_generate_explanation_prompt(task, reference_text, generated_text, examples: List[Dict[str, Any]], use_other_examples: bool) -> str: 

    template_data = {
        "task": task,
        "reference_text": reference_text,
        "generated_text": generated_text,
        "examples": examples,
        "use_other_examples": use_other_examples
    }

    template = jinja_env.get_template("generate_explanations_template.jinja")

    generate_explanation = template.render(**template_data)

    empty_template = template.render(
        task="<task>",
        reference_text="<reference_text>",
        generated_text="<generated_text>"
    )

    return clear_excess_newlines(generate_explanation), empty_template 