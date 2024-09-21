from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import os, json, time
import botocore.errorfactory
from loguru import logger
import torch
from openai import OpenAI
from .prompts import form_pairwise_eval_prompt
from .utils import extract_answer_and_explanation
import boto3

from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import RELATIVE_PROMPT_WO_REF
import botocore

MISTRAL_CHAT_TEMPLATE = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

MODELS_NO_SYSTEM_MESSAGE = ["mistral.mixtral-8x7b-instruct-v0:1"]

chat_templates = {"mistralai/Mistral-7B-Instruct-v0.2": MISTRAL_CHAT_TEMPLATE}




class UnifiedGenerationModel(ABC):

    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path

    @abstractmethod
    def generate(
        self,
        messages: List[
            Dict[str, str]
        ],  # openai messages format [{role: str, content: str}]
        gen_kwargs: Dict[str, Any],
    ) -> str:
        pass

    @abstractmethod
    def get_max_length(self) -> int:
        pass

    def prepare_batch_prediction_input(self, 
            text_a: str,
            text_b: str,
            demos: List[str],
            generate_explanation: bool = True, 
            custom_id: str = "request-1"
        ): 
        
        prompt = form_pairwise_eval_prompt(
            demos, text_a, text_b, generate_explanation=generate_explanation
        )

        messages = [{"content": prompt, "role": "user"}]
        request_payload = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-2024-08-06",
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0,
                "top_p": 1,
            }
        }

        return request_payload


    def eval_head_to_head(
        self,
        text_a: str,
        text_b: str,
        demos: List[str],
        generate_explanation: bool = True,
    ) -> Tuple[str]:
        prompt = form_pairwise_eval_prompt(
            demos, text_a, text_b, generate_explanation=generate_explanation
        )

        messages = [{"content": prompt, "role": "user"}]

        outputs = self.generate(
            messages,
            gen_kwargs={
                "max_length": 1024,
                "temperature": 0,
                "do_sample": False,
                "top_p": 1,
            },
        )

        answer, explanation = extract_answer_and_explanation(outputs, generate_explanation)

        if answer not in ["A", "B"]:
            logger.error(
                f"Invalid answer: {answer}.\nExplanation: {explanation}.\nTrying again..."
            )
            return self.eval_head_to_head(
                text_a, text_b, demos, generate_explanation=generate_explanation
            )

        return answer, explanation


class BedRockModel(UnifiedGenerationModel):
    # reference: https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started-api-ex-python.html#getting-started-api-ex-python-converse

    def __init__(self, model_name_or_path: str):
        super().__init__(model_name_or_path)
        self.model_name_or_path = model_name_or_path
        self.client = boto3.client("bedrock-runtime")

    def get_max_length(self) -> int:

        known_lengths = {
            "mistral.mixtral-8x7b-instruct-v0:1": 131_072,
            "mistral.mistral-7b-instruct-v0:2": 32_768,
            "mistral.mistral-large-2402-v1:0": 32_768,
            "meta.llama3-8b-instruct-v1:0": 8_000,
            "anthropic.claude-3-5-sonnet-20240620-v1:0": 200_000,
            "anthropic.claude-3-sonnet-20240229-v1:0": 200_000,
            "anthropic.claude-3-haiku-20240307-v1:0": 200_000,
        }

        length = known_lengths.get(self.model_name_or_path, None)

        if length is None:
            raise ValueError(
                f"Model {self.model_name_or_path} not found in known lengths"
            )

        return length

    def generate(
        self,
        messages: List[Dict[str, str]],  # messages in openai format
        gen_kwargs: Dict[str, Any],
    ) -> str:

        # format messages to bedrock format
        system_message, bedrock_messages = self._format_bedrock_messages(messages)

        # generate response with bedrock api
        try: 
            response = self._generate(system_message, bedrock_messages, gen_kwargs)
        except Exception as e:
            logger.error(f"ThrottlingException: {e}. Sleeping for 10s and trying again...")
            time.sleep(10)
            response = self.generate(messages, gen_kwargs)

        return response

    def _format_bedrock_messages(
        self, openai_messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:

        # bedrock messages take on a slightly different format: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#conversation-inference-call

        if openai_messages[0]["role"] == "system":
            system_message = [{"text": openai_messages[0]["content"]}]
            openai_messages = openai_messages[1:]
        else:
            system_message = None

        bedrock_messages = []

        for message in openai_messages:

            if message["role"] != "assistant" and message["role"] != "user":
                raise ValueError(
                    f"Invalid role: {message['role']}. Must be either 'assistant' or 'user'"
                )

            bedrock_messages.append(
                {"role": message["role"], "content": [{"text": message["content"]}]}
            )

        return system_message, bedrock_messages

    def _generate(
        self,
        system_message: str,
        messages: List[Dict[str, str]],
        gen_kwargs: Dict[str, Any],
    ) -> str:

        inference_config = {
            "maxTokens": gen_kwargs["max_length"],
            "temperature": gen_kwargs["temperature"],
            "topP": gen_kwargs["top_p"],
        }

        if (
            self.model_name_or_path in MODELS_NO_SYSTEM_MESSAGE
            or system_message is None
        ):

            if system_message:
                # add system message to the first user message and separate by \n\n
                messages[0]["content"][0][
                    "text"
                ] = f"{system_message[0]['text']}\n\n{messages[0]['content'][0]['text']}"

            response = self.client.converse(
                modelId=self.model_name_or_path,
                messages=messages,
                inferenceConfig=inference_config,
            )

        else:

            response = self.client.converse(
                modelId=self.model_name_or_path,
                system=system_message,
                messages=messages,
                inferenceConfig=inference_config,
            )

        response_text = response["output"]["message"]["content"][0]["text"]

        usage = response["usage"]
        latency = response["metrics"]["latencyMs"] / 1000

        logger.debug(f"Usage: {usage}")
        logger.debug(f"Latency: {latency}s")

        return response_text


class HuggingfaceModel(UnifiedGenerationModel):

    def __init__(self, model_name_or_path: str, device: str):
        super().__init__(model_name_or_path)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.chat_template = chat_templates[model_name_or_path]

        self.max_input_token_length = (
            self.tokenizer.model_max_length
        )  # doesn't seem to be correct
        logger.info(
            f"Model `{model_name_or_path}`'s max input token length: {self.max_input_token_length}"
        )

    def get_max_length(self) -> int:
        return self.max_input_token_length

    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        return self.tokenizer(text, return_tensors="pt")

    def _cleanup(self, text: str) -> str:

        tokens_to_remove = ["<s>", "</s>", "[INST]", "[/INST]", "<pad>", "<eos>"]

        for token in tokens_to_remove:
            text = text.replace(token, "")

        return text

    def generate(
        self, messages: List[Dict[str, str]], gen_kwargs: Dict[str, Any]
    ) -> str:

        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        untokenized_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        output = self.model.generate(inputs, **gen_kwargs)

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = self._cleanup(response)
        untokenized_input = self._cleanup(untokenized_input)

        response = response.split(untokenized_input)[
            1
        ].strip()  # mistral model provides the input as part of the response, so we need to remove it

        return response


class PrometheusModel:
    # reference: https://github.com/prometheus-eval/prometheus-eval/tree/main/libs/prometheus-eval

    def __init__(
        self,
        model_name_or_path: str = "prometheus-eval/prometheus-7b-v2.0",
        num_gpus: int = torch.cuda.device_count()
    ):
        self.model_name_or_path = model_name_or_path  # one of "prometheus-eval/prometheus-7b-v2.0", "prometheus-eval/prometheus-8x7b-v2.0"
        self.model = VLLM(
            model=model_name_or_path,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=num_gpus,
        )
        self.judge = PrometheusEval(
            model=self.model, relative_grade_template=RELATIVE_PROMPT_WO_REF
        )
        self.rubric = "Does the response make sense and is it stylistically similar to the author's writing?"
        self.instruction = """You are an impartial evaluator. Below are samples of an author's writing and two options. Which option is more likely to have been written by the author based on style similarity to the samples given as AUTHOR'S WRITING?

> AUTHOR'S WRITING:
{demos_text}"""

    def batch_eval_head_to_head(
        self, responses_from_a: List[str], responses_from_b: List[str], demos: List[str]
    ):

        demos_text = "\n\n".join(
            [f"SAMPLE #{i+1}:\n{demo}" for i, demo in enumerate(demos)]
        )

        prompt = self.instruction.format(demos_text=demos_text)

        instructions = [prompt for _ in range(len(responses_from_a))]

        feedbacks, scores = self.judge.relative_grade(
            instructions=instructions,
            responses_A=responses_from_a,
            responses_B=responses_from_b,
            rubric=self.rubric,
        )

        return scores, feedbacks

    def eval_head_to_head(
        self,
        text_a: str,
        text_b: str,
        demos: List[str],
        generate_explanation: bool = True,
    ) -> Tuple[str]:

        demos_text = "\n\n".join(
            [f"SAMPLE #{i+1}:\n{demo}" for i, demo in enumerate(demos)]
        )

        prompt = self.instruction.format(demos_text=demos_text)

        data = {
            "instruction": prompt,
            "response_A": text_a,
            "response_B": text_b,
            "rubric": self.rubric,
        }

        feedback, score = self.judge.single_relative_grade(**data)

        return score, feedback


class OpenAIModel(UnifiedGenerationModel):

    def __init__(
        self,
        model_name_or_path: str,
    ):
        super().__init__(model_name_or_path)
        self.client = OpenAI()


    def get_max_length(self) -> int:
        return 128_000  # from https://platform.openai.com/docs/models/gpt-4o

    def generate(self, messages: List[Dict[str, str]], gen_kwargs) -> str:

        try: 
            completion = self.client.chat.completions.create(
                model=self.model_name_or_path,
                messages=messages,
                max_tokens=gen_kwargs["max_length"],
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                # no top_k for openai
            )

            response = completion.choices[0].message.content
        except Exception as e: 
            logger.error(f"ThrottlingException: {e}. Sleeping for 5s and trying again...")
            time.sleep(5)
            response = self.generate(messages, gen_kwargs)

        return response
