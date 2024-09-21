from iota.generation_models import HuggingfaceModel, chat_templates
from test_iota_base import TestIotaBase

import torch


class TestIotaMistral(TestIotaBase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.config["model_name_or_path"] = "mistralai/Mistral-7B-Instruct-v0.2"
        cls.model = HuggingfaceModel(cls.config["model_name_or_path"], device=device)

    def test_transformers_model_generation(self):

        sample_messages = [
            {
                "role": "user",
                "content": """Perform the target task while using the following examples as stylistic guides.

### Example
Task: Give me 3 different things that a smartphone can do better than a landline phone.
Desirable Output: Smartphones can do many things that landline phones can't. They can access the internet, send text messages, and run apps. Would you like to learn more about other differences between smartphones and landline phones?

Target Task: How old is the Greek language?""",
            }
        ]

        output = self.model.generate(
            sample_messages,
            gen_kwargs={
                "max_length": self.config["max_tokens"],
                "do_sample": False,
            },
        )

        expected_greedy_output = "The Greek language, one of the oldest living languages in the world, has a rich history that spans over 3,000 years. While it's difficult to pinpoint an exact date for its origin, scholars generally agree that the earliest written records of the Greek language can be traced back to around 1400 BCE. Would you like to know more about the fascinating history of the Greek language or learn about its influence on modern languages?"

        assert output == expected_greedy_output

    def test_mistral_model_template(self):

        chat_template = self.model.tokenizer.chat_template
        assert chat_template == chat_templates[self.config["model_name_or_path"]]

        test_messages = [
            [{"role": "user", "content": "I want to book a flight to Paris"}],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "I want to book a flight to Paris"},
            ],
        ]

        tokenized_messages = [
            "<s>[INST] I want to book a flight to Paris [/INST]",
            "<s>[INST] You are a helpful assistant.\n\nI want to book a flight to Paris [/INST]",
        ]

        for messages, expected_text in zip(test_messages, tokenized_messages):
            templated_message = self.model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            assert templated_message == expected_text
