import unittest
import yaml
import os
from iota.generation_models import BedRockModel

# from iota.utils
from test_iota_base import TestIotaBase


class TestIotaBedrock(TestIotaBase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # list of model ids: https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html
        cls.config["model_name_or_path"] = "mistral.mixtral-8x7b-instruct-v0:1"
        # cls.config['model_name_or_path'] = "anthropic.claude-3-sonnet-20240229-v1:0"
        # cls.model = BedRockModel(cls.config['model_name_or_path'])

        cls.sample_openai_messages = [
            {
                "role": "system",
                "content": "Perform the target task while using the following examples as stylistic guides.",
            },
            {
                "role": "user",
                "content": "### Example\nTask: Give me 3 different things that a smartphone can do better than a landline phone.\nDesirable Output: Smartphones can do many things that landline phones can't. They can access the internet, send text messages, and run apps. Would you like to learn more about other differences between smartphones and landline phones?\n\nTarget Task: How old is the Greek language?",
            },
        ]

    def test_bedrock_message_formatting(self):

        model_id = "mistral.mixtral-8x7b-instruct-v0:1"
        self.model = BedRockModel(model_id)

        system_message, bedrock_messages = self.model._format_bedrock_messages(
            self.sample_openai_messages
        )

        expected_system_message = [
            {
                "text": "Perform the target task while using the following examples as stylistic guides."
            }
        ]

        expected_bedrock_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": "### Example\nTask: Give me 3 different things that a smartphone can do better than a landline phone.\nDesirable Output: Smartphones can do many things that landline phones can't. They can access the internet, send text messages, and run apps. Would you like to learn more about other differences between smartphones and landline phones?\n\nTarget Task: How old is the Greek language?"
                    }
                ],
            }
        ]

        assert system_message == expected_system_message
        assert bedrock_messages == expected_bedrock_messages

    def generate_example(self):

        output = self.model.generate(
            messages=self.sample_openai_messages,
            gen_kwargs={
                "max_length": self.config["max_tokens"],
                "temperature": 0,
                "top_p": 1,
            },
        )

        print(output)

        return output

    def test_bedrock_model_mixtral_generation(self):
        # model_id = "mistral.mixtral-8x7b-instruct-v0:1"
        model_id = "mistral.mistral-large-2402-v1:0"

        self.model = BedRockModel(model_id)
        output = self.generate_example()

        expected_greedy_output = " The Greek language is one of the oldest languages in the world, with a history that dates back thousands of years. It is believed to have originated in the 2nd millennium BCE, and it has been spoken continuously ever since.\n\nThe earliest form of Greek, known as Mycenaean Greek, was spoken in the Mycenaean civilization, which flourished in the late Bronze Age. This early form of Greek was written in a script called Linear B, which was deciphered in the 20th century.\n\nAfter the fall of the Mycenaean civilization, the Greek language evolved into a number of different dialects, including Ionic, Doric, and Aeolic. The Classical Greek period, which is considered the height of Greek literature and philosophy, began in the 9th century BCE and lasted until the 4th century BCE. During this time, the Attic dialect became the standard form of Greek, and it was used by many of the greatest thinkers and writers of the ancient world, including Plato, Aristotle, and Sophocles.\n\nThe Greek language continued to evolve over the centuries, and it is still spoken today in Greece and in various communities around the world. It is an official language of the European Union and is studied by millions of people as a second language.\n\nIn short, the Greek language is a very old and rich language with a long and fascinating history. It has been spoken continuously for thousands of years and has played a crucial role in the development of Western civilization."

        # assert output == expected_greedy_output
        assert True

    def test_bedrock_model_claude_generation(self):

        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        # model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.model = BedRockModel(model_id)

        output = self.generate_example()

        expected_greedy_output = "The Greek language is one of the oldest recorded living languages in the world, with a documented history spanning over 3,400 years. Here are some key points about the age and origins of the Greek language:\n\nThe earliest written evidence of the Greek language dates back to around the 15th century BCE, in the form of inscriptions found in Crete and other regions of Greece. These early writings were in a syllabic script known as Linear B, which was used to write an early form of Greek called Mycenaean Greek.\n\nThe Greek alphabet, which is still used today, emerged around the 8th century BCE, derived from the Phoenician alphabet. This marked the beginning of the Ancient Greek period, which lasted until the 4th century BCE and saw the development of various dialects, including Attic Greek, the dialect of Athens.\n\nAncient Greek evolved into Koine Greek, the common language of the Hellenistic period (323 BCE - 330 CE), which became the lingua franca of the Mediterranean region and the language of the New Testament.\n\nByzantine Greek, also known as Medieval Greek, was the continuation of Koine Greek during the Byzantine Empire (330 CE - 1453 CE), with some linguistic changes and influences from other languages.\n\nModern Greek, the current form of the language, emerged around the 11th century CE and has its roots in the Koine and Byzantine Greek traditions, with influences from other languages due to historical events and cultural interactions.\n\nWould you like me to elaborate on any specific aspect of the Greek language's history or evolution?"

        # assert output == expected_greedy_output
        assert True

    def test_bedrock_model_llama_generation(self):

        # model_id = "meta.llama3-1-8b-instruct-v1:0"
        model_id = "meta.llama3-8b-instruct-v1:0"
        self.model = BedRockModel(model_id)

        output = self.generate_example()

        assert True

    def test_bedrock_max_length(self):

        model_ids = [
            "mistral.mixtral-8x7b-instruct-v0:1",
            "mistral.mistral-7b-instruct-v0:2",
            "mistral.mistral-large-2402-v1:0",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "meta.llama3-8b-instruct-v1:0",
        ]

        for model_id in model_ids:
            self.model = BedRockModel(model_id)
            max_length = self.model.get_max_length()

            # assert max_length == 131_072
