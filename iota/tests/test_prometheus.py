# usage: export VLLM_WORKER_MULTIPROC_METHOD=spawn ;  pytest test_prometheus.py 
from iota.generation_models import PrometheusModel, chat_templates
from test_iota_base import TestIotaBase
from iota.utils import load_data
import torch

class TestPrometheus(TestIotaBase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.config["model_name_or_path"] = "prometheus-eval/prometheus-7b-v2.0"
        cls.model = PrometheusModel(cls.config["model_name_or_path"], num_gpus=1)

        cls.train_data = load_data(
            cls.config["dataset"], "train", "all"
        )

    def test_prometheus(self):

        a_author_id = 0
        b_author_id = 1  

        text_a = self.train_data[a_author_id][0]['output']
        text_b = self.train_data[b_author_id][0]['output']
        demos = [sample['output'] for sample in self.train_data[a_author_id][1:-1]]

        score, feedback = self.model.eval_head_to_head(
            text_a=text_a,
            text_b=text_b,
            demos=demos,
        )

        assert score in "AB"

