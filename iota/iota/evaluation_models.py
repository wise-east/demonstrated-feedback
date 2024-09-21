import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from typing import List
from loguru import logger
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class EmbeddingModel(ABC):

    @abstractmethod
    def __call__(self, texts) -> torch.Any:
        pass

    @abstractmethod
    def get_corpus_embeddings(self, texts: List[str]) -> torch.Tensor:
        pass

    def compute_similarity(self, emb1: np.array, emb2: np.array):
        return cosine_similarity(emb1, emb2)[0][0]
        
    def compute_max_similarity(self, candidate_embedding:np.array, comparison_embeddings: List[np.ndarray]):
        assert comparison_embeddings[0].shape == candidate_embedding.shape
        comparison_embeddings = np.array(comparison_embeddings).reshape(len(comparison_embeddings), -1)
        return cosine_similarity(comparison_embeddings, candidate_embedding).max()

# get LUAR embeddings (https://aclanthology.org/2021.emnlp-main.70.pdf)
class UAREmbeddingModel(EmbeddingModel):

    def __init__(
        self,
        model_name_or_path: str = "rrivera1849/LUAR-MUD",
        max_length: int = 512,
        device_id: int = 0,
    ):

        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.device = device
        self.max_length = max_length

    def __call__(self, text: str) -> torch.Any:
        return self.get_sample_embeddings(text)

    def get_sample_embeddings(self, text:str) -> torch.Tensor:
        return self.get_batch_embeddings([[text]])

    def get_corpus_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get the embeddings for a list of documents

        Args:
            texts: List of documents (strings)

        Returns:
            torch.Tensor: Embeddings of shape (num_documents, embedding_dim)
        """

        return self.get_batch_embeddings([texts])

    def get_batch_embeddings(self, episode_texts: List[List[str]]) -> torch.Tensor:
        """
        Get the embeddings for a batch of episodes (collection of documents)

        Args:
            episode_texts: List of episodes, where each episode is a list of documents (strings)

        Returns:
            torch.Tensor: Embeddings of shape (num_episodes, num_documents, embedding_dim)
        """

        batch_size = len(episode_texts)
        episode_length = len(episode_texts[0])

        # breakpoint() 

        # all episodes (collection of documents) should have the same number of documents. if not, embed them separately
        assert all(len(episode) == episode_length for episode in episode_texts)

        texts = [text for episode in episode_texts for text in episode]

        tokenized_text = self.tokenizer(
            texts,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self.device)

        tokenized_text["input_ids"] = tokenized_text["input_ids"].reshape(
            batch_size, episode_length, -1
        )
        tokenized_text["attention_mask"] = tokenized_text["attention_mask"].reshape(
            batch_size, episode_length, -1
        )

        with torch.no_grad():
            embeddings = self.model(**tokenized_text)

        embeddings = embeddings.cpu()

        return embeddings  # torch.Size([batch_size, hidden_size])


# get style embeddings: https://github.com/nlpsoc/Style-Embeddings?tab=readme-ov-file
class StyleEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_name_or_path: str = "AnnaWegmann/Style-Embedding",
        device_id: int = 0,
    ):
        self.model = SentenceTransformer(model_name_or_path, device=f"cuda:{device_id}")

    def __call__(self, text: str) -> torch.Any:
        return self.get_sample_embeddings(text)

    def get_sample_embeddings(self, text: str) -> torch.Tensor:
        return self.model.encode(text).reshape(1, -1)

    def get_corpus_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get the embeddings for a list of documents

        Args:
            texts: List of documents (strings)

        Returns:
            torch.Tensor: Embeddings of shape (num_documents, embedding_dim)
        """

        return self.model.encode(texts).mean(axis=0).reshape(1, -1)


# get average NLL and compare two outputs with a model
class LMProbabilitiesEvaluator:

    def __init__(
        self,
        model_name_or_path: str = "mistralai/Mistral-7B-v0.3",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.device = device

    def get_probabilities(self, text: str, max_length: int = 200) -> torch.Tensor:

        tokenized_text = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = tokenized_text["input_ids"]

        # raise warning if max_length is larger than input_ids
        if max_length > len(input_ids[0]):
            logger.warning(
                f"Warning: max_length {max_length} is larger than the input_ids length. Effective max_length will be to {len(input_ids[0])}"
            )
            max_length = len(input_ids[0])

        # truncate by max_length
        if max_length > 0:
            input_ids = input_ids[:, :max_length]

        with torch.no_grad():

            outputs = self.model(input_ids)[0]
            # probs = torch.log(outputs.logits.softmax(dim=-1)/100).detach()
            # sum_probs = probs.sum(dim=-1)

            sequence_probability = (
                torch.prod(
                    torch.softmax(outputs[0], dim=-1)[
                        torch.arange(len(input_ids[0])), input_ids[0]
                    ]
                ).item()
                * 100
            )

            probabilities = torch.softmax(outputs[0], dim=-1)
            token_probabilities = probabilities[
                torch.arange(len(input_ids[0])), input_ids[0]
            ]

            # calmp to avoid log(0)
            epsilon = 1e-10
            token_probabilities = torch.clamp(token_probabilities, min=epsilon)

            average_token_probability = token_probabilities.mean().item() * 100
            avg_nll = -torch.sum(torch.log(token_probabilities)) / max_length

        return sequence_probability, average_token_probability, avg_nll
