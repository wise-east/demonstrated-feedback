import unittest
from iota.evaluation_models import StyleEmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity


class TestStyleEmbeddings(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.model = StyleEmbeddingModel()

    def test_get_embedding(self):
        texts = ["r u a fan of them or something?", "Are you one of their fans?"]
        embeddings = self.model(texts)

        self.assertEqual(embeddings.shape, (2, 768))

        cosin_sim = cosine_similarity(
            embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1)
        )[0][0]

        print(f"cosine similarity between informal text and formal text: {cosin_sim}")

        # check that the cosine similarity is less than 0.5 (it should be ~0.4828)
        self.assertLess(cosin_sim, 0.1)
