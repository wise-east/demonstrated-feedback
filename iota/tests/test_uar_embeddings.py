import unittest
from iota.evaluation_models import UAREmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity


class TestLUAREmbeddings(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.model = UAREmbeddingModel()

    def test_get_embedding(self):
        episode_texts = [
            # things teenagers may say
            ["Ayeee", "what's up", "bro"],
            # things shakespearians may say
            [
                "To be or not to be",
                "that is the question",
                "whether tis nobler in the mind to suffer",
            ],
        ]
        embeddings = self.model(episode_texts)
        self.assertEqual(embeddings.shape, (2, 512))

        cosin_sim = cosine_similarity(
            embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1)
        )[0][0]

        print(
            f"cosine similarity between teenagers text and shakespearians: {cosin_sim}"
        )

        # check that the cosine similarity is less than 0.5 (it should be ~0.4828)
        self.assertLess(cosin_sim, 0.5)
