## test whether probabilites for text that makes sense are higher than for text that does not make sense

import unittest
from iota.evaluation_models import LMProbabilitiesEvaluator


class TestLMProbs(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        cls.lm_probs_evaluator = LMProbabilitiesEvaluator(device="cpu")

    def test_lm_probs(self):
        # text that makes sense
        author_text = "I absolutely feel as though the Catholic Church has to change its ways in order to adapt to the 21st century. Two major issues come to my mind when I think of the Catholic Church; those issues are Gay marriage and abortion. I am a very liberal person and I see eye to eye with many of the political views held my leftist in the political world. Thus I believe that gays should be allowed to marry just as any other couple that is in love should have the right to marriage and obviously I also believe that women should have the choice to abort a pregnancy should the circumstance present itself. The church does not believe that either should be allowed and its justification comes from religious doctrine that was given or written for us thousands of years ago. I don't want to seem like I am mooching the church but to be honest things change drastically in a matter of decades let alone in a matter of centuries. We have the proper medical procedures to perform abortions with out harming the women. Furthermore, to quote church doctrine that is older than time, and say that a man can not marry another man because in the bible it says that marriage is between a man and a women is preposterous. In the biblical times if somebody were to come out and pronounce there homosexuality they could be subject to blood letting or a lobotomy, you know to get the \"gay out of them.\" Heck if you want to go back even further there are numerous written accounts of homosexual acts within the Spartan male communities as well as with in the Greek schools for men. Older Spartans would regularly engage in acts of homosexuality with younger Spartans and it was perfectly permissible. This leads me right into my next topic which would be older clergy men, specifically priests over stepping their bounds and engaging in sexual acts with younger church going boys. What is the difference between Sparta and the churches in Boston where Priests were raping boys? In Sparta it was part of the culture, in America it is a clear act of rape preformed by a priest that is sexually uneasy due to years of celibacy and perhaps in many cases unrequited love. The Catholic Church needs to start to allow their priests to marry. It may seem ridiculous due to centuries of church rule against it, however, a married sexually fulfilled priest could perform his job just as well as a celibate one, and without the worry of perhaps raping a boy. Rabbi's and ministers of all other religions are permitted to marry and live normal lives all the while devoting themselves to their work and their God. Along the same lines, women are also allowed to be Rabbi's and Ministers, to not allow women to be Catholic Priests is a major gender discrimination that our country has overlooked for far too long. It is time for the Church to meet the times, steps need to be taken to modernize what is now becoming an outdated major religious movement."

        # text that doesn't make sense
        gibberish_text = "I will not be as much as incarnated morally as to condone or denounce any other mans marriage preferences because for me to go totally around would be unjustly accusing my self of hypocritization should any woman catch there flirtacious blances of my winking eye or discover a covert females letter to me in the wooded area behind the athletic center, porkey as that as much as any of us may love the traditionaic view of marriage being between a man and female for banning such activities to be legaly incorparated into any other country's laws would be a discriminatory and mentaly unhealthy path to our world's future self improvement and develiompment. If men where suppose to beat men with sticks as part of thier marital rapeing, then only men beating men with sticks shall marry other men beaten only with men's sticks shall be allowed to rape from that day forth. Porkey kinkle is wise as only a real huneman understanding world porgy should be allowed to pory in the porgey pond the pokey Porkeys of waddell. Hallow brook schools powerfull debate frat."

        # gibberish_text ="asdfjiczk;jf wiozdjfaklwej adsifzxdflkweq aksdjflkasjdlfk jasdkfjaskdjflkasdjfiwe lksdj  asdjfk 454 burrat lasdivk"

        author_seq_prob, author_avg_tok_prob, author_avg_nll = (
            self.lm_probs_evaluator.get_probabilities(author_text)
        )
        gibberish_seq_prob, gibberish_avg_tok_prob, gibberish_avg_nll = (
            self.lm_probs_evaluator.get_probabilities(gibberish_text)
        )

        print(
            f"Author text: sequence probability: {author_seq_prob:.3f}% \t| average token probability: {author_avg_tok_prob:.3f}% \t| average NLL: {author_avg_nll:.3f}"
        )
        print(
            f"Gibberish text: sequence probability: {gibberish_seq_prob:.3f}% \t| average token probability: {gibberish_avg_tok_prob:.3f}% \t| average NLL: {gibberish_avg_nll:.3f}"
        )
