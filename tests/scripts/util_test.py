from allennlp.common.testing import AllenNlpTestCase
from scripts.util import build_dataset_skip_gram, build_dataset_skip_phrase

class TestDataUtil(AllenNlpTestCase):
    def test_build_dataset_skip_gram(self):
        skipgram_data = build_dataset_skip_gram('tests/fixtures/lines.en')
        assert len(skipgram_data) == 4
        assert skipgram_data[3].split()[-2] == 'disgusting'  

    def test_build_dataset_skip_phrase(self):
        skipphrase_data = build_dataset_skip_phrase('tests/fixtures/lines.en')
        assert len(skipphrase_data) == 4
