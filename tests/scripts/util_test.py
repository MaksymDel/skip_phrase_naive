import os
from allennlp.common.testing import AllenNlpTestCase

from scripts.util import *

class TestDataUtil(AllenNlpTestCase):
    def test_build_dataset_skip_gram(self):
        out_path = 'tests/fixtures/lines-skipgram.en'
        try:
            os.remove(out_path)
        except OSError:
            pass
        build_dataset_skip_gram('tests/fixtures/lines.en', out_path, 2)
        with open(out_path, 'r') as f:
            examples = f.readlines()
            assert examples[-1] == '! disgusting'

    def test_process_line_skip_gram(self):
        l = "Jack of all trades , master of none ."
        training_examples = process_line_skip_gram(l, 2)
        
        assert len(training_examples) == 30
        assert training_examples[0] == ('Jack', 'of')
        assert training_examples[1] == ('Jack', 'all')

        assert training_examples[2] == ('of', 'Jack')
        assert training_examples[3] == ('of', 'all')
        assert training_examples[4] == ('of', 'trades')

        assert training_examples[5] == ('all', 'Jack')
        assert training_examples[6] == ('all', 'of')
        assert training_examples[7] == ('all', 'trades')
        assert training_examples[8] == ('all', ',')
        
        assert training_examples[9] == ('trades', 'of')
        assert training_examples[10] == ('trades', 'all')
        assert training_examples[11] == ('trades', ',')
        assert training_examples[12] == ('trades', 'master')

        assert training_examples[13] == (',', 'all')
        assert training_examples[14] == (',', 'trades')
        assert training_examples[15] == (',', 'master')
        assert training_examples[16] == (',', 'of')

        assert training_examples[17] == ('master', 'trades')
        assert training_examples[18] == ('master', ',')
        assert training_examples[19] == ('master', 'of')
        assert training_examples[20] == ('master', 'none')
        
        assert training_examples[21] == ('of', ',')
        assert training_examples[22] == ('of', 'master')
        assert training_examples[23] == ('of', 'none')
        assert training_examples[24] == ('of', '.')

        assert training_examples[25] == ('none', 'master')
        assert training_examples[26] == ('none', 'of')
        assert training_examples[27] == ('none', '.')

        assert training_examples[28] == ('.', 'of')
        assert training_examples[29] == ('.', 'none')

        l = "you like"
        training_examples = process_line_skip_gram(l, 2)
        assert training_examples[0] == ('you', 'like')
        assert training_examples[1] == ('like', 'you')

        l = "try"
        training_examples = process_line_skip_gram(l, 2)
        assert training_examples == None 

    def test_build_dataset_skip_phrase(self):
        skipphrase_data = build_dataset_skip_phrase('tests/fixtures/lines.en')
        pass