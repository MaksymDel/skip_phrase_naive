# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase

from skip_phrase.dataset_readers import SkipGramExamplesDatasetReader


class TestSkipGramExamplesDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = SkipGramExamplesDatasetReader()
        dataset = reader.read('tests/fixtures/lines-skipgram.en')

        instance1 = "the@@@quick".split('@@@')
        instance2 = "the@@@brown".split('@@@')
        instance3 = "quick@@@the".split('@@@')
        instance4 = "quick@@@brown".split('@@@')


        assert len(dataset.instances) == 128
        fields = dataset.instances[0].fields
        assert [t.text for t in fields["pivot_phrase"].tokens] == [instance1[0]]
        assert fields["context_word"].label == instance1[1]

        fields = dataset.instances[1].fields
        assert [t.text for t in fields["pivot_phrase"].tokens] == [instance2[0]]
        assert fields["context_word"].label == instance2[1]

        fields = dataset.instances[2].fields
        assert [t.text for t in fields["pivot_phrase"].tokens] == [instance3[0]]
        assert fields["context_word"].label == instance3[1]

        fields = dataset.instances[3].fields
        assert [t.text for t in fields["pivot_phrase"].tokens] == [instance4[0]]
        assert fields["context_word"].label == instance4[1]


        # skip-phrase mini-test
        fields = reader.text_to_instance("i go to", "school").fields
        assert [t.text for t in fields["pivot_phrase"].tokens] == ['i', 'go', 'to']
        assert fields["context_word"].label == 'school'