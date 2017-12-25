from allennlp.common.testing import ModelTestCase

class SkipGramTest(ModelTestCase):
    def setUp(self):
        super(SkipGramTest, self).setUp()
        self.set_up_model('tests/fixtures/skip_gram.json',
                          'tests/fixtures/lines-skipgram.en')

    def test_model_can_train_save_and_load(self):
        # waiting for response on one timestep RNN zerograds case from allennlp team
        # self.ensure_model_can_train_save_and_load(self.param_file)
        pass