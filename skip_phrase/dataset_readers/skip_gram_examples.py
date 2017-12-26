from typing import Dict
import logging

from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer 
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import WordSplitter, JustSpacesWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("skip_gram_examples")
class SkipGramExamplesDatasetReader(DatasetReader):
    """
    Reads a file containing pairs of words separated by `@@@` and extracted from some text corpus,
    and creates a dataset suitable for skip-gram model training.

    You can use `scripts/build_dataset.py` file to extract such word pairs from your data.

    Expected format for each input line: "pivot_phrase@@@context_word" (without quotes)
    
    The output of ``read`` is a list of ``Instance`` s with the fields:
        pivot_phrase: ``TextField``
            Phrase or just word we are using as input 
        context_word: ``TextField``
            Word we are trying to predict

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split pivot token if it is a phrase
        Default is JustSpacesWordSplitter since we expect examples dataset to be
        truekased and tokenized externaly.
    pivot_phrase_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    context_word_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 pivot_phrase_token_indexers: Dict[str, TokenIndexer] = None,
                 context_word_token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self._tokenizer = tokenizer or WordTokenizer()
        # should become phrase words token indexer in skip-phrase 
        self._pivot_phrase_token_indexers = pivot_phrase_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._context_word_token_indexers = context_word_token_indexers or {"tokens": SingleIdTokenIndexer()}

    def read(self, file_path):
        instances = []
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(tqdm.tqdm(data_file.readlines())):
                line = line.strip("\n")
                if not line:
                    continue
                line = line.split("@@@")
                pivot_phrase = line[0]
                context_word = line[1] 
                instances.append(self.text_to_instance(pivot_phrase, context_word))
        if not instances:
            raise ConfigurationError("No instances read!")
        return Dataset(instances)
        
    def text_to_instance(self, pivot_phrase: str, context_word: str) -> Instance:
        # tokenizing and indexing of the pivot phrase should occure here
        tokenized_pivot_phrase = self._tokenizer.tokenize(pivot_phrase)
        pivot_phrase_field = TextField(tokenized_pivot_phrase, self._pivot_phrase_token_indexers)
        if context_word is not None:
            context_word_field = LabelField(context_word, label_namespace="shared_words_vocab") # do not hardcode it here
        fields = {'pivot_phrase': pivot_phrase_field, 'context_word': context_word_field}
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'SkipGramExamplesDatasetReader':
        tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        phrase_token_indexer = TokenIndexer.dict_from_params(params.pop('pivot_phrase_token_indexers', {}))
        target_word_indexer = TokenIndexer.dict_from_params(params.pop('context_word_token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(tokenizer=tokenizer, pivot_phrase_token_indexers=phrase_token_indexer, 
                                        context_word_token_indexers=target_word_indexer)

