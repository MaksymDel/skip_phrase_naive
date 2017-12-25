from typing import Dict
import json
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

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("skip_gram_examples")
class SkipGramExamplesDatasetReader(DatasetReader):
    """
    Reads a file containing pairs of words separated by `@@@` and extracted from some text corpus,
    and creates a dataset suitable for skip-gram model training.

    You can use `scripts/build_dataset.py` file to extract such word pairs from your data.

    Expected format for each input line: "word1@@@word2" (without quotes)
    
    The output of ``read`` is a list of ``Instance`` s with the fields:
        pivot_word: ``TextField``
        context_word: ``TextField``

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    

