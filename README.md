# skip_gram_phrase
A PyTroch implementation of the word2vec model with ability to embed phrases in compositional way. Built using  NLP library.

Package uses:
* python 3.6
* [allennlp](https://github.com/allenai/allennlp) - high quality NLP library
* [pylint](https://www.pylint.org/) for style checks: run `bash scripts/pylint.sh` 
* [mypy](http://mypy-lang.org/) for type checks: run `bash scripts/mypy.sh`  
* [pytest](https://docs.pytest.org/en/latest/) for unit testing: run `pytest -v` 

#### This project essentially reimplements following papers:
* [Efficient Estimation of Word Representations in Vector Space (Skip-Gram model)](https://arxiv.org/pdf/1301.3781.pdf)
* [Distributed Representations of Words and Phrases and their Compositionality (Negative Sampling & Common Words Subsampling)](https://arxiv.org/pdf/1310.4546.pdf)
* [Exploring phrase-compositionality in skip-gram models (Compositional Phrase Embeddings)](https://arxiv.org/pdf/1607.06208.pdf)
* Allows usage of RNNs and CNNs to do compositional ohrase embeddings


#### Usage:
1. Run 

####: Requirments:
1. Run `bash scripts/install_requirements.sh` to install allennlp and nltk 
