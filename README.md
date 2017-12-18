# Skip-gram for phrases
A PyTroch implementation of the word2vec model with ability to embed phrases in compositional way.

That is, original word2vec package embeds only small portion of ngrams in atomic way (e.g. `right_away`) while this package allows
to get an embedding for any ngram in compositional way (e.g. `right` + `away`). 

Negative sampling, subsampling, and sparse gradient updates are used to achieve computational efficiency.

#### This project essentially reimplements following papers:
* [Exploring phrase-compositionality in skip-gram models](https://arxiv.org/pdf/1607.06208.pdf) (Compositional Phrase Embeddings)
* [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf) (Negative Sampling & Common Words Subsampling)
* [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) (Skip-Gram model)

and also allows using RNNs and CNNs for compositional phrase embeddings

#### Usage:
1. Run 

#### Requirments:
1. Run `bash scripts/install_requirements.sh` to install allennlp and nltk 

#### Packages used:
* python 3.6
* [allennlp](https://github.com/allenai/allennlp) - high quality NLP library
* [pylint](https://www.pylint.org/) for style checks: run `bash scripts/pylint.sh` 
* [mypy](http://mypy-lang.org/) for type checks: run `bash scripts/mypy.sh`  
* [pytest](https://docs.pytest.org/en/latest/) for unit testing: run `pytest -v` 
