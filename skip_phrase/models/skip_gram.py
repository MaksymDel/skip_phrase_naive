from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from torch.nn.modules.linear import Linear
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util

@Model.register("skip_gram")
class SkipGram(Model):
    """
    This ``Model`` implements skip gram algorithm. We assume we're given a
    pivot_phrase and we predict context_word.
    The basic model structure: we'll embed all tokens of the pivot_phrase separatly, and
    encode it with Seq2VecEncoder, getting a single vector representing the content of this phrase.
    We'll then project this content vector onto vocabulary to make a guess of the true context word.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    pivot_phrase_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the pivot_phrase to a vector.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 pivot_phrase_encoder: Seq2VecEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SkipGram, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("shared_words_vocab")
        self.pivot_phrase_encoder = pivot_phrase_encoder
        self.projection_layer = Linear(self.pivot_phrase_encoder.get_output_dim(), self.num_classes, bias=True)

        self.loss = torch.nn.CrossEntropyLoss()

        if text_field_embedder.get_output_dim() != pivot_phrase_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the pivot_phrase_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            title_encoder.get_input_dim()))

        initializer(self)

    @overrides
    def forward(self,
                pivot_phrase: Dict[str, torch.LongTensor],
                context_word: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        embedded_pivot_phrase = self.text_field_embedder(pivot_phrase)
        pivot_phrase_mask = util.get_text_field_mask(pivot_phrase)
        encoded_pivot_phrase = self.pivot_phrase_encoder(embedded_pivot_phrase, pivot_phrase_mask)

        logits = self.projection_layer(encoded_pivot_phrase)
        reshaped_log_probs = logits.view(-1, self.num_classes)

        class_probabilities = F.softmax(reshaped_log_probs, dim=-1)

        output_dict = {"logits": reshaped_log_probs, "class_probabilities": class_probabilities, "encoder_outputs": encoded_pivot_phrase}

        if context_word is not None:
            loss = self.loss(reshaped_log_probs, context_word.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SkipGram':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        pivot_phrase_encoder = Seq2VecEncoder.from_params(params.pop("pivot_phrase_encoder"))
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   pivot_phrase_encoder=pivot_phrase_encoder,
                   initializer=initializer,
                   regularizer=regularizer)