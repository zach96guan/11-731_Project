from typing import Dict, Optional, List, Any

import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, InputVariationalDropout
from allennlp.modules.matrix_attention.legacy_matrix_attention import LegacyMatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, masked_softmax, weighted_sum, replace_masked_values
from allennlp.training.metrics import PearsonCorrelation


@Model.register("esim_mt")
class ESIM(Model):
    """
    This ``Model`` implements the ESIM sequence model described in `"Enhanced LSTM for Natural Language Inference"
    <https://www.semanticscholar.org/paper/Enhanced-LSTM-for-Natural-Language-Inference-Chen-Zhu/83e7654d545fbbaaf2328df365a781fb67b841b4>`_
    by Chen et al., 2017.
    This code was taken from the AllenNLP repo, and modified for predicting (continuous) scores for MT system outputs

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``mtref`` and ``mtsys`` ``TextFields`` we get as input to the
        model.
    encoder : ``Seq2SeqEncoder``
        Used to encode the mtref and mtsys.
    similarity_function : ``SimilarityFunction``
        This is the similarity function used when computing the similarity matrix between encoded
        words in the mtref and words in the mtsys.
    projection_feedforward : ``FeedForward``
        The feedforward network used to project down the encoded and enhanced mtref and mtsys.
    inference_encoder : ``Seq2SeqEncoder``
        Used to encode the projected mtref and mtsys for prediction.
    output_feedforward : ``FeedForward``
        Used to prepare the concatenated mtref and mtsys for prediction.
    output_logit: FeedForward,
	legacy input that does nothing
    dropout : ``float``, optional (default=0.5)
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 similarity_function: SimilarityFunction,
                 projection_feedforward: FeedForward,
                 inference_encoder: Seq2SeqEncoder,
                 output_feedforward: FeedForward,
                 output_logit: FeedForward,
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder

        self._matrix_attention = LegacyMatrixAttention(similarity_function)
        self._projection_feedforward = projection_feedforward

        self._inference_encoder = inference_encoder

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            self.rnn_input_dropout = InputVariationalDropout(dropout)
        else:
            self.dropout = None
            self.rnn_input_dropout = None

        self._output_feedforward = output_feedforward

        self._num_labels = 1

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        check_dimensions_match(encoder.get_output_dim() * 4, projection_feedforward.get_input_dim(),
                               "encoder output dim", "projection feedforward input")
        check_dimensions_match(projection_feedforward.get_output_dim(), inference_encoder.get_input_dim(),
                               "proj feedforward output dim", "inference lstm input dim")

        self._metric = PearsonCorrelation()
        self._loss = torch.nn.MSELoss()

        initializer(self)

    def forward(self,  # type: ignore
                ref: Dict[str, torch.LongTensor],
                mt: Dict[str, torch.LongTensor],
                score: torch.IntTensor = None  # pylint:disable=unused-argument
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        ref : Dict[str, torch.LongTensor]
            From a ``TextField``
        mt : Dict[str, torch.LongTensor]
            From a ``TextField``
        score : torch.IntTensor, optional (default = None)
            From a ``NumericField`` 

        Returns
        -------
        An output dictionary consisting of:

        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # print(worker)
        embedded_mtref = self._text_field_embedder(ref)
        embedded_mtsys = self._text_field_embedder(mt)
        mtref_mask = get_text_field_mask(ref).float()
        mtsys_mask = get_text_field_mask(mt).float()

        # apply dropout for LSTM
        if self.rnn_input_dropout:
            embedded_mtref = self.rnn_input_dropout(embedded_mtref)
            embedded_mtsys = self.rnn_input_dropout(embedded_mtsys)

        # encode mtref and mtsys

        # Shape: (batch_size, mtref/sys_length, modeldim*2 =600)
        encoded_mtref = self._encoder(embedded_mtref, mtref_mask)
        encoded_mtsys = self._encoder(embedded_mtsys, mtsys_mask)

        # Shape: (batch_size, mtref_length, mtsys_length)
        similarity_matrix = self._matrix_attention(encoded_mtref, encoded_mtsys)

        # Shape: (batch_size, mtref_length, mtsys_length)
        p2h_attention = masked_softmax(similarity_matrix, mtsys_mask)
        # Shape: (batch_size, mtref_length, embedding_dim)
        attended_mtsys = weighted_sum(encoded_mtsys, p2h_attention)

        # Shape: (batch_size, mtsys_length, mtref_length)
        h2p_attention = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), mtref_mask)
        # Shape: (batch_size, mtsys_length, embedding_dim)
        attended_mtref = weighted_sum(encoded_mtref, h2p_attention)

        # the "enhancement" layer
        # Shape: (batch_size, mtref/sys_length, modeldim *2 * 4=2400)
        mtref_enhanced = torch.cat(
                [encoded_mtref, attended_mtsys,
                 encoded_mtref - attended_mtsys,
                 encoded_mtref * attended_mtsys],
                dim=-1
        )
        mtsys_enhanced = torch.cat(
                [encoded_mtsys, attended_mtref,
                 encoded_mtsys - attended_mtref,
                 encoded_mtsys * attended_mtref],
                dim=-1
        )

        # The projection layer down to the model dimension.  Dropout is not applied before
        # projection.

        # Shape: (batch_size, mtref/sys_length, modeldim =300)
        projected_enhanced_mtref = self._projection_feedforward(mtref_enhanced)
        projected_enhanced_mtsys = self._projection_feedforward(mtsys_enhanced)


        # Run the inference layer
        if self.rnn_input_dropout:
            projected_enhanced_mtref = self.rnn_input_dropout(projected_enhanced_mtref)
            projected_enhanced_mtsys = self.rnn_input_dropout(projected_enhanced_mtsys)
        # Shape: (batch_size, mtref/sys_length, modeldim*2 =600)
        v_ai = self._inference_encoder(projected_enhanced_mtref, mtref_mask)
        v_bi = self._inference_encoder(projected_enhanced_mtsys, mtsys_mask)

        # The pooling layer -- max and avg pooling.
        # (batch_size, model_dim*2 = 600)
        v_a_max, _ = replace_masked_values(
                v_ai, mtref_mask.unsqueeze(-1), -1e7
        ).max(dim=1)
        # (batch_size, model_dim * 2 = 600)
        v_b_max, _ = replace_masked_values(
                v_bi, mtsys_mask.unsqueeze(-1), -1e7
        ).max(dim=1)
        # (batch_size, model_dim * 2 = 600)
        v_a_avg = torch.sum(v_ai * mtref_mask.unsqueeze(-1), dim=1) / torch.sum(
                mtref_mask, 1, keepdim=True)
        # (batch_size, model_dim * 2 = 600)
        v_b_avg = torch.sum(v_bi * mtsys_mask.unsqueeze(-1), dim=1) / torch.sum(
                mtsys_mask, 1, keepdim=True)

        # Now concat
        # (batch_size, model_dim * 2 * 4)
        v_all = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        # the final MLP -- apply dropout to input, and MLP applies to output & hidden
        if self.dropout:
            v_all = self.dropout(v_all)

        pred  = self._output_feedforward(v_all)

        output_dict = {'pred': pred}

        if score is not None:
            loss = self._loss(pred, score)
            self._metric(pred, score)
            output_dict["loss"] = loss

        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'pearson': self._metric.get_metric(reset)}
