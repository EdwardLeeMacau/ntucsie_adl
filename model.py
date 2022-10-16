from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import RNN, Embedding, Linear
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.num_directions = 2 if bidirectional is True else 1
        self.hidden_size = hidden_size
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.rnn = RNN(
            input_size=self.embed.embedding_dim, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
            batch_first=True,
        )
        self.activation = nn.Tanh()
        self.classifier = nn.Sequential(
            Linear(self.encoder_output_size, num_class),
        )

    # State dict keeps keys and model parameters only.
    # To reconstruct the model, model metadata is needed.
    @classmethod
    def from_pretrained(cls, state_dict):
        raise NotImplementedError

    @property
    def encoder_output_size(self) -> int:
        return self.num_directions * self.hidden_size

    def forward(self, batch: dict) -> torch.Tensor:
        # Extract data from batch
        token, length = batch['token'], batch['length']

        embedded = self.embed(token)

        # Do not need ONNX compatibility. No needed enforce_sorted.
        out = pack_padded_sequence(embedded, length, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(out)
        out, length = pad_packed_sequence(out, batch_first=True)
        out = self.mask(out, length)

        out = self.classifier(self.activation(out))

        return out

    # https://discuss.pytorch.org/t/requesting-help-with-padding-packing-lstm-for-simple-classification-task/127713/4
    def mask(self, tensor, lengths):
        idx = torch.arange(max(lengths)).repeat(tensor.shape[0], 1)
        mask = idx == torch.unsqueeze(lengths - 1, axis = 1)
        return tensor[mask]

class SeqTagger(SeqClassifier):
    def forward(self, batch: dict) -> torch.Tensor:
        # Extract data from batch
        token, length = batch['token'], batch['length']

        embedded = self.embed(token)

        # Do not need ONNX compatibility. No needed enforce_sorted.
        out = pack_padded_sequence(embedded, length, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(out)
        out, length = pad_packed_sequence(out, batch_first=True)

        # Shape: [batch, length, num_class]
        out = self.classifier(self.activation(out))

        return out

# https://pytorch.org/vision/0.12/_modules/torchvision/ops/focal_loss.html
# Focus on hard targets
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "none") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default = 0.25
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
