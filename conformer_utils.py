# Copyright 2024 plumiume.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TypeVar, Generic

import torch
import torch.nn as nn
from torch import Tensor, Size
import torchaudio

_ConvNd = TypeVar('_ConvNd', bound=nn.Conv1d | nn.Conv2d | nn.Conv3d)
_ConvTransposeNd = TypeVar('_ConvTransposeNd', bound=nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d)

def _batch_check(*batches: Size) -> Size:
    """Check that the given batch sizes are equal.

    Args:
        *batches (Size): Variable number of batch sizes to check.

    Returns:
        Size: The common batch size if all input sizes are equal.

    Raises:
        TypeError: If any of the batch sizes are not equal.
    """
    if not batches:
        return

    b0, *bs = batches
    for b in bs:
        if b0 != b:
            raise TypeError(f"Batch size mismatch: {b0} does not match {b}")

    return b0

def ctc_decode(x: Tensor, xlen: Tensor, blank: int = 0, padding_value: int = 0) -> tuple[Tensor, Tensor]:
    """Perform Connectionist Temporal Classification (CTC) decoding.

    Args:
        x (Tensor): The input tensor of shape (batch, time, features).
        xlen (Tensor): Lengths of the input sequences in `x`.
        blank (int, optional): The index representing the blank label. Defaults to 0.
        padding_value (int, optional): The value used for padding sequences. Defaults to 0.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing:
            - `y` (Tensor): The decoded output tensor.
            - `ylen` (Tensor): Lengths of the decoded sequences.
    """
    batch = _batch_check(x.shape[:-1], xlen.shape)

    size = torch.tensor(batch).prod().item()

    x = x.reshape(size, -1)
    xlen = xlen.reshape(size)

    y = nn.utils.rnn.pad_sequence(
        [
            torch.unique_consecutive(xi[xi != blank][:xleni])
            for xi, xleni in zip(x, xlen)
        ],
        batch_first=True, padding_value=padding_value
    ).reshape(*batch, -1)

    ylen = (y != padding_value).sum(-1)

    return y, ylen

def edit_distans(a: Tensor, alen: Tensor, b: Tensor, blen: Tensor) -> Tensor:
    """Compute the edit distance between two sequences of tensors.

    Args:
        a (Tensor): First sequence tensor.
        alen (Tensor): Lengths of the sequences in `a`.
        b (Tensor): Second sequence tensor.
        blen (Tensor): Lengths of the sequences in `b`.

    Returns:
        Tensor: The computed edit distances between each pair of sequences in `a` and `b`.
    """
    batch = _batch_check(a.shape[:-1], alen.shape, b.shape[:-1], blen.shape)

    size = torch.tensor(batch).prod().item()

    a = a.reshape(size, -1)
    alen = alen.reshape(size)
    b = b.reshape(size, -1)
    blen = blen.reshape(size)

    y = torch.tensor(
        [
            torchaudio.functional.edit_distance(ai[:aleni], bi[:bleni])
            for ai, aleni, bi, bleni in zip(a, alen, b, blen)
        ],
        device=a.device
    ).reshape(batch)

    return y

def word_error_rate(r: Tensor, rlen: Tensor, h: Tensor, hlen: Tensor) -> Tensor:
    """Compute the Word Error Rate (WER) between reference and hypothesis sequences.

    Args:
        r (Tensor): Reference sequences.
        rlen (Tensor): Lengths of the reference sequences.
        h (Tensor): Hypothesis sequences.
        hlen (Tensor): Lengths of the hypothesis sequences.

    Returns:
        Tensor: The computed Word Error Rate (WER) for each sequence.
    """
    dist = edit_distans(r, rlen, h, hlen)

    wer = dist / rlen

    return wer

def conv_size(size: Tensor, kernel_size: Tensor, stride: Tensor, padding: Tensor, dilation: Tensor) -> Tensor:
    """Compute the output size of a convolution operation.

    Args:
        size (Tensor): Input size.
        kernel_size (Tensor): Size of the convolution kernel.
        stride (Tensor): Stride of the convolution.
        padding (Tensor): Padding added to both sides of the input.
        dilation (Tensor): Dilation applied to the kernel.

    Returns:
        Tensor: The output size after the convolution operation.
    """
    return (size - (kernel_size - 1) * dilation + 2 * padding).div(stride, rounding_mode='floor') + 1

def conv_transpose_size(size: Tensor, kernel_size: Tensor, stride: Tensor, padding: Tensor, dilation: Tensor, output_padding: Tensor) -> Tensor:
    """Compute the output size of a transposed convolution operation.

    Args:
        size (Tensor): Input size.
        kernel_size (Tensor): Size of the convolution kernel.
        stride (Tensor): Stride of the transposed convolution.
        padding (Tensor): Padding added to both sides of the input.
        dilation (Tensor): Dilation applied to the kernel.
        output_padding (Tensor): Additional size added to the output.

    Returns:
        Tensor: The output size after the transposed convolution operation.
    """
    return (size - 1) * stride + (kernel_size - 1) * dilation - 2 * padding + output_padding + 1

class ConvSize(nn.Module, Generic[_ConvNd]):
    """Module for computing the output size of a convolution operation.

    Args:
        conv_module (_ConvNd): Convolution module (Conv1d, Conv2d, Conv3d).
    """

    def __init__(self, conv_module: _ConvNd):
        super().__init__()
        self.kernel_size = torch.nn.Parameter(torch.tensor(conv_module.kernel_size), requires_grad=False)
        self.stride = torch.nn.Parameter(torch.tensor(conv_module.stride), requires_grad=False)
        self.padding = torch.nn.Parameter(torch.tensor(conv_module.padding), requires_grad=False)
        self.dilation = torch.nn.Parameter(torch.tensor(conv_module.dilation), requires_grad=False)

    def forward(self, size: Tensor, tgt_dim: int | slice = slice(None)) -> Tensor:
        """Compute the output size of a convolution operation for a given input size.

        Args:
            size (Tensor): Input size.
            tgt_dim (int | slice, optional): Target dimension. Defaults to slice(None).

        Returns:
            Tensor: The output size after the convolution operation.
        """
        return conv_size(
            size,
            self.kernel_size[tgt_dim],
            self.stride[tgt_dim],
            self.padding[tgt_dim],
            self.dilation[tgt_dim]
        )

class ConvTransposeSize(nn.Module, Generic[_ConvTransposeNd]):
    """Module for computing the output size of a transposed convolution operation.

    Args:
        conv_module (_ConvNd): Transposed convolution module (ConvTranspose1d, ConvTranspose2d, ConvTranspose3d).
    """

    def __init__(self, conv_module: _ConvNd):
        super().__init__()
        self.kernel_size = nn.Parameter(conv_module.kernel_size, requires_grad=False)
        self.stride = nn.Parameter(conv_module.stride, requires_grad=False)
        self.padding = nn.Parameter(conv_module.padding, requires_grad=False)
        self.dilation = nn.Parameter(conv_module.dilation, requires_grad=False)
        self.output_dilation = nn.Parameter(conv_module.output_padding, requires_grad=False)

    def forward(self, size: Tensor, tgt_dim: int | slice = slice(None)) -> Tensor:
        """Compute the output size of a transposed convolution operation for a given input size.

        Args:
            size (Tensor): Input size.
            tgt_dim (int | slice, optional): Target dimension. Defaults to slice(None).

        Returns:
            Tensor: The output size after the transposed convolution operation.
        """
        return conv_transpose_size(
            size,
            self.kernel_size[tgt_dim],
            self.stride[tgt_dim],
            self.padding[tgt_dim],
            self.dilation[tgt_dim],
            self.output_dilation[tgt_dim]
        )
