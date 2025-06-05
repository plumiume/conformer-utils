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

from typing import TypeVar, Generic, Literal, overload

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
        return torch.Size()

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

    size = int(torch.tensor(batch).prod().item())

    x = x.reshape(size, -1)
    xlen = xlen.reshape(size)

    results1: list[tuple[Tensor]] = [
        torch.unique_consecutive(xi[xi != blank][:xleni], return_inverse=True)
        for xi, xleni in zip(x, xlen)
    ]
    outputs = [ur[0] for ur in results1]

    y = nn.utils.rnn.pad_sequence(
        outputs, batch_first=True, padding_value=padding_value
    ).reshape(*batch, -1)

    ylen = (y != padding_value).sum(-1)

    return y, ylen

def ctc_decode_with_indices(x: Tensor, xlen: Tensor, blank: int = 0, padding_value: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    """Perform Connectionist Temporal Classification (CTC) decoding with inverse indices.
    Args:
        x (Tensor): The input tensor of shape (batch, time, features).
        xlen (Tensor): Lengths of the input sequences in `x`.
        blank (int, optional): The index representing the blank label. Defaults to 0.
        padding_value (int, optional): The value used for padding sequences. Defaults to 0.
    Returns:
        tuple[Tensor, Tensor, Tensor]: A tuple containing:
            - `y` (Tensor): The decoded output tensor.
            - `ylen` (Tensor): Lengths of the decoded sequences.
            - `inverse_indices` (Tensor): Indices mapping the original sequence to the decoded sequence.
    """

    batch = _batch_check(x.shape[:-1], xlen.shape)

    size = int(torch.tensor(batch).prod().item())

    x = x.reshape(size, -1)
    xlen = xlen.reshape(size)

    results2: list[tuple[Tensor, Tensor]] = [
        torch.unique_consecutive(xi[xi != blank][:xleni], return_inverse=True)
        for xi, xleni in zip(x, xlen)
    ]
    outputs = [ur[0] for ur in results2]
    inverse_indices = [ur[1] for ur in results2]

    y = nn.utils.rnn.pad_sequence(
        outputs, batch_first=True, padding_value=padding_value
    ).reshape(*batch, -1)

    ylen = (y != padding_value).sum(-1)

    return y, ylen, torch.stack(inverse_indices, dim=0)

def edit_distans(a: Tensor, alen: Tensor, b: Tensor, blen: Tensor) -> Tensor:
    """Compute the edit distance between two sequences of tensors.

    Args:
        a (Tensor): First sequence tensor, with shape (*B, L).
        alen (Tensor): Lengths of the sequences in `a`, with shape (*B).
        b (Tensor): Second sequence tensor, with shape (*B, L').
        blen (Tensor): Lengths of the sequences in `b`, with shape (*B).

    Returns:
        Tensor: The computed edit distances between each pair of sequences in `a` and `b`.
    """
    batch = _batch_check(a.shape[:-1], alen.shape, b.shape[:-1], blen.shape)
    size = int(torch.tensor(batch).prod().item())

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

def word_error_rate(r: Tensor, rlen: Tensor, h: Tensor, hlen: Tensor, normalize: bool = True) -> Tensor:
    """Compute the Word Error Rate (WER) between reference and hypothesis sequences.

    Args:
        r (Tensor): Reference sequences, with shape (*B, L).
        rlen (Tensor): Lengths of the reference sequences, with shape (*B).
        h (Tensor): Hypothesis sequences, with shape (*B, L').
        hlen (Tensor): Lengths of the hypothesis sequences, with shape (*B).
        normalize (bool, optional): Whether to normalize the WER by the length of the reference sequences. Defaults to True.

    Returns:
        Tensor: The computed Word Error Rate (WER) for each sequence.
    """
    dist = edit_distans(r, rlen, h, hlen)

    if normalize:
        return dist / rlen
    else:
        return dist

# def is_error(r: Tensor, rlen: Tensor, h: Tensor, hlen: Tensor) -> Tensor:

#     batch = _batch_check(r.shape[:-1], rlen.shape, h.shape[:-1], hlen.shape)
#     size = torch.tensor(batch).prod().item()

#     r = r.reshape(size, -1)
#     rlen = rlen.reshape(size)
#     h = h.reshape(size, -1)
#     hlen = hlen.reshape(size)

#     return torch.tensor(
#         [
#             torch.unique(torch.cat([r[:rlen]]))
#             for ri, rleni, hi, hlenti in zip(r, rlen, h, hlen)
#         ]
#         device=r.device
#     ).reshape(batch)

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
    return torch.floor_divide(size + 2 * padding - dilation * (kernel_size - 1) - 1, stride) + 1

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
    return (size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

class ConvSize(nn.Module, Generic[_ConvNd]):
    """Module for computing the output size of a convolution operation.
    """

    def __init__(self, conv_module: _ConvNd):
        """"
        Args:
            conv_module (_ConvNd): Convolution module (Conv1d, Conv2d, Conv3d).
        """
        super().__init__()
        self.kernel_size = torch.nn.Parameter(torch.tensor(conv_module.kernel_size), requires_grad=False)
        self.stride = torch.nn.Parameter(torch.tensor(conv_module.stride), requires_grad=False)
        self.padding = torch.nn.Parameter(torch.tensor(conv_module.padding), requires_grad=False)
        self.dilation = torch.nn.Parameter(torch.tensor(conv_module.dilation), requires_grad=False)

    @overload
    def forward(self, tensor: Tensor, tgt_dim: int | slice = slice(None)) -> Tensor: ...
    @overload
    def forward(self, size: Size, tgt_dim: int | slice = slice(None)) -> Size: ...
    @overload
    def forward(self, args: tuple[Tensor, int | slice], /) -> tuple[Tensor, int | slice]: ...
    def forward(self, arg1: Tensor | Size | tuple[Tensor | Size, int | slice], arg2: int | slice = slice(None)) -> Tensor | Size | tuple[Tensor, int | slice]: # type: ignore[reportInconsistentOverload]
        """Compute the output size of a convolution operation for a given input size.

        This method calculates the resulting size after applying a convolution operation
        based on the input size, target dimensions, and convolution module parameters.

        Args:
            arg1 (Tensor | Size | tuple): Input size, which can be:
                - A `Tensor` representing the input size for each dimension.
                - A `Size` object (torch.Size) representing the input size in PyTorch's native format.
                - A tuple containing:
                    - A `Tensor` as the first element.
                    - A target dimension (`int` or `slice`) as the second element.
            arg2 (int | slice, optional): Target dimension(s) to apply the convolution. 
                Ignored if `arg1` is a tuple. Defaults to `slice(None)` (all dimensions).

        Returns:
            Tensor | Size | tuple: The computed output size:
                - If the input is a `Tensor`, returns a `Tensor` with the output size.
                - If the input is a `Size`, returns a `Size` object.
                - If the input is a tuple, returns a tuple containing the computed size and target dimensions.

        Raises:
            TypeError: If `arg1` is not one of the supported types:
                `Tensor`, `Size`, or `tuple` of `Tensor` and target index.
        """

        if isinstance(arg1, tuple):
            # for torch.nn.Sequential
            tensor, tgt_dim = arg1
            return self._forward_tensor(torch.tensor(tensor), tgt_dim), tgt_dim
        elif isinstance(arg1, Tensor):
            return self._forward_tensor(arg1, arg2)
        elif isinstance(arg1, Size):
            return self._forward_size(arg1)
        else:
            raise TypeError()

    def _forward_tensor(self, tensor: Tensor, tgt_dim: int | slice):
        return conv_size(tensor, self.kernel_size[tgt_dim], self.stride[tgt_dim], self.padding[tgt_dim], self.dilation[tgt_dim])

    def _forward_size(self, size: Size):
        change_dimention = len(self.kernel_size)
        target = size[-change_dimention:]
        unchange = size[:-change_dimention]
        changed = Size(conv_size(torch.tensor(target), self.kernel_size, self.stride, self.padding, self.dilation).tolist())
        return Size(unchange + changed)

    def extra_repr(self):
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}"

class ConvTransposeSize(nn.Module, Generic[_ConvTransposeNd]):
    """Module for computing the output size of a transposed convolution operation.

    Args:
        conv_module (_ConvNd): Transposed convolution module (ConvTranspose1d, ConvTranspose2d, ConvTranspose3d).
    """

    def __init__(self, conv_module: _ConvTransposeNd):
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
