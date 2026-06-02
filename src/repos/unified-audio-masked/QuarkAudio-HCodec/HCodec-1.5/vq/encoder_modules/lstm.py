# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LSTM layers module."""

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    # def forward(self, x):
    #     x = x.permute(2, 0, 1)
    #     y, _ = self.lstm(x)
    #     if self.skip:
    #         y = y + x
    #     y = y.permute(1, 2, 0)
    #     return y

    # 修改transpose顺序
    def forward(self, x, padding_mask = None):
        # # 插入reshape
        # x = x.reshape(x.shape)
        x1 = x.permute(2, 0, 1)

        if padding_mask is not None:
            lengths = padding_mask.sum(dim=1).cpu()
            lengths = lengths.clamp(min=1)
            packed_input = pack_padded_sequence(
                x1, lengths, enforce_sorted=False
            )
            
            # 2. Run the bidirectional LSTM safely
            packed_output, _ = self.lstm(packed_input)
            
            # 3. Unpack it back to full padded shape (T, B, C)
            y, _ = pad_packed_sequence(
                packed_output, total_length=x1.shape[0]
            )
        else:
            y, _ = self.lstm(x1)
            
        y = y.permute(1, 2, 0)
        if self.skip:
            y = y + x
        return y
