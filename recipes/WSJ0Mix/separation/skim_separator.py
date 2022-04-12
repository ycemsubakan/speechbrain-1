from collections import OrderedDict
from typing import List
from typing import Tuple
from typing import Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex

# from espnet2.enh.layers.skim import SkiM
from espnet2.enh.separator.abs_separator import AbsSeparator


# An implementation of SkiM model described in
# "SkiM: Skipping Memory LSTM for Low-Latency Real-Time Continuous Speech Separation"
# (https://arxiv.org/abs/2201.10800)
#

import torch.nn as nn

from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoder,
    PositionalEncoding,
    get_lookahead_mask
)

from espnet2.enh.layers.dprnn import merge_feature
from espnet2.enh.layers.dprnn import SingleRNN
from espnet2.enh.layers.dprnn import split_feature

# from espnet2.enh.layers.tcn import choose_norm
# from speechbrain.lobes.models.conv_tasnet import choose_norm
import copy


EPS = torch.finfo(torch.get_default_dtype()).eps


def choose_norm(norm_type, channel_size, shape="BDT"):
    """The input of normalization will be (M, C, K), where M is batch size.
    C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size, shape=shape)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size, shape=shape)
    elif norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)
    elif norm_type == "GN":
        return nn.GroupNorm(1, channel_size, eps=1e-8)
    else:
        raise ValueError("Unsupported normalization type")


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)."""

    def __init__(self, channel_size, shape="BDT"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()
        assert shape in ["BDT", "BTD"]
        self.shape = shape

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """Forward.
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """

        assert y.dim() == 3

        if self.shape == "BTD":
            y = y.transpose(1, 2).contiguous()

        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTD":
            cLN_y = cLN_y.transpose(1, 2).contiguous()

        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)."""

    def __init__(self, channel_size, shape="BDT"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()
        assert shape in ["BDT", "BTD"]
        self.shape = shape

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """Forward.
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        if self.shape == "BTD":
            y = y.transpose(1, 2).contiguous()

        mean = y.mean(dim=(1, 2), keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=(1, 2), keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTD":
            gLN_y = gLN_y.transpose(1, 2).contiguous()
        return gLN_y


class MemLSTM(nn.Module):
    """the Mem-LSTM of SkiM
    args:
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the LSTM layers are bidirectional.
            Default is False.
        mem_type: 'hc', 'h', 'c' or 'id'.
            It controls whether the hidden (or cell) state of
            SegLSTM will be processed by MemLSTM.
            In 'id' mode, both the hidden and cell states will
            be identically returned.
        norm_type: gLN, cLN. cLN is for causal implementation.
    """

    def __init__(
        self,
        hidden_size,
        dropout=0.0,
        bidirectional=False,
        mem_type="hc",
        norm_type="cLN",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.input_size = (int(bidirectional) + 1) * hidden_size
        self.mem_type = mem_type

        assert mem_type in [
            "hc",
            "h",
            "c",
            "id",
        ], f"only support 'hc', 'h', 'c' and 'id', current type: {mem_type}"

        if mem_type in ["hc", "h"]:
            self.h_net = SingleRNN(
                "LSTM",
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.h_norm = choose_norm(
                norm_type=norm_type, channel_size=self.input_size, shape="BTD"
            )
        if mem_type in ["hc", "c"]:
            self.c_net = SingleRNN(
                "LSTM",
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            self.c_norm = choose_norm(
                norm_type=norm_type, channel_size=self.input_size, shape="BTD"
            )

    def extra_repr(self) -> str:
        return f"Mem_type: {self.mem_type}, bidirectional: {self.bidirectional}"

    def forward(self, hc, S):
        # hc = (h, c), tuple of hidden and cell states from SegLSTM
        # shape of h and c: (d, B*S, H)
        # S: number of segments in SegLSTM

        if self.mem_type == "id":
            ret_val = hc
        else:
            h, c = hc
            d, BS, H = h.shape
            B = BS // S
            h = h.transpose(1, 0).contiguous().view(B, S, d * H)  # B, S, dH
            c = c.transpose(1, 0).contiguous().view(B, S, d * H)  # B, S, dH
            if self.mem_type == "hc":
                h = h + self.h_norm(self.h_net(h))
                c = c + self.c_norm(self.c_net(c))
            elif self.mem_type == "h":
                h = h + self.h_norm(self.h_net(h))
                c = torch.zeros_like(c)
            elif self.mem_type == "c":
                h = torch.zeros_like(h)
                c = c + self.c_norm(self.c_net(c))

            h = h.view(B * S, d, H).transpose(1, 0).contiguous()
            c = c.view(B * S, d, H).transpose(1, 0).contiguous()
            ret_val = (h, c)

        if not self.bidirectional:
            # for causal setup
            causal_ret_val = []
            for x in ret_val:
                x_ = torch.zeros_like(x)
                x_[:, 1:, :] = x[:, :-1, :]
                causal_ret_val.append(x_)
            ret_val = tuple(causal_ret_val)

        return ret_val


class SegLSTM(nn.Module):

    """the Seg-LSTM of SkiM
    args:
        input_size: int, dimension of the input feature.
            The input should have shape (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the LSTM layers are bidirectional.
            Default is False.
        norm_type: gLN, cLN. cLN is for causal implementation.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        dropout=0.0,
        bidirectional=False,
        norm_type="cLN",
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)
        self.norm = choose_norm(
            norm_type=norm_type, channel_size=input_size, shape="BTD"
        )

    def forward(self, input, hc):
        # input shape: B, T, H

        B, T, H = input.shape

        if hc is None:
            # In fist input SkiM block, h and c are not available
            d = self.num_direction
            h = torch.zeros(d, B, self.hidden_size).to(input.device)
            c = torch.zeros(d, B, self.hidden_size).to(input.device)
        else:
            h, c = hc

        output, (h, c) = self.lstm(input, (h, c))
        output = self.dropout(output)
        output = self.proj(output.contiguous().view(-1, output.shape[2])).view(
            input.shape
        )
        output = input + self.norm(output)

        return output, (h, c)


class SBTransformerBlock_wnormandskip(nn.Module):
    """A wrapper for the SpeechBrain implementation of the transformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of layers.
    d_model : int
        Dimensionality of the representation.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Dimensionality of positional feed forward.
    input_shape : tuple
        Shape of input.
    kdim : int
        Dimension of the key (Optional).
    vdim : int
        Dimension of the value (Optional).
    dropout : float
        Dropout rate.
    activation : str
        Activation function.
    use_positional_encoding : bool
        If true we use a positional encoding.
    norm_before: bool
        Use normalization before transformations.

    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> block = SBTransformerBlock(1, 64, 8)
    >>> x = block(x)
    >>> x.shape
    torch.Size([10, 100, 64])
    """

    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        input_shape=None,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation="relu",
        use_positional_encoding=False,
        norm_before=False,
        attention_type="regularMHA",
        causal=False,
        use_norm=True,
        use_skip=True,
        norm_type="cLN",
    ):
        super(SBTransformerBlock_wnormandskip, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = nn.ReLU
        elif activation == "gelu":
            activation = nn.GELU
        else:
            raise ValueError("unknown activation")

        self.causal = causal

        self.mdl = TransformerEncoder(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            input_shape=input_shape,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
            normalize_before=norm_before,
            causal=causal,
            attention_type=attention_type,
        )

        self.use_norm = use_norm
        self.use_skip = use_skip

        if use_norm:
            self.norm = choose_norm(
                norm_type=norm_type, channel_size=d_model, shape="BTD"
            )

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(
                input_size=d_model, max_len=100000
            )

    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters

        """
        src_mask = get_lookahead_mask(x) if self.causal else None

        if self.use_positional_encoding:
            pos_enc = self.pos_enc(x)
            out = self.mdl(x + pos_enc, src_mask=src_mask)[0]
        else:
            out = self.mdl(x, src_mask=src_mask)[0]

        if self.use_norm:
            out = self.norm(out)
        if self.use_skip:
            out = out + x

        return out


class SkiM(nn.Module):
    """Skipping Memory Net
    args:
        input_size: int, dimension of the input feature.
            Input shape shoud be (batch, length, input_size)
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_blocks: number of basic SkiM blocks
        segment_size: segmentation size for splitting long features
        bidirectional: bool, whether the RNN layers are bidirectional.
        mem_type: 'hc', 'h', 'c', 'id' or None.
            It controls whether the hidden (or cell) state of SegLSTM
            will be processed by MemLSTM.
            In 'id' mode, both the hidden and cell states will
            be identically returned.
            When mem_type is None, the MemLSTM will be removed.
        norm_type: gLN, cLN. cLN is for causal implementation.
        seg_overlap: Bool, whether the segmentation will reserve 50%
            overlap for adjacent segments.Default is False.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.0,
        num_blocks=2,
        segment_size=20,
        bidirectional=True,
        mem_type="hc",
        norm_type="gLN",
        seg_overlap=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.segment_size = segment_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.mem_type = mem_type
        self.norm_type = norm_type
        self.seg_overlap = seg_overlap

        assert mem_type in [
            "hc",
            "h",
            "c",
            "id",
            None,
        ], f"only support 'hc', 'h', 'c', 'id', and None, current type: {mem_type}"

        self.seg_lstms = nn.ModuleList([])
        for i in range(num_blocks):
            self.seg_lstms.append(
                SegLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    norm_type=norm_type,
                )
            )
        if self.mem_type is not None:
            self.mem_lstms = nn.ModuleList([])
            for i in range(num_blocks - 1):
                self.mem_lstms.append(
                    MemLSTM(
                        hidden_size,
                        dropout=dropout,
                        bidirectional=bidirectional,
                        mem_type=mem_type,
                        norm_type=norm_type,
                    )
                )
        self.output_fc = nn.Sequential(
            nn.PReLU(), nn.Conv1d(input_size, output_size, 1)
        )

    def forward(self, input):
        # input shape: B, T (S*K), D
        B, T, D = input.shape

        if self.seg_overlap:
            input, rest = split_feature(
                input.transpose(1, 2), segment_size=self.segment_size
            )  # B, D, K, S
            input = input.permute(0, 3, 2, 1).contiguous()  # B, S, K, D
        else:
            input, rest = self._padfeature(input=input)
            input = input.view(B, -1, self.segment_size, D)  # B, S, K, D
        B, S, K, D = input.shape

        assert K == self.segment_size

        output = input.view(B * S, K, D).contiguous()  # BS, K, D
        hc = None
        for i in range(self.num_blocks):
            output, hc = self.seg_lstms[i](output, hc)  # BS, K, D
            if self.mem_type and i < self.num_blocks - 1:
                hc = self.mem_lstms[i](hc, S)

        if self.seg_overlap:
            output = output.view(B, S, K, D).permute(0, 3, 2, 1)  # B, D, K, S
            output = merge_feature(output, rest)  # B, D, T
            output = self.output_fc(output).transpose(1, 2)

        else:
            output = output.view(B, S * K, D)[:, :T, :]  # B, T, D
            output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)

        return output

    def _padfeature(self, input):
        B, T, D = input.shape
        rest = self.segment_size - T % self.segment_size

        if rest > 0:
            input = torch.nn.functional.pad(input, (0, 0, 0, rest))
        return input, rest


class SkiM_general(nn.Module):
    """Skipping Memory Net with the modified insegment transformation mechanism
    args:
        input_size: int, dimension of the input feature.
            Input shape shoud be (batch, length, input_size)
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_blocks: number of basic SkiM blocks
        segment_size: segmentation size for splitting long features
        bidirectional: bool, whether the RNN layers are bidirectional.
        mem_type: 'hc', 'h', 'c', 'id' or None.
            It controls whether the hidden (or cell) state of SegLSTM
            will be processed by MemLSTM.
            In 'id' mode, both the hidden and cell states will
            be identically returned.
            When mem_type is None, the MemLSTM will be removed.
        norm_type: gLN, cLN. cLN is for causal implementation.
        seg_overlap: Bool, whether the segmentation will reserve 50%
            overlap for adjacent segments.Default is False.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.0,
        num_blocks=2,
        segment_size=20,
        bidirectional=True,
        mem_type=None,
        norm_type="gLN",
        seg_overlap=False,
        seg_model=None,
        mem_model=None,
        mem_attmodel=None,
        use_dummy_timep=False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.segment_size = segment_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.mem_type = mem_type
        self.norm_type = norm_type
        self.seg_overlap = seg_overlap

        assert mem_type in [
            "hc",
            "h",
            "c",
            "id",
            "tf",
            None,
        ], f"only support 'hc', 'h', 'c', 'id', 'tf' and None, current type: {mem_type}"

        self.seg_model = nn.ModuleList([])
        for i in range(num_blocks):
            self.seg_model.append(copy.deepcopy(seg_model))

        if self.mem_type is not None:
            self.mem_model = nn.ModuleList([])
            for i in range(num_blocks - 1):
                self.mem_model.append(copy.deepcopy(mem_model))

        self.mem_attmodel = mem_attmodel
        if self.mem_attmodel is not None:
            self.mem_attmodel = nn.ModuleList([])
            for i in range(num_blocks - 1):
                self.mem_attmodel.append(copy.deepcopy(mem_attmodel))
        self.use_dummy_timep = use_dummy_timep

        self.output_fc = nn.Sequential(
            nn.PReLU(), nn.Conv1d(input_size, output_size, 1)
        )

    def forward(self, input):
        # input shape: B, T (S*K), D
        B, T, D = input.shape

        if self.seg_overlap:
            input, rest = split_feature(
                input.transpose(1, 2), segment_size=self.segment_size
            )  # B, D, K, S
            input = input.permute(0, 3, 2, 1).contiguous()  # B, S, K, D
        else:
            input, rest = self._padfeature(input=input)
            input = input.view(B, -1, self.segment_size, D)  # B, S, K, D
        B, S, K, D = input.shape

        assert K == self.segment_size

        output = input.view(B * S, K, D).contiguous()  # BS, K, D

        hc = torch.zeros(
            output.shape[0], 1, output.shape[-1], device=output.device
        )
        if self.use_dummy_timep:
            output = torch.cat([output, 
                                torch.ones(output.shape[0], 1, output.shape[-1], device=output.device)
                                ], dim=1)

        for i in range(self.num_blocks):
            #if self.mem_attmodel:
            #    if i < (self.num_blocks - 1):
            #        att = self.mem_attmodel[i](output + hc)

            output = self.seg_model[i](output + hc)  # BS, K, D
            if self.mem_type and i < self.num_blocks - 1:

                #import pdb; pdb.set_trace()
                if self.use_dummy_timep:
                    hc = output[:, -1, :].unsqueeze(0)
                    output = output[:, :-1, :]
                else:
                    if self.mem_attmodel:
                        att = self.mem_attmodel[i](output) 
                        hc = att.mean(1).unsqueeze(0)
                    else:
                        hc = output.mean(1).unsqueeze(0)

                hc = self.mem_model[i](hc).permute(1, 0, 2)

        if self.seg_overlap:
            output = output.view(B, S, K, D).permute(0, 3, 2, 1)  # B, D, K, S
            output = merge_feature(output, rest)  # B, D, T
            output = self.output_fc(output).transpose(1, 2)

        else:
            output = output.view(B, S * K, D)[:, :T, :]  # B, T, D
            output = self.output_fc(output.transpose(1, 2)).transpose(1, 2)

        return output

    def _padfeature(self, input):
        B, T, D = input.shape
        rest = self.segment_size - T % self.segment_size

        if rest > 0:
            input = torch.nn.functional.pad(input, (0, 0, 0, rest))
        return input, rest


class SkiMSeparator(AbsSeparator):
    """Skipping Memory (SkiM) Separator
    Args:
        input_dim: input feature dimension
        causal: bool, whether the system is causal.
        num_spk: number of target speakers.
        nonlinear: the nonlinear function for mask estimation,
            select from 'relu', 'tanh', 'sigmoid'
        layer: int, number of SkiM blocks. Default is 3.
        unit: int, dimension of the hidden state.
        segment_size: segmentation size for splitting long features
        dropout: float, dropout ratio. Default is 0.
        mem_type: 'hc', 'h', 'c', 'id' or None.
            It controls whether the hidden (or cell) state of
            SegLSTM will be processed by MemLSTM.
            In 'id' mode, both the hidden and cell states
            will be identically returned.
            When mem_type is None, the MemLSTM will be removed.
        seg_overlap: Bool, whether the segmentation will reserve 50%
            overlap for adjacent segments. Default is False.
    """

    def __init__(
        self,
        input_dim: int,
        causal: bool = True,
        num_spk: int = 2,
        nonlinear: str = "relu",
        layer: int = 3,
        unit: int = 512,
        segment_size: int = 20,
        dropout: float = 0.0,
        mem_type: str = "hc",
        seg_overlap: bool = False,
    ):

        super().__init__()

        self._num_spk = num_spk

        self.segment_size = segment_size

        if mem_type not in ("hc", "h", "c", "id", None):
            raise ValueError("Not supporting mem_type={}".format(mem_type))

        self.skim = SkiM(
            input_size=input_dim,
            hidden_size=unit,
            output_size=input_dim * num_spk,
            dropout=dropout,
            num_blocks=layer,
            bidirectional=(not causal),
            norm_type="cLN" if causal else "gLN",
            segment_size=segment_size,
            seg_overlap=seg_overlap,
            mem_type=mem_type,
        )

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

    def forward(
        self, input: Union[torch.Tensor, ComplexTensor],  # ilens: torch.Tensor
    ):

        # -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:

        """Forward.
        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]
        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input

        feature = feature.permute(0, 2, 1)

        B, T, N = feature.shape

        processed = self.skim(feature)  # B,T, N

        processed = processed.view(B, T, N, self.num_spk)
        masks = self.nonlinear(processed).unbind(dim=3)

        mask_tensor = torch.stack([m.permute(0, 2, 1) for m in masks])
        # masked = [input * m for m in masks]

        # others = OrderedDict(
        #  zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        # )

        return mask_tensor

    @property
    def num_spk(self):
        return self._num_spk


class SkiMSeparator_General(AbsSeparator):
    """Skipping Memory (SkiM) Separator with generalized processing
    Args:
        input_dim: input feature dimension
        causal: bool, whether the system is causal.
        num_spk: number of target speakers.
        nonlinear: the nonlinear function for mask estimation,
            select from 'relu', 'tanh', 'sigmoid'
        layer: int, number of SkiM blocks. Default is 3.
        unit: int, dimension of the hidden state.
        segment_size: segmentation size for splitting long features
        dropout: float, dropout ratio. Default is 0.
        mem_type: 'hc', 'h', 'c', 'id' or None.
            It controls whether the hidden (or cell) state of
            SegLSTM will be processed by MemLSTM.
            In 'id' mode, both the hidden and cell states
            will be identically returned.
            When mem_type is None, the MemLSTM will be removed.
        seg_overlap: Bool, whether the segmentation will reserve 50%
            overlap for adjacent segments. Default is False.
    """

    def __init__(
        self,
        input_dim: int,
        causal: bool = True,
        num_spk: int = 2,
        nonlinear: str = "relu",
        layer: int = 3,
        unit: int = 512,
        segment_size: int = 20,
        dropout: float = 0.0,
        mem_type: str = "hc",
        seg_overlap: bool = False,
        seg_model=None,
        mem_model=None,
        mem_attmodel=None,
        use_dummy_timep=False
    ):

        super().__init__()

        self._num_spk = num_spk

        self.segment_size = segment_size

        if mem_type not in ("hc", "h", "c", "id", "tf", None):
            raise ValueError("Not supporting mem_type={}".format(mem_type))

        self.skim = SkiM_general(
            input_size=input_dim,
            hidden_size=unit,
            output_size=input_dim * num_spk,
            dropout=dropout,
            num_blocks=layer,
            bidirectional=(not causal),
            norm_type="cLN" if causal else "gLN",
            segment_size=segment_size,
            seg_overlap=seg_overlap,
            mem_type=mem_type,
            seg_model=seg_model,
            mem_model=mem_model,
            mem_attmodel=mem_attmodel,
            use_dummy_timep=use_dummy_timep
        )

        if nonlinear not in ("sigmoid", "relu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

    def forward(
        self, input: Union[torch.Tensor, ComplexTensor],  # ilens: torch.Tensor
    ):

        # -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:

        """Forward.
        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]
        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """

        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        else:
            feature = input

        feature = feature.permute(0, 2, 1)

        B, T, N = feature.shape

        processed = self.skim(feature)  # B,T, N

        processed = processed.view(B, T, N, self.num_spk)
        masks = self.nonlinear(processed).unbind(dim=3)

        mask_tensor = torch.stack([m.permute(0, 2, 1) for m in masks])
        # masked = [input * m for m in masks]

        # others = OrderedDict(
        #  zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        # )

        return mask_tensor

    @property
    def num_spk(self):
        return self._num_spk
