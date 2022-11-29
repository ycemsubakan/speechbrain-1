import torch
import torch.nn as nn
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.lobes.models.dual_path import (
        Encoder, SBTransformerBlock, Dual_Path_Model, Decoder
    )

class SepDecoder(nn.Module):
    def __init__(
        self, enc_kernel_size=16, enc_outchannels=256, out_channels=256
    ):
        """
        Implements decoding to generate interpretation from raw audio input
        using a SepFormer.
        Takes pooled latent representations to condition the separation.
        """
        super().__init__()

        self.encoder = Encoder(
            kernel_size=enc_kernel_size, out_channels=enc_outchannels
        )

        self.SBtfintra = SBTransformerBlock(
            num_layers=8,
            d_model=out_channels,
            nhead=8,
            d_ffn=1024,
            dropout=0,
            use_positional_encoding=True,
            norm_before=True,
        )

        self.SBtfinter = SBTransformerBlock(
            num_layers=8,
            d_model=out_channels,
            nhead=8,
            d_ffn=1024,
            dropout=0,
            use_positional_encoding=True,
            norm_before=True,
        )

        self.masknet = Dual_Path_Model(
            num_spks=2,
            in_channels=enc_outchannels,
            out_channels=out_channels,
            num_layers=2,
            K=250,
            intra_model=self.SBtfintra,
            inter_model=self.SBtfinter,
            norm="ln",
            linear_layer_after_inter_intra=False,
            skip_around_intra=True,
        )

        self.decoder = Decoder(
            in_channels=enc_outchannels,
            out_channels=1,
            kernel_size=16,
            stride=8,
            bias=False,
        )

    @staticmethod
    def _check_shapes(x, psi_out):
        for i in range(x.dim() - 1):
            # -1 only if broadcasting in time!
            assert x.shape[i] == psi_out.shape[i], f"{i+1}-th of input and psi_out doesn't match. Got {psi_out.shape} expect {x.shape}."

    def forward(self, x, psi_out):
        # psi_out is expected to be sth like (2, B, out_channels, T_strided)
        mix_w = self.encoder(x)
        
        # check shapes before conditioning
        self._check_shapes(mix_w, psi_out[0])
        self._check_shapes(mix_w, psi_out[1])

        # condition with psi_out
        mix_w = mix_w * psi_out[0] + psi_out[1]

        # generates masks for garbage and interpretation
        est_mask = self.masknet(mix_w) 
        
        mix_w = torch.stack([mix_w] * 2) # stack to avoid broadcasting errors
        sep_h = mix_w * est_mask

        # decode from latent space to time domain
        est_source = torch.cat(
            [self.decoder(sep_h[i]).unsqueeze(-1) for i in range(2)], dim=-1
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = x.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source

if __name__ == "__main__":
    x = torch.randn(1, 16000 * 2)
    psi_out = torch.randn(2, 1, 256, 1)
    m = SepDecoder()
    from torchinfo import summary
    summary(m)
    m(x, psi_out)

