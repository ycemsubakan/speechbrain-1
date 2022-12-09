import torch
import torch.nn as nn
import torch.nn.functional as F

# import speechbrain as sb
from speechbrain.lobes.models.dual_path import (
    Encoder,
    SBTransformerBlock,
    Dual_Path_Model,
    Decoder,
)


class Psi(nn.Module):
    def __init__(self, n_comp=256, in_embed_dims=[2048, 1024, 512]):
        super().__init__()
        """
        Input to this Module for Cnn14 will have shape (2048, W, H), (1024, W, H)
        and (512, 2W, 2H). For the broadcasting variant of the conditioning,
        the decoder expects two tensors of shape (B, C, 1) -- C=256 for now.
        """
        self.conv_1 = nn.Conv2d(
            in_embed_dims[2],
            in_embed_dims[1],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        # self.bn = nn.BatchNorm2d(in_embed_dims[1])
        self.relu = nn.ReLU()

        self.out = nn.Linear(in_embed_dims[0] * 2, n_comp)
        # self.out_bn = nn.BatchNorm1d(n_comp)

        self.fc = nn.Linear(in_embed_dims[0] * 2, n_comp)
        self.out = nn.Linear(n_comp, n_comp)
        # self.out_bn = nn.BatchNorm1d(n_comp)

    def forward(self, f_i):
        batch_size = f_i[0].shape[0]
        # f_I is a tuple of hidden representations
        x3 = self.relu(self.conv_1(f_i[2]))

        comb = torch.cat(
            (
                F.adaptive_avg_pool2d(x3, (1, 1)),
                F.adaptive_avg_pool2d(f_i[1], (1, 1)),
                F.adaptive_avg_pool2d(f_i[0], (1, 1)),
            ),
            dim=1,
        )
        comb = comb.view(batch_size, -1)

        temp = self.relu(self.fc(comb))
        psi_out = self.out(temp).view(2, batch_size, -1, 1)

        return psi_out  # relu here is not appreciated for reconstruction


class SepDecoder(nn.Module):
    def __init__(
        self,
        enc_kernel_size=16,
        enc_outchannels=256,
        out_channels=256,
        d_ffn=1024,
        nhead=8,
        num_layers_tb=8,
        num_spks=2,
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
            num_layers=num_layers_tb,
            d_model=out_channels,
            nhead=nhead,
            d_ffn=d_ffn,
            dropout=0,
            use_positional_encoding=True,
            norm_before=True,
        )

        self.SBtfinter = SBTransformerBlock(
            num_layers=num_layers_tb,
            d_model=out_channels,
            nhead=nhead,
            d_ffn=d_ffn,
            dropout=0,
            use_positional_encoding=True,
            norm_before=True,
        )

        self.masknet = Dual_Path_Model(
            num_spks=num_spks,
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
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            bias=False,
        )

    @staticmethod
    def _check_shapes(x, psi_out):
        for i in range(x.dim() - 1):
            # -1 only if broadcasting in time!
            assert (
                x.shape[i] == psi_out.shape[i]
            ), f"{i+1}-th of input and psi_out doesn't match. Got {psi_out.shape} expect {x.shape}."

    def forward(self, x, psi_out):
        # psi_out is expected to be sth like (2, B, out_channels, 1)
        mix_w = self.encoder(x)

        # check shapes before conditioning
        self._check_shapes(mix_w, psi_out[0])
        self._check_shapes(mix_w, psi_out[1])

        # condition with psi_out
        # mix_w = mix_w * psi_out[0] + psi_out[1]
        mix_w = mix_w * psi_out[0] + psi_out[1]

        # generates masks for garbage and interpretation
        est_mask = self.masknet(mix_w)

        mix_w = torch.stack([mix_w] * 2)  # stack to avoid broadcasting errors
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
    f_i = [
        torch.randn(1, 2048, 6, 2),
        torch.randn(1, 1024, 6, 2),
        torch.randn(1, 512, 12, 4),
    ]
    psi = Psi()
    psi_out = psi(f_i)

    m = SepDecoder()
    m(x, psi_out)
    # from torchinfo import summary
    # summary(m)
    # m(x, psi_out)
