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

from vq_functions import vq, vq_st


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
        return_masks=False,
    ):
        """
        Implements decoding to generate interpretation from raw audio input
        using a SepFormer.
        Takes pooled latent representations to condition the separation.
        """
        super().__init__()

        self.return_masks = return_masks
        self.num_spks = num_spks
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

        mix_w = torch.stack(
            [mix_w] * self.num_spks
        )  # stack to avoid broadcasting errors
        sep_h = mix_w * est_mask

        # decode from latent space to time domain
        est_source = torch.cat(
            [
                self.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = x.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        if self.return_masks:
            return est_source, sep_h
        else:
            return est_source


class Theta_sep(nn.Module):
    def __init__(self, n_comp=100, T=431, num_classes=50):
        super().__init__()
        self.hard_att = nn.Linear(
            256, 1
        )  # collapse time axis using "attention" based pooling
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes), nn.Softmax(dim=1)
        )

    def forward(self, masks):
        """psi_out is of shape n_batch x n_comp x T
        collapse time axis using "attention" based pooling"""

        masks_pooled = masks.mean(-1)

        # theta_out = self.hard_att(masks_pooled).squeeze(2)
        theta_out = masks_pooled.squeeze(2)
        # print(theta_out.shape)
        # input()
        theta_out = self.classifier(theta_out)
        # print(theta_out.shape)
        # input()
        return theta_out


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        hs = []

        x = self.conv1(x)
        x = F.relu(x)
        hs.append(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        hs.append(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output, hs


class PsiMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Input to this Module for Cnn14 will have shape (2048, W, H), (1024, W, H)
        and (512, 2W, 2H). For the broadcasting variant of the conditioning,
        the decoder expects two tensors of shape (B, C, 1) -- C=256 for now.
        """
        self.convt1 = nn.ConvTranspose2d(
            64, 32, kernel_size=6, stride=2, padding=0,
        )
        self.convt2 = nn.ConvTranspose2d(
            32, 32, kernel_size=3, stride=1, padding=0,
        )
        # self.conv1 = nn.Conv2d(
        #    64,
        #    64,
        #    kernel_size=3,
        #    stride=1,
        #    padding='same',
        # )
        self.convt1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, hs, labels=None):
        h2up = self.convt1(hs[1])
        # h2up = F.relu(h2up)
        h1up = self.convt2(hs[0])
        # h1up = F.relu(h1up)

        hcat = torch.cat([h2up, h1up], dim=1)
        # hconv = self.conv1(hcat)
        # hconv = F.relu(hcat)
        hconv = hcat

        xhat = self.convt1x1(hconv)
        # xhat = torch.sigmoid(xhat)
        return xhat, hcat


class VQEmbedding(nn.Module):
    def __init__(self, K, D, numclasses=10, activate_class_partitioning=True):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1.0 / K, 1.0 / K)
        self.numclasses = numclasses
        self.activate_class_partitioning = activate_class_partitioning

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x, labels=None):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(
            z_e_x_,
            self.embedding.weight.detach(),
            labels,
            self.numclasses,
            self.activate_class_partitioning,
        )
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(
            self.embedding.weight, dim=0, index=indices
        )
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(
            0, 3, 1, 2
        )  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VectorQuantizedPSI(nn.Module):
    def __init__(self, dim=256, K=512, activate_class_partitioning=True):
        super().__init__()
        # self.encoder = nn.Sequential(
        #    nn.Conv2d(input_dim, dim, 4, 2, 1),
        #    nn.BatchNorm2d(dim),
        #    nn.ReLU(True),
        #    nn.Conv2d(dim, dim, 4, 2, 1),
        #    ResBlock(dim),
        #    ResBlock(dim),
        # )

        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2,)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0,)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2,)

        self.codebook = VQEmbedding(
            K, dim, activate_class_partitioning=activate_class_partitioning
        )

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

        self.apply(weights_init)

    def encode(self, hs):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(
            0, 3, 1, 2
        )  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, hs, labels):
        h1 = self.conv2(hs[0])
        h1 = F.relu(h1)
        h3 = self.conv3(h1)
        h3 = F.relu(h3)
        # h2up = F.relu(h2up)

        h2 = self.conv1(hs[1])
        h2 = F.relu(h2)
        # h1up = F.relu(h1up)

        hcat = torch.cat([h2, h3], dim=1)

        # z_e_x = self.encoder(hcat)
        z_q_x_st, z_q_x = self.codebook.straight_through(hcat, labels)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, hcat, z_q_x


class MNISTSeparator(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, dim, 3, padding="same"),
            ResBlock(dim),
            ResBlock(dim),
            nn.Conv2d(dim, 1, 3, padding="same"),
        )

    def forward(self, x):
        xhat = self.encoder(x)
        return xhat


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
