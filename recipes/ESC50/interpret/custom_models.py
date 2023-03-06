
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from speechbrain.lobes.models.dual_path import (
	Encoder,
	SBTransformerBlock,
	Dual_Path_Model,
	Decoder,
)

from vq_functions import vq, vq_st

class PIQAnalogPSI_Audio_L2I(nn.Module):
    def __init__(
        self,
        dim=128,
        K=100,
        numclasses=50,
        use_adapter=False,
        adapter_reduce_dim=True,
    ):
        super().__init__()
        # self.encoder = nn.Sequential(
        #    nn.Conv2d(input_dim, dim, 4, 2, 1),
        #    nn.BatchNorm2d(dim),
        #    nn.ReLU(True),
        #    nn.Conv2d(dim, dim, 4, 2, 1),
        #    ResBlock(dim),
        #    ResBlock(dim),
        # )

        # self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2,)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0,)
        # self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.use_adapter = use_adapter
        self.adapter_reduce_dim = adapter_reduce_dim
        if use_adapter:
            self.adapter = ResBlockAudio(dim)

            if adapter_reduce_dim:
                self.down = nn.Conv2d(dim, dim, 4, (2, 2), 1)
                self.up = nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1)

        self.decoder = nn.Sequential(
            # ResBlock(dim),
            # ResBlock(dim),
            # nn.ReLU(True),
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(dim, dim, 3, (2, 2), 1),
            # nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.BatchNorm2d(dim),
            # nn.LeakyReLU(),
            # nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1),
            # nn.BatchNorm2d(dim),
            # nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1),
            # nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, 1, 12, 1, 1),
            nn.ReLU(),
            #nn.BatchNorm2d(dim),
            # nn.LeakyReLU(),
            # nn.Linear(505, 513),
            nn.Linear(513, K),
            nn.ReLU(),
            # nn.Softplus(),
        )
        self.apply(weights_init)

    def forward(self, hs):
        if self.use_adapter:
            hcat = self.adapter(hs)
        else:
            hcat = hs

        if self.adapter_reduce_dim:
            hcat = self.down(hcat)
            z_q_x_st = self.up(hcat)
            out = self.decoder(z_q_x_st)
        else:
            out = self.decoder(hcat)

        return out, hcat


class VectorQuantizedPSI_Audio(nn.Module):
	def __init__(
		self,
		dim=128,
		K=512,
		numclasses=50,
		activate_class_partitioning=True,
		shared_keys=5,
		use_adapter=False,
		adapter_reduce_dim=True,
	):
		super().__init__()
		# self.encoder = nn.Sequential(
		#	 nn.Conv2d(input_dim, dim, 4, 2, 1),
		#	 nn.BatchNorm2d(dim),
		#	 nn.ReLU(True),
		#	 nn.Conv2d(dim, dim, 4, 2, 1),
		#	 ResBlock(dim),
		#	 ResBlock(dim),
		# )

		# self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2,)
		# self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0,)
		# self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

		self.codebook = VQEmbedding(
			K,
			dim,
			numclasses=numclasses,
			activate_class_partitioning=activate_class_partitioning,
			shared_keys=shared_keys,
		)
		self.use_adapter = use_adapter
		self.adapter_reduce_dim = adapter_reduce_dim
		if use_adapter:
			self.adapter = ResBlock(dim)

			if adapter_reduce_dim:
				self.down = nn.Conv2d(dim, dim, 4, (2, 2), 1)
				self.up = nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1)

		self.decoder = nn.Sequential(
			# ResBlock(dim),
			# ResBlock(dim),
			# nn.ReLU(True),
			# nn.LeakyReLU(),
			nn.ConvTranspose2d(dim, dim, 3, (2, 2), 1),
			# nn.BatchNorm2d(dim),
			nn.ReLU(True),
			nn.BatchNorm2d(dim),
			# nn.LeakyReLU(),
			# nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1),
			# nn.BatchNorm2d(dim),
			# nn.ReLU(True),
			nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1),
			# nn.BatchNorm2d(dim),
			nn.ReLU(),
			nn.BatchNorm2d(dim),
			# nn.LeakyReLU(),
			nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1),
			nn.ReLU(),
			nn.BatchNorm2d(dim),
			# nn.LeakyReLU(),
			nn.ConvTranspose2d(dim, dim, 4, (2, 2), 1),
			nn.ReLU(),
			nn.BatchNorm2d(dim),
			nn.ConvTranspose2d(dim, 1, 12, 1, 1),
			# nn.ReLU(),
			# nn.LeakyReLU(),
			# nn.Linear(505, 513),
			# nn.ReLU(),
			# nn.Linear(513, 513),
			# nn.Softplus(),
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
		if self.use_adapter:
			hcat = self.adapter(hs)
		else:
			hcat = hs

		if self.adapter_reduce_dim:
			hcat = self.down(hcat)
			z_q_x_st, z_q_x = self.codebook.straight_through(hcat, labels)
			z_q_x_st = self.up(z_q_x_st)
		else:
			z_q_x_st, z_q_x = self.codebook.straight_through(hcat, labels)
		x_tilde = self.decoder(z_q_x_st)
		return x_tilde, hcat, z_q_x


class custom_classifier(nn.Module):
	def __init__(self, dim=128, num_classes=50):
		super().__init__()
		self.lin1 = nn.Linear(dim, dim)
		self.lin2 = nn.Linear(dim, num_classes)

	def forward(self, z):
		z = F.relu(self.lin1(z))
		yhat = (self.lin2(z)).unsqueeze(1) 
		return yhat

class Conv2dEncoder_v2(nn.Module):
	def __init__(self, dim=256):
		super().__init__()
		# self.encoder = nn.Sequential(
		self.conv1 = nn.Conv2d(1, dim, 4, 2, 1)
		self.bn1 = nn.BatchNorm2d(dim)
		# nn.ReLU(True),
		# nn.LeakyReLU(),
		# nn.Conv2d(dim, dim, 4, 2, 1),
		# nn.BatchNorm2d(dim),
		# nn.ReLU(True),
		self.conv2 = nn.Conv2d(dim, dim, 4, 2, 1)
		self.bn2 = nn.BatchNorm2d(dim)
		# nn.ReLU(True),
		# nn.LeakyReLU(),
		self.conv3 = nn.Conv2d(dim, dim, 4, 2, 1)
		self.bn3 = nn.BatchNorm2d(dim)
		# nn.ReLU(True),
		# nn.LeakyReLU(),
		self.conv4 = nn.Conv2d(dim, dim, 4, 2, 1)
		self.bn4 = nn.BatchNorm2d(dim)

		self.resblock = ResBlockAudio(dim)
		# self.resblock2 = ResBlock(dim)

		self.nonl = nn.ReLU()

	def forward(self, x):
		# x = x.unsqueeze(1)
		h1 = self.conv1(x)
		h1 = self.bn1(h1)
		h1 = self.nonl(h1)

		h2 = self.conv2(h1)
		h2 = self.bn2(h2)
		h2 = self.nonl(h2)

		h3 = self.conv3(h2)
		h3 = self.bn3(h3)
		h3 = self.nonl(h3)

		h4 = self.conv4(h3)
		h4 = self.bn4(h4)
		h4 = self.nonl(h4)

		h4 = self.resblock(h4)
		# h4 = self.nonl(h4)
		# h4 = h4.mean((2, 3)).unsqueeze(1)
		return h4, [h1, h2, h3, h4]


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

		return psi_out	# relu here is not appreciated for reconstruction


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
		collapse time axis using "attention" based pooling
		"""

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
		self.conv3 = nn.Conv2d(64, 64, 3, 1)
		self.conv4 = nn.Conv2d(64, 128, 3, 1)
		self.dropout1 = nn.Dropout(0.25)
		self.dropout2 = nn.Dropout(0.5)
		self.fc1 = nn.Linear(2048, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		hs = []

		x = self.conv1(x)
		x = F.relu(x)
		# hs.append(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		# hs.append(x)
		x = self.dropout1(x)
		x = self.conv3(x)
		x = F.relu(x)
		hs.append(x)
		x = self.conv4(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		hs.append(x)
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
		#	 64,
		#	 64,
		#	 kernel_size=3,
		#	 stride=1,
		#	 padding='same',
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
	def __init__(
		self,
		K,
		D,
		numclasses=10,
		activate_class_partitioning=True,
		shared_keys=10,
	):
		super().__init__()
		self.embedding = nn.Embedding(K, D)

		uniform_mat = torch.round(
			torch.linspace(-0.5, numclasses - 0.51, K)
		).unsqueeze(
			1
		)  # .to(self.embedding.device) * 5
		# uniform_mat = uniform_mat.repeat(1, D) * 0.02

		self.embedding.weight.data.uniform_(-1.0 / K, 1.0 / K)
		# self.embedding.weight.data += uniform_mat

		self.numclasses = numclasses
		self.activate_class_partitioning = activate_class_partitioning
		self.shared_keys = shared_keys

	def forward(self, z_e_x):
		z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
		latents = vq(z_e_x_, self.embedding.weight, training=self.training)
		return latents

	def straight_through(self, z_e_x, labels=None):
		z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
		z_q_x_, indices = vq_st(
			z_e_x_,
			self.embedding.weight.detach(),
			labels,
			self.numclasses,
			self.activate_class_partitioning,
			self.shared_keys,
			self.training,
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


class ResBlockAudio(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.block = nn.Sequential(
			# nn.ReLU(True),
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
	def __init__(
		self, dim=128, K=512, activate_class_partitioning=True, shared_keys=5
	):
		super().__init__()
		self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0,)
		self.conv3 = nn.Conv2d(128, dim, kernel_size=3, stride=1, padding=1)
		self.conv_mix = nn.Conv2d(256, 256, kernel_size=3, padding="same")

		self.codebook = VQEmbedding(
			K,
			dim,
			activate_class_partitioning=activate_class_partitioning,
			shared_keys=shared_keys,
		)

		self.decoder = nn.Sequential(
		ResBlock(dim),
		ResBlock(dim),
		nn.ReLU(True),
		nn.ConvTranspose2d(dim, dim, 3, 2, 1),
		nn.BatchNorm2d(dim),
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
		# h1 = self.conv1(hs[0])
		# h1 = F.relu(h1)
		h3 = self.conv3(hs[1])
		# h3 = F.relu(h3)
		## h2up = F.relu(h2up)
		#
		h2 = self.conv1(hs[0])
		h2 = F.relu(h2)
		## h1up = F.relu(h1up)
		#
		# hcat = torch.cat([h2, h3], dim=1)
		# hcat = self.conv_mix(hcat)

		hcat = h3

		# hcat = hs[1]
		# hcat = self.conv3(hs[1])
		# hcat = self.norm(hcat)
		# hcat = F.relu(hcat)

		# hcat = torch.cat([hcat, h1], dim=1)

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


class NMFEncoderMNIST(nn.Module):
	def __init__(self, K=100):
		super().__init__()
		padding = "valid"
		self.enc = nn.Sequential(
			nn.Conv2d(1, 32, 3, 2, padding=padding),
			nn.ReLU(),
			nn.Conv2d(32, 64, 3, 2, padding=padding),
			nn.ReLU(),
			nn.Conv2d(64, K, 3, 2, padding=padding),
			nn.ReLU(),
		)

	def forward(self, x):
		x = self.enc(x)

		return F.adaptive_avg_pool2d(x, (1, 1))


class NMFDecoderMNIST(nn.Module):
	def __init__(self, N_COMP=100, H=28, W=28, init_file=None, device="cuda"):
		super(NMFDecoderMNIST, self).__init__()

		self.W = nn.Parameter(
			0.01 * torch.rand(N_COMP, H, W), requires_grad=True
		)
		self.activ = nn.ReLU()

		if init_file is not None:
			# handle numpy or torch
			if ".pt" in init_file:
				self.W.data = torch.load(
					init_file, map_location=torch.device(device)
				)
			else:
				temp = np.load(init_file)
				self.W.data = torch.as_tensor(temp).float()

	def forward(self, inp):
		# Assume input of shape n_batch x n_comp x T
		W = self.activ(self.W)

		output = torch.sum(W.unsqueeze(0) * inp, dim=1, keepdim=True)
		return output

	def return_W(self, dtype="numpy"):
		W = self.activ(self.W)

		if dtype == "numpy":
			return W.cpu().data.numpy()
		else:
			return W


class PSIMNIST(nn.Module):
	def __init__(
		self,
		dim=128,
		N_COMP=200,
		activate_class_partitioning=True,
		shared_keys=5,
	):
		super().__init__()
		self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
		# self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0,)
		self.conv3 = nn.Conv2d(256, N_COMP, kernel_size=3, stride=2, padding=1)

		self.apply(weights_init)

	def forward(self, hs):
		# hcat = hs[1]
		h1 = self.conv1(hs[0])
		h1 = F.adaptive_avg_pool2d(h1, (hs[1].shape[2], hs[1].shape[3]))

		hcat = torch.cat((h1, hs[1]), dim=1)

		hcat = self.conv3(hcat)

		return F.relu(
			F.adaptive_avg_pool2d(hcat, (1, 1))
		)  # output should be B x K x 1 x 1


class ThetaMNIST(nn.Module):
	def __init__(self, num_classes=10, N_COMP=100):
		super().__init__()
		self.lin = nn.Linear(N_COMP, num_classes, bias=False)

	def forward(self, x):
		x = x.view(x.shape[0], -1)

		return F.softmax(self.lin(x), dim=1)


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
