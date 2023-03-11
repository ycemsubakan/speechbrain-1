import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

from speechbrain.lobes.models.dual_path import (
	Encoder,
	SBTransformerBlock,
	Dual_Path_Model,
	Decoder,
)



def get_irrelevant_regions(labels, K, num_classes, N_shared=5, stage="TRAIN"):
	# we can make this more uniform
	uniform_mat = torch.round(
		torch.linspace(-0.5, num_classes - 0.51, K - N_shared)
	).to(labels.device)

	uniform_mat = uniform_mat.unsqueeze(0).repeat(labels.shape[0], 1)

	labels_expanded = labels.unsqueeze(1).repeat(1, K - N_shared)

	irrelevant_regions = uniform_mat != labels_expanded

	if stage == "TRAIN":
		irrelevant_regions = (
			torch.cat(
				[
					irrelevant_regions,
					torch.ones(irrelevant_regions.shape[0], N_shared).to(
						labels.device
					),
				],
				dim=1,
			)
			== 1
		)
	else:
		irrelevant_regions = (
			torch.cat(
				[
					irrelevant_regions,
					torch.zeros(irrelevant_regions.shape[0], N_shared).to(
						labels.device
					),
				],
				dim=1,
			)
			== 1
		)

	return irrelevant_regions


class VectorQuantization(Function):
	@staticmethod
	def forward(
		ctx,
		inputs,
		codebook,
		labels=None,
		num_classes=10,
		activate_class_partitioning=True,
		shared_keys=10,
		training=True,
	):
		with torch.no_grad():
			embedding_size = codebook.size(1)
			inputs_size = inputs.size()
			inputs_flatten = inputs.view(-1, embedding_size)

			labels_expanded = labels.reshape(-1, 1, 1).repeat(
				1, inputs_size[1], inputs_size[2]
			)
			labels_flatten = labels_expanded.reshape(-1)

			irrelevant_regions = get_irrelevant_regions(
				labels_flatten,
				codebook.shape[0],
				num_classes,
				N_shared=shared_keys,
				stage="TRAIN" if training else "VALID",
			)

			codebook_sqr = torch.sum(codebook**2, dim=1)
			inputs_sqr = torch.sum(inputs_flatten**2, dim=1, keepdim=True)

			# Compute the distances to the codebook
			distances = torch.addmm(
				codebook_sqr + inputs_sqr,
				inputs_flatten,
				codebook.t(),
				alpha=-2.0,
				beta=1.0,
			)

			# intervene and boost the distances for irrelevant codes
			if activate_class_partitioning:
				distances[irrelevant_regions] = torch.inf

			_, indices_flatten = torch.min(distances, dim=1)
			indices = indices_flatten.view(*inputs_size[:-1])
			ctx.mark_non_differentiable(indices)

			return indices

	@staticmethod
	def backward(ctx, grad_output):
		raise RuntimeError(
			"Trying to call `.grad()` on graph containing "
			"`VectorQuantization`. The function `VectorQuantization` "
			"is not differentiable. Use `VectorQuantizationStraightThrough` "
			"if you want a straight-through estimator of the gradient."
		)


class VectorQuantizationStraightThrough(Function):
	@staticmethod
	def forward(
		ctx,
		inputs,
		codebook,
		labels=None,
		num_classes=10,
		activate_class_partitioning=True,
		shared_keys=10,
		training=True,
	):
		indices = vq(
			inputs,
			codebook,
			labels,
			num_classes,
			activate_class_partitioning,
			shared_keys,
			training,
		)
		indices_flatten = indices.view(-1)
		ctx.save_for_backward(indices_flatten, codebook)
		ctx.mark_non_differentiable(indices_flatten)

		codes_flatten = torch.index_select(
			codebook, dim=0, index=indices_flatten
		)
		codes = codes_flatten.view_as(inputs)

		return (codes, indices_flatten)

	@staticmethod
	def backward(
		ctx,
		grad_output,
		grad_indices,
		labels=None,
		num_classes=None,
		activate_class_partitioning=True,
		shared_keys=10,
		training=True,
	):
		grad_inputs, grad_codebook = None, None

		if ctx.needs_input_grad[0]:
			# Straight-through estimator
			grad_inputs = grad_output.clone()
		if ctx.needs_input_grad[1]:
			# Gradient wrt. the codebook
			indices, codebook = ctx.saved_tensors
			embedding_size = codebook.size(1)

			grad_output_flatten = grad_output.contiguous().view(
				-1, embedding_size
			)
			grad_codebook = torch.zeros_like(codebook)
			grad_codebook.index_add_(0, indices, grad_output_flatten)

		return (grad_inputs, grad_codebook, None, None, None, None, None)

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



vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]



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
		x = x.unsqueeze(1)
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

		return h4

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

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find("Conv") != -1:
		try:
			nn.init.xavier_uniform_(m.weight.data)
			m.bias.data.fill_(0)
		except AttributeError:
			print("Skipping initialization of ", classname)


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
