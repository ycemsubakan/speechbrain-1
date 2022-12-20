import torch
from torch.autograd import Function


def get_irrelevant_regions(labels, K, num_classes, N_shared=5, stage="TRAIN"):
    # we can make this more uniform
    uniform_mat = torch.round(
            torch.linspace(-0.5, num_classes - 0.51, K - N_shared)
    ).to(labels.device)

    uniform_mat = uniform_mat.unsqueeze(0).repeat(labels.shape[0], 1)

    labels_expanded = labels.unsqueeze(1).repeat(1, K - N_shared)

    irrelevant_regions = uniform_mat != labels_expanded

    if stage == "TRAIN":
        irrelevant_regions = torch.cat(
                [irrelevant_regions, torch.ones(irrelevant_regions.shape[0], N_shared).to(labels.device)],
                dim=1
        ) == 1
    else:
        irrelevant_regions = torch.cat(
                [irrelevant_regions, torch.zeros(irrelevant_regions.shape[0], N_shared).to(labels.device)],
                dim=1
        ) == 1

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
        training=True
    ):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            labels_expanded = labels.reshape(-1, 1, 1).repeat(1, 7, 7)
            labels_flatten = labels_expanded.reshape(-1)

            irrelevant_regions = get_irrelevant_regions(
                labels_flatten,
                codebook.shape[0],
                num_classes,
                N_shared=shared_keys,
                stage="TRAIN" if training else "VALID"
            )

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

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
        training=True
    ):
        indices = vq(
            inputs,
            codebook,
            labels,
            num_classes,
            activate_class_partitioning,
            shared_keys,
            training
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
        training=True
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


vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]
