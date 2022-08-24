#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on wsjmix the
dataset. The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer.yaml
> python train.py hparams/dualpath_rnn.yaml
> python train.py hparams/convtasnet.yaml

The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both wsj2mix and
wsj3mix.


Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import os
import sys
import torch
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import logging


# Define training procedure
class Separation(sb.Brain):
    def compute_forward(self, chunks):
        """Forward computations """

        h = self.hparams.encoder(chunks).permute(0, 2, 1)

        # h_pooled = self.hparams.stat_pooling(h)
        # h_hat, _ = self.hparams.rnn(h_pooled.permute(1, 0, 2))

        h_hat = h.mean(dim=1, keepdim=True).permute(1, 0, 2)
        chunks_hat = self.hparams.decoder(h_hat.permute(1, 2, 0))
        # chunks_hat = chunks_hat - chunks_hat.mean(dim=-1, keepdim=True)
        # chunks_hat = chunks_hat / ( (chunks_hat**2).mean(dim=-1, keepdim=True).sqrt() + 1e-10)

        # windows = torch.hann_window(self.hparams.window_len, device=self.device).unsqueeze(0).unsqueeze(0)
        # chunks_hat = chunks_hat * windows
        # chunks_hat = chunks_hat / (chunks_hat.abs().max(dim=-1, keepdim=True)[0] + 1e-10)

        return chunks_hat, h_hat

    def compute_objectives(self, predictions, targets):
        """Computes the sinr loss"""
        return self.hparams.loss(targets, predictions)

    def _chunk(self, input, K):
        """The segmentation stage splits

        Arguments
        ---------
        K : int
            Length of the chunks.
        input : torch.Tensor
            Tensor with dim [B, N, L].

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, L = input.shape
        P = K // 2

        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.zeros(B, gap, device=input.device)
            input = torch.cat([input, pad], dim=1)

        _pad = torch.zeros(B, P, device=input.device)
        input = torch.cat([_pad, input, _pad], dim=-1)

        # [B, N, K, S]
        input1 = input[:, :-P].reshape(B, -1, K)
        input2 = input[:, P:].reshape(B, -1, K)
        input = torch.cat([input1, input2], dim=2).reshape(B, -1, K)

        return input, gap

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        signal = batch.sig.data.to(self.device)
        # signal = torch.arange(1, 100).unsqueeze(0)
        chunks, gap = self._chunk(signal, self.hparams.window_len)

        # chunks_norm = chunks / (torch.sqrt((chunks**2).mean(dim=-1, keepdim=True)) + 1e-10)
        # chunks = chunks_norm / (chunks_norm.abs().max(dim=-1, keepdim=True)[0] + 1e-10)

        windows = (
            torch.hann_window(self.hparams.window_len, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        windowed_chunks = chunks * windows
        chunks = windowed_chunks.squeeze().unsqueeze(1)
        # chunks = chunks / (chunks.abs().max(dim=-1, keepdim=True)[0] + 1e-10)

        chunks_in = chunks[:-1]
        chunks_target = chunks[:-1]

        chunks_hat, h_hat = self.compute_forward(chunks_in)
        # loss = self.compute_objectives(predictions, targets)
        loss = ((chunks_target - chunks_hat).abs()).mean()

        if (
            loss < self.hparams.loss_upper_lim and loss.nelement() > 0
        ):  # the fix for computational problems
            loss.backward()
            # if self.hparams.clip_grad_norm >= 0:
            #    torch.nn.utils.clip_grad_norm_(
            #        self.modules.parameters(), self.hparams.clip_grad_norm
            #    )
            self.optimizer.step()
        else:
            self.nonfinite_count += 1
            logger.info(
                "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                    self.nonfinite_count
                )
            )
            loss.data = torch.tensor(0).to(self.device)
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""

        signal = batch.sig.data.to(self.device)
        # signal = torch.arange(1, 100).unsqueeze(0)
        chunks, gap = self._chunk(signal, self.hparams.window_len)

        # chunks_norm = chunks / (torch.sqrt((chunks**2).mean(dim=-1, keepdim=True)) + 1e-10)
        # chunks = chunks_norm / (chunks_norm.abs().max(dim=-1, keepdim=True)[0] + 1e-10)

        windows = (
            torch.hann_window(self.hparams.window_len, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        windowed_chunks = chunks * windows

        chunks = windowed_chunks.squeeze().unsqueeze(1)
        # chunks = chunks / (chunks.abs().max(dim=-1, keepdim=True)[0] + 1e-10)

        chunks_in = chunks[:-1]
        chunks_target = chunks[:-1]

        chunks_hat, h_hat = self.compute_forward(chunks_in)
        # loss = self.compute_objectives(predictions, targets)
        loss = (chunks_target - chunks_hat).abs().mean()

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=[15, 4], dpi=100)

        N = 3
        for i in range(N):
            plt.subplot(N, 2, 2 * i + 1)
            plt.plot(chunks_hat[10 + i].cpu().squeeze().data)

            plt.subplot(N, 2, 2 * i + 2)
            plt.plot(chunks_target[10 + i].cpu().squeeze().data)
        plt.savefig("debug_fig.png", format="png")
        plt.close(fig)

        # Manage audio file saving
        # if stage == sb.Stage.TEST and self.hparams.save_audio:
        #     if hasattr(self.hparams, "n_audio_to_save"):
        #         if self.hparams.n_audio_to_save > 0:
        #             self.save_audio(snt_id[0], mixture, targets, predictions)
        #             self.hparams.n_audio_to_save += -1
        #     else:
        #         self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speedperturb(
                    targets[:, :, i], targ_lens
                )
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length within the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts
        )

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):

                    # Apply Separation
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            batch.mix_sig, targets, sb.Stage.TEST
                        )

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets
                    )
                    sisnr_i = sisnr - sisnr_baseline

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0].t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture_signal[0].t().detach().cpu().numpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):

            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )


def dataio_prep(hparams):
    """Creates data processing pipeline"""

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    datasets = [train_data, valid_data]

    # 2. Provide audio pipelines
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        # sig = torch.sin(2*torch.pi*(2/8)*torch.arange(1, sig.shape[0]))
        sig = torch.ones(sig.shape[0])
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    return train_data, valid_data


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Check if wsj0_tr is set with dynamic mixing
    if hparams["dynamic_mixing"] and not os.path.exists(
        hparams["base_folder_dm"]
    ):
        print(
            "Please, specify a valid base_folder_dm folder when using dynamic mixing"
        )
        sys.exit(1)

    # Data preparation
    from recipes.TTSo.prepare_data import create_data_csv  # noqa

    run_on_main(
        create_data_csv,
        kwargs={
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
        },
    )

    # Create dataset objects
    train_data, valid_data = dataio_prep(hparams)

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if not hparams["test_only"]:
        # Training
        separator.fit(
            separator.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )

    # Eval
    # separator.evaluate(test_data, min_key="si-snr")
    # separator.save_results(test_data)
