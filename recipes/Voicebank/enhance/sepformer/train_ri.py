#!/usr/bin/env/python3
"""Recipe for training a speech enhancement system with the Voicebank dataset.

To run this recipe, do the following:
> python train.py hparams/{hyperparam_file}.yaml

Authors
 * Szu-Wei Fu 2020
"""
import os
import sys
import torch
import torchaudio
import torch.nn.functional as F
import speechbrain as sb
from pesq import pesq
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.processing.features import spectral_magnitude
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.utils.distributed import run_on_main
import speechbrain.nnet.schedulers as schedulers

from speechbrain.dataio.batch import PaddedBatch, PaddedData

# Brain class for speech enhancement training
class SEBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the enhanced output."""
        batch = batch.to(self.device)
        noisy_wavs, lens = batch.noisy_sig

        # cut signal, get the random starting point
        #if stage == sb.Stage.TRAIN:
        if self.hparams.use_freq_domain:

            if self.hparams.spect_use_type == 'ri':
                mix_w = self.compute_feats_ri(noisy_wavs)
                mix_w = mix_w.permute(0, 2, 1)
                predict_spec = self.modules.masknet(mix_w).squeeze(0)
                predict_spec = torch.stack([predict_spec[:, :self.hparams.N_fft_effective, :], predict_spec[:, self.hparams.N_fft_effective:, :]], dim=-1).permute(0, 2, 1, 3)
                est_source = self.hparams.compute_ISTFT(predict_spec)

            elif self.hparams.spect_use_type == 'complex_mask':
                EPS = 1e-10
                mix_stft = self.hparams.compute_STFT(noisy_wavs)
                mix_w_complex = torch.view_as_complex(mix_stft)
                mix_w_mag = mix_w_complex.abs()
                mix_w_phase = mix_w_complex / (mix_w_mag + EPS)

                mix_w = self.compute_feats_ri(noisy_wavs)
                mix_w = mix_w.permute(0, 2, 1)
                predict_spec = self.modules.masknet(mix_w).squeeze(0)
                predict_spec = predict_spec.permute(0, 2, 1)

                predict_real = predict_spec[:, :, :self.hparams.N_fft_effective]
                predict_im = predict_spec[:, :, self.hparams.N_fft_effective:]
                predict = torch.stack([predict_real, predict_im], dim=-1)
                mask_complex = torch.view_as_complex(predict)
                mask_mag = torch.tanh(mask_complex.abs())
                mask_phase = mask_complex / (mask_complex.abs() + EPS)

                predict_spec = (mask_mag * mix_w_mag) * mask_phase * mix_w_phase
                predict_spec = torch.view_as_real(predict_spec)
                est_source = self.hparams.compute_ISTFT(predict_spec)
            else:
                mix_w = self.compute_feats(noisy_wavs)
                mix_w = mix_w.permute(0, 2, 1)
                est_mask = self.modules.masknet(mix_w).squeeze(0)

                predict_spec = (mix_w * est_mask).permute(0, 2, 1)

                if self.hparams.spect_transform == 'log':
                    est_source = self.hparams.resynth(torch.expm1(predict_spec),
                                                      noisy_wavs)
                elif self.hparams.spect_transform == 'sqrt':
                    est_source = self.hparams.resynth(predict_spec.pow(2),
                                                      noisy_wavs)
                else:
                    est_source = self.hparams.resynth(predict_spec,
                                                      noisy_wavs)

                # est_source = est_source[:, :T_origin]
            est_source = est_source.squeeze(-1)

        else:
            mix_w = self.hparams.Encoder(mix)
            est_mask = self.modules.masknet(mix_w)

            mix_w = torch.stack([mix_w] * self.hparams.num_spks)
            sep_h = mix_w * est_mask
            est_source = torch.cat(
                [
                    self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                    for i in range(self.hparams.num_spks)
                ],
                dim=-1,
            )


            # T changed after conv1d in encoder, fix it here
            T_origin = noisy_wavs.size(1)
            T_est = est_source.size(1)
            if T_origin > T_est:
                est_source = F.pad(est_source, (0, T_origin - T_est))
            else:
                est_source = est_source[:, :T_origin]
            est_source = est_source.squeeze(-1)

            
            predict_spec = self.compute_feats(est_source)

        # Also return predicted wav
        # predict_wav = self.hparams.resynth(
        #    torch.expm1(predict_spec), noisy_wavs
        # )

        return predict_spec, est_source

    def compute_feats(self, wavs):
        """Feature computation pipeline"""
        feats = self.hparams.compute_STFT(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        if self.hparams.spect_transform == 'log':
            feats = torch.log1p(feats)
        elif self.hparams.spect_transform == 'sqrt':
            feats = torch.sqrt(feats)
        else:
            feats = feats
            
        return feats


    def compute_feats_ri(self, wavs):
        """Feature computation pipeline for RI estimation"""
        feats = self.hparams.compute_STFT(wavs)
        feats = torch.cat([feats[:, :, :, 0], feats[:, :, :, 1]], dim=2)
                    
        return feats




    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        noisy_sig, noisy_sig_lens = batch.noisy_sig[0].to(self.device), batch.noisy_sig[1].to(self.device)
        targets, targets_lens = batch.clean_sig[0].to(self.device), batch.clean_sig[1].to(self.device)

        noisy_sig, self.randstart = self.cut_signals(noisy_sig)
        targets = targets[:, self.randstart:self.randstart + self.hparams.training_signal_len]

        if self.hparams.use_speed_perturb:
            noise = noisy_sig - targets
            #batch_speed = torch.concat([noisy_sigs, targets], dim=0)
            #lens = torch.concat([noisy_sigs_lens, targets_lens])
            targets_speed = self.hparams.speedperturb(targets, torch.ones(noise.shape[0], device=self.device))

            #torchaudio.save('noisy.wav', noisy_sigs[0:1].cpu().data, 16000)
            #torchaudio.save('clean.wav', targets[0:1].cpu().data, 16000)

            #torchaudio.save('noisy_speed.wav', batch_speed[0:1].cpu().data, 16000)
            #torchaudio.save('mixminusclean.wav', (noisy_sigs - targets)[0:1].cpu().data, 16000)

            len_noise = noise.shape[1]
            len_targets = targets_speed.shape[1]
            min_len = min(len_noise, len_targets)

            targets_speed = targets_speed[:, :min_len]
            noise = noise[:, :min_len]


            # shuffle the noise
            Nbatch = noise.shape[0]           
            randperm = torch.randperm(Nbatch)
            if self.hparams.use_dm:
                noise = noise[randperm, :]
            
            # normalize noises
            if self.hparams.normalize_levels:
                en_noise = ((noise**2).sum(-1, keepdim=True).sqrt())
                en_targets = (targets_speed**2).sum(-1, keepdim=True).sqrt()
                
                eps = 1e-8
                #noise = noise / (en_noise + eps)           
                #targets_speed = targets_speed / (en_targets + eps)

                random_gain = (1 + torch.randn(Nbatch, 1) * 0.3).to(self.device)
                targets_speed = targets_speed * random_gain

                #en_noise = ((noise**2).sum(-1, keepdim=True).sqrt())
                #en_targets = (targets_speed**2).sum(-1, keepdim=True).sqrt()

                #print( (en_targets / en_noise).sqrt() )
                #print( (en_targets / en_noise).sqrt().mean())

            # add the noise
            noisy_sig = targets_speed[:, :min_len] + noise[:, :min_len]
            
            #N_batch = noisy_sigs.shape[0]
            #noisy_sigs = batch_speed[:N_batch, :]
            #targets = batch_speed[N_batch:, :]
            
            padded_targets = PaddedData(targets_speed, targets_lens)
            padded_nsigs = PaddedData(noisy_sig, noisy_sig_lens)
            batch.clean_sig = padded_targets
            batch.noisy_sig = padded_nsigs

            #batch = PaddedBatch([clean_sig, noisy_sigs], padded_keys=['noisy_sig', ])

            #batch.clean_sig[0] = targets
            #batch.noisy_sigs[0] = noisy_sigs

        predictions = self.compute_forward(
            batch, sb.Stage.TRAIN
        )
        loss = self.compute_objectives(predictions, batch, stage=sb.Stage.TRAIN)

        if self.hparams.threshold_byloss:
            th = self.hparams.threshold
            loss_to_keep = loss[loss > th]
            if loss_to_keep.nelement() > 0:
                loss = loss_to_keep.mean()
        else:
            loss = loss.mean()

        if (
            loss < self.hparams.loss_upper_lim and loss.nelement() > 0
        ):  # the fix for computational problems
            loss.backward()
            if self.hparams.clip_grad_norm >= 0:
                torch.nn.utils.clip_grad_norm_(
                    self.modules.parameters(), self.hparams.clip_grad_norm
                )
            self.optimizer.step()
        else:
            self.nonfinite_count += 1
            print(
                "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                    self.nonfinite_count
                )
            )
            loss.data = torch.tensor(0.0).to(self.device)
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs"""
        predict_spec, predict_wav = predictions
        clean_wavs, lens = batch.clean_sig
        # noisy_wavs, lens = batch.noisy_sig

        #if stage == sb.Stage.TRAIN:
            #clean_wavs, _ = self.cut_signals(clean_wavs, randstart=self.randstart)

        if getattr(self.hparams, "waveform_target", False):
            loss = self.hparams.compute_cost(clean_wavs.unsqueeze(-1), predict_wav.unsqueeze(-1))
            self.loss_metric.append(
                batch.id, clean_wavs.unsqueeze(-1), predict_wav.unsqueeze(-1)
                )
        else:
            if self.hparams.spect_use_type in ['ri', 'complex_mask']:
                clean_spec = self.hparams.compute_STFT(clean_wavs)
                len1 = clean_spec.shape[1]
                len2 = predict_spec.shape[1]
                if len1 != len2:
                    min_len = min(len1, len2)
                    clean_spec = clean_spec[:, :min_len]
                    predict_spec = predict_spec[:, :min_len]

                if getattr(self.hparams, "use_timedom_loss", False):
                    len1 = clean_wavs.shape[1]
                    len2 = predict_wav.shape[1]
                    if len1 != len2:
                        min_len = min(len1, len2)
                        clean_wavs = clean_wavs[:, :min_len]
                        predict_wav = predict_wav[:, :min_len]

                    loss = self.hparams.compute_cost(clean_wavs.unsqueeze(-1), predict_wav.unsqueeze(-1)).mean()
                else:
                    loss = self.hparams.compute_cost(predict_spec[0], clean_spec[0]) + \
                           self.hparams.compute_cost(predict_spec[1], clean_spec[1])
            else:
                clean_spec = self.compute_feats(clean_wavs)
                len1 = clean_spec.shape[1]
                len2 = predict_spec.shape[1]
                if len1 != len2:
                    min_len = min(len1, len2)
                    clean_spec = clean_spec[:, :min_len]
                    predict_spec = predict_spec[:, :min_len]

                loss = self.hparams.compute_cost(predict_spec, clean_spec)
            
            if getattr(self.hparams, "use_timedom_loss", False):
                self.loss_metric.append(
                    batch.id, predict_spec, clean_spec
                )
            else:
                self.loss_metric.append(
                    batch.id, predict_spec, clean_spec, reduction="batch"
                )

        # torchaudio.save('clean.wav', clean_wavs[0:1].cpu().data, 16000)
        # torchaudio.save('predict_wav.wav', predict_wav[0:1].cpu().data, 16000)
        # torchaudio.save('noisy_wav.wav', noisy_wavs[0:1].cpu().data, 16000)

        # import pdb; pdb.set_trace()

        if stage != sb.Stage.TRAIN:

            # Evaluate speech quality/intelligibility
            self.stoi_metric.append(
                batch.id, predict_wav, clean_wavs, lens, reduction="batch"
            )
            self.pesq_metric.append(
                batch.id, predict=predict_wav, target=clean_wavs, lengths=lens
            )

            # Write wavs to file
            if stage == sb.Stage.TEST:
                lens = lens * clean_wavs.shape[1]
                for name, pred_wav, length in zip(batch.id, predict_wav, lens):
                    name += ".wav"
                    enhance_path = os.path.join(
                        self.hparams.enhanced_folder, name
                    )
                    torchaudio.save(
                        enhance_path,
                        torch.unsqueeze(pred_wav[: int(length)].cpu(), 0),
                        16000,
                    )

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch"""
        self.loss_metric = MetricStats(metric=self.hparams.compute_cost)
        self.stoi_metric = MetricStats(metric=stoi_loss)

        # Define function taking (prediction, target) for parallel eval
        def pesq_eval(pred_wav, target_wav):
            """Computes the PESQ evaluation metric"""
            return pesq(
                fs=16000,
                ref=target_wav.numpy(),
                deg=pred_wav.numpy(),
                mode="wb",
            )

        if stage != sb.Stage.TRAIN:
            self.pesq_metric = MetricStats(
                metric=pesq_eval, n_jobs=1, batch_eval=False
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_stats = {"loss": self.loss_metric.scores}
        else:
            stats = {
                "loss": stage_loss,
                "pesq": self.pesq_metric.summarize("average"),
                "stoi": -self.stoi_metric.summarize("average"),
            }

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

            if self.hparams.use_tensorboard:
                valid_stats = {
                    "loss": self.loss_metric.scores,
                    "stoi": self.stoi_metric.scores,
                    "pesq": self.pesq_metric.scores,
                }
                self.hparams.tensorboard_train_logger.log_stats(
                    {"Epoch": epoch}, self.train_stats, valid_stats
                )
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": current_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["pesq"])

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def cut_signals(self, signal, randstart=None):
        """This function selects a random segment of a given length withing the mixture.
        The corresponding targets are selected accordingly"""

        if randstart is None:
            randstart = torch.randint(
                0,
                1 + max(0, signal.shape[1] - self.hparams.training_signal_len),
                (1,),
            ).item()
        signal = signal[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return signal, randstart


    def add_speed_perturb(self, targets, mix, target_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1

        if self.hparams.use_speedperturb:
            import pdb; pdb.set_trace()
            # Performing speed change (independently on each source)
            
            #new_targets = []
            #recombine = True

            #for i in range(targets.shape[-1]):
            #    new_target = self.hparams.speedperturb(
            #        targets[:, :, i], targ_lens
            #    )
            #    new_targets.append(new_target)
            #    if i == 0:
            #        min_len = new_target.shape[-1]
            #    else:
            #        if new_target.shape[-1] < min_len:
            #            min_len = new_target.shape[-1]

            #if self.hparams.use_rand_shift:
            #    # Performing random_shift (independently on each source)
            #    recombine = True
            #    for i in range(targets.shape[-1]):
            #        rand_shift = torch.randint(
            #            self.hparams.min_shift, self.hparams.max_shift, (1,)
            #        )
            #        new_targets[i] = new_targets[i].to(self.device)
            #        new_targets[i] = torch.roll(
            #            new_targets[i], shifts=(rand_shift[0],), dims=1
            #        )

            ## Re-combination
            #if recombine:
            #    if self.hparams.use_speedperturb:
            #        targets = torch.zeros(
            #            targets.shape[0],
            #            min_len,
            #            targets.shape[-1],
            #            device=targets.device,
            #            dtype=torch.float,
            #        )
            #    for i, new_target in enumerate(new_targets):
            #        targets[:, :, i] = new_targets[i][:, 0:min_len]

        #mix = targets.sum(-1)
        return mix, targets


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    # Define audio pipelines
    @sb.utils.data_pipeline.takes("noisy_wav")
    @sb.utils.data_pipeline.provides("noisy_sig")
    def noisy_pipeline(noisy_wav):
        return sb.dataio.dataio.read_audio(noisy_wav)

    @sb.utils.data_pipeline.takes("clean_wav")
    @sb.utils.data_pipeline.provides("clean_sig")
    def clean_pipeline(clean_wav):
        return sb.dataio.dataio.read_audio(clean_wav)

    # Define datasets
    datasets = {}
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[noisy_pipeline, clean_pipeline],
            output_keys=["id", "noisy_sig", "clean_sig"],
        )

    # Sort train dataset
    if hparams["sorting"] == "ascending" or hparams["sorting"] == "descending":
        datasets["train"] = datasets["train"].filtered_sorted(
            sort_key="length", reverse=hparams["sorting"] == "descending"
        )
        hparams["dataloader_options"]["shuffle"] = False
    elif hparams["sorting"] != "random":
        raise NotImplementedError(
            "Sorting must be random, ascending, or descending"
        )

    return datasets


def create_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


# Recipe begins!
if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Data preparation
    from voicebank_prepare import prepare_voicebank  # noqa

    run_on_main(
        prepare_voicebank,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["output_folder"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Create dataset objects
    datasets = dataio_prep(hparams)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs"]
        )

    # Create the folder to save enhanced files (+ support for DDP)
    run_on_main(create_folder, kwargs={"folder": hparams["enhanced_folder"]})

    se_brain = SEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Load latest checkpoint to resume training
    if not hparams['test_only']:
        se_brain.fit(
            epoch_counter=se_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )

    # Load best checkpoint for evaluation
    test_stats = se_brain.evaluate(
        test_set=datasets["test"],
        max_key="pesq",
        test_loader_kwargs=hparams["dataloader_options"],
    )
