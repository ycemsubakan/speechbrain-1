#!/usr/bin/python3
import os
import sys
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from esc50_prepare import prepare_esc50
from speechbrain.utils.metric_stats import MetricStats
from os import makedirs
import torch.nn.functional as F
from speechbrain.processing.NMF import spectral_phase
from speechbrain.nnet.losses import cal_si_snr
import scipy.io.wavfile as wavf
import numpy as np

eps = 1e-10


class InterpreterESC50Brain(sb.core.Brain):
    """Class for sound class embedding training" """

    def save_interpretations(self, wavs, reconstructions, pred_classes, batch=None):
        """ get the interpratation for a given wav file."""        
        # save reconstructed and original spectrograms
        makedirs(
            os.path.join(
                self.hparams.output_folder, f"audios_from_interpretation",
            ),
            exist_ok=True,
        )
        
        pred_cl = torch.argmax(pred_classes[0], dim=0).item()
        current_class_ind = batch.class_string_encoded.data[0].item()
        current_class_name = self.hparams.label_encoder.ind2lab[
            current_class_ind
        ]
        predicted_class_name = self.hparams.label_encoder.ind2lab[pred_cl]
        torchaudio.save(
            os.path.join(
                self.hparams.output_folder,
                f"audios_from_interpretation",
                f"original_tc_{current_class_name}_pc_{predicted_class_name}.wav",
            ),
            wavs[0].unsqueeze(0).clone().cpu(),
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(
                self.hparams.output_folder,
                f"audios_from_interpretation",
                f"interpretation_tc_{current_class_name}_pc_{predicted_class_name}.wav",
            ),
            reconstructions[0, :, 0].unsqueeze(0).clone().cpu(),
            self.hparams.sample_rate,
        )
        
        torchaudio.save(
            os.path.join(
                self.hparams.output_folder,
                f"audios_from_interpretation",
                f"garbage_tc_{current_class_name}_pc_{predicted_class_name}.wav",
            ),
            reconstructions[0, :, 1].unsqueeze(0).clone().cpu(),
            self.hparams.sample_rate,
        )


        torchaudio.save(
            os.path.join(
                self.hparams.output_folder,
                f"audios_from_interpretation",
                f"sum_tc_{current_class_name}_pc_{predicted_class_name}.wav",
            ),
            reconstructions.sum(-1)[0, :].unsqueeze(0).clone().cpu(),
            self.hparams.sample_rate,
        )

    def overlap_test(self, batch):
        """interpration test with overlapped audio"""
        wavs, _ = batch.sig
        wavs = wavs.to(self.device)

        s1 = wavs[0]
        s2 = wavs[1]

        # create the mixture with s2 being the noise (lower gain)
        mix = (s1 + (s2 * 0.2)).unsqueeze(0)

        # get the interpretation spectrogram, phase, and the predicted class
        X_int, X_stft_phase, pred_cl = self.interpret_computation_steps(mix)

        X_stft_phase_sb = torch.cat(
            (
                torch.cos(X_stft_phase).unsqueeze(-1),
                torch.sin(X_stft_phase).unsqueeze(-1),
            ),
            dim=-1,
        )

        temp = X_int.transpose(0, 1).unsqueeze(0).unsqueeze(-1)

        X_wpsb = temp * X_stft_phase_sb
        x_int_sb = self.modules.compute_istft(X_wpsb)

        # save reconstructed and original spectrograms
        # epoch = self.hparams.epoch_counter.current
        current_class_ind = batch.class_string_encoded.data[0].item()
        current_class_name = self.hparams.label_encoder.ind2lab[
            current_class_ind
        ]
        predicted_class_name = self.hparams.label_encoder.ind2lab[pred_cl]

        noise_class_ind = batch.class_string_encoded.data[1].item()
        noise_class_name = self.hparams.label_encoder.ind2lab[noise_class_ind]

        out_folder = os.path.join(
            self.hparams.output_folder,
            f"overlap_test",
            f"tc_{current_class_name}_nc_{noise_class_name}_pc_{predicted_class_name}",
        )
        makedirs(
            out_folder, exist_ok=True,
        )

        torchaudio.save(
            os.path.join(out_folder, "mixture.wav"),
            mix,
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "source.wav"),
            s1.unsqueeze(0),
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "noise.wav"),
            s2.unsqueeze(0),
            self.hparams.sample_rate,
        )

        torchaudio.save(
            os.path.join(out_folder, "interpretation.wav"),
            x_int_sb,
            self.hparams.sample_rate,
        )

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + sound classifier.
        Data augmentation and environmental corruption are applied to the
        input sound.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        X_stft = self.modules.compute_stft(wavs)
        X_stft_power = sb.processing.features.spectral_magnitude(
            X_stft, power=self.hparams.spec_mag_power
        )
        X_logmel = self.modules.compute_fbank(X_stft_power)

        # Embeddings + sound classifier
        embeddings, f_I = self.hparams.embedding_model(X_logmel)
        predictions = self.hparams.classifier(embeddings).squeeze(1)
        
        psi_out = self.modules.psi(f_I)  # generate nmf activations

        #  generate log-mag spectrogram
        reconstructed = self.modules.decoder(wavs, psi_out)
        # self.hparams.tensorboard_train_logger.log_audio("interpret", reconstructed[0, :, 0], sample_rate=self.hparams.sample_rate)
        # self.hparams.tensorboard_train_logger.log_audio("garbage", reconstructed[0, :, 1], sample_rate=self.hparams.sample_rate)
       
        # here select only interepretation source
        interpretation = reconstructed[..., 0].clone()
        interpretation = interpretation / torch.max(interpretation, dim=1, keepdim=True).values

        r_stft = self.modules.compute_stft(interpretation)
        r_stft_power = sb.processing.features.spectral_magnitude(
            r_stft, power=self.hparams.spec_mag_power
        )
        r_logmel = self.modules.compute_fbank(r_stft_power)

        # generate classifications from interpretation 
        embeddings_reconstruction, _ = self.hparams.embedding_model(r_logmel)
        theta_out = self.hparams.classifier(embeddings_reconstruction).squeeze(1)

        if stage == sb.Stage.VALID:
            # save some samples
            if (
                self.hparams.epoch_counter.current
                % self.hparams.interpret_period
            ) == 0 and self.hparams.save_interpretations:
                wavs = wavs[0].unsqueeze(0)
                self.save_interpretations(wavs, reconstructed, predictions, batch)
                #self.interpret_sample(wavs, batch)
                #self.overlap_test(batch)

        return (reconstructed, wavs), (predictions, theta_out)

    def compute_objectives(self, pred, batch, stage):
        """Computes the loss using class-id as label."""
        (
            (reconstructions, wavs),
            (classification_out, theta_out),
        ) = pred

        uttid = batch.id
        classid, _ = batch.class_string_encoded

        batch = batch.to(self.device)
        wavs, lens = batch.sig

        if stage == sb.Stage.VALID or stage == sb.Stage.TEST:
            self.top_3_fidelity.append(batch.id, theta_out, classification_out)
            self.faithfulness.append(batch.id, wavs, reconstructions, classification_out)
        
        self.acc_metric.append(
            uttid, predict=classification_out, target=classid, length=lens
        )
        loss_rec = ((wavs.t().unsqueeze(-1) - reconstructions.sum(-1).t().unsqueeze(-1))**2).mean()
        # loss_rec = loss_rec * 0.01  # scaling to match fdi range..

        #loss_rec = ((reconstructions.sum(-1) - wavs) ** 2).mean()
        #loss_nmf = self.hparams.alpha * loss_nmf

        # HERE add energy term on garbage reconstruction
        #loss_nmf += self.hparams.beta * (time_activations).abs().mean()

        if stage != sb.Stage.TEST:
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        loss_fdi = 2 * (F.softmax(classification_out, dim=1) * -torch.log(F.softmax(theta_out, dim=1))).mean()
        #print(f"fdi {loss_fdi.item()} - rec {loss_rec.item()}")

        return loss_rec + loss_fdi

    def on_stage_start(self, stage, epoch=None):
        def accuracy_value(predict, target, length):
            """Computes Accuracy"""
            # predict = predict.argmax(1, keepdim=True)
            nbr_correct, nbr_total = sb.utils.Accuracy.Accuracy(
                predict.unsqueeze(1), target, length
            )
            acc = torch.tensor([nbr_correct / nbr_total])
            return acc

        @torch.no_grad()
        def compute_fidelity(theta_out, predictions):
            """ Computes top-`k` fidelity of interpreter. """
            predictions = F.softmax(predictions, dim=1)
            theta_out = F.softmax(theta_out, dim=1)

            pred_cl = torch.argmax(predictions, dim=1)
            k_top = torch.topk(theta_out, k=self.hparams.k_fidelity, dim=1)[1]

            # 1 element for each sample in batch, is 0 if pred_cl is in top k
            temp = (k_top - pred_cl.unsqueeze(1) == 0).sum(1)

            return temp

        @torch.no_grad()
        def compute_faithfulness(wavs, reconstructions, predictions):
            garbage = wavs - reconstructions[..., 0]
            X_stft = self.modules.compute_stft(garbage).to(self.device)
            X_stft_power = sb.processing.features.spectral_magnitude(
                X_stft, power=self.hparams.spec_mag_power
            )

            X2_logmel = self.modules.compute_fbank(X_stft_power)

            embeddings, _ = self.hparams.embedding_model(X2_logmel)
            predictions_masked = self.hparams.classifier(embeddings).squeeze(1)

            predictions = F.softmax(predictions, dim=1)
            predictions_masked = F.softmax(predictions_masked, dim=1)

            # get the prediction indices
            pred_cl = predictions.argmax(dim=1, keepdim=True)

            # get the corresponding output probabilities
            predictions_selected = torch.gather(
                predictions, dim=1, index=pred_cl
            )
            predictions_masked_selected = torch.gather(
                predictions_masked, dim=1, index=pred_cl
            )

            faithfulness = (
                predictions_selected - predictions_masked_selected
            ).squeeze()

            return faithfulness

        self.top_3_fidelity = MetricStats(metric=compute_fidelity)
        self.faithfulness = MetricStats(metric=compute_faithfulness)
        self.acc_metric = sb.utils.metric_stats.MetricStats(
            metric=accuracy_value, n_jobs=1
        )

        return super().on_stage_start(stage, epoch)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Plots in subplots the values of `self.batch_to_plot` and saves the
        plot to the experiment folder. `self.hparams.output_folder`"""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_stats = {
                "loss": self.train_loss,
                "acc": self.acc_metric.summarize("average"),
            }

        if stage == sb.Stage.VALID:
            current_fid = self.top_3_fidelity.summarize("average")
            old_lr, new_lr = self.hparams.lr_annealing(
                [self.optimizer], epoch, -current_fid
            )
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            valid_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "top-3_fid": current_fid,
                "faithfulness_median": torch.Tensor(
                    self.faithfulness.scores
                ).median(),
                "faithfulness_mean": torch.Tensor(
                    self.faithfulness.scores
                ).mean(),
            }

            if self.hparams.use_tensorboard:
                self.hparams.tensorboard_train_logger.log_stats(
                    {"Epoch": epoch}, self.train_stats, valid_stats
                )
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=valid_stats, max_keys=["top-3_fid"]
            )

        if stage == sb.Stage.TEST:
            current_fid = self.top_3_fidelity.summarize("average")
            test_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize("average"),
                "top-3_fid": current_fid,
                "faithfulness_median": torch.Tensor(
                    self.faithfulness.scores
                ).median(),
                "faithfulness_mean": torch.Tensor(
                    self.faithfulness.scores
                ).mean(),
            }

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch}, test_stats=test_stats
            )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_audio_folder = hparams["audio_data_folder"]
    config_sample_rate = hparams["sample_rate"]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    # TODO  use SB implementation but need to make sure it give the same results as PyTorch
    # resampler = sb.processing.speech_augmentation.Resample(orig_freq=latest_file_sr, new_freq=config_sample_rate)
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        wave_file = data_audio_folder + "/{:}".format(wav)

        sig, read_sr = torchaudio.load(wave_file)

        # If multi-channels, downmix it to a mono channel
        sig = torch.squeeze(sig)
        if len(sig.shape) > 1:
            sig = torch.mean(sig, dim=0)

        # Convert sample rate to required config_sample_rate
        if read_sr != config_sample_rate:
            # Re-initialize sampler if source file sample rate changed compared to last file
            if read_sr != hparams["resampler"].orig_freq:
                hparams["resampler"] = torchaudio.transforms.Resample(
                    orig_freq=read_sr, new_freq=config_sample_rate
                )
            # Resample audio
            sig = hparams["resampler"].forward(sig)

        # the librosa version
        fs, inp_audio = wavf.read(wave_file)
        inp_audio = inp_audio.astype(np.float32)
        inp_audio = inp_audio / inp_audio.max()

        if hparams['limit_training_signal_len']:
            randstart = torch.randint(
                0,
                1 + max(0, inp_audio.shape[0] - hparams['training_signal_len']),
                (1,),
            ).item()
            inp_audio = inp_audio[
                randstart : randstart + hparams['training_signal_len']
            ]

        return torch.from_numpy(inp_audio)

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("class_string")
    @sb.utils.data_pipeline.provides("class_string", "class_string_encoded")
    def label_pipeline(class_string):
        yield class_string
        class_string_encoded = label_encoder.encode_label_torch(class_string)
        yield class_string_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "class_string_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="class_string",
    )

    return datasets, label_encoder


if __name__ == "__main__":

    # # This flag enables the inbuilt cudnn auto-tuner
    # torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Tensorboard logging
    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs_folder"]
        )

    run_on_main(
        prepare_esc50,
        kwargs={
            "data_folder": hparams["data_folder"],
            "audio_data_folder": hparams["audio_data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "train_fold_nums": hparams["train_fold_nums"],
            "valid_fold_nums": hparams["valid_fold_nums"],
            "test_fold_nums": hparams["test_fold_nums"],
            "skip_manifest_creation": hparams["skip_manifest_creation"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    datasets, label_encoder = dataio_prep(hparams)
    hparams["label_encoder"] = label_encoder

    class_labels = list(label_encoder.ind2lab.values())
    print("Class Labels:", class_labels)

    Interpreter_brain = InterpreterESC50Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if "pretrained_esc50" in hparams:
        run_on_main(hparams["pretrained_esc50"].collect_files)
        hparams["pretrained_esc50"].load_collected()
        
    hparams["embedding_model"].to(hparams["device"])
    hparams["classifier"].to(hparams["device"])
    hparams["theta"].to(hparams["device"])

    # classifier is fixed here
    hparams["embedding_model"].eval()
    hparams["classifier"].eval()
    hparams["theta"].to(hparams["device"])

    if not hparams["test_only"]:
        Interpreter_brain.fit(
            epoch_counter=Interpreter_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    else:
        # Load the best checkpoint for evaluation
        test_stats = Interpreter_brain.evaluate(
            test_set=datasets["test"],
            min_key="error",
            progressbar=True,
            test_loader_kwargs=hparams["dataloader_options"],
        )
