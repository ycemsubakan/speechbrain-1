#!/usr/bin/python3
"""Recipe for training sound class embeddings (e.g, xvectors) using the UrbanSound8k.
We employ an encoder followed by a sound classifier.

To run this recipe, use the following command:
> python train_class_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hparams/train_x_vectors.yaml (for standard xvectors)
    hparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Authors
    * David Whipps 2021
    * Ala Eddine Limame 2021

Based on VoxCeleb By:
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import os
import sys
import torch
import torchaudio
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from esc50_prepare import prepare_esc50
from sklearn.metrics import confusion_matrix
import numpy as np
from confusion_matrix_fig import create_cm_fig

import librosa
from librosa.core import stft
import scipy.io.wavfile as wavf
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import itertools as it
import matplotlib.pyplot as plt

EPS = 1e-10


class MNISTIntBrain(sb.core.Brain):
    """Class for sound class embedding training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + sound classifier.
        Data augmentation and environmental corruption are applied to the
        input sound.
        """
        images = batch[0].to(self.device)
        labels = batch[1].to(self.device)

        pred, hs = self.hparams.classifier(images)
        class_pred = pred.argmax(dim=1)

        if self.hparams.use_vq:
            xhat, hcat, z_q_x = self.hparams.psi_model(hs, class_pred)
        else: 
            xhat, hcat = self.hparams.psi_model(hs, class_pred)
            z_q_x = None

        if hasattr(self.hparams, 'separator'):
            garbage = self.hparams.separator(images)
        else:
            garbage = 0

        return pred, xhat, hcat, z_q_x, garbage


    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using class-id as label.
        """

        predictions, xhat, hcat, z_q_x, garbage = predictions

        lens = torch.ones(batch[1].shape[0]).to(self.device)
        images = batch[0].to(self.device)
        labels = batch[1].to(self.device)
        uttid = torch.rand(batch[1].shape[0])

        loss = F.nll_loss(predictions, labels)

        # if there is a separator, we need to add sigmoid to the sum
        if hasattr(self.hparams, 'separator'):
            xhat = F.sigmoid(xhat + garbage)

            loss_fid = (-torch.exp(predictions - torch.logsumexp(predictions, dim=1, keepdim=True)) * self.hparams.classifier(xhat)[0]).mean()
        else:
            xhat = F.sigmoid(xhat)
            loss_fid = 0

        rec_loss = (
            -images * torch.log(xhat + EPS)
            - (1 - images) * torch.log(1 - xhat + EPS)
        ).mean()
        if self.hparams.use_vq: 
            loss_vq = F.mse_loss(z_q_x, hcat.detach())
            loss_commit = F.mse_loss(hcat, z_q_x.detach())
        else: 
            loss_vq = 0 
            loss_commit = 0

        # Concatenate labels (due to data augmentation)
        # loss = self.hparams.compute_cost(predictions, classid, lens)

        if stage != sb.Stage.TEST:
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(uttid, predictions, labels, reduce=False)

        # Confusion matrices
        if stage != sb.Stage.TRAIN:
            y_true = labels.cpu().detach().numpy()
            y_pred = predictions.cpu().detach().numpy().argmax(-1)

        # Compute Accuracy using MetricStats
        self.acc_metric.append(
            uttid, predict=predictions, target=labels, lengths=lens
        )

        return rec_loss + loss_vq + loss_commit + loss_fid

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(metric=F.nll_loss)

        # Compute Accuracy using MetricStats
        # Define function taking (prediction, target, length) for eval
        def accuracy_value(predict, target, lengths):
            """Computes Accuracy"""
            preds = predict.argmax(-1)
            nbr_correct = (preds == target).sum()
            nbr_total = target.shape[0]

            acc = torch.tensor([nbr_correct / nbr_total])
            return acc

        self.acc_metric = sb.utils.metric_stats.MetricStats(
            metric=accuracy_value, n_jobs=1
        )

        # Confusion matrices
        if stage == sb.Stage.VALID:
            self.valid_confusion_matrix = np.zeros(
                shape=(self.hparams.out_n_neurons, self.hparams.out_n_neurons),
                dtype=int,
            )
        if stage == sb.Stage.TEST:
            self.test_confusion_matrix = np.zeros(
                shape=(self.hparams.out_n_neurons, self.hparams.out_n_neurons),
                dtype=int,
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Compute/store important stats
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_stats = {
                "loss": self.train_loss,
                "acc": self.acc_metric.summarize(
                    "average"
                ),  # "acc": self.train_acc_metric.summarize(),
            }
        # Summarize Valid statistics from the stage for record-keeping.
        elif stage == sb.Stage.VALID:
            valid_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize(
                    "average"
                ),  # "acc": self.valid_acc_metric.summarize(),
            }
        # Summarize Test statistics from the stage for record-keeping.
        else:
            test_stats = {
                "loss": stage_loss,
                "acc": self.acc_metric.summarize(
                    "average"
                ),  # "acc": self.test_acc_metric.summarize(),
            }

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(
                [self.optimizer], epoch, current_loss=stage_loss
            )
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )
            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=valid_stats, min_keys=["error"]
            )

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            # Per class accuracy from Test confusion matrix

            self.hparams.train_logger.log_stats(
                {
                    "Epoch loaded": self.hparams.epoch_counter.current,
                    # "\n Per Class Accuracy": per_class_acc_arr_str,
                    # "\n Confusion Matrix": "\n{:}\n".format(
                    #    self.test_confusion_matrix
                    # ),
                },
                test_stats=test_stats,
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
        # if self.noise:
        #     energy_signal = (inp_audio ** 2).mean()
        #     noise = np.random.normal(0, 0.05, inp_audio.shape[0])
        #     energy_noise = (noise ** 2).mean()
        #     const = np.sqrt(energy_signal / energy_noise)
        #     noise = const * noise
        #     inp_audio = inp_audio + noise

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

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

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

    lam = lambda x: torch.clamp(x, 0, 1)
    train_kwargs = {"batch_size": 64}
    test_kwargs = {"batch_size": 128}
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lam)
            # transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    mnist_train = torchvision.datasets.MNIST(
        root=".", download="True", transform=transform
    )
    mnist_valid = torchvision.datasets.MNIST(
        root=".", download="True", train=False, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(mnist_train, **train_kwargs)
    valid_loader = torch.utils.data.DataLoader(mnist_valid, **test_kwargs)

    # Dataset IO prep: creating Dataset objects and proper encodings for phones

    mnistbrain = MNISTIntBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if "pretrained_mnist" in hparams:
        run_on_main(hparams["pretrained_mnist"].collect_files)
        hparams["pretrained_mnist"].load_collected()
        hparams["classifier"].to(hparams["device"])
        hparams["classifier"].eval()

    if not hparams["test_only"]:
        mnistbrain.fit(
            epoch_counter=mnistbrain.hparams.epoch_counter,
            train_set=train_loader,
            valid_set=valid_loader,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )

    # for x, y in it.islice(train_loader, 0, 1, 1):
    #    _, xhat, _, _ = mnistbrain.compute_forward([x, y], 'test')
    # torchvision.utils.save_image(xhat, 'notrreconstructions.png')
    #
    ## Load the best checkpoint for evaluation
    # test_stats = mnistbrain.evaluate(
    #    test_set=valid_loader,
    #    max_key="acc",
    #    progressbar=True,
    #    test_loader_kwargs=hparams["dataloader_options"],
    # )

    mnistbrain.checkpointer.recover_if_possible(
        max_key="acc", device=torch.device(mnistbrain.device),
    )

    # for x, y in it.islice(valid_loader, 0, 1, 1):
    #    _, xhat, _, _ = mnistbrain.compute_forward([x, y], 'test')
    #    torchvision.utils.save_image(xhat, 'reconstructions.png')

    for x, y in it.islice(valid_loader, 0, 1, 1):
        mask = y == 0
        x0, y0 = x[mask], y[mask]
        N0 = mask.sum().item()

        mask = y == 1
        x1, y1 = x[mask], y[mask]
        N1 = mask.sum().item()

        Nmin = min(N0, N1)

        mix_0 = torch.clamp(x0[:Nmin] + x1[:Nmin], 0, 1)
        mix_1 = torch.clamp(0.3 * x0[:Nmin] + 0.7 * x1[:Nmin], 0, 1)

        preds0, xhat0, _, _, garbage0 = mnistbrain.compute_forward(
            [mix_0, y0[:Nmin]], "test"
        )
        preds1, xhat1, _, _, garbage1 = mnistbrain.compute_forward(
            [mix_1, y1[:Nmin]], "test"
        )

        if not hparams["use_vq"]:
            xhat0, xhat1 = F.sigmoid(xhat0), F.sigmoid(xhat1)
            garbage0, garbage1 = F.sigmoid(garbage0), F.sigmoid(garbage1)

            torchvision.utils.save_image(garbage0, "garbage0.png")
            torchvision.utils.save_image(garbage1, "garbage1.png")

        preds0_cl = preds0.argmax(1)
        print(preds0_cl)
        exppreds0 = torch.exp(
            preds0 - torch.logsumexp(preds0, dim=1, keepdim=True)
        )
        print((-exppreds0 * preds0).sum(1))

        preds1_cl = preds1.argmax(1)
        print(preds1_cl)
        exppreds1 = torch.exp(
            preds1 - torch.logsumexp(preds1, dim=1, keepdim=True)
        )
        print((-exppreds1 * preds1).sum(1))

    torchvision.utils.save_image(xhat0, "reconstructions0.png")
    torchvision.utils.save_image(xhat1, "reconstructions1.png")

    torchvision.utils.save_image(mix_0, "mix_0.png")
    torchvision.utils.save_image(mix_1, "mix_1.png")
