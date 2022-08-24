"""
The .csv preperation functions for WSJ0-Mix.

Author
 * Cem Subakan 2020

 """

import os
import csv


def prepare_wsjmix(
    datapath,
    savepath,
    n_spks=2,
    skip_prep=False,
    librimix_addnoise=False,
    fs=8000,
):
    """
    Prepared wsj2mix if n_spks=2 and wsj3mix if n_spks=3.

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
        librimix_addnoise: If True, add whamnoise to librimix datasets
    """

    if skip_prep:
        return

    if "wsj" in datapath:

        if n_spks == 2:
            assert (
                "2speakers" in datapath
            ), "Inconsistent number of speakers and datapath"
            # create_wsj_csv(datapath, savepath)
        elif n_spks == 3:
            assert (
                "3speakers" in datapath
            ), "Inconsistent number of speakers and datapath"
            # create_wsj_csv_3spks(datapath, savepath)
        else:
            raise ValueError("Unsupported Number of Speakers")
    else:
        print("Creating a csv file for a custom dataset")
        create_custom_dataset(datapath, savepath)


def create_custom_dataset(
    datapath,
    savepath,
    dataset_name="custom",
    set_types=["train", "valid", "test"],
    folder_names={
        "source1": "source1",
        "source2": "source2",
        "mixture": "mixture",
    },
):
    """
    This function creates the csv file for a custom source separation dataset
    """

    for set_type in set_types:
        mix_path = os.path.join(datapath, set_type, folder_names["mixture"])
        s1_path = os.path.join(datapath, set_type, folder_names["source1"])
        s2_path = os.path.join(datapath, set_type, folder_names["source2"])

        files = os.listdir(mix_path)

        mix_fl_paths = [os.path.join(mix_path, fl) for fl in files]
        s1_fl_paths = [os.path.join(s1_path, fl) for fl in files]
        s2_fl_paths = [os.path.join(s2_path, fl) for fl in files]

        csv_columns = [
            "ID",
            "duration",
            "mix_wav",
            "mix_wav_format",
            "mix_wav_opts",
            "s1_wav",
            "s1_wav_format",
            "s1_wav_opts",
            "s2_wav",
            "s2_wav_format",
            "s2_wav_opts",
            "noise_wav",
            "noise_wav_format",
            "noise_wav_opts",
        ]

        with open(
            os.path.join(savepath, dataset_name + "_" + set_type + ".csv"), "w"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for i, (mix_path, s1_path, s2_path) in enumerate(
                zip(mix_fl_paths, s1_fl_paths, s2_fl_paths)
            ):

                row = {
                    "ID": i,
                    "duration": 1.0,
                    "mix_wav": mix_path,
                    "mix_wav_format": "wav",
                    "mix_wav_opts": None,
                    "s1_wav": s1_path,
                    "s1_wav_format": "wav",
                    "s1_wav_opts": None,
                    "s2_wav": s2_path,
                    "s2_wav_format": "wav",
                    "s2_wav_opts": None,
                }
                writer.writerow(row)


def create_data_csv(datapath, savepath):

    """
    This function creates the csv files to get the speechbrain data loaders for the wsj0-2mix dataset.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """

    csv_columns = [
        "ID",
        "duration",
        "wav",
        "wav_format",
        "wav_opts",
    ]
    files = os.listdir(datapath)
    fl_paths = [os.path.join(datapath, fl) for fl in files]

    with open(savepath + "/dataset.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for i, (fl_path) in enumerate(fl_paths):

            row = {
                "ID": i,
                "duration": 1.0,
                "wav": fl_path,
                "wav_format": "wav",
                "wav_opts": None,
            }
            writer.writerow(row)

    with open(savepath + "/dataset_prototype.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for i, (fl_path) in enumerate(fl_paths[:10]):
            row = {
                "ID": i,
                "duration": 1.0,
                "wav": fl_path,
                "wav_format": "wav",
                "wav_opts": None,
            }
            writer.writerow(row)
