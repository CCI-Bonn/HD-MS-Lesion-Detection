import pytest
import os
import sys
import urllib
import nibabel as nib
import numpy as np
import pandas as pd

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import datasets as data
from torch.utils.data import DataLoader


@pytest.fixture
def randomized_dataset_nii(tmp_path):
    filetype = "nii.gz"
    # generate labels
    label_dir = tmp_path / "labels"
    if not os.path.isdir(label_dir):
        label_dir.mkdir()

    label_file_path = label_dir / "labels.csv"

    data_dir = tmp_path / "data"
    if not os.path.isdir(data_dir):
        data_dir.mkdir()

    max_nb_samples = 500
    max_nb_series = 5
    avg_fr_missing_series = 0.1
    avg_fr_missing_data = 0.1
    avg_fr_missing_labels = 0.1
    avg_fr_exclusions = 0.1
    max_label_len = 10

    total_nb_samples = int(round(np.random.rand() * max_nb_samples))
    while total_nb_samples < 100:
        total_nb_samples = int(round(np.random.rand() * max_nb_samples))

    label_len = int(round(np.random.rand() * max_label_len))
    while label_len < 2:
        label_len = int(round(np.random.rand() * max_label_len))

    nb_series = int(round(np.random.rand() * max_nb_series))
    while nb_series < 2:
        nb_series = int(round(np.random.rand() * max_nb_series))

    missing_labels = []
    missing_series = []
    missing_samples = []

    labelfile_text = "ID,Label\n"

    for i in range(total_nb_samples):

        if np.random.uniform() > avg_fr_missing_labels:
            labelfile_text += "sample" + str(i) + ","
            pos_place = int(round(np.floor(np.random.rand() * label_len)))
            for place in range(label_len):
                if place != pos_place:
                    labelfile_text += "0.0 "
                else:
                    labelfile_text += "1.0 "
            labelfile_text = labelfile_text[:-1] + "\n"
        else:
            missing_labels.append("sample" + str(i))

    label_file_path.write_text(labelfile_text)

    # gen data
    for i in range(total_nb_samples):

        if np.random.uniform() > avg_fr_missing_data:
            sample_dir = data_dir / ("sample" + str(i))
            sample_dir.mkdir()
            for ser in range(nb_series):
                if np.random.uniform() > avg_fr_missing_series:
                    if filetype == "nii.gz":
                        series_path = sample_dir / ("series" + str(ser) + ".nii.gz")
                        arr = np.zeros((10, 10), np.float)
                        img = nib.Nifti1Image(arr, np.eye(4))
                        nib.save(img, series_path)
                    if filetype == "npy":
                        series_path = sample_dir / ("series" + str(ser) + ".npy")
                        arr = np.zeros((10, 10), np.float)
                        np.save(series_path, arr)
                else:
                    missing_series.append("sample" + str(i))
        else:
            missing_samples.append("sample" + str(i))

    exclusions = []
    for i in range(total_nb_samples):
        if np.random.uniform() < avg_fr_exclusions:
            exclusions.append("sample" + str(i))

    series_list = []
    for ser in range(nb_series):
        series_list.append("series" + str(ser))

    return (
        label_file_path,
        data_dir,
        series_list,
        missing_labels,
        missing_samples,
        missing_series,
        exclusions,
    )


def test_VolumeDataset_rand_split_correctly(randomized_dataset_nii):
    (
        label_file_path,
        data_dir,
        series_list,
        missing_labels,
        missing_samples,
        missing_series,
        exclusions,
    ) = randomized_dataset_nii
    print(label_file_path)
    dset = data.VolumeDataset(
        data_dir,
        label_file_path,
        series_list,
        exclusions=exclusions,
        transform=None,
        file_format=None,
        mask_series={},
    )
    splitfraction = 0.0
    while splitfraction < 0.5 and splitfraction > 0.8:
        splitfraction = 1.0 / round(np.random.rand() * 10)
    new_name_1 = "newgrp1"
    new_name_2 = "newgrp2"
    dset.rand_split("base", new_name_1, new_name_2, fraction=splitfraction)
    for item in dset.groups[new_name_1]:
        assert not item in dset.groups[new_name_2]
    assert len(dset.groups[new_name_1]) == int(
        round(len(dset.groups["base"]) * splitfraction)
    )
    assert len(dset.groups[new_name_2]) == int(
        round(len(dset.groups["base"]) * (1.0 - splitfraction))
    )
    assert len(dset.groups[new_name_1]) + len(dset.groups[new_name_2]) == len(
        dset.groups["base"]
    )


def test_VolumeDataset_crossval_split_correctly(randomized_dataset_nii):
    (
        label_file_path,
        data_dir,
        series_list,
        missing_labels,
        missing_samples,
        missing_series,
        exclusions,
    ) = randomized_dataset_nii
    dset = data.VolumeDataset(
        data_dir,
        label_file_path,
        series_list,
        exclusions=exclusions,
        transform=None,
        file_format=None,
        mask_series={},
    )
    folds = int(round(np.random.rand() * 5))
    while folds < 3:
        folds = int(round(np.random.rand() * 5))

    dset.crossval_split("base", folds)
    for i in range(folds):
        for item in dset.groups["val" + str(i)]:
            assert not item in dset.groups["train" + str(i)]
            for j in range(folds):
                if j != i:
                    assert not item in dset.groups["val" + str(j)]
        for item in dset.groups["train" + str(i)]:
            assert not item in dset.groups["val" + str(i)]

    lens_train = []
    lens_val = []
    for i in range(folds):
        lens_train.append(len(dset.groups["train" + str(i)]))
        lens_val.append(len(dset.groups["val" + str(i)]))

    ltrain_set = set(lens_train)
    lval_set = set(lens_val)

    assert len(ltrain_set) <= 2
    assert len(lval_set) <= 2

    len1_train = ltrain_set.pop()
    try:
        len2_train = ltrain_set.pop()
        assert np.abs(len1_train - len2_train) <= 1
    except KeyError:
        pass

    len1_val = lval_set.pop()
    len2_val = None
    try:
        len2_val = lval_set.pop()
        assert np.abs(len1_val - len2_val) <= 1
    except KeyError:
        pass

    if len2_val == None:
        assert len(dset.groups["base"]) // folds == len1_val
    else:
        assert len(dset.groups["base"]) // folds == min(len1_val, len2_val)


def test_VolumeDataset_draw_from_correctly(randomized_dataset_nii):
    (
        label_file_path,
        data_dir,
        series_list,
        missing_labels,
        missing_samples,
        missing_series,
        exclusions,
    ) = randomized_dataset_nii
    dset = data.VolumeDataset(
        data_dir,
        label_file_path,
        series_list,
        exclusions=exclusions,
        transform=None,
        file_format=None,
        mask_series={},
    )
    assert dset.sampling_group == "base"
    dl = DataLoader(dset, batch_size=4, shuffle=True, num_workers=0)
    data_list = []
    for i_batch, sample_batched in enumerate(dl):
        data_list += sample_batched["_idents"]
        for sname in series_list:
            assert sname in sample_batched["_meta"].keys()

    for entry in data_list:
        assert entry in dset.groups["base"]
        assert entry not in dset.groups["_no_data"]
        assert entry not in dset.groups["no_label"]
        assert entry not in dset.groups["_incomplete_data"]


def test_VolumeDataset_group_inits(randomized_dataset_nii):
    (
        label_file_path,
        data_dir,
        series_list,
        missing_labels,
        missing_samples,
        missing_series,
        exclusions,
    ) = randomized_dataset_nii
    dset = data.VolumeDataset(
        data_dir,
        label_file_path,
        series_list,
        exclusions=exclusions,
        transform=None,
        file_format=None,
        mask_series={},
    )
    for entry in missing_labels:
        assert entry not in list(pd.read_csv(label_file_path).iloc[:, 0])
        assert entry not in dset.groups["base"]
        if (
            entry not in missing_samples
            and entry not in missing_series
            and entry not in exclusions
        ):
            assert entry in dset.groups["no_label"]

    for entry in missing_samples:
        assert entry not in dset.groups["base"]
        if entry not in missing_labels:
            assert entry in dset.groups["_no_data"]

    for entry in missing_series:
        assert entry not in dset.groups["base"]
        if entry not in missing_samples and entry not in missing_labels:
            assert entry in dset.groups["_incomplete_data"]

    for entry in exclusions:
        assert entry not in dset.groups["base"]
        if (
            entry not in missing_samples
            and entry not in missing_labels
            and entry not in missing_series
        ):
            assert entry in dset.groups["_excluded"]

    for entry in list(pd.read_csv(label_file_path).iloc[:, 0]):
        if (
            (entry not in missing_samples)
            and (entry not in missing_labels)
            and (entry not in missing_series)
            and (entry not in exclusions)
        ):
            assert entry in dset.groups["base"]

