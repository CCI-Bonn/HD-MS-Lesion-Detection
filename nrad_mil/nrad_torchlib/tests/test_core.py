import pytest
import os
import sys
import requests
import hashlib
import random
import shutil
import csv
import numpy as np

from nrad_torchlib import core
from nrad_torchlib.datasets import VolumeDataset


MNIST_BASE_DIR = "/srv/DATA/RAID/hagen/nrad_torchlib_test/mnist/"
N_TRAIN = 1000
N_TEST = 100


def sha256(fp):
    hash_md5 = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_test_data(dl_dir=MNIST_BASE_DIR):

    # load from google storage, because that's the one I know and we get numpy directly
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    file_hash = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
    file_path = os.path.join(dl_dir, "mnist.npz")

    if not os.path.isdir(dl_dir):
        os.makedirs(dl_dir)

    if not os.path.isfile(file_path):
        print("downloading...")
        sess = requests.Session()
        req = sess.get(url)
        with open(file_path, "wb") as f:
            f.write(req.content)

    assert sha256(file_path) == file_hash

    with np.load(file_path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    return x_train, y_train, x_test, y_test


def save_dataset(n_train, n_test, save_dir_base=MNIST_BASE_DIR):

    random.seed()

    x_train, y_train, x_test, y_test = get_test_data()

    traindata_full = [(x_train[i], y_train[i]) for i in range(len(x_train))]
    random.shuffle(traindata_full)
    testdata_full = [(x_test[i], y_test[i]) for i in range(len(x_test))]
    random.shuffle(testdata_full)

    save_dir_train = os.path.join(save_dir_base, "train")
    save_dir_test = os.path.join(save_dir_base, "test")

    if os.path.isdir(save_dir_train):
        shutil.rmtree(save_dir_train)
    if os.path.isdir(save_dir_test):
        shutil.rmtree(save_dir_test)

    os.makedirs(save_dir_train)
    os.makedirs(save_dir_test)

    train_labels_str = "ID,Label\n"
    i = 0
    while i < n_train:
        label = np.zeros(10, np.float)
        label[traindata_full[i][1]] = 1.0
        label = str(label).strip("[]")
        data = traindata_full[i][0]
        train_labels_str += "{},{}\n".format(i, label)
        os.makedirs(os.path.join(save_dir_train, str(i)))
        fp = os.path.join(save_dir_train, str(i), "chn0.npy")
        np.save(fp, data)
        i += 1
    with open(os.path.join(save_dir_base, "train_labels.csv"), "w") as f:
        f.write(train_labels_str)

    test_labels_str = "ID,Label\n"
    i = 0
    while i < n_test:
        label = np.zeros(10, np.float)
        label[testdata_full[i][1]] = 1.0
        label = str(label).strip("[]")
        data = testdata_full[i][0]
        test_labels_str += "{},{}\n".format(i, label)
        os.makedirs(os.path.join(save_dir_test, str(i)))
        fp = os.path.join(save_dir_test, str(i), "chn0.npy")
        np.save(fp, data)
        i += 1
    with open(os.path.join(save_dir_base, "test_labels.csv"), "w") as f:
        f.write(test_labels_str)


def test():

    if os.path.isdir(os.path.join(MNIST_BASE_DIR, "simple_training")):
        shutil.rmtree(os.path.join(MNIST_BASE_DIR, "simple_training"))
    if os.path.isdir(os.path.join(MNIST_BASE_DIR, "crossval_training")):
        shutil.rmtree(os.path.join(MNIST_BASE_DIR, "crossval_training"))
    if os.path.isdir(os.path.join(MNIST_BASE_DIR, "simple_training_test")):
        shutil.rmtree(os.path.join(MNIST_BASE_DIR, "simple_training_test"))
    if os.path.isdir(os.path.join(MNIST_BASE_DIR, "crossval_training_test")):
        shutil.rmtree(os.path.join(MNIST_BASE_DIR, "crossval_training_test"))

    save_dataset(N_TRAIN, N_TEST, save_dir_base=MNIST_BASE_DIR)

    prep_dir = os.path.join(MNIST_BASE_DIR, "data_prep")

    prep = core.Preprocessing(
        dataset=("VolumeDataset", [], {}),
        dataset_root_dir=os.path.join(MNIST_BASE_DIR, "train"),
        label_file_path=os.path.join(MNIST_BASE_DIR, "train_labels.csv"),
        channel_list=["chn0"],
        exclusions=None,
        file_format="npy",
        mask_series={},
        loader=("GenericDataLoader", [], {}),
        num_workers=10,
        batch_size=16,
        shuffle=True,
        img_transforms=[("ToFloatTransform", [], {})],
        out_dir=prep_dir,
    )
    prep.run()

    simple = core.SimpleTrainingRun(
        base_dir=os.path.join(MNIST_BASE_DIR, "simple_training"),
        dataset=("VolumeDataset", [], {}),
        dataset_root_dir=prep_dir,
        label_file_path=os.path.join(MNIST_BASE_DIR, "train_labels.csv"),
        channel_list=["chn0"],
        exclusions=None,
        file_format="npy",
        mask_series={},
        loader=("GenericDataLoader", [], {}),
        num_workers=10,
        batch_size=16,
        shuffle=True,
        trainer=("ClassificationModelTrainer", [], {},),
        augmentations=[("CenterCropTransform", [(32, 32)], {})],
        adapter=("CustomDataAdapter", ["data", "labels"], {}),
        model=("ExampleConvNet", [2, 4], {"test_kwarg": "hallo"}),
        lossfunction=("CrossEntropyLoss", [], {}),
        optimizer=("SGD", [], {"lr": 0.01}),
        scheduler=("ReduceLROnPlateau", [], {"verbose": True}),
        reporter=("MulticlassClassificationReporter", [], {}),
        target_measure="acc",
        checkpoint_tool=("CheckpointTool", [], {"cop_str": ">"}),
        split_fraction=0.8,
        seed=123456,
        epochs=1,
        force_base_dir=False,
        devices=["cuda:0", "cuda:1"],
        verbose=False,
    )
    simple.run()

    cross = core.CrossvalTrainingRun(
        base_dir=os.path.join(MNIST_BASE_DIR, "crossval_training"),
        dataset=("VolumeDataset", [], {}),
        dataset_root_dir=prep_dir,
        label_file_path=os.path.join(MNIST_BASE_DIR, "train_labels.csv"),
        channel_list=["chn0"],
        exclusions=None,
        file_format="npy",
        mask_series={},
        loader=("GenericDataLoader", [], {}),
        num_workers=10,
        batch_size=16,
        shuffle=True,
        trainer=("ClassificationModelTrainer", [], {},),
        augmentations=[("CenterCropTransform", [(32, 32)], {})],
        adapter=("CustomDataAdapter", ["data", "labels"], {}),
        model=("ExampleConvNet", [2, 4], {"test_kwarg": "hallo"}),
        lossfunction=("CrossEntropyLoss", [], {}),
        optimizer=("SGD", [], {"lr": 0.01}),
        scheduler=("ReduceLROnPlateau", [], {"verbose": True}),
        reporter=("MulticlassClassificationReporter", [], {}),
        target_measure="acc",
        checkpoint_tool=("CheckpointTool", [], {"cop_str": ">"}),
        n_folds=5,
        seed=123456,
        epochs=1,
        force_base_dir=False,
        devices=["cuda:0", "cuda:1"],
        verbose=False,
    )
    cross.run()

    t = core.Inference(
        out_dir=os.path.join(MNIST_BASE_DIR, "simple_training_test"),
        model_dir=os.path.join(MNIST_BASE_DIR, "simple_training", "checkpoints"),
        epochs=1,
        testrunner=("ClassificationModelTestrunner", [], {},),
        augmentations=[
            ("ToFloatTransform", [], {}),
            ("CenterCropTransform", [(32, 32)], {}),
        ],
        dataset=("VolumeDataset", [], {}),
        dataset_root_dir=os.path.join(MNIST_BASE_DIR, "test"),
        label_file_path=os.path.join(MNIST_BASE_DIR, "test_labels.csv"),
        channel_list=["chn0"],
        exclusions=None,
        file_format="npy",
        mask_series={},
        loader=("GenericDataLoader", [], {}),
        num_workers=10,
        batch_size=16,
        shuffle=False,
        adapter=("CustomDataAdapter", ["data", "labels"], {}),
        checkpoint_tool=("CheckpointTool", [], {}),
        lossfunction=("CrossEntropyLoss", [], {}),
        reporter=("MulticlassClassificationReporter", [], {}),
        testoptions=["best", "last"],
        devices=["cuda:0", "cuda:1"],
        verbose=False,
    )
    t.run()

    """tc = core.TestRun(
        base_dir=os.path.join(MNIST_BASE_DIR, "crossval_training"),
        out_dir=os.path.join(MNIST_BASE_DIR, "crossval_training_test"),
        testrunner=("ClassificationModelTestrunner", [], {},),
        epochs=1,
        augmentations=[
            ("ToFloatTransform", [], {}),
            ("CenterCropTransform", [(32, 32)], {}),
        ],
        dataset=(
            "VolumeDataset",
            [],
            {
                "root_dir": os.path.join(MNIST_BASE_DIR, "test"),
                "labels_file": os.path.join(MNIST_BASE_DIR, "test_labels.csv"),
                "channel_list": ["chn0"],
                "exclusions": None,
                "mask_series": {},
            },
        ),
        loader=(
            "GenericDataLoader",
            [],
            {"num_workers": 10, "batch_size": 16, "shuffle": True,},
        ),
        adapter=("CustomDataAdapter", ["data", "labels"], {}),
        checkpoint_tool=("CheckpointTool", [], {}),
        lossfunction=("CrossEntropyLoss", [], {}),
        reporter=("MulticlassClassificationReporter", [], {}),
        testoptions=["best", "last"],
        devices=["cpu"],
        verbose=False,
    )
    # tc.run()"""

    """ds = VolumeDataset(
        root_dir=os.path.join(MNIST_BASE_DIR, "train"),
        labels_file=os.path.join(MNIST_BASE_DIR, "train_labels.csv"),
        channel_list=["chn0"],
    )
    ds.crossval_split("base", cross.n_folds, cross.seed)
    for fold_folder in os.listdir(cross.base_dir):
        if os.path.isdir(os.path.join(cross.base_dir, fold_folder)) and (
            ("train_val" in fold_folder)
        ):
            fold = int(fold_folder.replace("train_val_fold", ""))
            cur_path = os.path.join(cross.base_dir, fold_folder, "history")
            csv_list_train = sorted(
                [f for f in os.listdir(cur_path) if f.endswith(".csv") and "train" in f]
            )
            for csv_train in csv_list_train:
                with open(os.path.join(cur_path, csv_train), "r") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        assert (
                            row["ID"] in ds.groups["train" + str(fold)]
                            and not row["ID"] in ds.groups["val" + str(fold)]
                        )

            csv_list_val = sorted(
                [f for f in os.listdir(cur_path) if f.endswith(".csv") and "val" in f]
            )
            for csv_val in csv_list_val:
                with open(os.path.join(cur_path, csv_val), "r") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        assert (
                            row["ID"] in ds.groups["val" + str(fold)]
                            and not row["ID"] in ds.groups["train" + str(fold)]
                        )"""


if __name__ == "__main__":
    test()
