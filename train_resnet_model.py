import os
import ray
import nrad_torchlib.core as ntl
import numpy as np
from ray import tune
from sklearn.model_selection import train_test_split, KFold 
import glob
import torch
import nrad_torchlib
from collections import OrderedDict
import pickle
import sys
sys.path.append('.')
from utilities import get_split_deterministic

TRAINING_SET_BASEDIR = (
    "/path/data/combined_dataset/"          
)

PREPROCESSED_TRAINING_DATA_DIR = (
    "/path/data/combined_dataset/"
)
EXCLUSIONS = []
TRAINING_BASE_DIR = "/path/projects"
TRAINING_NAME = "resnet_model"

source_exams = os.listdir(TRAINING_SET_BASEDIR)
patients = set([os.path.split(exam)[-1].split('_')[0] for exam in source_exams if 'OAS' not in exam])
fold = 0
train,val = get_split_deterministic(patients,fold=fold, num_splits=5, random_state=12345)

train_exams=[exam for exam in os.listdir(TRAINING_SET_BASEDIR) if exam.split('_')[0] in train ]
val_exams=[exam for exam in os.listdir(TRAINING_SET_BASEDIR) if exam.split('_')[0] in val ]
oasis_exams = [exam for exam in os.listdir('/path/data/combined_dataset/') if 'OAS' in exam] # list of synthetic exams

train_exams = train_exams + oasis_exams # include CGAN generated synthetic FLAIR images for training


train_settings = {}
train_settings["name"] = TRAINING_NAME 
train_settings["base_dir"] = TRAINING_BASE_DIR
train_settings["dataset"] = ("VolumeDataset", [], {"get_split":True,"train_list":train_exams,"val_list":val_exams})
train_settings["dataset_root_dir"] = PREPROCESSED_TRAINING_DATA_DIR
train_settings[
    "label_file_path"
] = "./combined_labels_fold.csv"   # path to label csv file which has two columns, name of exams and class labels ((0,1) or (1,0))
train_settings["get_split"]=True
train_settings["channel_list"] = ["FLAIR_2_reg", "FLAIR_1"] # file names of input images
train_settings["exclusions"] = EXCLUSIONS
train_settings["file_format"] = "nii.gz"
train_settings["mask_series"] = {"mask":"brain_mask"}   # brain mask for center cropping 
train_settings["loader"] = ("GenericDataLoader", [], {})
train_settings["num_workers"] = 64
train_settings["batch_size"] = 8
train_settings["shuffle"] = True
train_settings["augmentations"] = [
    ("ChangeSpacingTransform",[(1,1,1)], {}),
    ("CenterCropTransform", [(180, 180, 180)], {"mask_name":"mask"}),
    (
        "SpatialTransform",
        [],
        {
            "translation": ((0.25, 0.0), (0.25, 0.0),(0.25, 0.0)),
            "rotation": ((0.25 * np.pi, 0.0), (0.25 * np.pi, 0.0),(0.25 * np.pi, 0.0)),
            "p_tra": 0.2,
            "p_rot": 0.2,
            "p_shear": 0.0,
        },
    ),
    ("StdMeanTransform", [], {})
]
train_settings["eval_augmentations"] = [
    ("ChangeSpacingTransform",[(1,1,1)], {}),
    ("CenterCropTransform", [(180, 180, 180)], {"mask_name":"mask"}),
    ("StdMeanTransform", [], {})
]
train_settings["model"] = ("ResNet18_3D", [], {"num_channels": 2, "num_classes": 2})
train_settings["adapter"] = ("MultiClassDataAdapter", ["data", "labels"], {})
train_settings["report_adapter"] = ("MultiClassReportAdapter", [], {})
train_settings["lossfunction"] = ("CrossEntropyLoss", [], {})
train_settings["optimizer"] = ("Adam", [], {"lr": 0.000025, "weight_decay": 0.0001})
train_settings["scheduler"] = (
    "ReduceLROnPlateau",
    [],
    {"verbose": True, "mode": "max", "factor":0.9, "patience": 5},
)
train_settings["reporter"] = (
    "MulticlassClassificationReporter",
    [],
    {},
)
train_settings["target_measure"] = "acc"
train_settings["metric_eval_mode"] = "max"
train_settings["checkpoint_tool"] = ("CheckpointTool", [], {"cop_str": ">"})
train_settings["trainer"] = ("ClassificationModelTrainer", [], {"use_class_weights":True,"load_pretrained":False,'model_path':None})
train_settings["split_fraction"] = None
train_settings["seed"] = 12345
train_settings["epochs"] = 100
train_settings["devices"] = ["cuda:0","cuda:1"]
train_settings["verbose"] = True

tr = ntl.Project(base_dir=TRAINING_BASE_DIR) 
tr.add_experiment('s', TRAINING_NAME, **train_settings)
tr.run_experiment(TRAINING_NAME)