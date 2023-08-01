import pandas as pd
import os
from scipy.special import softmax
from sklearn.metrics import recall_score, accuracy_score, fbeta_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import ray
import nrad_torchlib.core as ntl
import matplotlib.pyplot as plt
import numpy as np
from ray import tune
from sklearn.metrics import plot_roc_curve

inf_base_dir='/path/projects/'


projects = ntl.Project(inf_base_dir)

projects.setup_inference(
    "predictions_project_name",
    "project_name",
    dataset_root_dir="/path/data/test_dataset/",
    label_file_path="/path/labels.csv" # the csv file has two columns, name of exams and class labels ((0,1) or (1,0))
)

projects.run_experiment("predictions_project_name") # predictions will be saved as a csv file in the folder '/path/projects/predictions_project_name/history'