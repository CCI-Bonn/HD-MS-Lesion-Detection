"""                                                                                                            
:AUTHOR: Hagen Meredig                                                                                         
:ORGANIZATION: Department of Neuroradiology, Heidelberg Univeristy Hospital                                    
:CONTACT: Hagen.Meredig@med.uni-heidelberg.de                                                                  
:SINCE: August 18, 2021                                                                            
"""

import os
import csv
import json
import shutil
import ray
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv
from datetime import datetime
from copy import deepcopy
from copy import copy
from collections import OrderedDict
from ray import tune
from . import datasets
from . import transforms
from . import trainers
from . import models
from . import optimizers
from . import schedulers
from . import adapters
from . import lossfunctions
from . import testrunners
from . import loaders
from . import reporters
from . import tools
from . import optim


def _get_saved_settings(base_dir, settings_file_name):

    settings_file_name = settings_file_name

    if os.path.isfile(os.path.join(base_dir, settings_file_name)):
        s = None
        with open(
            os.path.join(base_dir, settings_file_name),
            "r",
        ) as f:
            s = json.load(f)
        return s
    else:
        raise Exception("No settings found at {}".format(base_dir))


def _save_settings(base_dir, settings_file_name, settings):

    if os.path.isfile(os.path.join(base_dir, settings_file_name)):
        os.remove(os.path.join(base_dir, settings_file_name))

    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    with open(os.path.join(base_dir, settings_file_name), "w") as f:
        json.dump(settings, f, indent=4)


def _check_for_settings_file(base_dir, settings_file_name):
    if os.path.isfile(os.path.join(base_dir, settings_file_name)):
        raise Exception(
            "Settings already present at {}. Use override=True or load_settings=True depending on what you want to do".format(
                base_dir
            )
        )


class Preprocessing(object):

    default_settings = {}
    default_settings["dataset"] = ("", [], {})
    default_settings["dataset_root_dir"] = ""
    default_settings["label_file_path"] = None
    default_settings["channel_list"] = []
    default_settings["exclusions"] = []
    default_settings["file_format"] = None
    default_settings["mask_series"] = {}
    default_settings["loader"] = ("", [], {})
    default_settings["num_workers"] = 0
    default_settings["batch_size"] = 1
    default_settings["img_transforms"] = []
    default_settings["base_dir"] = ""
    default_settings["load_settings"] = False
    default_settings["verbose"] = False
    default_settings["override"] = False

    def __init__(self, *input_args, **input_kwargs):

        self.input_kwargs = deepcopy(self.__class__.default_settings)
        for key in input_kwargs.keys():
            self.input_kwargs[key] = deepcopy(input_kwargs[key])

        if self.input_kwargs["load_settings"]:
            self.input_kwargs = _get_saved_settings(
                self.input_kwargs["base_dir"], "preprocessing_settings.json"
            )
        else:
            if self.input_kwargs["override"]:
                self.input_kwargs["override"] = False
                _save_settings(
                    self.input_kwargs["base_dir"],
                    "preprocessing_settings.json",
                    self.input_kwargs,
                )
            else:
                _check_for_settings_file(
                    self.input_kwargs["base_dir"], "preprocessing_settings.json"
                )
                _save_settings(
                    self.input_kwargs["base_dir"],
                    "preprocessing_settings.json",
                    self.input_kwargs,
                )

        # create and compose transforms
        trfs = []
        for transform_classname, args, kwargs in self.input_kwargs["img_transforms"]:
            trfs.append(getattr(transforms, transform_classname)(*args, **kwargs))
        self.img_transforms = tv.transforms.Compose(trfs)

        # create dataset
        dataset_classname, args, kwargs = self.input_kwargs["dataset"]
        self.dataset = getattr(datasets, dataset_classname)(
            *args,
            transform=self.img_transforms,
            root_dir=self.input_kwargs["dataset_root_dir"],
            labels_file=self.input_kwargs["label_file_path"],
            channel_list=self.input_kwargs["channel_list"],
            exclusions=self.input_kwargs["exclusions"],
            file_format=self.input_kwargs["file_format"],
            mask_series=self.input_kwargs["mask_series"],
            **kwargs,
        )

        # create loader
        loader_classname, args, kwargs = self.input_kwargs["loader"]
        self.loader = getattr(loaders, loader_classname)(
            self.dataset,
            *args,
            num_workers=self.input_kwargs["num_workers"],
            batch_size=self.input_kwargs["batch_size"],
            shuffle=False,
            **kwargs,
        )

        self.base_dir = self.input_kwargs["base_dir"]
        self.load_settings = self.input_kwargs["load_settings"]
        self.verbose = self.input_kwargs["verbose"]

        if self.loader.dataset.labels_file == None:
            self.loader.dataset.sample_from("no_label")

    def run(self, force: bool = False):

        if (
            os.path.isfile(os.path.join(self.base_dir, ".preprocessing_did_run"))
            and not force
        ):
            if self.verbose:
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Run preprocessing blocked, as it has already been run in this directory. \
                    Use run(force=True) to override.",
                )
            return

        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Running preprocessing...",
        )
        # run preprocessing
        counter = 1
        for _, batch in enumerate(self.loader, 0):
            for ib in range(len(batch["_idents"])):
                ident = batch["_idents"][ib]
                if self.verbose:
                    print(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        ident,
                        "({} of {})".format(counter, len(self.loader.dataset)),
                    )
                # iterate over series and save
                i = 0
                for series in self.loader.dataset.channel_list:
                    if not os.path.isdir(os.path.join(self.base_dir, ident)):
                        os.makedirs(os.path.join(self.base_dir, ident))
                    data = batch["data"][ib][i].numpy()
                    filepath = os.path.join(self.base_dir, ident, series + ".npy")
                    np.save(filepath, data)
                    i += 1
                # iterate over maks_series and save
                i = 0
                for series in self.loader.dataset.mask_series.keys():
                    if not os.path.isdir(os.path.join(self.base_dir, ident)):
                        os.makedirs(os.path.join(self.base_dir, ident))
                    data = batch[series][ib].numpy()
                    filepath = os.path.join(
                        self.base_dir,
                        ident,
                        self.loader.dataset.mask_series[series] + ".npy",
                    )
                    np.save(filepath, data)
                    i += 1
                counter += 1

        with open(os.path.join(self.base_dir, ".preprocessing_did_run"), "w") as f:
            f.write(".")

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Done preprocessing.")

    def print_settings(self):

        for key in self.input_kwargs:
            print(key + ":", self.input_kwargs[key])


class SimpleTrainingRun(object):

    default_settings = {}
    default_settings["base_dir"] = ""
    default_settings["dataset"] = ("", [], {})
    default_settings["dataset_root_dir"] = ""
    default_settings["label_file_path"] = ""
    default_settings["channel_list"] = []
    default_settings["exclusions"] = []
    default_settings["file_format"] = None
    default_settings["mask_series"] = {}
    default_settings["loader"] = ("", [], {})
    default_settings["num_workers"] = 0
    default_settings["batch_size"] = 1
    default_settings["shuffle"] = True
    default_settings["augmentations"] = []
    default_settings["eval_augmentations"] = []
    default_settings["adapter"] = ("", [], {})
    default_settings["report_adapter"] = ("", [], {})
    default_settings["model"] = ("", [], {})
    default_settings["lossfunction"] = ("", [], {})
    default_settings["optimizer"] = ("", [], {})
    default_settings["scheduler"] = ("", [], {})
    default_settings["reporter"] = ("", [], {})
    default_settings["target_measure"] = "acc"
    default_settings["metric_eval_mode"] = "max"
    default_settings["checkpoint_tool"] = ("", [], {})
    default_settings["trainer"] = ("", [], {})
    default_settings["split_fraction"] = 0.8
    default_settings["seed"] = None
    default_settings["epochs"] = 1
    default_settings["verbose"] = False
    default_settings["load_settings"] = False
    default_settings["override"] = False

    def __init__(self, *input_args, **input_kwargs):

        self.input_kwargs = deepcopy(self.__class__.default_settings)
        for key in input_kwargs.keys():
            self.input_kwargs[key] = deepcopy(input_kwargs[key])

        if self.input_kwargs["load_settings"]:
            self.input_kwargs = _get_saved_settings(
                self.input_kwargs["base_dir"], "simple_training_settings.json"
            )
        else:
            if self.input_kwargs["override"]:
                self.input_kwargs["override"] = False
                _save_settings(
                    self.input_kwargs["base_dir"],
                    "simple_training_settings.json",
                    self.input_kwargs,
                )
            else:
                _check_for_settings_file(
                    self.input_kwargs["base_dir"], "simple_training_settings.json"
                )
                _save_settings(
                    self.input_kwargs["base_dir"],
                    "simple_training_settings.json",
                    self.input_kwargs,
                )

        self.name = self.input_kwargs["name"]
        self.base_dir = self.input_kwargs["base_dir"]

        # inject kwargs for dataset
        self.dataset = deepcopy(self.input_kwargs["dataset"])
        self.dataset[2]["root_dir"] = self.input_kwargs["dataset_root_dir"]
        self.dataset[2]["labels_file"] = self.input_kwargs["label_file_path"]
        self.dataset[2]["channel_list"] = self.input_kwargs["channel_list"]
        self.dataset[2]["exclusions"] = self.input_kwargs["exclusions"]
        self.dataset[2]["file_format"] = self.input_kwargs["file_format"]
        self.dataset[2]["mask_series"] = self.input_kwargs["mask_series"]
        if self.input_kwargs["split_fraction"] == None:
            self.dataset[2]["do_split"] = False
        else:
            self.dataset[2]["do_split"] = True
        self.dataset[2]["split_fraction"] = self.input_kwargs["split_fraction"]
        self.dataset[2]["seed"] = self.input_kwargs["seed"]

        # inject kwargs for loader
        self.loader = deepcopy(self.input_kwargs["loader"])
        self.loader[2]["num_workers"] = self.input_kwargs["num_workers"]
        self.loader[2]["batch_size"] = self.input_kwargs["batch_size"]
        self.loader[2]["shuffle"] = self.input_kwargs["shuffle"]

        self.augmentations = deepcopy(self.input_kwargs["augmentations"])
        self.eval_augmentations = deepcopy(self.input_kwargs["eval_augmentations"])
        self.adapter = deepcopy(self.input_kwargs["adapter"])
        self.report_adapter = deepcopy(self.input_kwargs["report_adapter"])
        self.model = deepcopy(self.input_kwargs["model"])
        self.lossfunction = deepcopy(self.input_kwargs["lossfunction"])
        self.optimizer = deepcopy(self.input_kwargs["optimizer"])
        self.scheduler = deepcopy(self.input_kwargs["scheduler"])

        # inject kwargs for reporter
        self.reporter_dir = os.path.join(self.base_dir, "history")
        self.reporter = deepcopy(self.input_kwargs["reporter"])
        self.reporter[2]["out_dir"] = self.reporter_dir

        # create main reporter
        reporter_classname, args, kwargs = self.reporter
        self.report = getattr(reporters, reporter_classname)(*args, **kwargs)

        # copy parpams
        self.checkpoint_tool = deepcopy(self.input_kwargs["checkpoint_tool"])
        self.trainer = deepcopy(self.input_kwargs["trainer"])
        self.target_measure = self.input_kwargs["target_measure"]
        self.metric_eval_mode = self.input_kwargs["metric_eval_mode"]
        self.split_fraction = self.input_kwargs["split_fraction"]
        self.seed = self.input_kwargs["seed"]
        self.epochs = self.input_kwargs["epochs"]
        self.verbose = self.input_kwargs["verbose"]
        self.load_settings = self.input_kwargs["load_settings"]
        self.override = self.input_kwargs["override"]

        self.trainers = []

        trainer_classname, args, kwargs = self.trainer
        self.trainers.append(
            getattr(trainers, trainer_classname)(
                *args,
                name=self.name,
                reporter_dir=self.reporter_dir,
                checkpoint_dir=os.path.join(self.base_dir, "checkpoints"),
                epochs=self.epochs,
                augmentations=self.augmentations,
                eval_augmentations=self.eval_augmentations,
                dataset=self.dataset,
                loader=self.loader,
                adapter=self.adapter,
                report_adapter=self.report_adapter,
                model=self.model,
                lossfunction=self.lossfunction,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                reporter=self.reporter,
                target_measure=self.target_measure,
                checkpoint_tool=self.checkpoint_tool,
                train_group="train",
                val_group="val",
                verbose=self.verbose,
                **kwargs,
            )
        )

        # eventually load tuning results
        self.analyses = []
        try:
            for trainer in self.trainers:
                local_dir = os.path.join(self.base_dir, "ray_tune_" + trainer.name)
                self.analyses.append(tune.Analysis(local_dir))
        except ValueError:
            pass

    def run(self, force: bool = False):

        if os.path.isfile(os.path.join(self.base_dir, ".st_did_run")) and not force:
            if self.verbose:
                raise Warning(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    + "\n"
                    + "Training run blocked, as it did run in this directory already. Use run(force=True) to override."
                )
            return

        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Starting simple training in {} ...".format(self.base_dir),
        )

        # execute training
        for trainer in self.trainers:
            trainer.run()

        with open(os.path.join(self.base_dir, ".st_did_run"), "w") as f:
            f.write(".")

        self.report.reload()

        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Done simple training in {}.".format(self.base_dir),
        )

    def optimize(
        self,
        hyper_parameters,
        resources_per_trial: dict = {"cpu": 1, "gpu": 0},
        grace_period: int = 1,
        reduction_factor: int = 2,
        num_samples: int = 20,
        force: bool = False,
    ):

        if os.path.isfile(os.path.join(self.base_dir, ".st_did_run")) and not force:
            if self.verbose:
                raise Warning(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    + "\n"
                    + "Optimized simple training run blocked, as it did run in this directory already. Use run(force=True) to override."
                )
            return

        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Starting optimized simple training in {} ...".format(self.base_dir),
        )

        optim.optimize(
            self,
            hyper_parameters,
            resources_per_trial=resources_per_trial,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
            num_samples=num_samples,
        )

        with open(os.path.join(self.base_dir, ".st_did_run"), "w") as f:
            f.write(".")

        self.report.reload()

        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Done running optimized simple training in {}.".format(self.base_dir),
        )

    def get_best_config(self, scope_select: str = "last"):

        ret = []

        for analysis in self.analyses:

            if self.metric_eval_mode == "max":
                logdir = analysis.dataframe()["logdir"][
                    analysis.dataframe()[self.target_measure][
                        analysis.dataframe()["training_iteration"] == self.epochs
                    ].idxmax()
                ]
            else:
                if self.metric_eval_mode == "min:":
                    logdir = analysis.dataframe()["logdir"][
                        analysis.dataframe()[self.target_measure][
                            analysis.dataframe()["training_iteration"] == self.epochs
                        ].idxmin()
                    ]
                else:
                    raise NameError("Unknown metric_eval_mode", self.metric_eval_mode)

            ret.append(analysis.get_all_configs()[logdir])

        return ret

    def print_settings(self):

        for key in self.input_kwargs:
            print(key + ":", self.input_kwargs[key])


class CrossvalTrainingRun(object):
    default_settings = {}
    default_settings["name"] = ""
    default_settings["base_dir"] = ""
    default_settings["dataset"] = ("", [], {})
    default_settings["dataset_root_dir"] = ""
    default_settings["label_file_path"] = ""
    default_settings["channel_list"] = []
    default_settings["exclusions"] = []
    default_settings["file_format"] = None
    default_settings["mask_series"] = {}
    default_settings["loader"] = ("", [], {})
    default_settings["num_workers"] = 0
    default_settings["batch_size"] = 1
    default_settings["shuffle"] = True
    default_settings["augmentations"] = []
    default_settings["eval_augmentations"] = []
    default_settings["adapter"] = ("", [], {})
    default_settings["report_adapter"] = ("", [], {})
    default_settings["model"] = ("", [], {})
    default_settings["lossfunction"] = ("", [], {})
    default_settings["optimizer"] = ("", [], {})
    default_settings["scheduler"] = ("", [], {})
    default_settings["reporter"] = ("", [], {})
    default_settings["target_measure"] = "acc"
    default_settings["metric_eval_mode"] = "max"
    default_settings["checkpoint_tool"] = ("", [], {})
    default_settings["trainer"] = ("", [], {})
    default_settings["n_folds"] = 5
    default_settings["seed"] = None
    default_settings["epochs"] = 1
    default_settings["num_gpus"] = 1.0
    default_settings["autostop"] = False
    default_settings["autostop_mode"] = "max"
    default_settings["autostop_measure"] = "acc"
    default_settings["convergence_window"] = 5
    default_settings["convergence_value"] = 0.95
    default_settings["plateau_window"] = 20
    default_settings["plateau_limit"] = 0.0
    default_settings["dashboard_host"] = "0.0.0.0"
    default_settings["dashboard_port"] = 8265
    default_settings["verbose"] = False
    default_settings["load_settings"] = False
    default_settings["override"] = False

    def __init__(self, *input_args, **input_kwargs):

        self.input_kwargs = deepcopy(self.__class__.default_settings)
        for key in input_kwargs.keys():
            self.input_kwargs[key] = deepcopy(input_kwargs[key])

        if self.input_kwargs["load_settings"]:
            self.input_kwargs = _get_saved_settings(
                self.input_kwargs["base_dir"], "cv_training_settings.json"
            )
        else:
            if self.input_kwargs["override"]:
                self.input_kwargs["override"] = False
                _save_settings(
                    self.input_kwargs["base_dir"],
                    "cv_training_settings.json",
                    self.input_kwargs,
                )
            else:
                _check_for_settings_file(
                    self.input_kwargs["base_dir"], "cv_training_settings.json"
                )
                _save_settings(
                    self.input_kwargs["base_dir"],
                    "cv_training_settings.json",
                    self.input_kwargs,
                )

        self.name = self.input_kwargs["name"]
        self.base_dir = self.input_kwargs["base_dir"]

        # inject kwargs for dataset
        self.dataset = deepcopy(self.input_kwargs["dataset"])
        self.dataset[2]["root_dir"] = self.input_kwargs["dataset_root_dir"]
        self.dataset[2]["labels_file"] = self.input_kwargs["label_file_path"]
        self.dataset[2]["channel_list"] = self.input_kwargs["channel_list"]
        self.dataset[2]["exclusions"] = self.input_kwargs["exclusions"]
        self.dataset[2]["file_format"] = self.input_kwargs["file_format"]
        self.dataset[2]["mask_series"] = self.input_kwargs["mask_series"]
        self.dataset[2]["do_crossval"] = True
        self.dataset[2]["folds"] = self.input_kwargs["n_folds"]
        self.dataset[2]["seed"] = self.input_kwargs["seed"]

        # inject kwargs for loader
        self.loader = deepcopy(self.input_kwargs["loader"])
        self.loader[2]["num_workers"] = self.input_kwargs["num_workers"]
        self.loader[2]["batch_size"] = self.input_kwargs["batch_size"]
        self.loader[2]["shuffle"] = self.input_kwargs["shuffle"]

        self.augmentations = deepcopy(self.input_kwargs["augmentations"])
        self.eval_augmentations = deepcopy(self.input_kwargs["eval_augmentations"])
        self.adapter = deepcopy(self.input_kwargs["adapter"])
        self.report_adapter = deepcopy(self.input_kwargs["report_adapter"])
        self.model = deepcopy(self.input_kwargs["model"])
        self.lossfunction = deepcopy(self.input_kwargs["lossfunction"])
        self.optimizer = deepcopy(self.input_kwargs["optimizer"])
        self.scheduler = deepcopy(self.input_kwargs["scheduler"])

        # inject kwargs for reporter
        self.reporter_dir = os.path.join(self.base_dir, "history")
        self.reporter = deepcopy(self.input_kwargs["reporter"])
        self.reporter[2]["out_dir"] = self.reporter_dir

        # create main reporter
        reporter_classname, args, kwargs = self.reporter
        self.report = getattr(reporters, reporter_classname)(*args, **kwargs)

        # copy parpams
        self.checkpoint_tool = deepcopy(self.input_kwargs["checkpoint_tool"])
        self.trainer = deepcopy(self.input_kwargs["trainer"])
        self.target_measure = self.input_kwargs["target_measure"]
        self.metric_eval_mode = self.input_kwargs["metric_eval_mode"]
        self.n_folds = self.input_kwargs["n_folds"]
        self.seed = self.input_kwargs["seed"]
        self.epochs = self.input_kwargs["epochs"]
        self.num_gpus = self.input_kwargs["num_gpus"]
        self.autostop = self.input_kwargs["autostop"]
        self.autostop_mode = self.input_kwargs["autostop_mode"]
        self.autostop_measure = self.input_kwargs["autostop_measure"]
        self.convergence_window = self.input_kwargs["convergence_window"]
        self.convergence_value = self.input_kwargs["convergence_value"]
        self.plateau_window = self.input_kwargs["plateau_window"]
        self.plateau_limit = self.input_kwargs["plateau_limit"]
        self.dashboard_host = self.input_kwargs["dashboard_host"]
        self.dashboard_port = self.input_kwargs["dashboard_port"]
        self.verbose = self.input_kwargs["verbose"]
        self.load_settings = self.input_kwargs["load_settings"]
        self.override = self.input_kwargs["override"]

        self.trainers = []

        for i in range(self.n_folds):

            trainer_classname, args, kwargs = self.trainer
            self.trainers.append(
                getattr(trainers, trainer_classname)(
                    *args,
                    name=self.name + "_fold" + str(i),
                    reporter_dir=self.reporter_dir,
                    checkpoint_dir=os.path.join(
                        self.base_dir, "checkpoints_fold" + str(i)
                    ),
                    epochs=self.epochs,
                    augmentations=self.augmentations,
                    eval_augmentations=self.eval_augmentations,
                    dataset=self.dataset,
                    loader=self.loader,
                    adapter=self.adapter,
                    report_adapter=self.report_adapter,
                    model=self.model,
                    lossfunction=self.lossfunction,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    reporter=self.reporter,
                    target_measure=self.target_measure,
                    checkpoint_tool=self.checkpoint_tool,
                    train_group="train" + str(i),
                    val_group="val" + str(i),
                    autostop=self.autostop,
                    autostop_mode=self.autostop_mode,
                    autostop_measure=self.autostop_measure,
                    convergence_window=self.convergence_window,
                    convergence_value=self.convergence_value,
                    plateau_window=self.plateau_window,
                    plateau_limit=self.plateau_limit,
                    verbose=self.verbose,
                    **kwargs,
                )
            )

        # eventually load tuning results
        self.analyses = []
        try:
            for trainer in self.trainers:
                local_dir = os.path.join(self.base_dir, "ray_tune_" + trainer.name)
                self.analyses.append(tune.Analysis(local_dir))
        except ValueError:
            pass

    def run(self, force: bool = False):

        self.start(force=force)
        self.finish()

    def start(self, force: bool = False):

        if os.path.isfile(os.path.join(self.base_dir, ".cv_did_run")) and not force:
            if self.verbose:
                raise Warning(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    + "\n"
                    + "Cross-validation training run blocked, as it did run in this directory already. Use run(force=True) to override."
                )
            return

        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Starting cross-validation training in {} ...".format(self.base_dir),
        )

        if not ray.is_initialized():
            ray.init(
                dashboard_host=self.dashboard_host, dashboard_port=self.dashboard_port
            )

        @ray.remote(num_gpus=self.num_gpus)
        def trainer_run(tr, **kwargs):
            tr.run()

        # execute training
        self._runs = []
        for trainer in self.trainers:
            self._runs.append(trainer_run.remote(trainer))

    def finish(self):

        ray.get(self._runs)

        with open(os.path.join(self.base_dir, ".cv_did_run"), "w") as f:
            f.write(".")

        self.ensemble_val()

        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Done cross-validation training in {}.".format(self.base_dir),
        )

    def optimize(
        self,
        hyper_parameters,
        resources_per_trial: dict = {"cpu": 1, "gpu": 0},
        grace_period: int = 1,
        reduction_factor: int = 2,
        num_samples: int = 20,
        force: bool = False,
    ):

        if os.path.isfile(os.path.join(self.base_dir, ".cv_did_run")) and not force:
            if self.verbose:
                raise Warning(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    + "\n"
                    + "Optimized cross-validation training run blocked, as it did run in this directory already. Use run(force=True) to override."
                )
            return

        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Starting optimized cross-validation training in {} ...".format(
                self.base_dir
            ),
        )

        if not ray.is_initialized():
            ray.init(
                dashboard_host=self.dashboard_host, dashboard_port=self.dashboard_port
            )

        optim.optimize(
            self,
            hyper_parameters,
            resources_per_trial=resources_per_trial,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
            num_samples=num_samples,
        )

        with open(os.path.join(self.base_dir, ".cv_did_run"), "w") as f:
            f.write(".")

        self.ensemble_val()

        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Done running optimized cross-validation training in {}.".format(
                self.base_dir
            ),
        )

    def get_best_config(self, scope_select: str = "last"):

        ret = []

        for analysis in self.analyses:

            if self.metric_eval_mode == "max":
                logdir = analysis.dataframe()["logdir"][
                    analysis.dataframe()[self.target_measure][
                        analysis.dataframe()["training_iteration"] == self.epochs
                    ].idxmax()
                ]
            else:
                if self.metric_eval_mode == "min:":
                    logdir = analysis.dataframe()["logdir"][
                        analysis.dataframe()[self.target_measure][
                            analysis.dataframe()["training_iteration"] == self.epochs
                        ].idxmin()
                    ]
                else:
                    raise NameError("Unknown metric_eval_mode", self.metric_eval_mode)

            ret.append(analysis.get_all_configs()[logdir])

        return ret

    def _sanity_check(self):

        dataset_classname, args, kwargs = self.dataset
        ds = getattr(datasets, dataset_classname)(*args, **kwargs)

        assert "base" in ds.groups.keys()

        for fold in range(self.n_folds):
            for epoch in range(self.epochs):

                assert "train" + str(fold) in ds.groups.keys()
                assert "val" + str(fold) in ds.groups.keys()

                for ident in ds.groups["train" + str(fold)]:
                    assert ident in self.report.get_epoch_data(
                        phase=self.name + "_fold" + str(fold) + "_train",
                        epoch=epoch,
                    )["idents"]
                for ident in ds.groups["val" + str(fold)]:
                    assert ident in self.report.get_epoch_data(
                        phase=self.name + "_fold" + str(fold) + "_val",
                        epoch=epoch,
                    )["idents"]

                # check report integrity
                for ident in self.report.get_epoch_data(
                    phase=self.name + "_fold" + str(fold) + "_val",
                    epoch=epoch,
                )["idents"]:
                    assert (
                        ident
                        in self.report.get_epoch_data(
                            phase=self.name + "_cv_last", epoch=0
                        )["idents"]
                    ), ident
                    assert (
                        ident
                        in self.report.get_epoch_data(
                            phase=self.name + "_cv_best", epoch=0
                        )["idents"]
                    ), ident
                    assert not ident in self.report.get_epoch_data(
                        phase=self.name + "_fold" + str(fold) + "_train",
                        epoch=epoch,
                    )["idents"]
                    for fold2 in range(self.n_folds):
                        for epoch2 in range(self.epochs):
                            if fold2 != fold:
                                assert not ident in self.report.get_epoch_data(
                                    phase=self.name + "_fold" + str(fold2) + "_val",
                                    epoch=epoch2,
                                )["idents"], ident
                    for epoch2 in range(self.epochs):
                        if epoch2 != epoch:
                            assert ident in self.report.get_epoch_data(
                                phase=self.name + "_fold" + str(fold) + "_val",
                                epoch=epoch2,
                            )["idents"], ident
                            assert not ident in self.report.get_epoch_data(
                                phase=self.name + "_fold" + str(fold) + "_train",
                                epoch=epoch2,
                            )["idents"], ident

    def print_settings(self):

        for key in self.input_kwargs:
            print(key + ":", self.input_kwargs[key])

    def ensemble_val(self):

        self.report.reload()

        best_epochs = []
        last_epochs = []
        for fold in range(self.n_folds):
            best_epochs.append(
                self.report.get_epoch_data(
                    phase=self.name + "_fold" + str(fold) + "_val",
                    epoch=self.report.best_epoch(
                        phase=self.name + "_fold" + str(fold) + "_val"
                    ),
                )
            )
            last_epochs.append(
                self.report.get_epoch_data(
                    phase=self.name + "_fold" + str(fold) + "_val",
                    epoch=self.report.last_epoch(
                        phase=self.name + "_fold" + str(fold) + "_val"
                    ),
                )
            )
        self.report.ensemble(best_epochs, self.name + "_cv_best")
        self.report.ensemble(last_epochs, self.name + "_cv_last")

    def setup_inference(self, **settings):

        settings_ = deepcopy(Inference.default_settings)
        for key in settings.keys():
            settings_[key] = settings[key]

        settings_["name"] = self.name + "_test"
        settings_["base_dir"] = os.path.join(self.base_dir)
        settings_["checkpoint_tool"] = self.checkpoint_tool
        settings_["reporter"] = self.reporter

        for fold in range(self.n_folds):
            settings_["model_dirs"]["fold" + str(fold)] = os.path.join(
                self.base_dir, "checkpoints_fold" + str(fold)
            )

        self.inference = Inference(**settings_)

    def run_inference(self, **settings):

        self.setup_inference(**settings)
        self.inference.run()


class Inference(object):
    default_settings = {}
    default_settings["name"] = ""
    default_settings["base_dir"] = ""
    default_settings["model_dirs"] = {}
    default_settings["epochs"] = 1
    default_settings["testrunner"] = ("", [], {})
    default_settings["augmentations"] = []
    default_settings["dataset"] = ("", [], {})
    default_settings["dataset_root_dir"] = ""
    default_settings["label_file_path"] = ""
    default_settings["channel_list"] = []
    default_settings["exclusions"] = []
    default_settings["file_format"] = None
    default_settings["mask_series"] = {}
    default_settings["loader"] = ("", [], {})
    default_settings["num_workers"] = 0
    default_settings["batch_size"] = 1
    default_settings["shuffle"] = True
    default_settings["adapter"] = ("", [], {})
    default_settings["report_adapter"] = ("", [], {})
    default_settings["checkpoint_tool"] = ("", [], {})
    default_settings["lossfunction"] = ("", [], {})
    default_settings["reporter"] = ("", [], {})
    default_settings["checkpoint_options"] = ["best", "last"]
    default_settings["verbose"] = False
    default_settings["load_settings"] = False
    default_settings["override"] = False

    def __init__(self, *input_args, **input_kwargs):

        self.input_kwargs = deepcopy(self.__class__.default_settings)
        for key in input_kwargs.keys():
            self.input_kwargs[key] = deepcopy(input_kwargs[key])

        if self.input_kwargs["load_settings"]:
            self.input_kwargs = _get_saved_settings(
                self.input_kwargs["base_dir"], "inference_settings.json"
            )
        else:
            if self.input_kwargs["override"]:
                self.input_kwargs["override"] = False
                _save_settings(
                    self.input_kwargs["base_dir"],
                    "inference_settings.json",
                    self.input_kwargs,
                )
            else:
                _check_for_settings_file(
                    self.input_kwargs["base_dir"], "inference_settings.json"
                )
                _save_settings(
                    self.input_kwargs["base_dir"],
                    "inference_settings.json",
                    self.input_kwargs,
                )

        self.name = self.input_kwargs["name"]
        self.base_dir = self.input_kwargs["base_dir"]
        self.model_dirs = self.input_kwargs["model_dirs"]
        self.epochs = self.input_kwargs["epochs"]
        self.testrunner = self.input_kwargs["testrunner"]

        # inject kwargs for dataset
        self.dataset = deepcopy(self.input_kwargs["dataset"])
        self.dataset[2]["root_dir"] = self.input_kwargs["dataset_root_dir"]
        self.dataset[2]["labels_file"] = self.input_kwargs["label_file_path"]
        self.dataset[2]["channel_list"] = self.input_kwargs["channel_list"]
        self.dataset[2]["exclusions"] = self.input_kwargs["exclusions"]
        self.dataset[2]["file_format"] = self.input_kwargs["file_format"]
        self.dataset[2]["mask_series"] = self.input_kwargs["mask_series"]

        # inject kwargs for loader
        self.loader = deepcopy(self.input_kwargs["loader"])
        self.loader[2]["num_workers"] = self.input_kwargs["num_workers"]
        self.loader[2]["batch_size"] = self.input_kwargs["batch_size"]
        self.loader[2]["shuffle"] = self.input_kwargs["shuffle"]

        self.augmentations = deepcopy(self.input_kwargs["augmentations"])
        self.adapter = deepcopy(self.input_kwargs["adapter"])
        self.report_adapter = deepcopy(self.input_kwargs["report_adapter"])
        self.checkpoint_tool = deepcopy(self.input_kwargs["checkpoint_tool"])
        self.lossfunction = deepcopy(self.input_kwargs["lossfunction"])

        # inject kwargs for reporter
        self.reporter_dir = os.path.join(self.base_dir, "history")
        self.reporter = deepcopy(self.input_kwargs["reporter"])
        self.reporter[2]["out_dir"] = self.reporter_dir

        # create main reporter
        reporter_classname, args, kwargs = self.reporter
        self.report = getattr(reporters, reporter_classname)(*args, **kwargs)

        self.checkpoint_options = self.input_kwargs["checkpoint_options"]
        self.verbose = self.input_kwargs["verbose"]
        self.load_settings = self.input_kwargs["load_settings"]
        self.override = self.input_kwargs["override"]

        self.testrunners = {}

        for model_name in self.model_dirs.keys():

            self.testrunners[model_name] = {}

            for checkpoint_option in self.checkpoint_options:

                # create testrunner
                testrunner_classname, args, kwargs = self.testrunner
                self.testrunners[model_name][checkpoint_option] = getattr(
                    testrunners, testrunner_classname
                )(
                    name=self.name + "_" + model_name + "_" + checkpoint_option,
                    reporter_dir=self.reporter_dir,
                    checkpoint_dir=self.model_dirs[model_name],
                    epochs=self.epochs,
                    augmentations=self.augmentations,
                    dataset=self.dataset,
                    loader=self.loader,
                    adapter=self.adapter,
                    report_adapter=self.report_adapter,
                    lossfunction=self.lossfunction,
                    checkpoint_tool=self.checkpoint_tool,
                    reporter=self.reporter,
                    verbose=self.verbose,
                    checkpoint_option=checkpoint_option,
                    *args,
                    **kwargs,
                )

    def run(self, force: bool = False):

        if (
            os.path.isfile(os.path.join(self.base_dir, ".inference_did_run"))
            and not force
        ):
            if self.verbose:
                raise Warning(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    + "\n"
                    + "Inference run blocked, as it did run in this directory already. Use run(force=True) to override."
                )
            return

        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Running inference of {} model directories with {} checkpoint_options in {}...".format(
                len(self.model_dirs.keys()), len(self.checkpoint_options), self.base_dir
            ),
        )

        # execute testing
        for model_name in self.testrunners.keys():
            for checkpoint_option in self.testrunners[model_name].keys():
                self.testrunners[model_name][checkpoint_option].run()

        with open(os.path.join(self.base_dir, ".inference_did_run"), "w") as f:
            f.write(".")

        self.ensemble()

        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Done running inference of {} model directories with {} checkpoint_options in {}.".format(
                len(self.model_dirs.keys()), len(self.checkpoint_options), self.base_dir
            ),
        )

    def ensemble(self):

        self.report.reload()

        for model_name in self.model_dirs.keys():
            for checkpoint_option in self.checkpoint_options:
                epoch_list = []
                for epoch in range(self.epochs):
                    epoch_list.append(
                        self.report.get_epoch_data(
                            phase=self.name
                            + "_"
                            + model_name
                            + "_"
                            + checkpoint_option,
                            epoch=epoch,
                        )
                    )
                self.report.ensemble(
                    epoch_list,
                    self.name + "_" + model_name + "_" + checkpoint_option + "_all_eps",
                )

        for checkpoint_option in self.checkpoint_options:
            data_list = []
            for model_name in self.model_dirs.keys():
                data_list.append(
                    self.report.get_epoch_data(
                        phase=self.name
                        + "_"
                        + model_name
                        + "_"
                        + checkpoint_option
                        + "_all_eps",
                        epoch=0,
                    )
                )
            self.report.ensemble(
                data_list, self.name + "_" + "ensembled_" + checkpoint_option
            )

    def print_settings(self):

        for key in self.input_kwargs:
            print(key + ":", self.input_kwargs[key])


class Project(object):
    def __init__(self, base_dir: str = None):

        self.base_dir = base_dir
        self.experiments = OrderedDict()
        if not os.path.isdir(self.base_dir):
            os.makedirs(self.base_dir)
        self.load_from_base_dir()

    def __getattribute__(self, name):

        if name in object.__getattribute__(self, "experiments").keys():
            return object.__getattribute__(self, "experiments")[name]
        else:
            return object.__getattribute__(self, name)

    def load_from_base_dir(self):

        for folder in os.listdir(self.base_dir):
            if os.path.isdir(os.path.join(self.base_dir, folder)):
                if os.path.isfile(
                    os.path.join(self.base_dir, folder, "simple_training_settings.json")
                ):
                    obj = SimpleTrainingRun
                elif os.path.isfile(
                    os.path.join(self.base_dir, folder, "cv_training_settings.json")
                ):
                    obj = CrossvalTrainingRun
                elif os.path.isfile(
                    os.path.join(self.base_dir, folder, "preprocessing_settings.json")
                ):
                    obj = Preprocessing
                elif os.path.isfile(
                    os.path.join(self.base_dir, folder, "inference_settings.json")
                ):
                    obj = Inference
                else:
                    print(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        f"No settings file found at {os.path.join(self.base_dir, folder)}. Skipping...",
                    )
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f"{obj} settings file found at {os.path.join(self.base_dir, folder)}. Initializing...",
                )
                try:
                    self.experiments[folder] = obj(
                        base_dir=os.path.join(self.base_dir, folder), load_settings=True
                    )
                except Exception as e:
                    print(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        f"Exception while loading experiment at {os.path.join(self.base_dir, folder)}:",
                        str(e),
                    )

                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f"Done.",
                )

    def rebase(self):

        self.experiments = OrderedDict()

        self.load_from_base_dir()

        for folder in os.listdir(self.base_dir):
            if os.path.isdir(os.path.join(self.base_dir, folder)):
                if folder in self.experiments.keys():

                    s = deepcopy(self.experiments[folder].input_kwargs)

                    if not s["name"] == folder:
                        s["name"] = folder
                    if not s["base_dir"] == os.path.join(self.base_dir, folder):
                        s["base_dir"] = os.path.join(self.base_dir, folder)

                    s["override"] = True
                    self.experiments[folder] = self.experiments[folder].__class__(**s)

    def add_experiment(
        self,
        experiment_type: str,
        experiment_name: str,
        copy_settings: str = None,
        **additional_settings,
    ):

        if experiment_name in self.experiments.keys():
            raise NameError("Name already exists.")

        if experiment_type not in (
            "simple_training",
            "s",
            "cross-validation",
            "crosval",
            "cv_training",
            "cv",
            "inference",
            "i",
            "optimization",
            "o",
        ):
            raise TypeError(f"Unknown experiment type: {experiment_type}")

        if experiment_type in ("simple_training", "s"):
            class_object = SimpleTrainingRun
        elif experiment_type in ("cross-validation", "crosval", "cv_training", "cv"):
            class_object = CrossvalTrainingRun
        elif experiment_type in ("inference", "i"):
            class_object = Inference

        s = deepcopy(class_object.default_settings)

        if copy_settings:
            for key in self.experiments[copy_settings].input_kwargs.keys():
                if key in s.keys():
                    s[key] = deepcopy(self.experiments[copy_settings].input_kwargs[key])

        for key in additional_settings.keys():
            s[key] = deepcopy(additional_settings[key])

        s["name"] = experiment_name
        s["base_dir"] = os.path.join(self.base_dir, experiment_name)

        self.experiments[experiment_name] = class_object(**s)

    def setup_inference(
        self,
        inference_name: str,
        base_exp_name: str = None,
        **additional_settings,
    ):

        if inference_name in self.experiments.keys():
            raise NameError("Name already exists.")

        if not base_exp_name in self.experiments.keys():
            raise NameError("base_exp_name doesn't exist.")

        s = deepcopy(Inference.default_settings)

        for key in self.experiments[base_exp_name].input_kwargs.keys():
            if key in s.keys():
                if key == "augmentations":
                    continue
                elif key == "epochs":
                    s[key] = 1
                else:
                    s[key] = deepcopy(self.experiments[base_exp_name].input_kwargs[key])
            if (
                key == "trainer"
                and self.experiments[base_exp_name].input_kwargs[key][0]
                == "ClassificationModelTrainer"
            ):
                s["testrunner"] = ("ClassificationModelTestrunner", [], {})
            elif (
                key == "trainer"
                and self.experiments[base_exp_name].input_kwargs[key][0]
                == "BasicTrainer"
            ):
                s["testrunner"] = ("BasicTestrunner", [], {})
            elif key == "eval_augmentations":
                s["augmentations"] = deepcopy(
                    self.experiments[base_exp_name].input_kwargs[key]
                )

        for key in additional_settings.keys():
            s[key] = deepcopy(additional_settings[key])

        s["name"] = inference_name
        s["base_dir"] = os.path.join(self.base_dir, inference_name)
        s["model_dirs"] = {}

        if isinstance(self.experiments[base_exp_name], CrossvalTrainingRun):
            for fold in range(self.experiments[base_exp_name].n_folds):
                s["model_dirs"]["fold" + str(fold)] = os.path.join(
                    self.experiments[base_exp_name].base_dir,
                    "checkpoints_fold" + str(fold),
                )
        elif isinstance(self.experiments[base_exp_name], SimpleTrainingRun):
            s["model_dirs"]["fold0"] = os.path.join(
                self.experiments[base_exp_name].base_dir,
                "checkpoints",
            )
        else:
            raise Exception(
                f"Unable to setup inference for this type of experiment {self.experiments[base_exp_name]}"
            )
        self.experiments[inference_name] = Inference(**s)

    def set_settings(self, experiment_name: str, **settings):

        if not experiment_name in self.experiments.keys():
            raise NameError(f"Experiment {experiment_name} doesn't exist.")

        s = deepcopy(self.experiments[experiment_name].input_kwargs)

        for key in settings.keys():
            s[key] = deepcopy(settings[key])

        s["override"] = True
        self.experiments[experiment_name] = self.experiments[experiment_name].__class__(
            **s
        )

    def reset_experiment(self, experiment_name: str):

        if not experiment_name in self.experiments.keys():
            raise NameError(f"Experiment {experiment_name} doesn't exist.")

        s = deepcopy(self.experiments[experiment_name].input_kwargs)

        shutil.rmtree(os.path.join(self.base_dir, experiment_name))

        s["override"] = True
        self.experiments[experiment_name] = self.experiments[experiment_name].__class__(
            **s
        )

    def delete_experiment(self, experiment_name: str):

        if not experiment_name in self.experiments.keys():
            raise NameError(f"Experiment {experiment_name} doesn't exist.")

        shutil.rmtree(os.path.join(self.base_dir, experiment_name))
        del self.experiments[experiment_name]

    def rename_experiment(self, experiment_name: str, new_name: str):

        if not experiment_name in self.experiments.keys():
            raise NameError(f"Experiment {experiment_name} doesn't exist.")

        if new_name in self.experiments.keys():
            raise NameError("Name already exists.")

        os.rename(
            os.path.join(self.base_dir, experiment_name),
            os.path.join(self.base_dir, new_name),
        )

        self.rebase()

    def run_experiment(self, name: str, raise_exceptions: bool = False):

        if not name in self.experiments.keys():
            raise NameError("Name doesn't exist.")

        try:
            self.experiments[name].run()
        except Exception as e:
            if raise_exceptions:
                raise e
            else:
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f"Exception in experiment run {name}:",
                    str(e),
                )

    def run_all(self):

        for key in self.experiments.keys():
            self.run_experiment(key)

    def show_summary(
        self,
        experiment_names: list = "all",
        experiment_name_contains: str = "",
        measure: str = "acc",
        class_select: any = None,
    ):

        if experiment_names == "all":
            experiment_names = sorted([e for e in self.experiments.keys()])

        if not experiment_name_contains == "":
            experiment_names = sorted(
                [e for e in self.experiments.keys() if experiment_name_contains in e]
            )

        msr = copy(measure)
        fn_name = "get_measure"
        cl_sel = False
        if "per_class" in msr:
            msr = msr.replace("per_class_", "")
            fn_name = "get_per_class_measure"
            cl_sel = True

        values = {}

        for experiment in experiment_names:
            if not experiment in self.experiments.keys():
                raise NameError(f"Experimen '{experiment}' doesn't exist.")

            values[experiment] = {}

            if isinstance(self.experiments[experiment], SimpleTrainingRun):
                rep = self.experiments[experiment].report

                try:
                    val_best = rep.__getattribute__(fn_name)(
                        msr,
                        phase=experiment + "_val",
                        epoch=rep.best_epoch(phase=experiment + "_val"),
                    )
                    if cl_sel:
                        val_best = val_best[str(class_select)]
                except KeyError:
                    val_best = 0.0

                try:
                    val_last = rep.__getattribute__(fn_name)(
                        msr,
                        phase=experiment + "_val",
                        epoch=rep.last_epoch(phase=experiment + "_val"),
                    )
                    if cl_sel:
                        val_last = val_last[str(class_select)]
                except KeyError:
                    val_last = 0.0

                values[experiment]["best"] = round(val_best, 3)
                values[experiment]["last"] = round(val_last, 3)

            elif isinstance(self.experiments[experiment], CrossvalTrainingRun):
                rep = self.experiments[experiment].report

                try:
                    val_best = rep.__getattribute__(fn_name)(
                        msr, phase=experiment + "_cv_best"
                    )
                    if cl_sel:
                        val_best = val_best[str(class_select)]
                except KeyError:
                    val_best = 0.0

                try:
                    val_last = rep.__getattribute__(fn_name)(
                        msr, phase=experiment + "_cv_last"
                    )
                    if cl_sel:
                        val_last = val_last[str(class_select)]
                except KeyError:
                    val_last = 0.0

                values[experiment]["best"] = round(val_best, 3)
                values[experiment]["last"] = round(val_last, 3)

            elif isinstance(self.experiments[experiment], Inference):
                rep = self.experiments[experiment].report
                try:
                    val_best = rep.__getattribute__(fn_name)(
                        msr, phase=experiment + "_ensembled_best"
                    )
                    if cl_sel:
                        val_best = val_best[str(class_select)]
                except KeyError:
                    val_best = 0.0
                try:
                    val_last = rep.__getattribute__(fn_name)(
                        msr, phase=experiment + "_ensembled_last"
                    )
                    if cl_sel:
                        val_last = val_last[str(class_select)]
                except KeyError:
                    val_last = 0.0

                values[experiment]["best"] = round(val_best, 3)
                values[experiment]["last"] = round(val_last, 3)

        x = np.arange(len(experiment_names))  # the label locations
        width = 0.3  # the width of the bars

        fig, ax = plt.subplots(figsize=(16, 10))

        rects_best = ax.bar(
            x - width / 2,
            [values[e]["best"] for e in experiment_names],
            width,
            label="best",
        )
        rects_last = ax.bar(
            x + width / 2,
            [values[e]["last"] for e in experiment_names],
            width,
            label="last",
        )

        # Add some text for labels, title and custom x-axis tick labels, etc.
        label_meaning = ""
        if cl_sel:
            label_meaning = f" for class {class_select}"
        ax.set_ylabel(f"{measure}{label_meaning}")
        ax.set_title("Summary")
        ax.set_xticks(x)
        ax.set_xticklabels([e for e in experiment_names])
        ax.legend(loc="lower right")

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    "{}".format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        autolabel(rects_best)
        autolabel(rects_last)

        plt.show()
