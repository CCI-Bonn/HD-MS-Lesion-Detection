import os
import torch
import shutil
import random
import hashlib
import json
import numpy as np
from datetime import datetime
from copy import deepcopy
from . import trainers
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from typing import Dict, Any


# from https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def inject_parameters(obj, params):
    def inject_param(obj_, param_name, param_value, _level=0):
        try:
            obj_[param_name] = param_value
        except Exception as exception:
            if isinstance(exception, TypeError):
                pass
            elif isinstance(exception, KeyError):
                for entry in obj_.keys():
                    inject_param(obj_[entry], param_name, param_value, _level + 1)
        print("found", param_name)

    for param_name in params.keys():
        print("Recursive search for", param_name)
        inject_param(obj, param_name, params[param_name])


def optimize(
    exp_instance,
    hyper_parameters,
    resources_per_trial: dict = {"cpu": 1, "gpu": 0},
    grace_period: int = 1,
    reduction_factor: int = 2,
    num_samples: int = 20,
):

    print(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Starting hyperparameter tuning in {} ...".format(exp_instance.base_dir),
    )

    for trainer in exp_instance.trainers:

        checkpoint_folder_name = (
            str(trainer.checkpoint_dir)
            .replace(exp_instance.base_dir, "")
            .replace("/", "")
            .replace("\\", "")
        )

        def train_eval(params):
            # maybe add autostop here too?

            inject_parameters(trainer, params)

            reporter_dir = os.path.join(tune.get_trial_dir(), "history")
            checkpoint_dir = os.path.join(
                tune.get_trial_dir(),
                checkpoint_folder_name,
            )

            trainer.reporter_dir = reporter_dir
            trainer.checkpoint_dir = checkpoint_dir

            for epoch in range(trainer.epochs):

                go_on = trainer.run_epoch()

                report_kwargs = {}
                report_kwargs["loss"] = trainer.live_objects["reporter"].get_measure(
                    "loss", phase=trainer.name + "_val", epoch=epoch
                )
                report_kwargs[trainer.target_measure] = trainer.live_objects[
                    "reporter"
                ].get_measure(
                    trainer.target_measure, phase=trainer.name + "_val", epoch=epoch
                )
                tune.report(**report_kwargs)

                if not go_on:
                    break

        if hyper_parameters == None:
            raise RuntimeError("hyper_parameters argument required")

        scheduler = ASHAScheduler(
            max_t=exp_instance.epochs,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
        )

        local_dir = os.path.join(exp_instance.base_dir, "ray_tune_" + trainer.name)

        trainer.opt_result = tune.run(
            tune.with_parameters(train_eval),
            resources_per_trial=resources_per_trial,
            config=hyper_parameters,
            metric=exp_instance.target_measure,
            mode=exp_instance.metric_eval_mode,
            num_samples=num_samples,
            scheduler=scheduler,
            local_dir=local_dir,
        )

        analysis = tune.Analysis(local_dir)

        if exp_instance.metric_eval_mode == "max":
            logdir = analysis.dataframe()["logdir"][
                analysis.dataframe()[exp_instance.target_measure][
                    analysis.dataframe()["training_iteration"] == exp_instance.epochs
                ].idxmax()
            ]
        else:
            if exp_instance.metric_eval_mode == "min":
                logdir = analysis.dataframe()["logdir"][
                    analysis.dataframe()[exp_instance.target_measure][
                        analysis.dataframe()["training_iteration"]
                        == exp_instance.epochs
                    ].idxmin()
                ]
            else:
                raise NameError(
                    "Unknown metric_eval_mode", exp_instance.metric_eval_mode
                )

        shutil.copytree(
            os.path.join(logdir, checkpoint_folder_name),
            os.path.join(exp_instance.base_dir, checkpoint_folder_name),
        )

        os.makedirs(os.path.join(exp_instance.base_dir, "history"), exist_ok=True)

        for fn in os.listdir(os.path.join(logdir, "history")):
            if os.path.isfile(os.path.join(logdir, "history", fn)):
                shutil.copyfile(
                    os.path.join(logdir, "history", fn),
                    os.path.join(exp_instance.base_dir, "history", fn),
                )
            if os.path.isdir(os.path.join(logdir, "history", fn)):
                shutil.copytree(
                    os.path.join(logdir, "history", fn),
                    os.path.join(exp_instance.base_dir, "history", fn),
                )
