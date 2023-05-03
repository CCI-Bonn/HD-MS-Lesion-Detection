"""                                                                                                            
:AUTHOR: Hagen Meredig                                                                                         
:ORGANIZATION: Department of Neuroradiology, Heidelberg Univeristy Hospital                                    
:CONTACT: Hagen.Meredig@med.uni-heidelberg.de                                                                  
:SINCE: August 18, 2021                                                                            
""" 
import os
import torch
import operator
import pickle
import numpy as np
from . import models
from . import optimizers
from . import core
from collections import OrderedDict
from itertools import starmap
import matplotlib.pyplot as plt


class CheckpointTool(object):
    def __init__(
        self,
        base_dir: str = "",
        cop_str: str = ">",
        model_init_params: tuple = ("", [], {}),
        optimizer_init_params: tuple = ("", [], {}),
    ):
        self.base_dir = base_dir
        self.cop_str = cop_str
        if cop_str == ">":
            self._comp_op = operator.gt
        if cop_str == ">=":
            self._comp_op = operator.ge
        if cop_str == "<":
            self._comp_op = operator.lt
        if cop_str == "<=":
            self._comp_op = operator.le

        self.model_params = model_init_params
        self.optimizer_params = optimizer_init_params

        best_epoch = None
        last_epoch = None
        if os.path.isdir(self.base_dir):
            for entry in os.listdir(self.base_dir):
                if "best" in entry:
                    best_epoch = int(entry.split("_")[2])
                if "last" in entry:
                    last_epoch = int(entry.split("_")[2])
        self.best_epoch = best_epoch
        self.last_epoch = last_epoch
        self.cur_best_val = None

    def save_init_params(self, params: tuple, name: str):
        fp = os.path.join(self.base_dir, "{}_initparams.pkl".format(name))
        if not os.path.isfile(fp):
            with open(
                fp,
                "wb",
            ) as f:
                pickle.dump(params, f)

    def save_state_dict(self, state_dict: dict, name: str, epoch: int, chkp_type: str):

        for fn in os.listdir(self.base_dir):
            if name in fn and chkp_type in fn and fn.endswith(".pt"):
                os.remove(os.path.join(self.base_dir, fn))
        torch.save(
            state_dict,
            os.path.join(
                self.base_dir,
                "{}_ep_{}_{}.pt".format(name, epoch, chkp_type),
            ),
        )

    def save(self, obj: any, name: str, epoch: int, chkp_type: str):
        if name == "model":
            params = self.model_params
        elif name == "optimizer":
            params = self.optimizer_params
        else:
            raise Exception("Unknown object name.")
        self.save_init_params(params, name)
        self.save_state_dict(obj.state_dict(), name, epoch, chkp_type)

    def load_init_params(self, name: str):

        ret = None
        for fn in os.listdir(self.base_dir):
            if fn.endswith("_initparams.pkl") and name in fn:
                with open(
                    os.path.join(self.base_dir, fn),
                    "rb",
                ) as f:
                    ret = pickle.load(f)
        return ret

    def load_state_dict(self, name: str, chkp_type: str):

        ret = None
        for fn in os.listdir(self.base_dir):
            if fn.startswith(name) and fn.endswith("{}.pt".format(chkp_type)):
                ret = torch.load(os.path.join(self.base_dir, fn))
        return ret

    def load(self, chkp_type: str):

        model = None
        model_sig = self.load_init_params("model")
        if not model_sig == None:
            model_classname, args, kwargs = model_sig
            model = getattr(models, model_classname)(*args, **kwargs)
            try:
                model.load_state_dict(self.load_state_dict("model", chkp_type))
            except:
                state_dict = self.load_state_dict("model", chkp_type)
                # if model was saved as data parallel create new OrderedDict that does not contain `module.`
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                model.load_state_dict(new_state_dict)

        optimizer = None
        optimizer_sig = self.load_init_params("optimizer")
        if not optimizer_sig == None:
            optimizer_classname, args, kwargs = optimizer_sig
            optimizer = getattr(optimizers, optimizer_classname)(
                model.parameters(), *args, **kwargs
            )
            optimizer.load_state_dict(self.load_state_dict("optimizer", chkp_type))

        return {"model": model, "optimizer": optimizer}

    def update(
        self,
        model: any = None,
        optimizer: any = None,
        epoch: int = 0,
        value: any = None,
    ):

        if not os.path.isdir(self.base_dir):
            os.makedirs(self.base_dir)

        save_best = False
        if self.cur_best_val == None or self._comp_op(value, self.cur_best_val):
            save_best = True
            self.cur_best_val = value
            self.best_epoch = epoch

        if model:
            self.save(model, "model", epoch, "last")
            if save_best:
                self.save(model, "model", epoch, "best")

        if optimizer:
            self.save(optimizer, "optimizer", epoch, "last")
            if save_best:
                self.save(optimizer, "optimizer", epoch, "best")

        self.last_epoch = epoch


class MILPredictor(object):
    def __init__(
        self,
        inference_instance=None,
        fold_names: list = ["fold0"],
        checkpoint_options: list = ["last"],
    ):
        self.inference_instance = inference_instance
        self.checkpoint_options = checkpoint_options
        self.fold_names = fold_names

    def predict(
        self, safe_path: str = None, show: bool = True, overlay_treshold: float = 0.05
    ):

        for model_name in self.inference_instance.testrunners.keys():
            if not model_name in self.fold_names:
                continue
            for option in self.inference_instance.testrunners[model_name].keys():

                if not option in self.checkpoint_options:
                    continue

                if show:
                    print("Using", option, "model of", model_name)

                runner = self.inference_instance.testrunners[model_name][option]
                runner.setup()
                model = runner.live_objects["model"]
                loader = runner.live_objects["loader"]

                for batch in iter(loader):
                    assert len(batch["data"]) == 1, "only defined for batch_size = 1"

                    len_bag = 0

                    for series_name in batch["_meta"].keys():
                        if not (series_name in batch.keys()):
                            if batch["_meta"][series_name]["len_tiles"] > len_bag:
                                len_bag = batch["_meta"][series_name]["len_tiles"]
                                positions = batch["_meta"][series_name]["tile_pos"][0]
                                orig_shape = batch["_meta"][series_name]["orig_shape"]

                    bag = torch.swapaxes(batch["data"][:, :, :len_bag], 1, 2)
                    positions = positions[:len_bag].detach().numpy()
                    label = torch.argmax(batch["labels"][0])
                    if torch.cuda.is_available():
                        bag = bag.to(torch.device("cuda"))
                    prediction = model(bag)
                    attention_weights = prediction[2][0].detach().cpu().numpy()
                    pred = float(prediction[0][0].detach().cpu().numpy())
                    bag = bag.squeeze(0).detach().cpu().numpy()

                    assert len(attention_weights) == len(bag)

                    if show:
                        print("\n\n\n\n\n")
                        print(batch["_idents"][0])
                        print("Label", float(label.item()))
                        print("Prediction", pred)
                        if round(pred) == label:
                            print("Correct")
                        else:
                            print("Wrong")

                    for channel in range(bag.shape[1]):

                        out_img = np.zeros(orig_shape[0].tolist())
                        out_attention = np.zeros(orig_shape[0].tolist() + [2])

                        for tile, position, weight in zip(
                            bag[:, channel], positions, attention_weights
                        ):
                            position = [int(i) for i in position]
                            end_pos = [
                                e for e in zip(position, out_img.shape, tile.shape)
                            ]
                            end_pos = [
                                i
                                for i in starmap(
                                    lambda p, o, t: p + t if (p + t) < o else o,
                                    end_pos,
                                )
                            ]
                            end_tile = [
                                e for e in zip(position, out_img.shape, tile.shape)
                            ]
                            end_tile = [
                                i
                                for i in starmap(
                                    lambda p, o, t: t if (p + t) < o else o - p,
                                    end_tile,
                                )
                            ]
                            out_indices = tuple(
                                starmap(
                                    lambda s, e: slice(s, e, 1),
                                    [e for e in zip(position, end_pos)],
                                )
                            )
                            tile_indices = tuple(
                                map(lambda e: slice(0, e, 1), end_tile)
                            )

                            out_img[out_indices] = tile[tile_indices]
                            out_attention[out_indices + (0,)] += weight
                            out_attention[out_indices + (1,)] += 1.0

                        slices = tuple(
                            [slice(0, e, 1) for e in out_attention.shape[:-1]]
                        )
                        out_attention[slices + (0,)] /= (
                            out_attention[slices + (1,)] + 1e-8
                        )
                        out_attention = out_attention[slices + (0,)]

                        if show:
                            if len(out_img.shape) == 3:  # 3D
                                slice_to_show = out_img.shape[2] // 2
                                img = out_img[:, :, slice_to_show]
                                attn = out_attention[:, :, slice_to_show]
                            else:
                                img = out_img
                                attn = out_attention

                            cma = plt.get_cmap("plasma")
                            attn[attn < 0.1][4] = 0.0
                            attn_color = cma(attn)

                            img -= img.min()
                            img /= img.max() + 1e-8
                            cmi = plt.get_cmap("gray")
                            img_color = cmi(img)

                            img_color[attn > overlay_treshold] = (
                                img_color[attn > overlay_treshold] * 0.7
                                + attn_color[attn > overlay_treshold] * 0.3
                            )

                            plt.imshow(img_color)
                            plt.show()

    def setup(self, fold_name: any = "fold0", option: str = "last"):

        if isinstance(self.inference_instance, core.Inference):
            assert fold_name in self.inference_instance.testrunners.keys()

            self.runner = self.inference_instance.testrunners[fold_name][option]
            self.runner.setup()
            self.model = self.runner.live_objects["model"]
            self.loader = self.runner.live_objects["loader"]
        elif isinstance(self.inference_instance, core.CrossvalTrainingRun):
            try:
                fold_name = int(fold_name)
            except ValueError:
                raise ValueError(
                    "For CrossvalTrainingRun fold_name is expected to be an int."
                )

            self.runner = self.inference_instance.trainers[fold_name]
            self.runner.setup()
            self.model = self.runner.live_objects["checkpoint_tool"].load(option)[
                "model"
            ]
            self.loader = self.runner.live_objects["loader"]

    def analyze(
        self,
        count: any = "all",
        n_channels: int = 4,
        chn1: int = 0,
        chn2: int = 3,
        cls: int = 0,
    ):

        high_attn_values = []
        class_summary_dicts = []
        for chn in range(n_channels):
            high_attn_values.append([])
            class_summary_dicts.append(
                {
                    0: {
                        "tilevalues_correct": [],
                        "attnweights_correct": [],
                        "tilevalues_wrong": [],
                        "attnweights_wrong": [],
                    },
                    1: {
                        "tilevalues_correct": [],
                        "attnweights_correct": [],
                        "tilevalues_wrong": [],
                        "attnweights_wrong": [],
                    },
                }
            )

        for model_name in self.inference_instance.testrunners.keys():
            if not model_name in self.fold_names:
                continue
            for option in self.inference_instance.testrunners[model_name].keys():

                if not option in self.checkpoint_options:
                    continue

                runner = self.inference_instance.testrunners[model_name][option]
                runner.setup()
                model = runner.live_objects["model"]
                loader = runner.live_objects["loader"]

                max_count = len(loader)
                if count != "all":
                    max_count = count

                counter = 0
                for batch in iter(loader):

                    if counter >= max_count:
                        break

                    assert len(batch["data"]) == 1, "only defined for batch_size = 1"

                    len_bag = 0

                    for series_name in batch["_meta"].keys():
                        if not (series_name in batch.keys()):
                            if batch["_meta"][series_name]["len_tiles"] > len_bag:
                                len_bag = batch["_meta"][series_name]["len_tiles"]
                                positions = batch["_meta"][series_name]["tile_pos"][0]
                                orig_shape = batch["_meta"][series_name]["orig_shape"]

                    bag = torch.swapaxes(batch["data"][:, :, :len_bag], 1, 2)
                    positions = positions[:len_bag].detach().numpy()
                    label = torch.argmax(batch["labels"][0])
                    if torch.cuda.is_available():
                        bag = bag.to(torch.device("cuda"))
                    prediction = model(bag)
                    attention_weights = prediction[2][0].detach().cpu().numpy()
                    pred = float(prediction[0][0].detach().cpu().numpy())
                    bag = bag.squeeze(0).detach().cpu().numpy()

                    pred_correct = round(pred) == label

                    assert len(attention_weights) == len(bag)

                    for channel in range(bag.shape[1]):

                        out_img = np.zeros(orig_shape[0].tolist())

                        attns = []
                        values = []

                        for tile, position, weight in zip(
                            bag[:, channel], positions, attention_weights
                        ):
                            position = [int(i) for i in position]
                            end_pos = [
                                e for e in zip(position, out_img.shape, tile.shape)
                            ]
                            end_pos = [
                                i
                                for i in starmap(
                                    lambda p, o, t: p + t if (p + t) < o else o,
                                    end_pos,
                                )
                            ]
                            end_tile = [
                                e for e in zip(position, out_img.shape, tile.shape)
                            ]
                            end_tile = [
                                i
                                for i in starmap(
                                    lambda p, o, t: t if (p + t) < o else o - p,
                                    end_tile,
                                )
                            ]
                            tile_indices = tuple(
                                map(lambda e: slice(0, e, 1), end_tile)
                            )

                            if pred_correct:
                                class_summary_dicts[channel][int(label)][
                                    "tilevalues_correct"
                                ].append(tile[tile_indices].mean())
                                class_summary_dicts[channel][int(label)][
                                    "attnweights_correct"
                                ].append(weight)

                                if int(label) == cls:
                                    attns.append(weight)
                                    values.append(tile[tile_indices].mean())
                            else:
                                class_summary_dicts[channel][int(label)][
                                    "tilevalues_wrong"
                                ].append(tile[tile_indices].mean())
                                class_summary_dicts[channel][int(label)][
                                    "attnweights_wrong"
                                ].append(weight)

                        if int(label) == cls and pred_correct:
                            arr = sorted([(i, j) for i, j in zip(attns, values)])
                            high_attn_values[channel].append(arr[-1][1])
                    counter += 1

        return class_summary_dicts, high_attn_values
