"""                                                                                                            
:AUTHOR: Hagen Meredig                                                                                         
:ORGANIZATION: Department of Neuroradiology, Heidelberg Univeristy Hospital                                    
:CONTACT: Hagen.Meredig@med.uni-heidelberg.de                                                                  
:SINCE: August 18, 2021                                                                            
""" 
import os
import torch
from . import models
from . import transforms
from . import datasets
from . import loaders
from . import adapters
from . import reporters
from . import lossfunctions
from . import tools
import torchvision as tv
import numpy as np
from datetime import datetime
from itertools import starmap


class BasicTestrunner(object):
    def __init__(
        self,
        name: str = "",
        reporter_dir: str = "",
        checkpoint_dir: str = "",
        epochs: int = 1,
        augmentations: any = None,
        dataset: any = None,
        loader: any = None,
        adapter: any = None,
        report_adapter: any = None,
        lossfunction: any = None,
        checkpoint_tool: any = None,
        reporter: any = None,
        verbose: bool = False,
        checkpoint_option: str = "best",
    ):

        self.name = name
        self.reporter_dir = reporter_dir
        self.checkpoint_dir = checkpoint_dir
        self.epochs = epochs
        self.augmentations = augmentations
        self.dataset = dataset
        self.loader = loader
        self.adapter = adapter
        self.report_adapter = report_adapter
        self.lossfunction = lossfunction
        self.checkpoint_tool = checkpoint_tool
        self.reporter = reporter
        self.verbose = verbose
        self.checkpoint_option = checkpoint_option

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

    def __repr__(self):

        ret = "ClassificationModelTestrunner\n"
        ret += f"name: {self.name}\n"
        ret += f"reporter_dir: {self.reporter_dir}\n"
        ret += f"checkpoint_dir: {self.checkpoint_dir}\n"
        ret += f"epochs: {self.epochs}\n"
        ret += f"augmentations: {self.augmentations}\n"
        ret += f"dataset: {self.dataset}\n"
        ret += f"loader: {self.loader}\n"
        ret += f"adapter: {self.adapter}\n"
        ret += f"report_adapter: {self.report_adapter}\n"
        ret += f"lossfunction: {self.lossfunction}\n"
        ret += f"checkpoint_tool: {self.checkpoint_tool}\n"
        ret += f"reporter: {self.reporter}\n"
        ret += f"verbose: {self.verbose}\n"
        ret += f"checkpoint_option: {self.checkpoint_option}\n"
        ret += f"device: {self.device}\n"

        return ret

    def test_epoch(self, i_epoch):

        with torch.no_grad():
            for data in iter(self.live_objects["loader"]):

                self.live_objects["model"].eval()

                # forward pass
                data_ = self.live_objects["adapter"](data)

                input = data_["input"]
                ground_truth = data_["ground_truth"]
                supplement = data_["supplement"]

                input = input.to(torch.device(self.device))
                if not ground_truth == None:
                    ground_truth = ground_truth.to(torch.device(self.device))

                output = self.live_objects["model"](input)
                loss = self.live_objects["lossfunction"](output, ground_truth)

                # write data
                report_data = self.live_objects["report_adapter"](
                    input=input,
                    ground_truth=ground_truth,
                    supplement=supplement,
                    output=output,
                    loss=loss,
                )

                self.live_objects["reporter"](
                    **report_data,
                    phase=self.name,
                    epoch=i_epoch,
                )

    def setup(self):

        self.live_objects = {}

        # create and compose augmentation transforms
        trfs = []
        for transform_classname, args, kwargs in self.augmentations:
            trfs.append(getattr(transforms, transform_classname)(*args, **kwargs))
        self.live_objects["augmentations"] = tv.transforms.Compose(trfs)

        # create dataset
        dataset_classname, args, kwargs = self.dataset
        self.live_objects["dataset"] = getattr(datasets, dataset_classname)(
            *args, transform=self.live_objects["augmentations"], **kwargs
        )

        # create loader
        loader_classname, args, kwargs = self.loader
        self.live_objects["loader"] = getattr(loaders, loader_classname)(
            self.live_objects["dataset"], *args, **kwargs
        )

        # create adatpter
        adapter_classname, args, kwargs = self.adapter
        self.live_objects["adapter"] = getattr(adapters, adapter_classname)(
            *args, **kwargs
        )

        # create report adatpter
        adapter_classname, args, kwargs = self.report_adapter
        self.live_objects["report_adapter"] = getattr(adapters, adapter_classname)(
            *args, **kwargs
        )

        # create reporter
        reporter_classname, args, kwargs = self.reporter
        kwargs["out_dir"] = self.reporter_dir
        self.live_objects["reporter"] = getattr(reporters, reporter_classname)(
            *args, **kwargs
        )

        # create lossfunction
        lossfunction_classname, args, kwargs = self.lossfunction
        self.live_objects["lossfunction"] = getattr(
            lossfunctions, lossfunction_classname
        )(*args, **kwargs)

        # create checkpoint tool
        checkpoint_tool_classname, args, kwargs = self.checkpoint_tool
        kwargs["base_dir"] = self.checkpoint_dir
        self.live_objects["checkpoint_tool"] = getattr(
            tools, checkpoint_tool_classname
        )(*args, **kwargs)

        # create model
        self.live_objects["model"] = self.live_objects["checkpoint_tool"].load(
            self.checkpoint_option
        )["model"]

        self.live_objects["model"] = self.live_objects["model"].to(
            torch.device(self.device)
        )
        if torch.cuda.device_count() > 1:
            self.live_objects["model"] = torch.nn.DataParallel(
                self.live_objects["model"]
            )
        self.live_objects["model"].eval()

    def run(self):

        self.setup()

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Starting inference run...")
        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Name: {}".format(self.name),
        )
        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Reporter directory: {}".format(self.live_objects["reporter"].out_dir),
        )
        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Checkpoint directory: {}".format(
                self.live_objects["checkpoint_tool"].base_dir
            ),
        )

        for ep in range(self.epochs):
            self.test_epoch(ep)
            self.live_objects["reporter"].save()

            if self.verbose:
                acc = self.live_objects["reporter"].get_measure(
                    "acc", phase=self.name, epoch=ep
                )
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Inference epoch {} acc {}".format(ep, acc),
                )

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Inference run done.")

        del self.live_objects


### depraceted ###
class ClassificationModelTestrunner(object):
    def __init__(
        self,
        name: str = "",
        reporter_dir: str = "",
        checkpoint_dir: str = "",
        epochs: int = 1,
        augmentations: any = None,
        dataset: any = None,
        loader: any = None,
        adapter: any = None,
        report_adapter: any = None,
        lossfunction: any = None,
        checkpoint_tool: any = None,
        reporter: any = None,
        verbose: bool = False,
        checkpoint_option: str = "best",
    ):

        self.name = name
        self.reporter_dir = reporter_dir
        self.checkpoint_dir = checkpoint_dir
        self.epochs = epochs
        self.augmentations = augmentations
        self.dataset = dataset
        self.loader = loader
        self.adapter = adapter
        self.report_adapter = report_adapter
        self.lossfunction = lossfunction
        self.checkpoint_tool = checkpoint_tool
        self.reporter = reporter
        self.verbose = verbose
        self.checkpoint_option = checkpoint_option

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

    def __repr__(self):

        ret = "ClassificationModelTestrunner\n"
        ret += f"name: {self.name}\n"
        ret += f"reporter_dir: {self.reporter_dir}\n"
        ret += f"checkpoint_dir: {self.checkpoint_dir}\n"
        ret += f"epochs: {self.epochs}\n"
        ret += f"augmentations: {self.augmentations}\n"
        ret += f"dataset: {self.dataset}\n"
        ret += f"loader: {self.loader}\n"
        ret += f"adapter: {self.adapter}\n"
        ret += f"report_adapter: {self.report_adapter}\n"
        ret += f"lossfunction: {self.lossfunction}\n"
        ret += f"checkpoint_tool: {self.checkpoint_tool}\n"
        ret += f"reporter: {self.reporter}\n"
        ret += f"verbose: {self.verbose}\n"
        ret += f"checkpoint_option: {self.checkpoint_option}\n"

        return ret

    def test_epoch(self, i_epoch):

        with torch.no_grad():
            for data in iter(self.live_objects["loader"]):

                self.live_objects["model"].eval()

                # forward pass
                inputs, ground_truth = self.live_objects["adapter"](data)

                inputs = inputs.to(torch.device(self.device))
                if not ground_truth == None:
                    ground_truth = ground_truth.to(torch.device(self.device))

                outputs = self.live_objects["model"](inputs)
                loss = self.live_objects["lossfunction"](outputs, ground_truth)

                # write data
                outputs, ground_truth, loss, inputs = self.live_objects[
                    "report_adapter"
                ](outputs, ground_truth, loss, inputs)

                self.live_objects["reporter"](
                    outputs,
                    ground_truth,
                    loss,
                    idents=data["_idents"],
                    phase=self.name,
                    epoch=i_epoch,
                    verbose=self.verbose,
                    batch_input=inputs,
                )

    def setup(self):

        self.live_objects = {}

        # create and compose augmentation transforms
        trfs = []
        for transform_classname, args, kwargs in self.augmentations:
            trfs.append(getattr(transforms, transform_classname)(*args, **kwargs))
        self.live_objects["augmentations"] = tv.transforms.Compose(trfs)

        # create dataset
        dataset_classname, args, kwargs = self.dataset
        self.live_objects["dataset"] = getattr(datasets, dataset_classname)(
            *args, transform=self.live_objects["augmentations"], **kwargs
        )

        # create loader
        loader_classname, args, kwargs = self.loader
        self.live_objects["loader"] = getattr(loaders, loader_classname)(
            self.live_objects["dataset"], *args, **kwargs
        )

        # create adatpter
        adapter_classname, args, kwargs = self.adapter
        self.live_objects["adapter"] = getattr(adapters, adapter_classname)(
            *args, **kwargs
        )

        # create report adatpter
        adapter_classname, args, kwargs = self.report_adapter
        self.live_objects["report_adapter"] = getattr(adapters, adapter_classname)(
            *args, **kwargs
        )

        # create reporter
        reporter_classname, args, kwargs = self.reporter
        kwargs["out_dir"] = self.reporter_dir
        self.live_objects["reporter"] = getattr(reporters, reporter_classname)(
            *args, **kwargs
        )

        # create lossfunction
        lossfunction_classname, args, kwargs = self.lossfunction
        self.live_objects["lossfunction"] = getattr(
            lossfunctions, lossfunction_classname
        )(*args, **kwargs)

        # create checkpoint tool
        checkpoint_tool_classname, args, kwargs = self.checkpoint_tool
        kwargs["base_dir"] = self.checkpoint_dir
        self.live_objects["checkpoint_tool"] = getattr(
            tools, checkpoint_tool_classname
        )(*args, **kwargs)

        # create model
        self.live_objects["model"] = self.live_objects["checkpoint_tool"].load(
            self.checkpoint_option
        )["model"]

        self.live_objects["model"] = self.live_objects["model"].to(
            torch.device(self.device)
        )
        if torch.cuda.device_count() > 1:
            self.live_objects["model"] = torch.nn.DataParallel(
                self.live_objects["model"]
            )
        self.live_objects["model"].eval()

    def run(self):

        self.setup()

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Starting inference run...")
        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Name: {}".format(self.name),
        )
        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Reporter directory: {}".format(self.live_objects["reporter"].out_dir),
        )
        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Checkpoint directory: {}".format(
                self.live_objects["checkpoint_tool"].base_dir
            ),
        )

        for ep in range(self.epochs):
            self.test_epoch(ep)
            self.live_objects["reporter"].save()

            if self.verbose:
                acc = self.live_objects["reporter"].get_measure(
                    "acc", phase=self.name, epoch=ep
                )
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Inference epoch {} acc {}".format(ep, acc),
                )

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Inference run done.")

        del self.live_objects
