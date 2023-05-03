"""                                                                                                            
:AUTHOR: Hagen Meredig                                                                                         
:ORGANIZATION: Department of Neuroradiology, Heidelberg Univeristy Hospital                                    
:CONTACT: Hagen.Meredig@med.uni-heidelberg.de                                                                  
:SINCE: August 18, 2021 
:UPDATED: April 7, 2022 by Chandrakanth Jayachandran Preetha. Modified ClassificationModelTrainer class to enable use of pre-trained model
""" 

import torch
from . import models
from . import optimizers
from . import schedulers
from . import transforms
from . import datasets
from . import loaders
from . import adapters
from . import reporters
from . import lossfunctions
from . import tools
import torchvision as tv
from collections import OrderedDict
import numpy as np
from datetime import datetime


class BasicTrainer(object):
    def __init__(
        self,
        name: str = "",
        reporter_dir: str = "",
        checkpoint_dir: str = "",
        epochs: int = 1,
        augmentations: any = None,
        eval_augmentations: any = None,
        dataset: any = None,
        loader: any = None,
        adapter: any = None,
        report_adapter: any = None,
        model: any = None,
        lossfunction: any = None,
        optimizer: any = None,
        scheduler: any = None,
        reporter: any = None,
        target_measure: str = "acc",
        checkpoint_tool: any = None,
        train_group: str = "train",
        val_group: str = "val",
        use_class_weights: bool = False,
        autostop: bool = False,
        autostop_mode: str = "max",
        autostop_measure: str = "acc",
        convergence_window: int = 5,
        convergence_value: int = 0.95,
        plateau_window: int = 20,
        plateau_limit: float = 0.0,
        verbose: bool = False,
    ):

        assert autostop_mode in ("min", "max"), "Unknown autostop_mode: " + str(
            autostop_mode
        )

        self.name = name
        self.reporter_dir = reporter_dir
        self.checkpoint_dir = checkpoint_dir
        self.epochs = epochs
        self.augmentations = augmentations
        self.eval_augmentations = eval_augmentations
        self.dataset = dataset
        self.loader = loader
        self.adapter = adapter
        self.report_adapter = report_adapter
        self.model = model
        self.lossfunction = lossfunction
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.reporter = reporter
        self.target_measure = target_measure
        self.checkpoint_tool = checkpoint_tool
        self.train_group = train_group
        self.val_group = val_group
        self.use_class_weights = use_class_weights
        self.autostop = autostop
        self.autostop_mode = autostop_mode
        self.autostop_measure = autostop_measure
        self.convergence_window = convergence_window
        self.convergence_value = convergence_value
        self.plateau_window = plateau_window
        self.plateau_limit = plateau_limit
        self.verbose = verbose

        self.class_weights = None
        self.current_epoch = 0
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

    def train_epoch(self, i_epoch):

        self.live_objects["loader"].dataset.sample_from(self.train_group)
        self.live_objects["loader"].dataset.eval_mode = False

        loss = None

        for data in iter(self.live_objects["loader"]):

            self.live_objects["model"].train()

            data_ = self.live_objects["adapter"](data)

            input = data_["input"]
            ground_truth = data_["ground_truth"]
            supplement = data_["supplement"]

            input = input.to(torch.device(self.device))
            ground_truth = ground_truth.to(torch.device(self.device))

            self.live_objects["optimizer"].zero_grad()

            # forward + backward + optimize
            output = self.live_objects["model"](input)
            loss = self.live_objects["lossfunction"](output, ground_truth)
            loss.backward()
            self.live_objects["optimizer"].step()

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
                phase=self.name + "_train",
                epoch=i_epoch,
            )

        if self.verbose:
            target_value = self.live_objects["reporter"].get_measure(
                self.target_measure, phase=self.name + "_train", epoch=i_epoch
            )
            epoch_loss = self.live_objects["reporter"].get_measure(
                "loss", phase=self.name + "_train", epoch=i_epoch
            )
            print(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"Epoch {i_epoch}: {self.train_group} loss: {epoch_loss} {self.target_measure}:{target_value}",
            )

    def val_epoch(self, i_epoch, ret: bool = False):

        self.live_objects["loader"].dataset.sample_from(self.val_group)
        self.live_objects["loader"].dataset.eval_mode = True

        with torch.no_grad():
            for data in iter(self.live_objects["loader"]):

                self.live_objects["model"].eval()

                # forward pass
                data_ = self.live_objects["adapter"](data)

                input = data_["input"]
                ground_truth = data_["ground_truth"]
                supplement = data_["supplement"]

                input = input.to(torch.device(self.device))
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
                    phase=self.name + "_val",
                    epoch=i_epoch,
                )

        target_value = self.live_objects["reporter"].get_measure(
            self.target_measure, phase=self.name + "_val", epoch=i_epoch
        )
        epoch_loss = self.live_objects["reporter"].get_measure(
            "loss", phase=self.name + "_val", epoch=i_epoch
        )

        self.live_objects["checkpoint_tool"].update(
            model=self.live_objects["model"],
            optimizer=self.live_objects["optimizer"],
            value=target_value,
            epoch=i_epoch,
        )
        if isinstance(
            self.live_objects["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            self.live_objects["scheduler"].step(target_value)
        else:
            self.live_objects["scheduler"].step()

        self.live_objects["reporter"].save()

        if self.verbose:
            print(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"Epoch {i_epoch}: {self.val_group} loss: {epoch_loss} {self.target_measure}:{target_value}",
            )

        if ret:
            return target_value

    def setup(self):

        self.live_objects = {}

        # create and compose augmentation transforms
        trfs = []
        for transform_classname, args, kwargs in self.augmentations:
            trfs.append(getattr(transforms, transform_classname)(*args, **kwargs))
        self.live_objects["augmentations"] = tv.transforms.Compose(trfs)

        # create and compose eval-augmentation transforms
        trfs = []
        for transform_classname, args, kwargs in self.eval_augmentations:
            trfs.append(getattr(transforms, transform_classname)(*args, **kwargs))
        self.live_objects["eval_augmentations"] = tv.transforms.Compose(trfs)

        # create dataset
        dataset_classname, args, kwargs = self.dataset
        self.live_objects["dataset"] = getattr(datasets, dataset_classname)(
            *args,
            transform=self.live_objects["augmentations"],
            eval_transform=self.live_objects["eval_augmentations"],
            **kwargs,
        )

        if self.verbose:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Dataset:")
            self.live_objects["dataset"].print_summary()

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
        if self.use_class_weights:
            self.class_weights = torch.FloatTensor(
                self.live_objects["dataset"].get_class_weights(self.train_group)
            )
            self.class_weights = self.class_weights.to(torch.device(self.device))
            self.live_objects["lossfunction"] = getattr(
                lossfunctions, lossfunction_classname
            )(*args, weight=self.class_weights, **kwargs)
        else:
            self.live_objects["lossfunction"] = getattr(
                lossfunctions, lossfunction_classname
            )(*args, **kwargs)

        # create model
        model_classname, args, kwargs = self.model
        self.live_objects["model"] = getattr(models, model_classname)(*args, **kwargs)

        self.live_objects["model"] = self.live_objects["model"].to(
            torch.device(self.device)
        )
        if torch.cuda.device_count() > 1:
            self.live_objects["model"] = torch.nn.DataParallel(
                self.live_objects["model"]
            )

        # create optimizer
        optimizer_classname, args, kwargs = self.optimizer
        self.live_objects["optimizer"] = getattr(optimizers, optimizer_classname)(
            self.live_objects["model"].parameters(), *args, **kwargs
        )

        # create scheduler
        scheduler_classname, args, kwargs = self.scheduler
        self.live_objects["scheduler"] = getattr(schedulers, scheduler_classname)(
            self.live_objects["optimizer"], *args, **kwargs
        )

        # create checkpoint tool
        checkpoint_tool_classname, args, kwargs = self.checkpoint_tool
        kwargs["base_dir"] = self.checkpoint_dir
        self.live_objects["checkpoint_tool"] = getattr(
            tools, checkpoint_tool_classname
        )(
            *args,
            model_init_params=self.model,
            optimizer_init_params=self.optimizer,
            **kwargs,
        )

    def run(self):

        for _ in range(self.epochs):
            if not self.run_epoch():
                break

        if self.verbose:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Training run done.")

        del self.live_objects

    def run_epoch(self):

        if not hasattr(self, "live_objects"):
            self.setup()

        if self.current_epoch == 0:
            if self.verbose:
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Starting first epoch of training run...",
                )
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Name: {}".format(self.name),
                )
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Reporter directory: {}".format(
                        self.live_objects["reporter"].out_dir
                    ),
                )
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Checkpoint directory: {}".format(
                        self.live_objects["checkpoint_tool"].base_dir
                    ),
                )

            if self.autostop:
                self._last_values_conv = []
                self._last_values_plateau = []

        self.train_epoch(self.current_epoch)
        self.val_epoch(self.current_epoch)

        self.current_epoch += 1

        if self.autostop:
            current_value = self.live_objects["reporter"].get_measure(
                self.autostop_measure,
                phase=self.name + "_train",
                epoch=self.current_epoch - 1,
            )
            self._last_values_conv.append(current_value)
            self._last_values_plateau.append(current_value)

            if len(self._last_values_conv) > self.convergence_window:
                del self._last_values_conv[0]
            if len(self._last_values_plateau) > self.plateau_window:
                del self._last_values_plateau[0]

            if self.autostop_mode == "max":
                if np.all(np.array(self._last_values_conv) >= self.convergence_value):
                    if self.verbose:
                        print(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Convergence reached. Stopping...",
                        )
                    return False
            else:
                if np.all(np.array(self._last_values_conv) <= self.convergence_value):
                    if self.verbose:
                        print(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Convergence reached. Stopping...",
                        )
                    return False

            if len(self._last_values_plateau) < self.plateau_window:
                return True
            elif self.autostop_mode == "max":

                mean0 = sum(
                    self._last_values_plateau[: self.plateau_window // 2]
                ) / len(self._last_values_plateau[: self.plateau_window // 2])
                mean1 = sum(
                    self._last_values_plateau[self.plateau_window // 2 :]
                ) / len(self._last_values_plateau[self.plateau_window // 2 :])

                if mean0 >= mean1 + self.plateau_limit:
                    if self.verbose:
                        print(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Plateau reached. Stopping...",
                        )
                    return False
            else:
                mean0 = sum(
                    self._last_values_plateau[: self.plateau_window // 2]
                ) / len(self._last_values_plateau[: self.plateau_window // 2])
                mean1 = sum(
                    self._last_values_plateau[self.plateau_window // 2 :]
                ) / len(self._last_values_plateau[self.plateau_window // 2 :])

                if mean0 <= mean1 + self.plateau_limit:
                    if self.verbose:
                        print(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Plateau reached. Stopping...",
                        )
                    return False

        return True


class ClassificationModelTrainer(object):
    def __init__(
        self,
        name: str = "",
        reporter_dir: str = "",
        checkpoint_dir: str = "",
        epochs: int = 1,
        augmentations: any = None,
        eval_augmentations: any = None,
        dataset: any = None,
        loader: any = None,
        adapter: any = None,
        report_adapter: any = None,
        model: any = None,
        load_pretrained=False,
        model_path=None,
        lossfunction: any = None,
        optimizer: any = None,
        scheduler: any = None,
        reporter: any = None,
        target_measure: str = "acc",
        checkpoint_tool: any = None,
        train_group: str = "train",
        val_group: str = "val",
        use_class_weights: bool = False,
        autostop: bool = False,
        autostop_mode: str = "max",
        autostop_measure: str = "acc",
        convergence_window: int = 5,
        convergence_value: int = 0.95,
        plateau_window: int = 20,
        plateau_limit: float = 0.0,
        verbose: bool = False,
    ):

        assert autostop_mode in ("min", "max"), "Unknown autostop_mode: " + str(
            autostop_mode
        )

        self.name = name
        self.reporter_dir = reporter_dir
        self.checkpoint_dir = checkpoint_dir
        self.epochs = epochs
        self.augmentations = augmentations
        self.eval_augmentations = eval_augmentations
        self.dataset = dataset
        self.loader = loader
        self.adapter = adapter
        self.load_pretrained = load_pretrained
        self.model_path = model_path
        self.report_adapter = report_adapter
        self.model = model
        self.lossfunction = lossfunction
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.reporter = reporter
        self.target_measure = target_measure
        self.checkpoint_tool = checkpoint_tool
        self.train_group = train_group
        self.val_group = val_group
        self.use_class_weights = use_class_weights
        self.autostop = autostop
        self.autostop_mode = autostop_mode
        self.autostop_measure = autostop_measure
        self.convergence_window = convergence_window
        self.convergence_value = convergence_value
        self.plateau_window = plateau_window
        self.plateau_limit = plateau_limit
        self.verbose = verbose

        self.class_weights = None
        self.current_epoch = 0
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:0"

    def train_epoch(self, i_epoch):

        self.live_objects["loader"].dataset.sample_from(self.train_group)
        self.live_objects["loader"].dataset.eval_mode = False

        loss = None

        for data in iter(self.live_objects["loader"]):

            self.live_objects["model"].train()

            inputs, ground_truth = self.live_objects["adapter"](data)

            inputs = inputs.to(torch.device(self.device))
            ground_truth = ground_truth.to(torch.device(self.device))

            self.live_objects["optimizer"].zero_grad()

            # forward + backward + optimize
            outputs = self.live_objects["model"](inputs)
            loss = self.live_objects["lossfunction"](outputs, ground_truth)
            loss.backward()
            self.live_objects["optimizer"].step()

            # write data
            outputs, ground_truth, loss, inputs = self.live_objects["report_adapter"](
                outputs, ground_truth, loss, inputs
            )

            self.live_objects["reporter"](
                outputs,
                ground_truth,
                loss,
                idents=data["_idents"],
                phase=self.name + "_train",
                epoch=i_epoch,
                verbose=self.verbose,
                batch_input=inputs,
            )

        target_value = self.live_objects["reporter"].get_measure(
            self.target_measure, phase=self.name + "_train", epoch=i_epoch
        )
        epoch_loss = self.live_objects["reporter"].get_measure(
            "loss", phase=self.name + "_train", epoch=i_epoch
        )

        if self.verbose:
            print(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"Epoch {i_epoch}: {self.train_group} loss: {epoch_loss} {self.target_measure}:{target_value}",
            )

    def val_epoch(self, i_epoch, ret: bool = False):

        self.live_objects["loader"].dataset.sample_from(self.val_group)
        self.live_objects["loader"].dataset.eval_mode = True

        with torch.no_grad():
            for data in iter(self.live_objects["loader"]):

                self.live_objects["model"].eval()

                # forward pass
                inputs, ground_truth = self.live_objects["adapter"](data)

                inputs = inputs.to(torch.device(self.device))
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
                    phase=self.name + "_val",
                    epoch=i_epoch,
                    verbose=self.verbose,
                    batch_input=inputs,
                )

        target_value = self.live_objects["reporter"].get_measure(
            self.target_measure, phase=self.name + "_val", epoch=i_epoch
        )
        epoch_loss = self.live_objects["reporter"].get_measure(
            "loss", phase=self.name + "_val", epoch=i_epoch
        )

        self.live_objects["checkpoint_tool"].update(
            model=self.live_objects["model"],
            optimizer=self.live_objects["optimizer"],
            value=target_value,
            epoch=i_epoch,
        )
        if isinstance(
            self.live_objects["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            self.live_objects["scheduler"].step(target_value)
        else:
            self.live_objects["scheduler"].step()

        self.live_objects["reporter"].save()

        if self.verbose:
            print(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"Epoch {i_epoch}: {self.val_group} loss: {epoch_loss} {self.target_measure}:{target_value}",
            )

        if ret:
            return target_value

    def setup(self):

        self.live_objects = {}

        # create and compose augmentation transforms
        trfs = []
        for transform_classname, args, kwargs in self.augmentations:
            trfs.append(getattr(transforms, transform_classname)(*args, **kwargs))
        self.live_objects["augmentations"] = tv.transforms.Compose(trfs)

        # create and compose eval-augmentation transforms
        trfs = []
        for transform_classname, args, kwargs in self.eval_augmentations:
            trfs.append(getattr(transforms, transform_classname)(*args, **kwargs))
        self.live_objects["eval_augmentations"] = tv.transforms.Compose(trfs)

        # create dataset
        dataset_classname, args, kwargs = self.dataset
        self.live_objects["dataset"] = getattr(datasets, dataset_classname)(
            *args,
            transform=self.live_objects["augmentations"],
            eval_transform=self.live_objects["eval_augmentations"],
            **kwargs,
        )

        if self.verbose:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Dataset:")
            self.live_objects["dataset"].print_summary()

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
        if self.use_class_weights:
            self.class_weights = torch.FloatTensor(
                self.live_objects["dataset"].get_class_weights(self.train_group)
            )
            self.class_weights = self.class_weights.to(torch.device(self.device))
            self.live_objects["lossfunction"] = getattr(
                lossfunctions, lossfunction_classname
            )(*args, weight=self.class_weights, **kwargs)
        else:
            self.live_objects["lossfunction"] = getattr(
                lossfunctions, lossfunction_classname
            )(*args, **kwargs)

        # create model
        model_classname, args, kwargs = self.model
        self.live_objects["model"] = getattr(models, model_classname)(*args, **kwargs)
        
        if self.load_pretrained:
            saved_model = torch.load(self.model_path)
            new_state_dict = OrderedDict()
            for k, v in saved_model.items():
                name = k[7:] # remove 'module.' of dataparallel
                new_state_dict[name]=v

            self.live_objects["model"].load_state_dict(new_state_dict)
            self.live_objects["model"]=self.live_objects["model"].to(torch.device(self.device))
    
        else:   
            self.live_objects["model"] = self.live_objects["model"].to(
                torch.device(self.device)
            )
        if torch.cuda.device_count() > 1:
            self.live_objects["model"] = torch.nn.DataParallel(
                self.live_objects["model"]
            )
        # create optimizer
        optimizer_classname, args, kwargs = self.optimizer
        self.live_objects["optimizer"] = getattr(optimizers, optimizer_classname)(
            list(self.live_objects["model"].parameters()), *args, **kwargs
        )

        # create scheduler
        scheduler_classname, args, kwargs = self.scheduler
        self.live_objects["scheduler"] = getattr(schedulers, scheduler_classname)(
            self.live_objects["optimizer"], *args, **kwargs
        )

        # create checkpoint tool
        checkpoint_tool_classname, args, kwargs = self.checkpoint_tool
        kwargs["base_dir"] = self.checkpoint_dir
        self.live_objects["checkpoint_tool"] = getattr(
            tools, checkpoint_tool_classname
        )(
            *args,
            model_init_params=self.model,
            optimizer_init_params=self.optimizer,
            **kwargs,
        )

    def run(self):

        for _ in range(self.epochs):
            if not self.run_epoch():
                break

        if self.verbose:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Training run done.")

        del self.live_objects

    def run_epoch(self):

        if not hasattr(self, "live_objects"):
            self.setup()

        if self.current_epoch == 0:
            if self.verbose:
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Starting first epoch of training run...",
                )
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Name: {}".format(self.name),
                )
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Reporter directory: {}".format(
                        self.live_objects["reporter"].out_dir
                    ),
                )
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Checkpoint directory: {}".format(
                        self.live_objects["checkpoint_tool"].base_dir
                    ),
                )

            if self.autostop:
                self._last_values_conv = []
                self._last_values_plateau = []

        self.train_epoch(self.current_epoch)
        self.val_epoch(self.current_epoch)

        self.current_epoch += 1

        if self.autostop:
            current_value = self.live_objects["reporter"].get_measure(
                self.autostop_measure,
                phase=self.name + "_train",
                epoch=self.current_epoch - 1,
            )
            self._last_values_conv.append(current_value)
            self._last_values_plateau.append(current_value)

            if len(self._last_values_conv) > self.convergence_window:
                del self._last_values_conv[0]
            if len(self._last_values_plateau) > self.plateau_window:
                del self._last_values_plateau[0]

            if self.autostop_mode == "max":
                if np.all(np.array(self._last_values_conv) >= self.convergence_value):
                    if self.verbose:
                        print(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Convergence reached. Stopping...",
                        )
                    return False
            else:
                if np.all(np.array(self._last_values_conv) <= self.convergence_value):
                    if self.verbose:
                        print(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Convergence reached. Stopping...",
                        )
                    return False

            if len(self._last_values_plateau) < self.plateau_window:
                return True
            elif self.autostop_mode == "max":

                mean0 = sum(
                    self._last_values_plateau[: self.plateau_window // 2]
                ) / len(self._last_values_plateau[: self.plateau_window // 2])
                mean1 = sum(
                    self._last_values_plateau[self.plateau_window // 2 :]
                ) / len(self._last_values_plateau[self.plateau_window // 2 :])

                if mean0 >= mean1 + self.plateau_limit:
                    if self.verbose:
                        print(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Plateau reached. Stopping...",
                        )
                    return False
            else:
                mean0 = sum(
                    self._last_values_plateau[: self.plateau_window // 2]
                ) / len(self._last_values_plateau[: self.plateau_window // 2])
                mean1 = sum(
                    self._last_values_plateau[self.plateau_window // 2 :]
                ) / len(self._last_values_plateau[self.plateau_window // 2 :])

                if mean0 <= mean1 + self.plateau_limit:
                    if self.verbose:
                        print(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Plateau reached. Stopping...",
                        )
                    return False

        return True
