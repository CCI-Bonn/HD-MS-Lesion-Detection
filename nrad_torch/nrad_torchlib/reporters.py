"""                                                                                                            
:AUTHOR: Hagen Meredig                                                                                         
:ORGANIZATION: Department of Neuroradiology, Heidelberg Univeristy Hospital                                    
:CONTACT: Hagen.Meredig@med.uni-heidelberg.de                                                                  
:SINCE: August 18, 2021                                                                            
""" 
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from scipy.special import softmax
from scipy.special import expit


class BasicMCCReporter(object):
    """This is the new base reporter class.
    With standard init params this does exactly the same as the former
    MulticlassClassificationReporter class.
    Use softmax_pred=False if the prediction entries are already softmaxed..."""

    def __init__(
        self,
        out_dir: str = "",
        input_report: bool = True,
        _ident_kw: str = "ID",
        _ground_truth_kw: str = "ground_truth",
        _prediction_kw: str = "prediction",
        _loss_kw: str = "batch_loss",
        softmax_pred: bool = True,
    ):

        self.out_dir = out_dir
        self.input_report = input_report
        self._ident_kw = _ident_kw
        self._ground_truth_kw = _ground_truth_kw
        self._prediction_kw = _prediction_kw
        self._loss_kw = _loss_kw
        self.softmax_pred = softmax_pred
        self.history = {}
        self.current_batch = {}

        if os.path.isdir(self.out_dir):
            for entry in os.listdir(self.out_dir):
                if entry.endswith(".csv"):
                    epoch = int(entry.split("_")[-1].replace(".csv", ""))
                    phase = entry.replace("_{}.csv".format(epoch), "")
                    with open(os.path.join(self.out_dir, entry), "r") as csvfile:
                        reader = csv.DictReader(csvfile)
                        table_content = {}
                        for row in reader:
                            for fieldname in row.keys():
                                try:
                                    if " " in row[fieldname]:
                                        fieldvalue = [
                                            float(i) for i in row[fieldname].split(" ")
                                        ]
                                    else:
                                        try:
                                            fieldvalue = float(row[fieldname])
                                        except ValueError:
                                            fieldvalue = row[fieldname]

                                    if not fieldname in table_content.keys():
                                        table_content[fieldname] = []

                                    table_content[fieldname].append(fieldvalue)
                                except Exception as e:
                                    print("Exception while loading:", e)
                                    print("csvfile", csvfile)
                                    print("row", row)
                                    print("fieldname", fieldname)

                    self(**table_content, phase=phase, epoch=epoch)

    def __call__(self, **kwargs):
        """reserved keywords: phase, epoch, input"""

        assert "phase" in kwargs.keys(), "must provide phase keyword"
        assert "epoch" in kwargs.keys(), "must provide epoch keyword"

        phase = kwargs["phase"]
        epoch = kwargs["epoch"]

        if not phase in self.history.keys():
            self.history[phase] = {}
        if not epoch in self.history[phase].keys():
            self.history[phase][epoch] = {}

        for fieldname in [
            i
            for i in kwargs.keys()
            if not (i == "phase" or i == "epoch" or i == "input")
        ]:
            if not fieldname in self.history[phase][epoch].keys():
                self.history[phase][epoch][fieldname] = []

            self.history[phase][epoch][fieldname] += kwargs[fieldname]

        if not phase in self.current_batch.keys():
            self.current_batch[phase] = 0

        # if specified, create input report consiting of the
        # first batch of the first epoch per phase saved as npy
        if (
            self.input_report
            and (self.current_batch[phase] == 0)
            and (epoch == 0)
            and "input" in kwargs.keys()
        ):

            if not os.path.isdir(self.out_dir):
                os.makedirs(self.out_dir)

            for sample in range(len(kwargs["input"])):
                for channel in range(len(kwargs["input"][sample])):
                    save_data = kwargs["input"][sample][channel]
                    if not isinstance(save_data, np.ndarray):
                        save_data = np.array(save_data)
                    np.save(
                        os.path.join(
                            self.out_dir, f"{phase}_smpl{sample}_chn{channel}.npy"
                        ),
                        save_data,
                    )

        self.current_batch[phase] += 1

    def reload(self):

        self.__init__(
            out_dir=self.out_dir,
            input_report=self.input_report,
            _ident_kw=self._ident_kw,
            _ground_truth_kw=self._ground_truth_kw,
            _prediction_kw=self._prediction_kw,
            _loss_kw=self._loss_kw,
            softmax_pred=self.softmax_pred,
        )

    def get_measure(self, name: str, phase: str = "default", epoch: int = 0):

        if name == "acc" or name == "accuracy":
            return self.accuracy(phase=phase, epoch=epoch)
        elif name == "loss":
            return self.loss(phase=phase, epoch=epoch)
        else:
            if name in self.history[phase][epoch].keys():
                return np.array(self.history[phase][epoch][name]).mean()
            else:
                raise KeyError(f"Measure {name} not found.")

    def get_per_class_measure(
        self,
        measure_name: str,
        phase: str = "default",
        epoch: int = 0,
        label_names: any = None,
    ):

        gts = []
        preds = []

        p_op = softmax if self.softmax_pred else lambda x: x

        for i in range(len(self.history[phase][epoch][self._ground_truth_kw])):
            gts.append(
                np.argmax(
                    np.array(self.history[phase][epoch][self._ground_truth_kw][i])
                )
            )
            preds.append(
                np.argmax(
                    np.array(p_op(self.history[phase][epoch][self._prediction_kw][i]))
                )
            )

        label_indices = [i for i in sorted(set(gts))]
        target_names = None
        if label_names != None:
            target_names = [str(i) + ": " + label_names[i] for i in label_indices]

        cr = classification_report(
            gts,
            preds,
            labels=label_indices,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        ret = {}
        for i in sorted(cr.keys()):
            if i not in ("accuracy", "macro avg", "weighted avg"):
                ret[i] = cr[i][measure_name]

        return ret

    def accuracy(self, phase: str = "default", epoch: int = 0):

        gts = []
        preds = []

        p_op = softmax if self.softmax_pred else lambda x: x

        for i in range(len(self.history[phase][epoch][self._ground_truth_kw])):
            gts.append(
                np.argmax(
                    np.array(self.history[phase][epoch][self._ground_truth_kw][i])
                )
            )
            preds.append(
                np.argmax(
                    np.array(p_op(self.history[phase][epoch][self._prediction_kw][i]))
                )
            )

        return accuracy_score(gts, preds)

    def loss(self, phase: str = "default", epoch: int = 0):

        losses = []

        for i in range(len(self.history[phase][epoch][self._ground_truth_kw])):
            losses.append(self.history[phase][epoch][self._loss_kw][i])

        loss = np.array(losses).mean()

        return loss

    def best_epoch(self, phase: str = "default", target: str = "acc"):

        if target == "acc":
            return np.argmax(
                [
                    self.accuracy(phase=phase, epoch=i)
                    for i in range(len(self.history[phase]))
                ]
            )
        elif target == "loss":
            return np.argmin(
                [
                    self.loss(phase=phase, epoch=i)
                    for i in range(len(self.history[phase]))
                ]
            )
        else:
            raise Exception("Unknown target value: " + target)

    def last_epoch(self, phase: str = "default"):

        return len(self.history[phase].keys()) - 1

    def save(self):

        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

        for phase in self.history.keys():
            for epoch in self.history[phase].keys():

                fp = os.path.join(self.out_dir, "{}_{}.csv".format(phase, epoch))

                with open(fp, "w") as f:
                    firstrow = ""
                    for fieldname in sorted(list(self.history[phase][epoch].keys())):
                        firstrow += fieldname + ","
                    firstrow = firstrow[:-1] + "\n"
                    f.write(firstrow)

                    # assume all lists have equal lenghth
                    firstkey = sorted(list(self.history[phase][epoch].keys()))[0]
                    for irow in range(len(self.history[phase][epoch][firstkey])):
                        rowstr = ""
                        for fieldname in sorted(
                            list(self.history[phase][epoch].keys())
                        ):
                            item = (
                                str(self.history[phase][epoch][fieldname][irow])
                                .replace("[", "")
                                .replace(",", "")
                                .replace("]", "")
                            )
                            rowstr += item + ","
                        rowstr = rowstr[:-1] + "\n"
                        f.write(rowstr)

    def plot_per_class_measure(
        self, measure_name: str, phases: any = "all", label_names: any = None
    ):

        if phases == "all":
            phases = self.history.keys()

        for phase in phases:
            epochs_list = []
            label_legend_list = []
            for ep in range(len(self.history[phase])):
                precs = self.get_per_class_measure(
                    measure_name, phase=phase, epoch=ep, label_names=label_names
                )
                epochs_list.append([])

                for key in precs.keys():
                    if not key in label_legend_list:
                        label_legend_list.append(key)

                for entry in label_legend_list:
                    epochs_list[-1].append(precs[entry])

            epochs_list = np.array(epochs_list)

            x = [i for i in range(len(epochs_list))]

            plt.title("Phase: " + phase)
            plt.xlabel("Epoch")
            plt.xticks([i for i in x if (i % (len(x) / 10) == 0)])
            plt.ylabel(measure_name.capitalize())
            for cl in range(epochs_list.shape[1]):
                plt.plot(x, epochs_list[:, cl])
            plt.legend(label_legend_list)
            plt.show()
            plt.close()

    def plot_roc(
        self, class_index, phase: str = "default", epoch: int = 0, class_name=""
    ):
        """Plot ROC curve. Correctly defined only for binary classification."""

        gts = []
        preds = []

        p_op = softmax if self.softmax_pred else lambda x: x

        for i in range(len(self.history[phase][epoch][self._ground_truth_kw])):
            gts.append(
                self.history[phase][epoch][self._ground_truth_kw][i][class_index]
            )
            preds.append(
                p_op(self.history[phase][epoch][self._prediction_kw][i])[class_index]
            )

        fpr, tpr, _ = roc_curve(gts, preds)
        auc = roc_auc_score(gts, preds)

        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="navy",
            lw=lw,
            label="AUC = %0.2f" % auc,
        )
        plt.plot([0, 1], [0, 1], color="gray", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve " + class_name)
        plt.legend(loc="lower right")
        plt.show()
        plt.close()

    def plot_measure(self, measure_name: str, phases: list = "all"):

        if phases == "all":
            phases = list(self.history.keys())

        value_list = []
        for phase in phases:
            value_list.append([])
            for ep in range(len(self.history[phase])):
                value_list[-1].append(
                    self.get_measure(measure_name, phase=phase, epoch=ep)
                )

        max_x = max([len(i) for i in value_list])
        x = [i for i in range(max_x)]

        for p in range(len(value_list)):

            while len(value_list[p]) < len(x):
                value_list[p].append(0.0)

            plt.plot(x, value_list[p], label=phases[p])

        plt.xlabel("Epoch")
        plt.xticks([i for i in x if (i % (len(x) / 10) == 0)])
        plt.ylabel(measure_name.capitalize())
        plt.title(f"{measure_name.capitalize()} plot")
        plt.legend(fontsize=6)
        plt.show()
        plt.close()

    def show_input_report(self, phases: list = "all"):

        npy_filenames = []

        for phase in phases:
            npy_filenames += [
                i for i in os.listdir(self.out_dir) if i.endswith(".npy") and phase in i
            ]
        npy_filenames = sorted(npy_filenames)

        for fn in npy_filenames:
            print(fn)
            data = np.load(os.path.join(self.out_dir, fn))

            if len(data.shape) == 3:  # 3D
                slice_to_show = data.shape[2] // 2
                data = data[:, :, slice_to_show]

            plt.imshow(data, cmap="gray")

            plt.show()

    def ensemble(self, data_list: tuple, name: str):

        value_keys = [k for k in data_list[0].keys() if not (k == self._ident_kw)]

        for i in range(len(data_list)):
            value_keys_i = [k for k in data_list[i].keys() if not (k == self._ident_kw)]
            assert len(value_keys_i) == len(
                value_keys
            ), "Tables seem to have inconsistent value fields"
            for value_key in value_keys_i:
                assert (
                    value_key in value_keys
                ), "Tables seem to have inconsistent value fields"

        data_dicts = []
        all_idents = []

        for data in data_list:
            idents = data[self._ident_kw]
            data_dicts.append({})
            for i in range(len(idents)):
                if not idents[i] in all_idents:
                    all_idents.append(idents[i])
                data_dicts[-1][idents[i]] = {}
                for value_key in value_keys:
                    data_dicts[-1][idents[i]][value_key] = data[value_key][i]

        self.history[name] = {0: {self._ident_kw: all_idents}}
        for value_key in value_keys:
            self.history[name][0][value_key] = []

        for ident in all_idents:
            for value_key in value_keys:
                summed_values = None
                i = -1
                for data in data_dicts:
                    i += 1
                    try:
                        value = data[ident][value_key]
                        if isinstance(value, list):
                            value = np.array(value)
                        if summed_values == None:
                            summed_values = [value, 1]
                        else:
                            summed_values[0] += value
                            summed_values[1] += 1
                    except KeyError:
                        pass
                summed_values[0] /= summed_values[1]
                if isinstance(summed_values[0], np.ndarray):
                    summed_values[0] = summed_values[0].tolist()
                self.history[name][0][value_key].append(summed_values[0])

        self.save()

    def get_epoch_data(self, phase: str, epoch: int):

        return self.history[phase][epoch]


class MulticlassClassificationReporter(object):
    def __init__(self, out_dir: str = "", input_report=True):

        self.out_dir = out_dir
        self.input_report = input_report
        self.history = {}
        self.current_batch = {}

        if os.path.isdir(self.out_dir):
            for entry in os.listdir(self.out_dir):
                if entry.endswith(".csv"):
                    epoch = int(entry.split("_")[-1].replace(".csv", ""))
                    phase = entry.replace("_{}.csv".format(epoch), "")
                    idents = []
                    outputs = []
                    ground_truths = []
                    losses = []
                    with open(os.path.join(self.out_dir, entry), "r") as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            idents.append(row["ID"])
                            outputs.append(
                                [float(i) for i in row["prediction"].split(" ")]
                            )
                            ground_truths.append(
                                None
                                if "None" in row["ground_truth"]
                                else [float(i) for i in row["ground_truth"].split(" ")]
                            )
                            losses.append(float(row["batch_loss"]))
                    self(
                        outputs,
                        ground_truths,
                        np.array(losses).mean(),
                        idents,
                        phase=phase,
                        epoch=epoch,
                    )

    def __call__(
        self,
        batch_output: list,
        batch_ground_truth: list,
        loss: float,
        idents: any = None,
        phase: str = "default",
        epoch: int = 0,
        batch_input: list = None,
        verbose: bool = False,
    ):

        if not phase in self.history.keys():
            self.history[phase] = {}
        if not epoch in self.history[phase].keys():
            self.history[phase][epoch] = {
                "idents": [],
                "predictions": [],
                "ground_truths": [],
                "losses": [],
            }
        if idents == None:
            idents = [""] * len(batch_ground_truth)
        self.history[phase][epoch]["idents"] += idents
        self.history[phase][epoch]["predictions"] += batch_output
        self.history[phase][epoch]["ground_truths"] += batch_ground_truth
        self.history[phase][epoch]["losses"] += [loss] * len(batch_ground_truth)

        if not phase in self.current_batch.keys():
            self.current_batch[phase] = 0

        # if specified, create input report consiting of the
        # first batch of the first epoch per phase saved as npy
        if (
            self.input_report
            and (self.current_batch[phase] == 0)
            and (epoch == 0)
            and not (batch_input == None)
        ):

            if not os.path.isdir(self.out_dir):
                os.makedirs(self.out_dir)

            for sample in range(len(batch_input)):
                for channel in range(len(batch_input[sample])):
                    save_data = batch_input[sample][channel]
                    if not isinstance(save_data, np.ndarray):
                        save_data = np.array(save_data)
                    np.save(
                        os.path.join(
                            self.out_dir, f"{phase}_smpl{sample}_chn{channel}.npy"
                        ),
                        save_data,
                    )

        self.current_batch[phase] += 1

    def reload(self):

        self.__init__(out_dir=self.out_dir)

    def get_measure(self, name: str, phase: str = "default", epoch: int = 0):

        if name == "acc" or name == "accuracy":
            return self.accuracy(phase=phase, epoch=epoch)
        if name == "loss":
            return self.loss(phase=phase, epoch=epoch)

    def get_per_class_measure(
        self,
        measure_name: str,
        phase: str = "default",
        epoch: int = 0,
        label_names: any = None,
    ):

        gts = []
        preds = []

        for i in range(len(self.history[phase][epoch]["ground_truths"])):
            gts.append(
                np.argmax(np.array(self.history[phase][epoch]["ground_truths"][i]))
            )
            preds.append(
                np.argmax(
                    np.array(softmax(self.history[phase][epoch]["predictions"][i]))
                )
            )

        label_indices = [i for i in sorted(set(gts))]
        target_names = None
        if label_names != None:
            target_names = [str(i) + ": " + label_names[i] for i in label_indices]

        cr = classification_report(
            gts,
            preds,
            labels=label_indices,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        ret = {}
        for i in sorted(cr.keys()):
            if i not in ("accuracy", "macro avg", "weighted avg"):
                ret[i] = cr[i][measure_name]

        return ret

    def accuracy(self, phase: str = "default", epoch: int = 0):

        gts = []
        preds = []

        for i in range(len(self.history[phase][epoch]["ground_truths"])):
            gts.append(
                np.argmax(np.array(self.history[phase][epoch]["ground_truths"][i]))
            )
            preds.append(
                np.argmax(
                    np.array(softmax(self.history[phase][epoch]["predictions"][i]))
                )
            )

        return accuracy_score(gts, preds)

    def loss(self, phase: str = "default", epoch: int = 0):

        losses = []

        for i in range(len(self.history[phase][epoch]["ground_truths"])):
            losses.append(self.history[phase][epoch]["losses"][i])

        loss = np.array(losses).mean()

        return loss

    def best_epoch(self, phase: str = "default", target: str = "acc"):

        if target == "acc":
            return np.argmax(
                [
                    self.accuracy(phase=phase, epoch=i)
                    for i in range(len(self.history[phase]))
                ]
            )
        elif target == "loss":
            return np.argmin(
                [
                    self.loss(phase=phase, epoch=i)
                    for i in range(len(self.history[phase]))
                ]
            )
        else:
            raise Exception("Unknown target value: " + target)

    def last_epoch(self, phase: str = "default"):

        return len(self.history[phase].keys()) - 1

    def save(self):

        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

        for phase in self.history.keys():
            for epoch in self.history[phase].keys():

                fp = os.path.join(self.out_dir, "{}_{}.csv".format(phase, epoch))
                if os.path.isfile(fp):
                    os.remove(fp)

                with open(fp, "w") as f:
                    f.write("ID,ground_truth,prediction,batch_loss\n")
                    for i in range(len(self.history[phase][epoch]["ground_truths"])):
                        ident = self.history[phase][epoch]["idents"][i]
                        gt = (
                            str(self.history[phase][epoch]["ground_truths"][i])
                            .replace("[", "")
                            .replace(",", "")
                            .replace("]", "")
                        )
                        pr = (
                            str(self.history[phase][epoch]["predictions"][i])
                            .replace("[", "")
                            .replace(",", "")
                            .replace("]", "")
                        )
                        lo = str(self.history[phase][epoch]["losses"][i])
                        f.write("{},{},{},{}\n".format(ident, gt, pr, lo))

    def plot_per_class_measure(
        self, measure_name: str, phases: any = "all", label_names: any = None
    ):

        if phases == "all":
            phases = self.history.keys()

        for phase in phases:
            epochs_list = []
            label_legend_list = []
            for ep in range(len(self.history[phase])):
                precs = self.get_per_class_measure(
                    measure_name, phase=phase, epoch=ep, label_names=label_names
                )
                epochs_list.append([])

                for key in precs.keys():
                    if not key in label_legend_list:
                        label_legend_list.append(key)

                for entry in label_legend_list:
                    epochs_list[-1].append(precs[entry])

            epochs_list = np.array(epochs_list)

            x = [i for i in range(len(epochs_list))]

            plt.title("Phase: " + phase)
            plt.xlabel("Epoch")
            plt.xticks([i for i in x if (i % (len(x) / 10) == 0)])
            plt.ylabel(measure_name.capitalize())
            for cl in range(epochs_list.shape[1]):
                plt.plot(x, epochs_list[:, cl])
            plt.legend(label_legend_list)
            plt.show()
            plt.close()

    def plot_roc(
        self, class_index, phase: str = "default", epoch: int = 0, class_name=""
    ):
        """Plot ROC curve. correctly defined only for binary classification."""

        gts = []
        preds = []

        for i in range(len(self.history[phase][epoch]["ground_truths"])):
            gts.append(self.history[phase][epoch]["ground_truths"][i][class_index])
            preds.append(
                softmax(self.history[phase][epoch]["predictions"][i])[class_index]
            )

        fpr, tpr, _ = roc_curve(gts, preds)
        auc = roc_auc_score(gts, preds)

        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="navy",
            lw=lw,
            label="AUC = %0.2f" % auc,
        )
        plt.plot([0, 1], [0, 1], color="gray", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve " + class_name)
        plt.legend(loc="lower right")
        plt.show()
        plt.close()

    def plot_measure(self, measure_name: str, phases: list = "all"):

        if phases == "all":
            phases = list(self.history.keys())

        value_list = []
        for phase in phases:
            value_list.append([])
            for ep in range(len(self.history[phase])):
                value_list[-1].append(
                    self.get_measure(measure_name, phase=phase, epoch=ep)
                )

        max_x = max([len(i) for i in value_list])
        x = [i for i in range(max_x)]

        for p in range(len(value_list)):

            while len(value_list[p]) < len(x):
                value_list[p].append(0.0)

            plt.plot(x, value_list[p], label=phases[p])

        plt.xlabel("Epoch")
        plt.xticks([i for i in x if (i % (len(x) / 10) == 0)])
        plt.ylabel(measure_name.capitalize())
        plt.title(f"{measure_name.capitalize()} plot")
        plt.legend(fontsize=6)
        plt.show()
        plt.close()

    def show_input_report(self, phases: list = "all"):

        npy_filenames = []

        for phase in phases:
            npy_filenames += [
                i for i in os.listdir(self.out_dir) if i.endswith(".npy") and phase in i
            ]
        npy_filenames = sorted(npy_filenames)

        for fn in npy_filenames:
            print(fn)
            data = np.load(os.path.join(self.out_dir, fn))

            if len(data.shape) == 3:  # 3D
                slice_to_show = data.shape[2] // 2
                data = data[:, :, slice_to_show]

            plt.imshow(data, cmap="gray")

            plt.show()

    def ensemble(self, data_list: tuple, name: str):

        predictions = {}
        ground_truths = {}
        losses = {}

        for data in data_list:
            for i in range(len(data["idents"])):
                ident = data["idents"][i]
                pred = data["predictions"][i]
                gt = data["ground_truths"][i]
                lo = data["losses"][i]
                if not ident in predictions.keys():
                    predictions[ident] = [
                        np.array(pred),
                        1,
                    ]
                else:
                    predictions[ident][0] += np.array(pred)
                    predictions[ident][1] += 1
                if not ident in ground_truths.keys():
                    ground_truths[ident] = [
                        np.array(gt),
                        1,
                    ]
                else:
                    ground_truths[ident][0] += np.array(gt)
                    ground_truths[ident][1] += 1
                if not ident in losses.keys():
                    losses[ident] = [
                        np.array(lo),
                        1,
                    ]
                else:
                    losses[ident][0] += float(lo)
                    losses[ident][1] += 1

        self.history[name] = {
            0: {"idents": [], "predictions": [], "ground_truths": [], "losses": []}
        }

        for ident in sorted(ground_truths.keys()):
            self.history[name][0]["idents"].append(ident)
            self.history[name][0]["predictions"].append(
                (predictions[ident][0] / predictions[ident][1]).tolist()
            )
            self.history[name][0]["ground_truths"].append(
                (ground_truths[ident][0] / ground_truths[ident][1]).tolist()
            )
            self.history[name][0]["losses"].append(
                float(losses[ident][0] / losses[ident][1])
            )

        self.save()

    def get_epoch_data(self, phase: str, epoch: int):

        return self.history[phase][epoch]


class MultilabelClassificationReporter(MulticlassClassificationReporter):
    def __init__(self, out_dir: str = "", input_report=True):

        super(MultilabelClassificationReporter, self).__init__(
            out_dir=out_dir, input_report=input_report
        )

    def accuracy(self, phase: str = "default", epoch: int = 0, threshold=0.5):

        gts = []
        preds = []

        for i in range(len(self.history[phase][epoch]["ground_truths"])):
            gts.append(np.array(self.history[phase][epoch]["ground_truths"][i]))
            sigmoids = expit(self.history[phase][epoch]["predictions"][i])
            preds.append(np.array([1.0 if i > threshold else 0.0 for i in sigmoids]))

        gts = np.array(gts)
        preds = np.array(preds)

        return accuracy_score(gts, preds)

    def get_per_class_measure(
        self,
        measure_name: str,
        phase: str = "default",
        epoch: int = 0,
        label_names: any = None,
        threshold=0.5,
    ):

        gts = []
        preds = []

        for i in range(len(self.history[phase][epoch]["ground_truths"])):
            gts.append(np.array(self.history[phase][epoch]["ground_truths"][i]))
            sigmoids = expit(self.history[phase][epoch]["predictions"][i])
            preds.append(np.array([1.0 if i > threshold else 0.0 for i in sigmoids]))

        gts = np.array(gts)
        preds = np.array(preds)

        label_indices = [i for i in range(len(gts[0]))]

        if label_names != None:
            target_names = [str(i) + ": " + label_names[i] for i in label_indices]
        else:
            target_names = [str(i) for i in label_indices]

        cm = multilabel_confusion_matrix(gts, preds)

        ret = {}

        for cl in range(cm.shape[0]):
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
            tn, fp, fn, tp = cm[cl].ravel()
            value = None
            if measure_name in ("acc", "accuracy"):
                value = (tp + tn) / (
                    tp + tn + fp + fn
                )  # (TP + TN) / (TP + TN + FP + FN)
            elif measure_name in ("sens", "sensitivity"):
                value = tp / (tp + fn)  # TP / (TP + FN)
            elif measure_name in ("spec", "specificity"):
                value = tn / (tn + fp)  # TN / (TN + FP)
            else:
                print("unknon measure_name")

            ret[target_names[cl]] = value

        return ret

    def plot_roc(
        self, class_index, phase: str = "default", epoch: int = 0, class_name=""
    ):

        gts = []
        preds = []

        for i in range(len(self.history[phase][epoch]["ground_truths"])):
            gts.append(self.history[phase][epoch]["ground_truths"][i][class_index])
            sigmoids = expit(self.history[phase][epoch]["predictions"][i])
            preds.append(sigmoids[class_index])

        fpr, tpr, _ = roc_curve(gts, preds)
        auc = roc_auc_score(gts, preds)

        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="navy",
            lw=lw,
            label="AUC = %0.2f" % auc,
        )
        plt.plot([0, 1], [0, 1], color="gray", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve " + class_name)
        plt.legend(loc="lower right")
        plt.show()
        plt.close()


class ImgToImgReporter(MulticlassClassificationReporter):
    def __init__(self, *args, **kwargs):
        super(ImgToImgReporter, self).__init__(*args, **kwargs)

    def __call__(
        self,
        batch_output: list,
        batch_ground_truth: list,
        loss: float,
        idents: any = None,
        phase: str = "default",
        epoch: int = 0,
        batch_input: list = None,
        verbose: bool = False,
    ):

        if not phase in self.history.keys():
            self.history[phase] = {}
        if not epoch in self.history[phase].keys():
            self.history[phase][epoch] = {
                "idents": [],
                "predictions": [],
                "ground_truths": [],
                "losses": [],
            }
        if idents == None:
            idents = [""] * len(batch_ground_truth)
        self.history[phase][epoch]["idents"] += idents
        self.history[phase][epoch]["predictions"] += [-1.0] * len(batch_ground_truth)
        self.history[phase][epoch]["ground_truths"] += [-1.0] * len(batch_ground_truth)
        self.history[phase][epoch]["losses"] += [loss] * len(batch_ground_truth)

        if not phase in self.current_batch.keys():
            self.current_batch[phase] = 0

        # if specified, create input report consiting of the
        # first batch of the first epoch per phase saved as npy
        if (
            self.input_report
            and (self.current_batch[phase] == 0)
            and (epoch == 0)
            and not (batch_input == None)
        ):

            if not os.path.isdir(self.out_dir):
                os.makedirs(self.out_dir)

            for sample in range(len(batch_input)):
                for channel in range(len(batch_input[sample])):
                    save_data = batch_input[sample][channel]
                    if not isinstance(save_data, np.ndarray):
                        save_data = np.array(save_data)
                    np.save(
                        os.path.join(
                            self.out_dir, f"{phase}_smpl{sample}_chn{channel}.npy"
                        ),
                        save_data,
                    )

        self.current_batch[phase] += 1


class MILMCReporter(MulticlassClassificationReporter):
    def __init__(self, *args, **kwargs):
        super(MILMCReporter, self).__init__(*args, **kwargs)

    def get_per_class_measure(
        self,
        measure_name: str,
        phase: str = "default",
        epoch: int = 0,
        label_names: any = None,
    ):

        gts = []
        preds = []

        for i in range(len(self.history[phase][epoch]["ground_truths"])):
            gts.append(
                np.argmax(np.array(self.history[phase][epoch]["ground_truths"][i]))
            )
            preds.append(
                np.argmax(np.array(self.history[phase][epoch]["predictions"][i]))
            )

        label_indices = [i for i in sorted(set(gts))]
        target_names = None
        if label_names != None:
            target_names = [str(i) + ": " + label_names[i] for i in label_indices]

        cr = classification_report(
            gts,
            preds,
            labels=label_indices,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        ret = {}
        for i in sorted(cr.keys()):
            if i not in ("accuracy", "macro avg", "weighted avg"):
                ret[i] = cr[i][measure_name]

        return ret

    def accuracy(self, phase: str = "default", epoch: int = 0):

        gts = []
        preds = []

        for i in range(len(self.history[phase][epoch]["ground_truths"])):
            gts.append(
                np.argmax(np.array(self.history[phase][epoch]["ground_truths"][i]))
            )
            preds.append(
                np.argmax(np.array(self.history[phase][epoch]["predictions"][i]))
            )

        return accuracy_score(gts, preds)

    def plot_roc(
        self, class_index, phase: str = "default", epoch: int = 0, class_name=""
    ):
        """Plot ROC curve. correctly defined only for binary classification."""

        gts = []
        preds = []

        for i in range(len(self.history[phase][epoch]["ground_truths"])):
            gts.append(self.history[phase][epoch]["ground_truths"][i][class_index])
            preds.append(self.history[phase][epoch]["predictions"][i][class_index])

        fpr, tpr, _ = roc_curve(gts, preds)
        auc = roc_auc_score(gts, preds)

        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="navy",
            lw=lw,
            label="AUC = %0.2f" % auc,
        )
        plt.plot([0, 1], [0, 1], color="gray", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve " + class_name)
        plt.legend(loc="lower right")
        plt.show()
        plt.close()
