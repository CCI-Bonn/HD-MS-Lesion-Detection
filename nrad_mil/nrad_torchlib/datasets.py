"""                                                                                                            
:AUTHOR: Hagen Meredig                                                                                         
:ORGANIZATION: Department of Neuroradiology, Heidelberg Univeristy Hospital                                    
:CONTACT: Hagen.Meredig@med.uni-heidelberg.de                                                                  
:SINCE: Aug 18, 2021 
:UPDATED: April 7, 2022 by Chandrakanth Jayachandran Preetha. Modified Dataset class to allow for training and validaion on pre-defined groups
"""

import os
import torch
import pydicom
import pandas as pd
import numpy as np
import nibabel as nib
from collections import OrderedDict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, KFold 
import random


class VolumeDataset(Dataset):
    """A volumetric dataset."""

    def __init__(
        self,
        root_dir=None,
        labels_file=None,
        channel_list=[],
        exclusions=None,
        transform=None,
        eval_transform=None,
        file_format=None,
        mask_series={},
        do_crossval=False,
        folds=None,
        do_split=False,
        split_fraction=None,
        get_split=False,
        train_list=None,
        val_list=None,
        seed=None,
    ):
        """
        Arguments
                ---------
        root_dir : str
            Parent directory with all subfolders containing the images.
        labels_file : str
            Path to the csv file with annotations.
        channel_list : list
            List of strings containing the filenames (without filetype extensions)
            to be loaded into the individual channels.
        exclusions : list
            A list of IDs which will be excluded.
        transform : callable
            Optional transform to be applied on a sample.
        file_format : str
            Fileformat in which the data is stored (npy, nii, nii.gz, dcm).
            If None, fileformat will be determined from the filename extension
            on the fly while loading (slower).
        mask_series : dict
            Dictionary mapping series names to individual identifiers.
            The given series are accessible in the sample data directly
            via thier given identifieres.
            Ex.: mask_series = {"brain_mask" : "mask.nii.gz"}
            VolumeDataset.__get_item__() will output a dict containing
            {"data" : img_data, ..., "brain_mask" : <data in mask.nii.gz of this sample>}
        """

        # init groups
        self.groups = {}
        self.groups["base"] = []
        self.groups["_excluded"] = []
        self.groups["_no_data"] = []
        self.groups["no_label"] = []
        self.groups["_incomplete_data"] = []

        # save parameters
        self.labels_file = labels_file
        self.root_dir = root_dir
        self.channel_list = channel_list
        self.exclusions = exclusions
        self.transform = transform
        self.eval_transform = eval_transform
        self.format = file_format
        self.mask_series = mask_series

        # load labels
        self.labels = {}
        if self.labels_file != None:
            self.labels_frame = pd.read_csv(self.labels_file, dtype=str)
            for row in range(self.labels_frame.shape[0]):

                ident = str(self.labels_frame.iloc[row, 0])
                label_str = self.labels_frame.iloc[row, 1]
                label_npy = np.array([float(i) for i in label_str.split(" ")])
                self.labels[ident] = label_npy

                # if there's no entry for this ID add to special group '_no_data'
                if not (ident in os.listdir(self.root_dir)):
                    self.groups["_no_data"].append(ident)

        # for all IDs in root_dir
        for ident in [
            i
            for i in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, i))
        ]:
            chnlst_complete = True
            channel_list_incl_masks = list(self.channel_list)
            for en in list(set(list(self.mask_series.values()))):
                if not (en in channel_list_incl_masks):
                    channel_list_incl_masks.append(en)
            for series_name in channel_list_incl_masks:
                # search for exact filename if fileformat is specified
                if (self.format != None) and not (self.format == "dcm"):
                    fn = series_name + "." + self.format
                    if not (fn in os.listdir(os.path.join(self.root_dir, ident))):
                        chnlst_complete = False
                else:
                    # otherwise search for filename without filetype extension
                    ident_series_list = [
                        f.split(".")[0]
                        for f in os.listdir(os.path.join(self.root_dir, ident))
                    ]
                    if not (series_name in ident_series_list):
                        chnlst_complete = False
            if chnlst_complete:
                if (self.exclusions != None) and (ident in self.exclusions):
                    self.groups["_excluded"].append(ident)
                else:
                    if ident in self.labels.keys():
                        self.groups["base"].append(ident)
                    else:
                        self.groups["no_label"].append(ident)
            else:
                self.groups["_incomplete_data"].append(ident)

        # sort groups  so random shuffle is reproducible if same random seed is used
        for gr in self.groups.keys():
            self.groups[gr].sort()

        # set default mode
        if self.labels_file == None:
            self.sampling_group = "no_label"
        else:
            self.sampling_group = "base"
        self.eval_mode = False

        if do_crossval:
            assert not do_split, "do_crossval and do split are mutually exclusive"
            self.crossval_split("base", folds, seed=seed)

        if do_split:
            assert not do_crossval, "do_crossval and do split are mutually exclusive"
            assert not split_fraction == None, "need to specify a split_fraction"
            self.rand_split("base", "train", "val", fraction=split_fraction, seed=seed)
         
        if get_split:
            assert not do_split, "another split method flagged true"
            assert not do_crossval, "another split method flagged true"
            self.get_split("base", "train", "val", train=train_list, validation=val_list)

    def __len__(self):
        return len(self.groups[self.sampling_group])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ident = self.groups[self.sampling_group][idx]
        sample_data = self.load_sample(ident)
        img_data = sample_data[0]
        img_meta = sample_data[1]

        if ident in self.labels.keys():
            label = self.labels[ident]
        else:
            label = np.array([0.0])

        sample = {
            "data": img_data,
            "labels": label,
            "_idents": ident,
            "_meta": img_meta,
        }

        if self.mask_series != None:
            for spec_ser in self.mask_series.keys():
                if os.path.isdir(os.path.join(self.root_dir, ident)):
                    ld = self.load_series(
                        ident, self.mask_series[spec_ser], self.format
                    )
                    sample[spec_ser] = ld[0]
                    sample["_meta"][spec_ser] = ld[1]

        if self.eval_mode:
            if self.eval_transform:
                sample = self.eval_transform(sample)
        else:
            if self.transform:
                sample = self.transform(sample)

        for entry in self.channel_list:
            if not torch.is_tensor(sample["data"][entry]):
                sample["data"][entry] = torch.from_numpy(sample["data"][entry])

        sample["data"] = torch.stack([sample["data"][i] for i in self.channel_list])

        return sample

    def load_sample(self, ident):
        ret_dat = {}
        ret_meta = OrderedDict()

        if os.path.isdir(os.path.join(self.root_dir, ident)):
            for ser in self.channel_list:
                ld = self.load_series(ident, ser, self.format)
                ret_dat[ser] = ld[0]
                ret_meta[ser] = ld[1]
        else:
            raise FileNotFoundError(
                "Directory "
                + str(os.path.join(self.root_dir, ident))
                + " doesn't exist."
            )

        return ret_dat, ret_meta

    def load_series(self, ident, ser, fmt):
        ret_dat = None
        ret_meta = None
        fmt_ = None

        if fmt == None:
            if os.path.isfile(os.path.join(self.root_dir, ident, ser + ".npy")):
                fmt_ = "npy"
            elif os.path.isfile(os.path.join(self.root_dir, ident, ser + ".nii.gz")):
                fmt_ = "nii.gz"
            elif os.path.isdir(os.path.join(self.root_dir, ident, ser)):
                fmt_ = "dcm"
            else:
                raise FileNotFoundError(
                    str(os.path.join(self.root_dir, ident, ser)) + " doesn't exist."
                )
        else:
            fmt_ = str(fmt)

        if fmt_ == "npy":
            ret_dat = np.load(os.path.join(self.root_dir, ident, ser + ".npy"))
            ret_meta = {}

        elif fmt_ == "nii.gz":
            nifti = nib.load(os.path.join(self.root_dir, ident, ser + ".nii.gz"))
            ret_dat = nifti.get_fdata()
            hdr_dict = {}
            for key in nifti.header:
                hdr_dict[key] = str(nifti.header[key])
            ret_meta = {"nifti_header": hdr_dict}

        elif fmt_ == "dcm":
            slices = []
            pixel_spacing = None
            slice_thickness = None
            spacing_between_slices = None
            i = 0
            for dcm_file in sorted(os.listdir(os.path.join(self.root_dir, ident, ser))):
                if dcm_file.endswith(".dcm"):
                    dcm = pydicom.dcmread(
                        os.path.join(self.root_dir, ident, ser, dcm_file), force=True
                    )
                    if hasattr(dcm, "pixel_array"):
                        if hasattr(dcm, "SliceLocation"):
                            slices.append((dcm.SliceLocation, dcm.pixel_array))
                        else:
                            slices.append((i, dcm.pixel_array))

                        if hasattr(dcm, "PixelSpacing"):
                            if pixel_spacing == None:
                                pixel_spacing = dcm.PixelSpacing
                            else:
                                if dcm.PixelSpacing != pixel_spacing:
                                    raise Warning(
                                        "Pixel spacings between dcm files of the same series differ! "
                                        + ident
                                        + ", ser"
                                    )

                        if hasattr(dcm, "SliceThickness"):
                            if slice_thickness == None:
                                slice_thickness = dcm.SliceThickness
                            else:
                                if dcm.SliceThickness != slice_thickness:
                                    raise Warning(
                                        "Slice thickness between dcm files of the same series differ! "
                                        + ident
                                        + ", ser"
                                    )
                        if hasattr(dcm, "SpacingBetweenSlices"):
                            if spacing_between_slices == None:
                                spacing_between_slices = dcm.SpacingBetweenSlices
                            else:
                                if dcm.SpacingBetweenSlices != spacing_between_slices:
                                    raise Warning(
                                        "Spacing between slices between dcm files of the same series differ! "
                                        + ident
                                        + ", ser"
                                    )
                i += 1

            slices = sorted(slices)
            dat = []
            for s in slices:
                dat.append(s[1])
            ret_dat = np.array(dat, np.float64)
            ret_dat = np.swapaxes(ret_dat, 0, 2)
            ret_dat = np.array(np.flip(ret_dat, axis=1))
            ret_meta = {
                "dicom_attributes": {
                    "pixel_spacing": pixel_spacing,
                    "slice_thickness": slice_thickness,
                    "spacing_between_slices": spacing_between_slices,
                }
            }

        return ret_dat, ret_meta

    def sample_from(self, group):
        if group in self.groups.keys():
            self.sampling_group = group
        else:
            raise Exception("Group " + group + " doesn´t exist.")

    def rand_split(self, orig_name, new_name_1, new_name_2, fraction=0.8, seed=None):
        """
        Randomly split group in two new groups with given names.
        Containing given by fractions of the original length.

        Arguments
        ---------
        orig_name : str
            The name of the original group from where to split.
        new_name_1 : str
            New name of first group.
        new_name_2 : str
            New name of second group.
        fraction : float
            Split fraction. Fhe first group of the newly created ones
            will have itemcount(originial_group) * fraction items.
            The second group gets the remaining.
            0 < fraction < 1.
        seed : int or None
            Optional seed for python RNG. For same seed, the split,
            if done again, will have same items in same order.

        """
        # get a sorted list of the group to split
        orig_group_list = sorted(list(self.groups[orig_name]))

        # random shuffle
        if not (seed == None):
            random.seed(seed)

        random.shuffle(orig_group_list)

        count_new_1 = int(round(fraction * len(orig_group_list)))

        # create new groups
        self.groups[new_name_1] = []
        self.groups[new_name_2] = []

        # fill first new group and delete inserted
        # items in original group
        for i in range(count_new_1):
            item = orig_group_list[i]
            self.groups[new_name_1].append(item)

        # and delete inserted items from original groups copied list
        for i in range(count_new_1):
            del orig_group_list[0]

        # the rest goes in the second new group
        for item in orig_group_list:
            self.groups[new_name_2].append(item)

    def crossval_split(self, group, folds, seed=None):
        # get a sorted list of the group to split
        orig_group_list = sorted(list(self.groups[group]))

        # random shuffle
        if not (seed == None):
            random.seed(seed)

        random.shuffle(orig_group_list)

        for i in range(folds):

            train_list = []
            val_list = []

            # first part of training group
            for j in range(i * (len(orig_group_list) // folds)):
                train_list.append(orig_group_list[j])

            # second part of training group
            start = (i + 1) * (len(orig_group_list) // folds)
            for j in range(
                len(orig_group_list) - (i + 1) * (len(orig_group_list) // folds)
            ):
                train_list.append(orig_group_list[start + j])

            # validation group
            start_val = i * (len(orig_group_list) // folds)
            for j in range(len(orig_group_list) // folds):
                val_list.append(orig_group_list[start_val + j])

            # create new groups
            self.groups["train" + str(i)] = sorted(train_list)
            self.groups["val" + str(i)] = sorted(val_list)

    def get_class_weights(self, group):

        if self.labels == None:
            raise Exception("This function is only available, if labels were given.")

        labels = []

        for item in self.groups[group]:
            labels.append(self.labels[item])

        class_weights = []

        for class_index in range(len(labels[0])):
            labels_for_class = [labels[i][class_index] for i in range(len(labels))]
            weight_for_class = compute_class_weight(
                "balanced", classes=np.unique(labels_for_class), y=labels_for_class
            )
            class_weights.append(weight_for_class[1])

        return class_weights

    def print_summary(self):
        print("Volume dataset")
        print("")
        print("root_dir:", self.root_dir)
        print("labels filepath:", self.labels_file)
        print("")

        if self.labels == None:
            for group in self.groups:
                print(
                    "Group '", group, "'", "with length:", len(self.groups[group]), "\n"
                )
            print(len(self.groups), "group(s) in total. No labels given.")
        else:
            print("\nLabel counts for individual groups:")

            for group in self.groups:

                if group[0] == "_" and len(self.groups[group]) == 0:
                    continue

                print(group)

                if group != "no_label":

                    counts_no_label = 0
                    counts = {}

                    for label in set(tuple(l) for l in self.labels.values()):
                        counts[label] = 0

                    for item in self.groups[group]:
                        if item in self.labels.keys():
                            counts[tuple(self.labels[item])] += 1
                        else:
                            counts_no_label += 1

                    sorted_keys = list(counts.keys())
                    sorted_keys.sort(key=lambda x: x.index(1.0))

                    for key in sorted_keys:
                        print(key, ":", counts[key])
                    if counts_no_label > 0:
                        print("without label:", counts_no_label)

                print("Items in this group:", len(self.groups[group]), "\n")

    def print_group_entries(self, group):
        for entry in self.groups[group]:
            print(entry)
            
    def get_split(self, orig_name, new_name_1, new_name_2, train, validation):
        """
        Splits a list of patient identifiers (or numbers) into train and validation based on input lists.

        """
        
        all_keys_sorted = list(self.groups[orig_name]) 
        self.groups[new_name_1] = list(set(self.groups[orig_name])&set(train))
        self.groups[new_name_2] = list(set(self.groups[orig_name])&set(validation))
        
