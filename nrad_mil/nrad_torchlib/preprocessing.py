"""                                                                                                            
:AUTHOR: Hagen Meredig                                                                                         
:ORGANIZATION: Department of Neuroradiology, Heidelberg Univeristy Hospital                                    
:CONTACT: Hagen.Meredig@med.uni-heidelberg.de                                                                  
:SINCE: August 18, 2021                                                                            
""" 
import os
import numpy as np
import nibabel as nib
import multiprocessing as mp
from copy import copy
from datetime import datetime
from skimage.transform import resize


def _change_spacing(data, original_spacing, min_spacing, order=3):

    target_spacing = (min_spacing,) * len(original_spacing)
    scale = (np.array(original_spacing) / np.array(target_spacing)).astype(float)
    new_shape = np.round(scale * data.shape).astype(int)

    reshaped = resize(data, new_shape, order=order, cval=0.0)

    return reshaped


def _respace_series(
    dataset_path,
    folder,
    series,
    mask,
    original_spacing,
    data_mask_name,
    lesion_mask_name,
    min_spacing,
    out_path,
):

    data = nib.load(os.path.join(dataset_path, folder, series)).get_fdata()

    # crop to mask
    nonzero = np.nonzero(mask > 0)
    nonzero = np.array([[np.min(i), np.max(i)] for i in nonzero])

    data = np.array(
        data[
            nonzero[0, 0] : nonzero[0, 1] + 1,
            nonzero[1, 0] : nonzero[1, 1] + 1,
            nonzero[2, 0] : nonzero[2, 1] + 1,
        ]
    )

    if series in (data_mask_name, lesion_mask_name):
        data = _change_spacing(data, original_spacing, min_spacing, order=0)
    else:
        data = _change_spacing(data, original_spacing, min_spacing, order=3)
        # TBD: inculde modality; if MR std-mean and clip, if CT, normalization must be performed with abs values

    np.save(
        os.path.join(
            out_path, folder, series.replace(".nii", "").replace(".gz", ".npy"),
        ),
        data,
    )


def _respace_folder(args):
    (
        dataset_path,
        folder,
        original_spacing,
        data_mask_name,
        lesion_mask_name,
        min_spacing,
        out_path,
    ) = args

    if not os.path.isdir(os.path.join(out_path, folder)):
        os.makedirs(os.path.join(out_path, folder))

    mask = nib.load(os.path.join(dataset_path, folder, data_mask_name)).get_fdata()
    for series in os.listdir(os.path.join(dataset_path, folder)):
        if series.endswith(".nii.gz"):
            _respace_series(
                dataset_path,
                folder,
                series,
                mask,
                original_spacing,
                data_mask_name,
                lesion_mask_name,
                min_spacing,
                out_path,
            )
    print(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{folder} done.",
    )


class Preprocessor(object):
    def __init__(
        self,
        modality: str,
        dataset_path: str,
        data_format: str,
        out_path: str,
        data_mask_name: str = None,
        lesion_mask_name: str = None,
        num_proc: int = None,
    ):

        self.modality = modality
        self.dataset_path = dataset_path
        self.data_format = data_format
        self.out_path = out_path
        self.data_mask_name = data_mask_name
        self.lesion_mask_name = lesion_mask_name
        self.num_proc = num_proc
        self.dataset_dict = {}
        self.spacing_discrep_list = []

    def get_min_spacing(self):

        min_spacing = None

        for folder in [
            i
            for i in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, i))
        ]:
            if (
                self.data_format == "nifti"
                or self.data_format == "nii"
                or self.data_format == "nii.gz"
            ):
                prev_spacing = None

                for series in os.listdir(os.path.join(self.dataset_path, folder)):
                    if not (series.endswith(".nii.gz") or series.endswith(".nii")):
                        continue

                    nifti = nib.load(os.path.join(self.dataset_path, folder, series))

                    hdr_dict = {}
                    for key in nifti.header:
                        hdr_dict[key] = str(nifti.header[key])

                    if "pixdim" in hdr_dict:
                        pixdim_list_raw = hdr_dict["pixdim"].split(" ")
                        pixdim_list = []
                        for entry in pixdim_list_raw:
                            entry = entry.replace("[", "")
                            entry = entry.replace("]", "")
                            if entry != "":
                                pixdim_list.append(float(entry))

                        spacing = tuple(pixdim_list[1:4])

                        if prev_spacing == None:
                            prev_spacing = copy(spacing)
                        else:
                            if not spacing == prev_spacing:
                                if not folder in self.spacing_discrep_list:
                                    self.spacing_discrep_list.append(folder)

                        if not folder in self.dataset_dict.keys():
                            self.dataset_dict[folder] = {}

                        self.dataset_dict[folder]["spacing"] = spacing

                        if min_spacing == None:
                            min_spacing = min(spacing)
                        else:
                            min_spacing = min(min_spacing, min(spacing))
                    else:
                        raise Exception("No pixdim info in nifti header.")

        if len(self.spacing_discrep_list) > 0:
            for entry in self.spacing_discrep_list:
                print(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f"Warning: Series for sample{entry} have discrepant spacing entries!\n"
                    f"Will be skipped in further preprocessing steps.",
                )

        self.min_spacing = min_spacing

    def respace_dataset(self):

        args_for_respace = []
        for folder in [
            i
            for i in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, i))
            and not i in self.spacing_discrep_list
            and i in self.dataset_dict.keys()
        ]:
            if (
                self.data_format == "nifti"
                or self.data_format == "nii"
                or self.data_format == "nii.gz"
            ):

                original_spacing = self.dataset_dict[folder]["spacing"]

                args_for_respace.append(
                    (
                        self.dataset_path,
                        folder,
                        original_spacing,
                        self.data_mask_name,
                        self.lesion_mask_name,
                        self.min_spacing,
                        self.out_path,
                    )
                )

        number_of_processes = self.num_proc
        if number_of_processes == None:
            number_of_processes = mp.cpu_count()

        pool = mp.Pool(processes=number_of_processes)
        pool.map(_respace_folder, args_for_respace)

    def get_max_sizes(self):

        self.max_lesion_size = None
        self.max_data_size = None

        for folder in [
            i
            for i in os.listdir(self.out_path)
            if os.path.isdir(os.path.join(self.out_path, i))
        ]:
            for series in os.listdir(os.path.join(self.out_path, folder)):

                if not series == self.lesion_mask_name.replace(".gz", "").replace(
                    ".nii", ".npy"
                ):
                    continue

                print(folder)
                data = np.load(os.path.join(self.out_path, folder, series))

                if self.max_lesion_size == None:
                    self.max_lesion_size = [0.0 for i in data.shape]

                if self.max_data_size == None:
                    self.max_data_size = [0.0 for i in data.shape]

                nonzero = np.nonzero(data > 0)
                lesion_size = np.array([np.min(i) - np.max(i) for i in nonzero])
                for e in range(len(lesion_size)):
                    if lesion_size[e] > self.max_lesion_size[e]:
                        self.max_lesion_size[e] = lesion_size[e]

                for e in range(len(data.shape)):
                    if data.shape[e] > self.max_data_size[e]:
                        self.max_data_size[e] = data.shape[e]

    def run(self):

        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"Starting preprocessing of {self.dataset_path}...",
        )

        self.get_min_spacing()

        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"Found min spacing of {self.min_spacing}. Changing spacing...",
        )

        self.respace_dataset()

        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"Changed spacing to {self.min_spacing} iso and saved to {self.out_path}.\n"
            f"Getting max sizes...",
        )

        self.get_max_sizes()

        print(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"Done Preprocessing.\n"
            f"Max data size: {self.max_data_size}.\n"
            f"Max lesion size: {self.max_lesion_size}.",
        )

