"""                                                                                                            
:AUTHOR: Hagen Meredig                                                                                         
:ORGANIZATION: Department of Neuroradiology, Heidelberg Univeristy Hospital                                    
:CONTACT: Hagen.Meredig@med.uni-heidelberg.de                                                                  
:SINCE: August 18, 2021                                                                            
""" 

import torch
import numpy as np
from skimage.transform import resize
from scipy.ndimage import affine_transform
from itertools import starmap


def affine_rotation_matrix(rotation, dtype=np.float32):
    """Creates affine rotation matrix in 2 or 3 dimensions."""

    assert (
        type(rotation) == float or len(rotation) == 3
    ), "only defined for 2 or 3 dimensions"

    if type(rotation) == float:
        theta = rotation

        rot_mat = [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]

        final = np.array(rot_mat, dtype)

    elif len(rotation) == 3:
        alpha, beta, gamma = rotation

        # rotation 3x3
        rot_alpha = [
            [np.cos(alpha), -np.sin(alpha), 0.0],
            [np.sin(alpha), np.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ]

        rot_beta = [
            [np.cos(beta), 0.0, np.sin(beta)],
            [0.0, 1.0, 0.0],
            [-np.sin(beta), 0.0, np.cos(beta)],
        ]

        rot_gamma = [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(gamma), -np.sin(gamma)],
            [0.0, np.sin(gamma), np.cos(gamma)],
        ]

        rot_alpha = np.array(rot_alpha, dtype)
        rot_beta = np.array(rot_beta, dtype)
        rot_gamma = np.array(rot_gamma, dtype)

        rot_interm = np.matmul(rot_alpha, rot_beta)
        rot_mat_3 = np.matmul(rot_interm, rot_gamma)

        # to rotation affine
        final = np.zeros((4, 4), dtype)

        final[:3, :3] = rot_mat_3
        final[3, 3] = 1.0

    return final


def affine_shear_matrix(shear, dtype=np.float32):
    """Creates affine shear matrix in 2 or 3 dimensions."""

    assert len(shear) == 2 or len(shear) == 3, "only defined for 2 or 3 dimensions"

    if len(shear) == 2:
        shearx, sheary = shear

        shear_mat = [[1.0, shearx, 0.0], [sheary, 1.0, 0.0], [0.0, 0.0, 1.0]]

        final = np.array(shear_mat, dtype)

    elif len(shear) == 3:

        shearxy, shearxz, shearyz = shear
        shearxy_x, shearxy_y = shearxy
        shearxz_x, shearxz_z = shearxz
        shearyz_y, shearyz_z = shearyz

        shear_mat = [
            [1.0, shearyz_z, shearxz_z, 0.0],
            [shearyz_y, 1.0, shearxy_y, 0.0],
            [shearxz_x, shearxy_x, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        final = np.array(shear_mat, dtype)

    return final


def affine_translation_matrix(translation, dtype=np.float32):
    """Creates affine translation matrix in 2 or 3 dimensions."""

    assert (
        len(translation) == 2 or len(translation) == 3
    ), "only defined for 2 or 3 dimensions"

    if len(translation) == 2:
        tx, ty = translation

        trans_mat = [[1.0, 0.0, ty], [0.0, 1.0, tx], [0.0, 0.0, 1.0]]

        final = np.array(trans_mat, dtype)

    elif len(translation) == 3:
        tx, ty, tz = translation

        trans_mat = [
            [1.0, 0.0, 0.0, tz],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tx],
            [0.0, 0.0, 0.0, 1.0],
        ]

        final = np.array(trans_mat, dtype)

    return final


def affine_scaling_matrix(scale, dtype=np.float32):
    """Creates affine scaling matrix in 2 or 3 dimensions."""

    assert len(scale) == 2 or len(scale) == 3, "only defined for 2 or 3 dimensions"

    if len(scale) == 2:
        sx, sy = scale

        scale_mat = [[sy, 0.0, 0.0], [0.0, sx, 0.0], [0.0, 0.0, 1.0]]

    elif len(scale) == 3:
        sx, sy, sz = scale

        scale_mat = [
            [sz, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sx, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    final = np.array(scale_mat, dtype)

    return final


def rand_value(val_range, val_zero):
    """Compute random value in val_range around val_zero."""

    return val_zero - val_range * 0.5 + val_range * np.random.uniform()


class SpatialTransform(object):
    def __init__(
        self,
        rotation=(0.0, 0.0),
        shear=((0.0, 0.0), (0.0, 0.0)),
        translation=((0.0, 0.0), (0.0, 0.0)),
        p_rot=1.0,
        p_shear=1.0,
        p_tra=1.0,
    ):
        """
        Spatial transforms using affine matrices, linear interpolation.

        Arguments
                ---------
                rotation : tuple of length 2 (2D) or tuple of 3 tuples of length 2 (3D)
                        Specifies rotation range and offset for rotation along midpoint of 2D image
            or along x, y and z axis respectively.
        shear : tuple of 2 tuples of lengt 2 or tuple of 3 tuples of 2 tuples of length 2.
            Principle of (range, offset) notation applies here as for rotation.
            Example in 2D:
            shear = ((0., 0.), (0., 0.)) -> (shear in x-direction in xy-plane, shear in y-direction in xy-plane)
            Example in 3D:
            shear = (((0., 0.), (0., 0.)), -> (shear in x-direction in xy-plane, shear in y-direction in xy-plane)
                     ((0., 0.), (0., 0.)), -> (shear in x-direction in xz-plane, shear in z-direction in xz-plane)
                     ((0., 0.), (0., 0.))) -> (shear in y-direction in yz-plane, shear in z-direction in yz-plane)
        translation : tuple of 2 tuples of length 2 or tuple of 3 tuples of length 2
            Translation along x and y axis (2D) or x, y and z axis (3D) in (range, offset) notation.
        p_rot : float
            Probability of rotation per sample [0., 1.]
        p_shear : float
            Probability of shear per sample [0., 1.]
        p_tra : float
            Probability of translation per sample [0., 1.]

        Returns
        -------
            The transformed input as torch tensor.
        """

        self.rotation = rotation
        self.shear = shear
        self.translation = translation
        self.p_rot = p_rot
        self.p_shear = p_shear
        self.p_tra = p_tra
        self._randomize()

    def _randomize(self):

        do_apply = False

        if len(self.rotation) == 2:
            self.current_rotation = rand_value(*self.rotation)
            self.current_shear = tuple(map(lambda x: rand_value(*x), self.shear))
            self.affine_matrix = torch.eye(3)
        else:
            self.current_rotation = tuple(map(lambda x: rand_value(*x), self.rotation))
            self.current_shear = tuple(
                map(lambda x: (rand_value(*x), rand_value(*x)), self.shear)
            )
            self.affine_matrix = torch.eye(4)

        self.current_translation = tuple(
            map(lambda x: rand_value(*x), self.translation)
        )

        if np.random.uniform() < self.p_tra:
            do_apply = True
            tra_mat = torch.from_numpy(
                affine_translation_matrix(self.current_translation)
            )
            self.affine_matrix = torch.matmul(self.affine_matrix, tra_mat)

        if np.random.uniform() < self.p_rot:
            do_apply = True
            rot_mat = torch.from_numpy(affine_rotation_matrix(self.current_rotation))
            self.affine_matrix = torch.matmul(self.affine_matrix, rot_mat)

        if np.random.uniform() < self.p_shear:
            do_apply = True
            shear_mat = torch.from_numpy(affine_shear_matrix(self.current_shear))
            self.affine_matrix = torch.matmul(self.affine_matrix, shear_mat)

        return do_apply

    def __call__(self, sample):

        np.random.seed()

        if self._randomize():

            inpt = None
            normal_series_list = []
            mask_series_list = []

            for series in sample["_meta"].keys():
                if series in sample["data"].keys():
                    normal_series_list.append(series)
                    if type(sample["data"][series]) == np.ndarray:
                        sample["data"][series] = torch.from_numpy(
                            sample["data"][series]
                        )
                if series in sample.keys():
                    mask_series_list.append(series)
                    if type(sample[series]) == np.ndarray:
                        sample[series] = torch.from_numpy(sample[series])

            i = 0
            for series in normal_series_list:
                if inpt == None:
                    inpt = torch.zeros(
                        (
                            1,
                            len(normal_series_list) + len(mask_series_list),
                            *sample["data"][series].shape,
                        )
                    )

                inpt[0, i] = sample["data"][series]
                i += 1

            for series in mask_series_list:
                if inpt == None:
                    inpt = torch.zeros(
                        (
                            1,
                            len(normal_series_list) + len(mask_series_list),
                            *sample[series].shape,
                        )
                    )
                inpt[0, i] = sample[series]
                i += 1

            if len(inpt.shape) == 4:
                matrices = torch.empty((inpt.shape[0], 2, 3))
            else:
                matrices = torch.empty((inpt.shape[0], 3, 4))

            for j in range(inpt.shape[0]):
                if len(inpt.shape) == 4:
                    matrices[j] = self.affine_matrix.clone()[:2, :]
                else:
                    matrices[j] = self.affine_matrix.clone()[:3, :]

            # see official pytorch doc of these two functions for further details
            affine_grid = torch.nn.functional.affine_grid(
                matrices, inpt.size(), align_corners=False
            )
            output = torch.nn.functional.grid_sample(
                inpt, affine_grid, align_corners=False
            )

            i = 0
            for series in normal_series_list:
                sample["data"][series] = output[0, i]
                i += 1

            for series in mask_series_list:
                sample[series] = torch.round(output[0, i])
                i += 1

        return sample


class SpatialTransformScipy(object):
    def __init__(
        self,
        rotation=(0.0, 0.0),
        shear=((0.0, 0.0), (0.0, 0.0)),
        translation=((0.0, 0.0), (0.0, 0.0)),
        p_rot=1.0,
        p_shear=1.0,
        p_tra=1.0,
        order=3,
        mode="2d",
    ):
        """
        Spatial transforms using affine matrices, linear interpolation.

        Arguments
                ---------
                rotation : tuple of length 2 (2D) or tuple of 3 tuples of length 2 (3D)
                        Specifies rotation range and offset for rotation along midpoint of 2D image
            or along x, y and z axis respectively.
        shear : tuple of 2 tuples of lengt 2 or tuple of 3 tuples of 2 tuples of length 2.
            Principle of (range, offset) notation applies here as for rotation.
            Example in 2D:
            shear = ((0., 0.), (0., 0.)) -> (shear in x-direction in xy-plane, shear in y-direction in xy-plane)
            Example in 3D:
            shear = (((0., 0.), (0., 0.)), -> (shear in x-direction in xy-plane, shear in y-direction in xy-plane)
                     ((0., 0.), (0., 0.)), -> (shear in x-direction in xz-plane, shear in z-direction in xz-plane)
                     ((0., 0.), (0., 0.))) -> (shear in y-direction in yz-plane, shear in z-direction in yz-plane)
        translation : tuple of 2 tuples of length 2 or tuple of 3 tuples of length 2
            Translation along x and y axis (2D) or x, y and z axis (3D) in (range, offset) notation.
        p_rot : float
            Probability of rotation per sample [0., 1.]
        p_shear : float
            Probability of shear per sample [0., 1.]
        p_tra : float
            Probability of translation per sample [0., 1.]

        Returns
        -------
            The transformed input as torch tensor.
        """

        self.rotation = rotation
        self.shear = shear
        self.translation = translation
        self.p_rot = p_rot
        self.p_shear = p_shear
        self.p_tra = p_tra
        self.order = order
        self.mode = mode
        self._randomize()

    def _randomize(self):

        do_apply = False

        if self.mode == "2d":
            assert len(self.rotation) == len(self.translation) == len(self.shear) == 2
            self.current_rotation = rand_value(*self.rotation)
            self.current_shear = tuple(map(lambda x: rand_value(*x), self.shear))
            self.affine_matrix = np.eye(3)
        elif self.mode == "3d":
            assert len(self.rotation) == len(self.translation) == len(self.shear) == 3
            self.current_rotation = tuple(map(lambda x: rand_value(*x), self.rotation))
            self.current_shear = tuple(
                map(lambda x: (rand_value(*x[0]), rand_value(*x[1])), self.shear)
            )
            self.affine_matrix = np.eye(4)

        self.current_translation = tuple(
            map(lambda x: rand_value(*x), self.translation)
        )

        if np.random.uniform() < self.p_tra:
            do_apply = True
            tra_mat = affine_translation_matrix(self.current_translation)
            self.affine_matrix = np.matmul(self.affine_matrix, tra_mat)

        if np.random.uniform() < self.p_rot:
            do_apply = True
            rot_mat = affine_rotation_matrix(self.current_rotation)
            self.affine_matrix = np.matmul(self.affine_matrix, rot_mat)

        if np.random.uniform() < self.p_shear:
            do_apply = True
            shear_mat = affine_shear_matrix(self.current_shear)
            self.affine_matrix = np.matmul(self.affine_matrix, shear_mat)

        return do_apply

    def __call__(self, sample):

        np.random.seed()

        if self._randomize():

            for series in sample["_meta"].keys():
                if series in sample["data"].keys():
                    transl0 = affine_translation_matrix(
                        [e // 2 for e in sample["data"][series].shape]
                    )
                    transl1 = affine_translation_matrix(
                        [-e // 2 for e in sample["data"][series].shape]
                    )
                    mat_final = np.matmul(
                        np.matmul(transl0, self.affine_matrix), transl1
                    )
                    sample["data"][series] = affine_transform(
                        sample["data"][series], mat_final, order=self.order
                    )
                    sample["data"][series] = torch.from_numpy(sample["data"][series])

                if series in sample.keys():
                    transl0 = affine_translation_matrix(
                        [e // 2 for e in sample[series].shape]
                    )
                    transl1 = affine_translation_matrix(
                        [-e // 2 for e in sample[series].shape]
                    )
                    mat_final = np.matmul(
                        np.matmul(transl0, self.affine_matrix), transl1
                    )
                    sample[series] = affine_transform(
                        sample[series],
                        mat_final,
                        order=0,
                    )
                    sample[series] = torch.from_numpy(sample[series])

        return sample


class ContrastTransform(object):
    def __init__(
        self,
        c_range=(0.8, 1.2),
        b_range=(0.0, 0.0),
        preserve_range=True,
        p_per_series=0.2,
        mask_name: any = None,
    ):
        """
        Contrast transform.

        Arguments
        ---------
        c_range : tuple of length 2
                    Specifies contrast range.
        b_range : tuple of length 2
                    Specifies brightness range as fractions of standard deviation.
        preserve_range : bool
            Preserve range of absolute values.
        p_per_series : float
            Probability of tranform per series [0., 1.]

        Returns
        -------
            Transformed input as torch tensor
        """
        self.c_range = c_range
        self.b_range = b_range
        self.preserve_range = preserve_range
        self.p_per_series = p_per_series
        self.mask_name = mask_name
        self._randomize()

    def _randomize(self):

        self.current_contrast = (
            self.c_range[0] + (self.c_range[1] - self.c_range[0]) * np.random.uniform()
        )
        self.current_brightness = (
            self.b_range[0] + (self.b_range[1] - self.b_range[0]) * np.random.uniform()
        )

    def __call__(self, sample):

        np.random.seed()

        for series in sample["_meta"]:
            inpt = None

            if series in sample["data"].keys():
                inpt = sample["data"][series]
            else:
                continue

            if type(inpt) == np.ndarray:
                inpt = torch.from_numpy(inpt)

            if np.random.uniform() < self.p_per_series:
                self._randomize()

                if self.mask_name != None:
                    mask = sample[self.mask_name]
                    the_range = inpt[mask > 0].min(), inpt[mask > 0].max()
                    std = inpt[mask > 0].std()
                    output = (
                        inpt * self.current_contrast
                    ) + self.current_brightness * std
                    if self.preserve_range:
                        output = output.clamp(*the_range)
                else:
                    the_range = inpt.min(), inpt.max()
                    std = inpt.std()
                    output = (
                        inpt * self.current_contrast
                    ) + self.current_brightness * std
                    if self.preserve_range:
                        output = output.clamp(*the_range)

                sample["data"][series] = output

        return sample


class ChangeSpacingTransform(object):
    """
    Change the spacing of the given data.

        Arguments
        ---------
        data : numpy.ndarray
                A numpy array of shape (x, y, z).
        original_spacing : tuple
                A tuple containing the original spacing of the data; (x, y, z).
        target_spacing : tuple
                A tuple containing the target spacing: (x, y, z).
        order : int
                Orderof interpolation ;  parameter passed to the resize function.
                Defaults to 3.

        Returns
        -------
        torch.Tensor
                The resized data with new spacing.
    """

    def __init__(self, target_spacing, source_spacing=None, mask_names=[], order=3):
        assert isinstance(target_spacing, (int, tuple, list))
        self.target_spacing = target_spacing
        self.source_spacing = source_spacing
        self.mask_names = mask_names

    def __call__(self, sample):
        img_data, meta = sample["data"], sample["_meta"]

        for series_name in meta.keys():
            original_spacing = None

            if self.source_spacing:
                original_spacing = self.source_spacing

            elif "spacing" in meta[series_name].keys():
                original_spacing = meta[series_name]["spacing"]

            elif "nifti_header" in meta[series_name].keys():
                if "pixdim" in meta[series_name]["nifti_header"]:
                    pixdim_list_raw = meta[series_name]["nifti_header"]["pixdim"].split(
                        " "
                    )
                    pixdim_list = []
                    for entry in pixdim_list_raw:
                        entry = entry.replace("[", "")
                        entry = entry.replace("]", "")
                        if entry != "":
                            pixdim_list.append(float(entry))

                    original_spacing = tuple(pixdim_list[1:4])
                else:
                    raise Exception("No pixdim info in nifti header.")

            elif "dicom_attributes" in meta[series_name].keys():
                pixel_spacing = [
                    float(i)
                    for i in meta[series_name]["dicom_attributes"]["pixel_spacing"]
                ]
                spacing_between_slices = float(
                    meta[series_name]["dicom_attributes"]["spacing_between_slices"]
                )
                original_spacing = (
                    pixel_spacing[1],
                    pixel_spacing[0],
                    spacing_between_slices,
                )
            else:
                raise Exception("No spacing information of existing data available.")

            if series_name in img_data.keys():
                img_data[series_name] = self._change_spacing(
                    img_data[series_name], original_spacing
                )
                img_data[series_name] = torch.from_numpy(img_data[series_name])

            if series_name in sample.keys():
                sample[series_name] = self._change_spacing(
                    sample[series_name], original_spacing
                )
                sample[series_name] = torch.from_numpy(sample[series_name])
                sample[series_name] = torch.round(sample[series_name])

            meta[series_name]["spacing"] = np.array(self.target_spacing)

        sample["data"] = img_data
        sample["_meta"] = meta

        return sample

    def _change_spacing(self, data, original_spacing, order=3):

        shape = data.shape[:3]
        new_shape = (np.array(original_spacing) / np.array(self.target_spacing)).astype(
            float
        )
        new_shape *= shape
        new_shape[:] = np.round(new_shape[:])
        new_shape = new_shape.astype(int)

        reshaped = resize(data, new_shape, order=order, cval=0.0)

        return reshaped


class ResizeTransform(object):
    """
    Resize images to given shape.

        Arguments
        ---------
    target_shape : list or tuple
                The target shape (x, y[, z]).
        order : int
                Order of interpolation;  parameter passed to the resize function.
                Defaults to 3.

    Returns
        -------
    torch.Tensor
                The resized data with new shape.
    """

    def __init__(self, target_shape, order=3):

        assert isinstance(target_shape, (tuple, list))

        self.target_shape = target_shape
        self.order = order

    def __call__(self, sample):

        img_data, meta = sample["data"], sample["_meta"]

        for series_name in meta.keys():

            if series_name in img_data.keys():
                img_data[series_name] = resize(
                    img_data[series_name], self.target_shape, order=self.order, cval=0.0
                )
                img_data[series_name] = torch.from_numpy(img_data[series_name])

            if series_name in sample.keys():
                sample[series_name] = resize(
                    sample[series_name], self.target_shape, order=self.order, cval=0.0
                )
                sample[series_name] = torch.from_numpy(sample[series_name])
                sample[series_name] = torch.round(sample[series_name])

            meta[series_name]["spacing"] = np.array([-1.0])

        sample["data"] = img_data
        sample["_meta"] = meta

        return sample


class CenterCropTransform(object):
    """
    Crop the image to a new shape. Incresing the size is possible (zero-fill).


        Arguments
        ---------
        target_shape : list or tuple
                The target shape (x, y[, z]).
        mask_name : str
                Optional name of a mask to use as center for cropping.
        If not supplied the image is cropped to the image center (ie shape // 2).

    Returns
        -------
    torch.Tensor
                The cropped data with new shape.

    """

    def __init__(self, target_shape, mask_name=None):

        assert isinstance(target_shape, (int, tuple, list))

        self.target_shape = target_shape
        self.mask_name = mask_name

    def __call__(self, sample):

        img_data, meta = sample["data"], sample["_meta"]

        center = None
        if self.mask_name:
            mask = sample[self.mask_name]
            if type(mask) == torch.Tensor:
                mask = mask.numpy()
            if np.any(mask):
                # find center of mask
                nonzero_ = np.nonzero(mask)
                nonzero = [[i.min(), i.max()] for i in nonzero_]
                midpoint = []
                for i in range(len(nonzero)):
                    midpoint.append(
                        nonzero[i][0] + ((nonzero[i][1] - nonzero[i][0]) // 2)
                    )
                center = np.array(midpoint)

        for series_name in meta.keys():

            if series_name in img_data.keys():
                img_data[series_name] = self._center_crop_to_target_shape(
                    img_data[series_name], center
                )

            if series_name in sample.keys():
                sample[series_name] = self._center_crop_to_target_shape(
                    sample[series_name], center
                )

        sample["data"] = img_data
        sample["_meta"] = meta

        return sample

    def _center_crop_to_target_shape(self, data, center):
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)
        orig_shape = data.shape
        if type(center) == type(None):
            center = np.array(orig_shape) // 2

        ret = torch.zeros(self.target_shape)

        start_src = []
        for i in range(len(orig_shape)):
            start_src.append(center[i] - (self.target_shape[i] // 2))

        end_src = []
        for i in range(len(orig_shape)):
            end_src.append(center[i] + (self.target_shape[i] // 2))

        start_dst = []
        for i in range(len(orig_shape)):
            if start_src[i] < 0:
                start_dst.append(-start_src[i])
                start_src[i] = 0
            else:
                start_dst.append(0)

        end_dst = []
        for i in range(len(orig_shape)):
            if end_src[i] > orig_shape[i]:
                end_dst.append(self.target_shape[i] - (end_src[i] - orig_shape[i]))
                end_src[i] = orig_shape[i]
            else:
                end_dst.append(self.target_shape[i])

        if len(orig_shape) == 2:
            ret[start_dst[0] : end_dst[0], start_dst[1] : end_dst[1]] = data[
                start_src[0] : end_src[0], start_src[1] : end_src[1]
            ]
        if len(orig_shape) == 3:
            ret[
                start_dst[0] : end_dst[0],
                start_dst[1] : end_dst[1],
                start_dst[2] : end_dst[2],
            ] = data[
                start_src[0] : end_src[0],
                start_src[1] : end_src[1],
                start_src[2] : end_src[2],
            ]

        return ret


class MaskTransform(object):
    """
    Mask images.

        Arguments
        ---------
        mask_name: str
        Name of the mask.

        Returns
        -------
    torch.Tensor
        The original images with values outside the mask region set to zero.

    """

    def __init__(self, mask_name):
        self.mask_name = mask_name

    def __call__(self, sample):
        img_data, meta = sample["data"], sample["_meta"]

        mask = sample[self.mask_name]

        for series_name in meta.keys():

            if series_name in img_data.keys():
                img_data[series_name][mask == 0] = 0.0

            if series_name in sample.keys():
                sample[series_name][mask == 0] = 0.0

        sample["data"] = img_data
        sample["_meta"] = meta

        return sample


class ExtractSliceTransform(object):
    """
    Extract a silce from a volume

        Arguments
        ---------
    position: float
        Relative position along the given axis from where to draw the slice.
        Default 0.5, the middle.
    mask_name: str
        Optional mask name to use as a reference volume from where to draw the slice.
    randomize: bool
        Randomize the position from where to draw the slice.
        The slice is randomly drawn from within following interval:
        position +/- (img(/mask).max - img(/mask).min) * lim_frac
        Default is False.
        lim_frac: float
        See 'randomize' argument. Default is 0.5

        Returns
        -------
    torch.Tensor
        The extracted slice.

    """

    def __init__(
        self, position=0.5, mask_name=None, randomize=False, axis=2, lim_frac=0.5
    ):
        self.position = position
        self.mask_name = mask_name
        self.randomize = randomize
        assert axis in (0, 1, 2)
        self.axis = axis
        self.lim_frac = lim_frac

    def __call__(self, sample):

        img_data, meta = sample["data"], sample["_meta"]

        center = None
        minpoint = None
        maxpoint = None
        slice_pos = None

        if self.mask_name:
            mask = sample[self.mask_name]
            if type(mask) == torch.Tensor:
                mask = mask.numpy()
            if np.any(mask):
                # find center of mask
                nonzero_ = np.nonzero(mask)
                nonzero = [[i.min(), i.max()] for i in nonzero_]
                midpoint = []
                maxpoint = []
                minpoint = []
                for i in range(len(nonzero)):
                    midpoint.append(
                        nonzero[i][0] + ((nonzero[i][1] - nonzero[i][0]) // 2)
                    )
                    minpoint.append(nonzero[i][0])
                    maxpoint.append(nonzero[i][1])
                center = np.array(midpoint)
                minpoint = np.array(minpoint)
                maxpoint = np.array(maxpoint)

                if self.randomize:
                    slice_pos = int(
                        round(minpoint[self.axis])
                        + ((maxpoint[self.axis] - minpoint[self.axis]) * self.position)
                        + (
                            (maxpoint[self.axis] - minpoint[self.axis])
                            * self.lim_frac
                            * (np.random.uniform() - 0.5)
                        )
                    )

                else:
                    slice_pos = int(
                        round(
                            minpoint[self.axis]
                            + (maxpoint[self.axis] - minpoint[self.axis])
                            * self.position
                        )
                    )

        for series_name in meta.keys():

            if series_name in img_data.keys():
                img_data[series_name] = self._extract_slice(
                    img_data[series_name], center, minpoint, maxpoint, slice_pos
                )
            if series_name in sample.keys():
                sample[series_name] = self._extract_slice(
                    sample[series_name], center, minpoint, maxpoint, slice_pos
                )

        sample["data"] = img_data
        sample["_meta"] = meta

        return sample

    def _extract_slice(self, data, center, minpoint, maxpoint, slice_pos):

        orig_shape = data.shape

        if type(center) == type(None):
            center = np.array(orig_shape) // 2

        if type(minpoint) == type(None):
            minpoint = [0 for i in range(len(orig_shape))]

        if type(maxpoint) == type(None):
            maxpoint = list(orig_shape)

        if type(slice_pos) == type(None):
            slice_pos = int(
                round(
                    minpoint[self.axis]
                    + (maxpoint[self.axis] - minpoint[self.axis]) * self.position
                )
            )

        if self.axis == 0:
            ret = data[slice_pos, :, :]
        elif self.axis == 1:
            ret = data[:, slice_pos, :]
        else:
            ret = data[:, :, slice_pos]

        return ret


class To25DTransform(object):
    def __call__(self, sample):

        img_data, meta = sample["data"], sample["_meta"]

        for series_name in meta.keys():

            if series_name in img_data.keys():

                plane0 = self._extract_slice(img_data[series_name], 0)
                plane1 = self._extract_slice(img_data[series_name], 1)
                plane2 = self._extract_slice(img_data[series_name], 2)
                img_data[series_name] = torch.from_numpy(
                    np.array([plane0, plane1, plane2])
                )
            if series_name in sample.keys():
                plane0 = self._extract_slice(sample[series_name], 0)
                plane1 = self._extract_slice(sample[series_name], 1)
                plane2 = self._extract_slice(sample[series_name], 2)
                sample[series_name] = torch.from_numpy(
                    np.array([plane0, plane1, plane2])
                )

        sample["data"] = img_data
        sample["_meta"] = meta

        return sample

    def _extract_slice(self, data, axis):

        orig_shape = data.shape

        minpoint = [0 for i in range(len(orig_shape))]

        maxpoint = list(orig_shape)

        slice_pos = int(round(minpoint[axis] + (maxpoint[axis] - minpoint[axis]) * 0.5))

        if axis == 0:
            ret = data[slice_pos, :, :]
        elif axis == 1:
            ret = data[:, slice_pos, :]
        else:
            ret = data[:, :, slice_pos]

        if isinstance(ret, np.ndarray):
            ret = np.array(ret)
        if isinstance(ret, torch.Tensor):
            ret = ret.numpy()

        return ret


class ToFloatTransform(object):
    """
    Cast to float values.

        Arguments

        Returns
        -------
    """

    def __call__(self, sample):
        img_data, meta = sample["data"], sample["_meta"]

        for series_name in meta.keys():

            if series_name in img_data.keys():
                img_data[series_name] = torch.FloatTensor(img_data[series_name])

            if series_name in sample.keys():
                sample[series_name] = torch.FloatTensor(sample[series_name])
                sample[series_name] = torch.round(sample[series_name])

        return sample


class IdentityTransform(object):
    """
    Returns the sample unaltered.

        Arguments

        Returns
        -------
    """

    def __call__(self, sample):

        return sample


class StdMeanTransform(object):
    """
    Normalize to std mean (Z-score normalization).

        Arguments
        ---------
        mask_name: str
        Optional name of a mask. Is specified only the values inside the mask volume/area
        are taken into account.

        Returns
        -------
    torch.Tensor
        The normalized images.

    """

    def __init__(self, mask_name: any = None):

        self.mask_name = mask_name

    def __call__(self, sample):

        img_data, meta = sample["data"], sample["_meta"]

        if self.mask_name != None:

            mask = sample[self.mask_name]

            for series_name in meta.keys():
                if series_name in img_data.keys():
                    mean = img_data[series_name][mask > 0.0].mean()
                    std = img_data[series_name][mask > 0.0].std()
                    img_data[series_name] -= mean
                    img_data[series_name] /= std + 1e-8
        else:
            for series_name in meta.keys():
                if series_name in img_data.keys():
                    mean = img_data[series_name].mean()
                    std = img_data[series_name].std()
                    img_data[series_name] -= mean
                    img_data[series_name] /= std + 1e-8

        sample["data"] = img_data

        return sample


class ClipTransform(object):
    """
    Clip image values.

        Arguments
        ---------
        mask_name: str
        Optional name of a mask. Is specified only the values inside the mask volume/area
        are taken into account.
    clip: float
        Float specifying the number of standard deviations of original values to clip to.
        clipped = -clip*std...0...+clip*std
    set_value: float
        Optional value to set the clipped voxels/pixels to.
        Default None. If None the values are set to the max / min allowed value.

        Returns
        -------
    torch.Tensor
        The images clipped to specified values.

    """

    def __init__(self, mask_name: any = None, clip: tuple = (1.0, 99.0)):

        self.mask_name = mask_name
        self.clip = clip

    def __call__(self, sample):

        img_data, meta = sample["data"], sample["_meta"]

        if self.mask_name != None:

            mask = sample[self.mask_name]

            if np.count_nonzero(mask) > 0:
                for series_name in meta.keys():
                    if series_name in img_data.keys():

                        clip_values = np.percentile(
                            img_data[series_name][mask > 0], self.clip
                        )
                        img_data[series_name][mask > 0] = np.clip(
                            img_data[series_name][mask > 0],
                            clip_values[0],
                            clip_values[1],
                        )
        else:
            for series_name in meta.keys():

                if series_name in img_data.keys():

                    clip_values = np.percentile(img_data[series_name], self.clip)

                    img_data[series_name] = np.clip(
                        img_data[series_name],
                        clip_values[0],
                        clip_values[1],
                    )

        sample["data"] = img_data

        return sample


class UnitNormTransform(object):
    """
    Normalize to [0..1].

        Arguments
        ---------
        mask_name: str
        Optional name of a mask. Is specified only the values inside the mask volume/area
        are taken into account.

        Returns
        -------
    torch.Tensor
        The normalized images.

    """

    def __init__(self, mask_name: any = None):

        self.mask_name = mask_name

    def __call__(self, sample):

        img_data, meta = sample["data"], sample["_meta"]

        if self.mask_name != None:

            mask = sample[self.mask_name]

            for series_name in meta.keys():
                if series_name in img_data.keys():
                    img_data[series_name][mask > 0.0] -= img_data[series_name][
                        mask > 0.0
                    ].min()
                    img_data[series_name][mask > 0.0] /= (
                        img_data[series_name][mask > 0.0].max() + 1e-8
                    )
        else:
            for series_name in meta.keys():
                if series_name in img_data.keys():
                    img_data[series_name] -= img_data[series_name].min()
                    img_data[series_name] /= img_data[series_name].max() + 1e-8

        sample["data"] = img_data

        return sample


class AddEpsilonTransform(object):
    """
    Add an epsilon.

        Arguments
        ---------
        epsilon: str
        Optional name of a mask. Is specified only the values inside the mask volume/area
        are taken into account.

        Returns
        -------
    torch.Tensor
        The images with added epsilon to all pixels/voxels.

    """

    def __init__(self, epsilon: float = 1e-8):

        self.epsilon = epsilon

    def __call__(self, sample):

        img_data, meta = sample["data"], sample["_meta"]

        for series_name in meta.keys():
            if series_name in img_data.keys():
                img_data[series_name] += self.epsilon

        sample["data"] = img_data

        return sample


class MILToBagTransform(object):
    """
    Convert image to a list (bag) of tiles of instance_shape size.

         Arguments
         ---------
         instance_shape: list/tuple
             Shape of individual instances in bag.
         overlap: float
             overlap fraction of tiles.

         Returns
         -------
     torch.Tensor
         The image as bag of tiles, each with shape of instance_shape.

    """

    def __init__(
        self,
        instance_shape: any = (28, 28),
        overlap: float = 0.5,
        _allow_constant_tiles: bool = False,
    ):

        self.instance_shape = instance_shape
        self.overlap = overlap
        self._allow_constant_tiles = _allow_constant_tiles
        self.instance_increment = tuple(
            [
                int(round(self.instance_shape[i] * (1.0 - self.overlap)))
                for i in range(len(instance_shape))
            ]
        )

    def __call__(self, sample):

        img_data, meta = sample["data"], sample["_meta"]

        for series_name in meta.keys():
            if series_name in img_data.keys():
                orig_shape = img_data[series_name].shape
                assert len(orig_shape) == len(self.instance_shape)

                tiled_dims = []
                for i in range(len(orig_shape)):
                    tiled_dims.append(orig_shape[i] // self.instance_increment[i])

                len_bag = tiled_dims[0] * tiled_dims[1]
                if len(tiled_dims) == 3:
                    len_bag *= tiled_dims[2]

                bag = torch.zeros(len_bag, *self.instance_shape)
                tile_pos = torch.zeros(len_bag, len(self.instance_shape))

                tiles = []
                positions = []

                if len(tiled_dims) == 2:
                    for i in range(tiled_dims[0]):
                        for j in range(tiled_dims[1]):

                            tile = torch.zeros(self.instance_shape)

                            start_pos = [
                                e * s for e, s in zip((i, j), self.instance_increment)
                            ]
                            end_pos = [
                                p + s for p, s in zip(start_pos, self.instance_shape)
                            ]

                            end_tile = [s for s in self.instance_shape]
                            end_tile = [
                                e
                                for e in starmap(
                                    lambda et, ep, os: et
                                    if ep < os
                                    else et - (ep - os),
                                    zip(end_tile, end_pos, orig_shape),
                                )
                            ]

                            end_pos = [
                                e
                                for e in starmap(
                                    lambda e, s: e if e <= s else s,
                                    zip(end_pos, orig_shape),
                                )
                            ]

                            img_indices = tuple(
                                starmap(
                                    lambda s, e: slice(s, e, 1),
                                    [e for e in zip(start_pos, end_pos)],
                                )
                            )
                            tile_indices = tuple(
                                map(lambda e: slice(0, e, 1), end_tile)
                            )

                            tile[tile_indices] = img_data[series_name][img_indices]

                            if self._allow_constant_tiles:
                                tiles.append(tile)
                                positions.append(torch.tensor(start_pos))
                            else:
                                if (
                                    not torch.all(tile == 0.0)
                                    and not len(torch.unique(tile)) == 1
                                ):
                                    tiles.append(tile)
                                    positions.append(torch.tensor(start_pos))

                elif len(tiled_dims) == 3:
                    for i in range(tiled_dims[0]):
                        for j in range(tiled_dims[1]):
                            for k in range(tiled_dims[2]):

                                tile = torch.zeros(self.instance_shape)

                                start_pos = [
                                    e * s
                                    for e, s in zip((i, j, k), self.instance_shape)
                                ]
                                end_pos = [
                                    (e + 1) * s
                                    for e, s in zip((i, j, k), self.instance_shape)
                                ]

                                end_tile = [s for s in self.instance_shape]
                                end_tile = [
                                    e
                                    for e in starmap(
                                        lambda et, ep, os: et
                                        if ep < os
                                        else et - (ep - os),
                                        zip(end_tile, end_pos, orig_shape),
                                    )
                                ]

                                end_pos = [
                                    e
                                    for e in starmap(
                                        lambda e, s: e if e <= s else s,
                                        zip(end_pos, orig_shape),
                                    )
                                ]

                                img_indices = tuple(
                                    starmap(
                                        lambda s, e: slice(s, e, 1),
                                        [e for e in zip(start_pos, end_pos)],
                                    )
                                )
                                tile_indices = tuple(
                                    map(lambda e: slice(0, e, 1), end_tile)
                                )

                                tile[tile_indices] = img_data[series_name][img_indices]

                                if self._allow_constant_tiles:
                                    tiles.append(tile)
                                    positions.append(torch.tensor(start_pos))
                                else:
                                    if (
                                        not torch.all(tile == 0.0)
                                        and not len(torch.unique(tile)) == 1
                                    ):
                                        tiles.append(tile)
                                        positions.append(torch.tensor(start_pos))

                for i_tile in range(len(tiles)):
                    bag[i_tile] = tiles[i_tile]
                    tile_pos[i_tile] = positions[i_tile]

                img_data[series_name] = bag
                sample["_meta"][series_name]["tile_pos"] = tile_pos
                sample["_meta"][series_name]["len_tiles"] = len(tiles)
                sample["_meta"][series_name]["orig_shape"] = torch.tensor(orig_shape)
                sample["_meta"][series_name]["tile_dim"] = torch.tensor(
                    self.instance_shape
                )

        sample["data"] = img_data

        return sample


class MILBagInstanceTransform(object):
    """
    Convert image to a list (bag) of tiles of instance_shape size.

         Arguments
         ---------
         operation: str
            Name of operation to be applied to instances (tiles).
            Possible choices: mean, min, max.

         Returns
         -------
     torch.Tensor
         The image as bag of tiles, each with shape of instance_shape.

    """

    def __init__(self, operation: str = "mean"):
        assert operation in ("mean", "min", "max"), "Unknown operation"
        self.operation = getattr(np, operation)

    def __call__(self, sample):
        img_data, meta = sample["data"], sample["_meta"]

        for series_name in meta.keys():
            if series_name in img_data.keys():
                for i in range(img_data[series_name].shape[0]):
                    img_data[series_name][i] = img_data[series_name][i].mean()

        sample["data"] = img_data

        return sample
