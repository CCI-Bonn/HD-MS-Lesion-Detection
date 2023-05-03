"""
:AUTHOR: Hagen Meredig
:ORGANIZATION: Department of Neuroradiology, Heidelberg Univeristy Hospital
:CONTACT: Hagen.Meredig@med.uni-heidelberg.de
:SINCE: August 18, 2021
"""

import torch
import numpy as np
from itertools import starmap


class GenericDataAdapter(object):
    def __init__(self, *args, **kwargs):
        self.args = args

    def __call__(self, batch):
        out = []
        for key in self.args:
            out.append(batch[key])
        return out


class MultiClassDataAdapter(object):
    def __init__(self, data_key, ground_truth_key):
        self.data_key = data_key
        self.ground_truth_key = ground_truth_key

    def __call__(self, batch):
        return batch[self.data_key], torch.max(batch[self.ground_truth_key], 1)[1]


class MultiLabelDataAdapter(object):
    def __init__(self, data_key, ground_truth_key):
        self.data_key = data_key
        self.ground_truth_key = ground_truth_key

    def __call__(self, batch):
        return batch[self.data_key], batch[self.ground_truth_key]


class AutoencoderDataAdapter(object):
    def __init__(self, data_key):
        self.data_key = data_key

    def __call__(self, batch):
        return batch[self.data_key], batch[self.data_key].detach().clone()



class MILDataAdapter(object):
    def __init__(self, data_key, ground_truth_key):
        self.data_key = data_key
        self.ground_truth_key = ground_truth_key

    def __call__(self, batch):

        assert len(batch["data"]) == 1, "only defined for batch_size = 1"

        len_bag = 0

        for series_name in batch["_meta"].keys():
            if not (series_name in batch.keys()):
                if batch["_meta"][series_name]["len_tiles"] > len_bag:
                    len_bag = batch["_meta"][series_name]["len_tiles"]

        bag = torch.swapaxes(batch["data"][:, :, : len_bag - 1], 1, 2)

        bag_dim = bag.size()
        bag_concat_chnl = (bag_dim[0],bag_dim[2]*bag_dim[1], 1,*bag_dim[3:])
        bag_out = torch.zeros(bag_concat_chnl)

        for chnl in range(bag_dim[2]):
            bag_out[:,bag_dim[1]*chnl:bag_dim[1]*(chnl+1),0,:,:,:] = bag[:,:,chnl,:,:,:]

        label = torch.argmax(batch[self.ground_truth_key][0])

        return bag_out, label



class MILDataAdapter_(object):
    def __init__(self, data_key, ground_truth_key):
        self.data_key = data_key
        self.ground_truth_key = ground_truth_key

    def __call__(self, batch):

        assert len(batch[self.data_key]) == 1, "only defined for batch_size = 1"

        len_bag = 0

        for series_name in batch["_meta"].keys():
            if not (series_name in batch.keys()):
                if batch["_meta"][series_name]["len_tiles"] > len_bag:
                    len_bag = batch["_meta"][series_name]["len_tiles"]
                    positions = batch["_meta"][series_name]["tile_pos"][0]
                    orig_shape = batch["_meta"][series_name]["orig_shape"]

        bag = torch.swapaxes(batch[self.data_key][:, :, : len_bag - 1], 1, 2)
        label = torch.argmax(batch[self.ground_truth_key][0])

        supplement = {}
        supplement["idents"] = batch["_idents"]
        supplement["positions"] = positions
        supplement["orig_shape"] = orig_shape
        supplement["segmentation"] = batch["segmentation"][0]

        return {
            "input": bag,
            "ground_truth": label,
            "supplement": supplement,
        }


class MILReportAdapter_(object):
    def __call__(
        self,
        input: any = None,
        ground_truth: any = None,
        supplement: any = None,
        output: any = None,
        loss: any = None,
    ):

        assert len(output[0]) == 1, "only defined for batch_size = 1"

        out = [[1.0 - output[0][0].item(), output[0][0].item()]]
        gt = [[1.0 - ground_truth.float().item(), ground_truth.float().item()]]
        lo = [loss.item()]
        inp = [i.tolist() for i in input.squeeze(0)]

        ret = {}
        ret["ID"] = supplement["idents"]
        ret["ground_truth"] = gt
        ret["prediction"] = out
        ret["batch_loss"] = lo
        ret["input"] = inp

        positions = supplement["positions"].detach().cpu().numpy()
        orig_shape = supplement["orig_shape"].detach().cpu().numpy()
        segmentation = supplement["segmentation"].detach().cpu().numpy()

        attention_weights = output[2][0].detach().cpu().numpy()

        out_attention = np.zeros(orig_shape[0].tolist() + [2])

        if len(input.shape) == 5:
            instance_shape = input.shape[-2:]
        elif len(input.shape) == 6:
            instance_shape = input.shape[-3:]
        else:
            raise ValueError(
                f"Expected input shape of length 5 (2D) or 6 (3D) but got {len(input.shape)}"
            )

        for i_tile in range(len(attention_weights)):
            weight = attention_weights[i_tile]
            position = [int(i) for i in positions[i_tile]]
            end_pos = [e for e in zip(position, orig_shape[0], instance_shape)]
            end_pos = [
                i
                for i in starmap(
                    lambda p, o, t: p + t if (p + t) < o else o,
                    end_pos,
                )
            ]
            end_tile = [e for e in zip(position, orig_shape[0], instance_shape)]
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

            out_attention[out_indices + (0,)] += weight / np.prod(instance_shape)
            out_attention[out_indices + (1,)] += 1.0

        slices = tuple([slice(0, e, 1) for e in out_attention.shape[:-1]])
        out_attention[slices + (0,)] /= out_attention[slices + (1,)] + 1e-8
        out_attention = out_attention[slices + (0,)]

        # now correlate with segmentation ...
        seg_nonseg_ratio = segmentation[segmentation > 0].sum() / (
            segmentation[segmentation == 0].sum() + 1e-8
        )
        nonseg_seg_ratio = segmentation[segmentation == 0].sum() / (
            segmentation[segmentation > 0].sum() + 1e-8
        )
        attn_seg = out_attention[segmentation > 0].sum() / (seg_nonseg_ratio + 1e-8)
        attn_nonseg = out_attention[segmentation == 0].sum() / (nonseg_seg_ratio + 1e-8)
        attn_mean = attention_weights.mean()
        attn_std = attention_weights.std()
        attn_min = attention_weights.min()
        attn_max = attention_weights.max()

        ret["attn_seg"] = [attn_seg.item()]
        ret["attn_nonseg"] = [attn_nonseg.item()]
        ret["attn_mean"] = [attn_mean.item()]
        ret["attn_std"] = [attn_std.item()]
        ret["attn_min"] = [attn_min.item()]
        ret["attn_max"] = [attn_max.item()]

        return ret


class MILDataAdapterAttnOnly_(object):
    def __init__(self, data_key, ground_truth_key):
        self.data_key = data_key
        self.ground_truth_key = ground_truth_key

    def __call__(self, batch):

        assert len(batch[self.data_key]) == 1, "only defined for batch_size = 1"

        len_bag = 0

        for series_name in batch["_meta"].keys():
            if not (series_name in batch.keys()):
                if batch["_meta"][series_name]["len_tiles"] > len_bag:
                    len_bag = batch["_meta"][series_name]["len_tiles"]
                    positions = batch["_meta"][series_name]["tile_pos"][0]
                    orig_shape = batch["_meta"][series_name]["orig_shape"]

        bag_ = torch.swapaxes(batch[self.data_key][:, :, : len_bag - 1], 1, 2)
        bag = torch.zeros((bag_.shape[0], bag_.shape[1], bag_.shape[2], 1))
        for i in range(len(bag[0])):
            for j in range(len(bag[0][i])):
                bag[0][i, j] = bag_[0][i, j].mean()

        label = torch.argmax(batch[self.ground_truth_key][0])

        supplement = {}
        supplement["idents"] = batch["_idents"]
        supplement["positions"] = positions
        supplement["orig_shape"] = orig_shape
        supplement["segmentation"] = batch["segmentation"][0]

        return {
            "input": bag,
            "ground_truth": label,
            "supplement": supplement,
        }


class MILReportAdapterAttnOnly_(object):
    def __init__(self, instance_shape: any = (16, 16)):
        self.instance_shape = instance_shape

    def __call__(
        self,
        input: any = None,
        ground_truth: any = None,
        supplement: any = None,
        output: any = None,
        loss: any = None,
    ):

        assert len(output[0]) == 1, "only defined for batch_size = 1"

        out = [[1.0 - output[0][0].item(), output[0][0].item()]]
        gt = [[1.0 - ground_truth.float().item(), ground_truth.float().item()]]
        lo = [loss.item()]
        inp = [i.tolist() for i in input.squeeze(0)]

        ret = {}
        ret["ID"] = supplement["idents"]
        ret["ground_truth"] = gt
        ret["prediction"] = out
        ret["batch_loss"] = lo
        ret["input"] = inp

        positions = supplement["positions"].detach().cpu().numpy()
        orig_shape = supplement["orig_shape"].detach().cpu().numpy()
        segmentation = supplement["segmentation"].detach().cpu().numpy()

        attention_weights = output[2][0].detach().cpu().numpy()

        out_attention = np.zeros(orig_shape[0].tolist() + [2])

        instance_shape = self.instance_shape

        for i_tile in range(len(attention_weights)):
            weight = attention_weights[i_tile]
            position = [int(i) for i in positions[i_tile]]
            end_pos = [e for e in zip(position, orig_shape[0], instance_shape)]
            end_pos = [
                i
                for i in starmap(
                    lambda p, o, t: p + t if (p + t) < o else o,
                    end_pos,
                )
            ]
            end_tile = [e for e in zip(position, orig_shape[0], instance_shape)]
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

            out_attention[out_indices + (0,)] += weight / np.prod(instance_shape)
            out_attention[out_indices + (1,)] += 1.0

        slices = tuple([slice(0, e, 1) for e in out_attention.shape[:-1]])
        out_attention[slices + (0,)] /= out_attention[slices + (1,)] + 1e-8
        out_attention = out_attention[slices + (0,)]

        # now correlate with segmentation ...
        seg_nonseg_ratio = segmentation[segmentation > 0].sum() / (
            segmentation[segmentation == 0].sum() + 1e-8
        )
        nonseg_seg_ratio = segmentation[segmentation == 0].sum() / (
            segmentation[segmentation > 0].sum() + 1e-8
        )
        attn_seg = out_attention[segmentation > 0].sum() / (seg_nonseg_ratio + 1e-8)
        attn_nonseg = out_attention[segmentation == 0].sum() / (nonseg_seg_ratio + 1e-8)
        attn_mean = attention_weights.mean()
        attn_std = attention_weights.std()
        attn_min = attention_weights.min()
        attn_max = attention_weights.max()

        ret["attn_seg"] = [attn_seg.item()]
        ret["attn_nonseg"] = [attn_nonseg.item()]
        ret["attn_mean"] = [attn_mean.item()]
        ret["attn_std"] = [attn_std.item()]
        ret["attn_min"] = [attn_min.item()]
        ret["attn_max"] = [attn_max.item()]

        return ret


class MultiLabelReportAdapter(object):
    def __init__(self):
        super(MultiLabelReportAdapter, self).__init__()

    def __call__(self, outputs, ground_truth, loss, inputs):

        out = [i.tolist() for i in outputs]
        gt = [i.tolist() for i in ground_truth]
        lo = loss.item()
        inp = [i.tolist() for i in inputs]

        return out, gt, lo, inp


class MultiClassReportAdapter(object):
    def __init__(self):
        super(MultiClassReportAdapter, self).__init__()

    def __call__(self, outputs, ground_truth, loss, inputs):

        out = [i.tolist() for i in outputs]
        gt = []
        for i in range(len(ground_truth)):
            v = [0.0] * len(out[i])
            v[int(ground_truth[i])] = 1.0
            gt.append(v)
        lo = loss.item()
        inp = [i.tolist() for i in inputs]

        return out, gt, lo, inp


class AutoencoderReportAdapter(object):
    def __call__(self, outputs, ground_truth, loss, inputs):

        out = [[-1.0] for i in outputs]
        gt = [[-1.0] for i in ground_truth]
        lo = loss.item()
        inp = [i.tolist() for i in inputs]

        return out, gt, lo, inp


class MILReportAdapter(object):
    def __call__(self, outputs, ground_truth, loss, inputs):

        assert len(outputs[0]) == 1, "only defined for batch_size = 1"

        out = [[1.0 - outputs[0][0].item(), outputs[0][0].item()]]
        gt = [[1.0 - ground_truth.float().item(), ground_truth.float().item()]]
        lo = loss.item()
        inp = [i.tolist() for i in inputs.squeeze(0)]

        return out, gt, lo, inp


"""class MILReportAdapter_(object):
    def __call__(self, output, ground_truth, loss, input):

        assert len(output[0]) == 1, "only defined for batch_size = 1"

        for series_name in data["_meta"].keys():
                    if not (series_name in data.keys()):
                        if data["_meta"][series_name]["len_tiles"] > len_bag:
                            len_bag = data["_meta"][series_name]["len_tiles"]
                            positions = data["_meta"][series_name]["tile_pos"][0]
                            orig_shape = data["_meta"][series_name]["orig_shape"]

                segmentation = data["segmentation"]

        attention_weights = output[2][0].detach().cpu().numpy()

        out_attention = np.zeros(orig_shape[0].tolist() + [2])

                for tile, position, weight in zip(
                    inputs[:, 0], positions, attention_weights
                ):
                    position = [int(i) for i in position]
                    end_pos = [e for e in zip(position, orig_shape[0], tile.shape)]
                    end_pos = [
                        i
                        for i in starmap(
                            lambda p, o, t: p + t if (p + t) < o else o,
                            end_pos,
                        )
                    ]
                    end_tile = [e for e in zip(position, orig_shape[0], tile.shape)]
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

                    out_attention[out_indices + (0,)] += weight
                    out_attention[out_indices + (1,)] += 1.0

                slices = tuple([slice(0, e, 1) for e in out_attention.shape[:-1]])
                out_attention[slices + (0,)] /= out_attention[slices + (1,)] + 1e-8
                out_attention = out_attention[slices + (0,)]

                # now correlate with segmentation ...


        out = [[1.0 - outputs[0][0].item(), outputs[0][0].item()]]
        gt = [[1.0 - ground_truth.float().item(), ground_truth.float().item()]]
        lo = loss.item()
        inp = [i.tolist() for i in inputs.squeeze(0)]

        return out, gt, lo, inp"""
