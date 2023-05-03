import os
import sys
import torch
import pytest
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .. import tools
from .. import models
from .. import lossfunctions
from .. import optimizers


# https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/5
def models_match(model_1, model_2):
    for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
        else:
            return True


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                print("Mismtach found at", key_item_1[0])
                return False
            else:
                raise Exception
    if models_differ == 0:
        print("Models match perfectly! :)")
        return True


def test_checkpoint_tool(tmp_path):

    ct = tools.CheckpointTool(
        base_dir=tmp_path,
        cop_str=">",
        model_init_params=("VGG11", [], {}),
        optimizer_init_params=("SGD", [], {"lr": 0.01}),
    )

    acc_list = []
    model_list = []
    opt_list = []

    epochs = random.randint(5, 10)

    for epoch in range(epochs):
        print(epoch)
        acc = random.random()
        model = models.VGG11()
        opt = optimizers.SGD(model.parameters(), lr=random.random())
        ct.update(model=model, optimizer=opt, epoch=epoch, value=acc)
        acc_list.append(acc)
        model_list.append(model)
        opt_list.append(opt)

    best_epoch = np.argmax(np.array(acc_list))
    last_epoch = len(acc_list) - 1

    assert ct.best_epoch == best_epoch
    assert ct.last_epoch == last_epoch

    best_model = model_list[best_epoch]
    best_opt = opt_list[best_epoch]

    last_model = model_list[last_epoch]
    last_opt = opt_list[last_epoch]

    load_best = ct.load("best")
    load_last = ct.load("last")

    assert compare_models(best_model, load_best["model"])
    assert compare_models(last_model, load_last["model"])
    if not best_epoch == last_epoch:
        assert not compare_models(best_model, load_last["model"])
        assert not compare_models(last_model, load_best["model"])
    else:
        test_checkpoint_tool(tmp_path)

