import pytest
import os
import sys
import numpy as np
import random

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import reporters


def test_MulticlassClassificationReporter(tmp_path):

    batch_size = random.randint(8, 64)
    batch_count = random.randint(10, 100)
    class_count = random.randint(2, 10)
    epochs = random.randint(5, 10)
    phases = ("train", "val", "something_else")

    data = {}

    c = 0

    for epoch in range(epochs):

        predictions = []
        for batch in range(batch_count):
            predictions.append([])
            for item in range(batch_size):
                predictions[-1].append([])
                for classitem in range(class_count):
                    predictions[-1][-1].append(random.random())

        ground_truths = []
        for batch in range(batch_count):
            ground_truths.append([])
            for item in range(batch_size):
                ground_truths[-1].append([0.0] * class_count)
                ground_truths[-1][-1][random.randint(0, class_count - 1)] = 1.0

        losses = []
        for batch in range(batch_count):
            losses.append(random.random() * 100.0)

        idents = []
        for batch in range(batch_count):
            idents.append([])
            for item in range(batch_size):
                idents[-1].append("ident_{}".format(c))
                c += 1

        data[epoch] = {
            "predictions": predictions,
            "ground_truths": ground_truths,
            "losses": losses,
            "idents": idents,
        }

    reporter = reporters.MulticlassClassificationReporter(out_dir=tmp_path)

    for phase in phases:
        for epoch in range(epochs):
            for batch in range(batch_count):
                preds = data[epoch]["predictions"][batch]
                gts = data[epoch]["ground_truths"][batch]
                los = data[epoch]["losses"][batch]
                idents = data[epoch]["idents"][batch]

                reporter(preds, gts, los, idents=idents, phase=phase, epoch=epoch)
        reporter.save()

    for phase in phases:
        for epoch in range(epochs):
            assert "{}_{}.csv".format(phase, epoch) in os.listdir(reporter.out_dir)

    rep2 = reporters.MulticlassClassificationReporter(out_dir=tmp_path)

    for phase in phases:
        for target in "acc", "loss":
            assert rep2.best_epoch(phase, target) == reporter.best_epoch(phase, target)
            assert rep2.last_epoch(phase) == reporter.last_epoch(phase)
        for epoch in range(epochs):
            for measure in "acc", "loss":
                assert (
                    rep2.get_measure(measure, phase, epoch)
                    - reporter.get_measure(measure, phase, epoch)
                ) < 0.000001
            for measure in "recall", "precision", "f1-score":
                assert np.all(
                    rep2.get_per_class_measure(measure, phase, epoch)
                    == reporter.get_per_class_measure(measure, phase, epoch)
                )

