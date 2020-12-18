import time

import pandas as pd
import os

import torch
from torch import nn

from analyze.deeppoly.analyzer import DeepPoly
from verifier import main as run_verifier
import glob


class Oracle:

    def __init__(self, path):
        self.gt = pd.read_csv(f"{path}/gt.txt", names=['net', 'img', 'output'])

    def ground_truth(self, net, test_name):
        image = os.path.basename(test_name)
        res = self.gt.loc[(self.gt["net"] == net) & (self.gt["img"] == image), "output"]
        if res.size > 0:
            return res.values[0]
        return None


class Evaluator:

    def __init__(self, test_folders):
        self.nns = sorted(map(lambda f: os.path.splitext(os.path.basename(f))[0], os.listdir("../mnist_nets")))
        self._test_folders = test_folders
        self._oracles = dict((folder, Oracle(folder)) for folder in test_folders)


    @staticmethod
    def get_test_cases(folder, nn_name):
        test_cases = glob.glob(f"{folder}/{nn_name}/*")
        return test_cases

    def test_all(self):
        self._test(self.nns)

    def test_fc(self):
        self.test("fc")

    def test_conv2d(self):
        self.test("conv")

    def test(self, startswith):
        self._test(list(filter(lambda n: n.startswith(startswith), self.nns)))

    def _test(self, nns):
        total = 0
        correct = 0
        sound = True
        score = 0
        max_score = 0

        for net in nns:
            print(f"\nverifying network '{net}'")
            for test_folder in self._test_folders:
                o = self._oracles[test_folder]
                folder_name = os.path.split(os.path.dirname(test_folder))[-1]
                for test in reversed(self.get_test_cases(test_folder, net)):
                    #if test == '..\\preliminary_test_cases\\/fc7\\img6_0.14400.txt': continue
                    test_name = os.path.basename(test)
                    start_time = time.time()
                    out = run_verifier(net=net, spec=test, verbose=False)
                    runtime = round(time.time() - start_time, 2)
                    gt = o.ground_truth(net, test)
                    sound = False if (gt == "not verified") and (out == "verified") else sound
                    max_score = max_score + 1 if gt == "verified" else max_score
                    point = 0
                    if out == gt:
                        status = "OK"
                        point = 1 if gt == "verified" else 0
                    elif gt == "not verified":
                        status = "KO - NOT SOUND!"
                        point = -2
                    else:
                        status = "FAILED"
                    point_msg = f"({point} point - expecting '{gt}' got '{out}')"
                    runtime_msg = f"runtime: {runtime} seconds"
                    print(f"- test {folder_name}/{test_name}: {status} {point_msg} - {runtime_msg}")
                    score += point
                    correct += (gt == out)
                    total += 1

        print("\n" + "="*50)
        print(f"FINAL SCORE = {score}/{max_score}  ({correct}/{total} correct, {round(correct / total * 100, 2)}%)")
        print("-" * 50)
        if not sound:
            print(f"WARNING: analyzer is NOT SOUND!")
        else:
            print(f"Analyzer seems sound!")
        print("=" * 50)

class ToyNNPaper(nn.Module):
    def __init__(self):
        super().__init__()

        linear1 = nn.Linear(2, 2, False)
        linear1.weight = torch.nn.Parameter(torch.tensor([[1.0, 1.0], [1.0, -1.0]]))
        relu = nn.ReLU()
        linear2 = nn.Linear(2, 2, False)
        linear2.weight = torch.nn.Parameter(torch.tensor([[1.0, 1.0], [1.0, -1.0]]))
        relu2 = nn.ReLU()
        linear3 = nn.Linear(2, 2)
        linear3.weight = torch.nn.Parameter(torch.tensor([[1.0, 1.0], [0.0, 1.0]]))
        linear3.bias = torch.nn.Parameter(torch.tensor([1.0, 0]))
        self.layers = nn.Sequential(linear1, relu, linear2, relu2, linear3)

    def forward(self, x):
        return self.layers(x)

def test_paper_deepoly():
    """
        Verifying toy NN of the Deep poly paper page 41:5
    """
    net = ToyNNPaper()
    inputs = torch.zeros((2, 1)).reshape(1, 2)
    from analyze.deeppoly.heuristic.factory import HeuristicFactory
    dp = DeepPoly(heuristic=HeuristicFactory(net, inputs, 0).create())
    res, ads, steps = dp.verify(net, inputs, 1, 0, domain_bounds=[-1, 1])

    expected_lower_bounds = [[-1, -1], [-2, -2], [0, 0], [0, -2], [0, 0], [1.0, 0], [1]]
    expected_upper_bounds = [[1, 1], [2, 2], [2, 2], [3, 2.0], [3, 2], [5.5, 2], None]

if __name__ == '__main__':
    e = Evaluator(glob.glob(f"../*test_cases/"))
    #e.test_fc()
    e.test("fc7")
    # e.test_conv2d()
    #e.test_all()
    #test_paper_deepoly()
