import pandas as pd
import os
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
        self.nns = map(lambda f: os.path.splitext(os.path.basename(f))[0], os.listdir("../mnist_nets"))
        self._test_folders = test_folders
        self._oracles = dict((folder, Oracle(folder)) for folder in test_folders)


    @staticmethod
    def get_test_cases(folder, nn_name):
        test_cases = glob.glob(f"{folder}/{nn_name}/*")
        return test_cases

    def test_all(self):
        self._test(self.nns)

    def test_fc(self):
        self._test(list(filter(lambda n: n.startswith("fc"), self.nns)))

    def test_conv2d(self):
        self._test(list(filter(lambda n: n.startswith("conv"), self.nns)))

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
                for test in self.get_test_cases(test_folder, net):
                    test_name = os.path.basename(test)
                    out = run_verifier(net=net, spec=test, verbose=False)
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
                    print(f"- test {folder_name}/{test_name}: {status}  ({point} point - expecting '{gt}' got '{out}')")
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



if __name__ == '__main__':
    e = Evaluator(glob.glob(f"../*test_cases/"))
    e.test_fc()
    # e.test_conv2d()
    #e.test_all()
