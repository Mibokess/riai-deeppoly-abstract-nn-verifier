import argparse
import torch
from networks import FullyConnected, Conv
import analyze.deeppoly.heuristic as H
import analyze.deeppoly.transform.relu.lambda_ as L
import numpy as np
DEVICE = 'cpu'
INPUT_SIZE = 28


def analyze(net, inputs, eps, true_label):
    from analyze.deeppoly import analyzer
    heuristic = H.Sequential([L.MinimizeArea(), L.Zonotope(), H.IterateOverArgs(L.Constant, np.linspace(0, 1, 10))])
    dp = analyzer.DeepPoly(heuristic)
    res, *_ = dp.verify(net, inputs, eps, true_label)
    return res



def main(net, spec, verbose=True):

    with open(spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(spec[:-4].split('/')[-1].split('_')[-1])

    if net == 'fc1':
        nn = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif net == 'fc2':
        nn = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif net == 'fc3':
        nn = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif net == 'fc4':
        nn = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif net == 'fc5':
        nn = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif net == 'fc6':
        nn = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    elif net == 'fc7':
        nn = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 100, 10]).to(DEVICE)
    elif net == 'conv1':
        nn = Conv(DEVICE, INPUT_SIZE, [(16, 3, 2, 1)], [100, 10], 10).to(DEVICE)
    elif net == 'conv2':
        nn = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif net == 'conv3':
        nn = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    else:
        assert False

    nn.load_state_dict(torch.load('../mnist_nets/%s.pt' % net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = nn(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    res = analyze(nn, inputs, eps, true_label)
    out = "verified" if res else 'not verified'
    if verbose:
        print(out)
    return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3'],
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    parser.add_argument('--verbose', type=bool, required=False, default=True, help='display the test result')
    args = parser.parse_args()

    main(**vars(args))

