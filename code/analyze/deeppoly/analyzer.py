import torch
from analyze import Analyzer
from analyze.deeppoly.heuristic import Heuristic, Implicit
from analyze.deeppoly.transform.factory import TransformerFactory
from analyze.deeppoly.transform.preprocess import Preprocessor
from analyze.deeppoly.transform.relu import ReluTransformer
from analyze.deeppoly.domain import AbstractDomain
from torch.nn.modules.activation import ReLU
from analyze.deeppoly import setup
from analyze.utils import Timer


class RobustnessProperty:

    def __init__(self, nb_features, true_label, comparison_fn):
        self._compare = comparison_fn
        self._transformer = self._create_transformer(nb_features, true_label)

    def verify(self, ads):
        ads = self._transformer.transform(ads)
        verified = self.verify_bounds(ads[-1].lower_bounds)
        return verified, ads

    def verify_bounds(self, lower_bounds):
        return torch.all(self._compare(lower_bounds, 0.0))

    @staticmethod
    def _create_transformer(nb_labels, true_label):
        layer = torch.nn.Linear(nb_labels, nb_labels - 1, False)
        weights = torch.zeros((nb_labels - 1, nb_labels))
        weights.fill_diagonal_(-1)
        weights[:, true_label] = 1
        layer.weight = torch.nn.Parameter(weights)
        return TransformerFactory.create(layer, backsubstitution=True)



class DeepPolyCoreEvaluator:

    def __init__(self, net, input, robustness_property):
        self.input = input
        self.net = net

        transformers = []
        relu_id = 0
        for i, layer in enumerate(net.layers):
            backsubst = i > 0 and isinstance(net.layers[i - 1], ReLU)
            transformer = TransformerFactory.create(layer, backsubst, relu_id)
            transformers.append(transformer)
            if isinstance(layer, ReLU):
                relu_id += 1

        self.transformers = transformers
        self.robustness = robustness_property

    def set_lambda_calculator(self, lambda_calculator):
        for tf in self.transformers:
            if isinstance(tf, ReluTransformer):
                tf.set_lambda_calculator(lambda_calculator)

    def set_backsubstitution_to(self, id):
        for tf in self.transformers:
            tf.set_backsubstitution_to(id)

    def get_nb_layers(self):
        return len(self.transformers)

    def get_relu_layer_ids(self):
        ids, id = [], 0
        for tf in self.transformers:
            if not isinstance(tf, Preprocessor):
                id += 1
            if isinstance(tf, ReluTransformer):
                ids.append(id)
        return ids

    def forward(self):
        ads = [self.input]
        steps = ["input"]
        for transformer in self.transformers:
            ads = transformer.transform(ads)
            steps.append(transformer.__class__.__name__)

        verified, ads = self.robustness.verify(ads)
        steps.append(RobustnessProperty.__name__)
        return verified, ads, steps



class DeepPoly(Analyzer):

    def __init__(self, heuristic):
        self._heuristic = Implicit.convert(heuristic)

    def verify(self, net, inputs, eps, true_label, domain_bounds=[0.0, 1.0], robustness_fn=setup.robustness_function):
        deeppoly = self._create_deep_poly_evaluator(net, inputs, eps, true_label, domain_bounds, robustness_fn)
        return self._heuristic.run(deeppoly)

    def _create_deep_poly_evaluator(self, net, inputs, eps, true_label, domain_bounds, robustness_fn):
        ini = AbstractDomain.create(inputs, eps, domain_bounds=domain_bounds)
        robustness = RobustnessProperty(net.layers[-1].out_features, true_label, robustness_fn)
        deeppoly = DeepPolyCoreEvaluator(net, ini, robustness)
        return deeppoly
