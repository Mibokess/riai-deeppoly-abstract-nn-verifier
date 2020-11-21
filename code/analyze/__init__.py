from abc import ABC, abstractmethod



class Analyzer(ABC):

    @abstractmethod
    def verify(self, net, inputs, eps, true_label):
        pass


class Sound(Analyzer):

    def verify(self, net, inputs, eps, true_label):
        return 0

