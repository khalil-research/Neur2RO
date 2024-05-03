from abc import ABC, abstractmethod


class SecondStageRelax(ABC):
    @abstractmethod
    def adversarial_problem(self):
        pass
