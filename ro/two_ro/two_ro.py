from abc import ABC, abstractmethod


class TwoStageRO(ABC):
    """
    Class for two-stage robust integer optimization problem
    """

    @abstractmethod
    def solve_second_stage(self, x, xi, gap=0.02, time_limit=600, threads=1, verbose=1):
        """ Solves the second-stage problem for a given first-stage, uncertainty pair. """
        pass
