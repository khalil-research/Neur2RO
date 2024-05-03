from abc import ABC, abstractmethod


class SSAProblem(ABC):
    @abstractmethod
    def set_eval_problem(self):
        pass

    @abstractmethod
    def eval_problem(self):
        pass

    @abstractmethod
    def get_obj(self):
        pass
