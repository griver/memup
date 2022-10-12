from abc import abstractmethod, ABC


class Metric(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        pass