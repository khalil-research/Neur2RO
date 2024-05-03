from enum import Enum


class DataManagerModes(Enum):
    GEN_INSTANCE = 1
    GEN_DATASET_P = 2
    GEN_DATASET_E = 3


class LearningModelTypes(Enum):
    pass


class LossPenaltyTypes(Enum):
    none = 0
    lasso = 1
    ridge = 2
    elastic = 3
