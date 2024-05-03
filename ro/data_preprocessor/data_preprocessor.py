from abc import ABC, abstractmethod

import time
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class DataPreprocessor(ABC):
    
    def __init__(self, cfg, model_type, predict_feas, device):
        """ Constructor for DataPreprocessor. """
        self.cfg = cfg
        self.model_type = model_type
        self.predict_feas = predict_feas
        self.device = device

        self.label_scaler = None
        self.feat_scaler = None


    @abstractmethod
    def get_set_encoder_dataset(self, data):
        """ Gets set encoder dataset.  """        
        pass


    def to_tensor(self, x):
        """ Returns input (x) as tensor. """
        return torch.from_numpy(x).float()


    def preprocess_data(self, tr_data, val_data):
        """ Function to preprocess training/validation data. """
        if self.model_type == "set_encoder":
            tr_dataset = self.get_set_encoder_dataset(tr_data)
            val_dataset = self.get_set_encoder_dataset(val_data)

        else:
            raise Exception(f"DataPreprocessor for model_type={self.model} must be implemented")

        return tr_dataset, val_dataset 


    def pad_features(self, feats, pad_dim):
        """ Pads features to be of dimension given. """
        feat_dim = feats.shape[1]
        item_dim = feats.shape[0]
       
        feats_padded = np.zeros((pad_dim, feat_dim))

        feats_padded[:item_dim,:] = feats

        return feats_padded
