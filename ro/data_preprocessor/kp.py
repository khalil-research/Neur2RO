import os
import pickle as pkl
import time
from multiprocessing import Manager, Pool

import gurobipy as gp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ro.two_ro.kp import Knapsack
from ro.utils.kp import get_path
from .data_preprocessor import DataPreprocessor


class KnapsackDataPreprocessor(DataPreprocessor):


    def __init__(self, cfg, model_type, predict_feas, device):
        """ Constructor for KnapsackDataPreprocessor.  """
        super(KnapsackDataPreprocessor, self).__init__(cfg, model_type, predict_feas, device)
        self.pad_size = self.cfg.n_items[-1]


    def get_set_encoder_dataset(self, dataset):
        """ Gets features ready to be used in Set-Based NN """
        x_features = []
        xi_features = []
        n_items_features = []     # note this is the same for x and y
        labels = []

        for sample in dataset:

            # decision/uncertianty
            x = sample['x']
            xi = sample['xi']

            # instance, number of items
            inst = sample['instance']
            n_items = inst['n_items']

            # features for each item
            inst_feats = list(map(lambda i: [
                                 inst['c'][i], 
                                 inst['p_bar'][i], 
                                 inst['p_hat'][i], 
                                 inst['f'][i], 
                                 inst['t'][i],  
                                 inst['C']], range(n_items)))    

            # build features for x and xi
            x_feats = []
            xi_feats = []
            for i in range(n_items):
                x_ft_i = [x[i]] + inst_feats[i]
                xi_ft_i = [xi[i]] + inst_feats[i]

                x_feats.append(x_ft_i)
                xi_feats.append(xi_ft_i)

            # scale features
            if self.feat_scaler is not None:
                x_feats = (x_feats - self.feat_scaler[0]) / ( self.feat_scaler[1] -  self.feat_scaler[0])
                xi_feats = (xi_feats - self.feat_scaler[0]) / ( self.feat_scaler[1] -  self.feat_scaler[0])

            # pad features
            x_feats = self.pad_features(x_feats, pad_dim = self.pad_size)
            xi_feats = self.pad_features(xi_feats, pad_dim = self.pad_size)

            # label
            label = sample['ss_obj']

            # scale label
            if self.label_scaler is not None:
                min_y, max_y = self.label_scaler[n_items]
                label = (label - min_y) / (max_y - min_y)

            # append to lists
            x_features.append(x_feats)
            xi_features.append(xi_feats)
            n_items_features.append(n_items)
            labels.append(label)

        # convert features to arrays
        x_features = np.array(x_features)
        xi_features = np.array(xi_features)
        n_items_features = np.array(n_items_features)
        labels = np.array(labels)

        # convert features to tensor
        x_features = self.to_tensor(x_features).to(self.device)
        xi_features = self.to_tensor(xi_features).to(self.device)
        n_items_features = self.to_tensor(n_items_features).to(self.device)
        labels = self.to_tensor(labels).to(self.device)

        # to pytorch dataset
        tensor_dataset = TensorDataset(x_features, xi_features, n_items_features, n_items_features, labels)

        return tensor_dataset


    def init_label_scaler(self, dataset):
        """ Compute min/max for n_item min/max scaling. """
        n_item_sets = self.cfg.n_items

        # compute scalers coefficients
        item_datasets = []
        item_labels = []
        for n_items in n_item_sets:

            ds_feats = list(filter(lambda x: x['instance']['n_items'] == n_items, dataset))
            ds_labels = list(map(lambda x: x['ss_obj'], ds_feats))
            
            item_datasets.append(ds_feats)
            item_labels.append(ds_labels)

        # compute scalers coefficients
        scaler_dict = {}
        for i in range(len(item_labels)):
            if len(item_labels[i]) > 0:
                label_min = np.min(item_labels[i])
                label_max = np.max(item_labels[i])
                scaler_dict[n_item_sets[i]] = (label_min, label_max)
            else:
                scaler_dict[n_item_sets[i]] = None, None

        # set label scaler
        self.label_scaler = scaler_dict


    def init_feature_scaler(self, dataset):
        """ Gets features ready to be used in Set-Based NN """
        set_features = []
        for sample in dataset:
            
            inst = sample['instance']
            n_items = inst['n_items']

            x_feats = sample['x']
            xi_feats = sample['xi']
            t1 = time.time()
            prob_feats = list(map(lambda x: [
                                 inst['c'][x], 
                                 inst['p_bar'][x], 
                                 inst['p_hat'][x], 
                                 inst['f'][x], 
                                 inst['t'][x],  
                                 inst['C']], range(n_items)))    

            feats = []
            for i in range(n_items):
                ft_i = [x_feats[i]] + prob_feats[i]
                feats.append(ft_i)

            set_features.append(np.array(feats))
        
        x_set = np.concatenate(set_features)
        feat_min = np.min(x_set, axis=0).reshape(-1)
        feat_max = np.max(x_set, axis=0).reshape(-1)

        # set min/max of x and xi to 0/1
        feat_min[0] = 0
        feat_max[0] = 1

        # set feature scaler
        self.feat_scaler =  feat_min, feat_max




