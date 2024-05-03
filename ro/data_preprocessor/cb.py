import os
import pickle as pkl
import time
from multiprocessing import Manager, Pool

import gurobipy as gp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ro.two_ro.cb import CapitalBudgeting
from ro.utils.cb import get_path
from .data_preprocessor import DataPreprocessor


class CapitalBudgetingDataPreprocessor(DataPreprocessor):


    def __init__(self, cfg, model_type, predict_feas, device):
        """ Constructor for KnapsackDataPreprocessor.  """
        super(CapitalBudgetingDataPreprocessor, self).__init__(cfg, model_type, predict_feas, device)
        self.pad_size = self.cfg.n_items[-1]


    def get_set_encoder_dataset(self, dataset):
        """ Gets features ready to be used in Set-Based NN """
        x_features = []
        xi_features = []
        n_items_features = []
        labels = []

        n_infeas = 0
        for sample in dataset:

            # decision/uncertainty
            x = sample['x']
            xi = sample['xi']

            # instance
            inst = sample['instance']
            n_items = inst['n_items']
            xi_dim = inst['xi_dim']

            # instance features
            inst_feats = list(map(lambda i: [inst['c_bar'][i], inst['r_bar'][i]], range(n_items)))

            # compute prod
            phi = np.array(list(inst['phi'].values()))
            psi = np.array(list(inst['psi'].values()))

            psi_prod = 1 + psi @ xi / 2
            phi_prod = 1 + phi @ xi / 2

            # include these if changing.  Not currently done in implementation
            # inst_feats = [inst['k'], inst['l'], inst['m']]       
    
            x_feats = []
            xi_feats = []
            for i in range(n_items):
                x_ft_i = [x[i]] + inst_feats[i]
                x_feats.append(x_ft_i)

            for i in range(n_items):
                xi_ft_i = [phi_prod[i], psi_prod[i]] + inst_feats[i]
                xi_feats.append(xi_ft_i)

            # scale features
            if self.feat_scaler is not None:
                x_feats = (x_feats - self.feat_scaler['x'][0]) / (self.feat_scaler['x'][1] - self.feat_scaler['x'][0])
                xi_feats = (xi_feats - self.feat_scaler['xi'][0]) / (self.feat_scaler['xi'][1] - self.feat_scaler['xi'][0])

            # pad features
            x_feats = self.pad_features(x_feats, self.pad_size)
            xi_feats = self.pad_features(xi_feats, self.pad_size)

            # label
            if self.predict_feas:
                raise Exception("CB feasibility prediction is not implemented.  ")
            
            else:
                label = sample[self.cfg.obj_label]

                if np.isnan(label):
                    n_infeas += 1
                    #print("  skipping infeasible sample. ")
                    continue

                # scale label
                if self.label_scaler is not None:
                    min_y, max_y = self.label_scaler[n_items]['obj']
                    label = (label - min_y) / (max_y - min_y)

            x_features.append(x_feats)
            xi_features.append(xi_feats)
            n_items_features.append(n_items)
            labels.append(label)

        print(f"  # n_infeasible = {n_infeas} / {len(dataset)}")
        x_features = np.array(x_features)
        xi_features = np.array(xi_features)
        n_items_features = np.array(n_items_features)

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
        item_obj_labels = []

        item_feas_labels = []
        for n_items in n_item_sets:
            ds_feats = list(filter(lambda x: x['instance']['n_items'] == n_items, dataset))
            obj_label = list(map(lambda x: x[self.cfg.obj_label], ds_feats))
            feas_label = list(map(lambda x: x[self.cfg.feas_label], ds_feats))

            item_datasets.append(ds_feats)
            item_obj_labels.append(obj_label)
            item_feas_labels.append(feas_label)

        # compute scalers coefficients
        scaler_dict = {}
        for i in range(len(item_obj_labels)):
            if len(item_obj_labels[i]) > 0:
                label_min = np.nanmin(item_obj_labels[i])
                label_max = np.nanmax(item_obj_labels[i])
                scaler_dict[n_item_sets[i]] = {'obj': (label_min, label_max)}
            else:
                scaler_dict[n_item_sets[i]] = None

        for i in range(len(item_feas_labels)):
            if len(item_feas_labels[i]) > 0:
                label_min = np.min(item_feas_labels[i])
                label_max = np.max(item_feas_labels[i])
                scaler_dict[n_item_sets[i]]['feas'] = (label_min, label_max)

        self.label_scaler = scaler_dict


    def init_feature_scaler(self, dataset):
        """ Gets features ready to be used in Set-Based NN """
        set_features_x = []
        set_features_xi = []
        for sample in dataset:

            x = sample['x']
            xi = sample['xi']

            inst = sample['instance']
            n_items = inst['n_items']

            xi_dim = inst['xi_dim']
            inst_feats = list(map(lambda i: [inst['c_bar'][i], inst['r_bar'][i]], range(n_items)))

            # compute prod
            phi = np.array(list(inst['phi'].values()))
            psi = np.array(list(inst['psi'].values()))

            psi_prod = 1 + psi @ xi / 2
            phi_prod = 1 + phi @ xi / 2

            # prob_feats = [inst['k'], inst['l'], inst['m']]        # if more than one instance
            # x = pt['x']
            # xi = pt['xi']

            x_feats = []
            xi_feats = []
            for i in range(n_items):
                x_ft_i = [x[i]] + inst_feats[i]
                x_feats.append(x_ft_i)
            set_features_x.append(np.array(x_feats))

            for i in range(n_items):
                xi_ft_i = [phi_prod[i], psi_prod[i]] + inst_feats[i]
                xi_feats.append(xi_ft_i)
            set_features_xi.append(np.array(xi_feats))

        x_set = np.concatenate(set_features_x)
        feat_min_x = np.min(x_set, axis=0).reshape(-1)
        feat_max_x = np.max(x_set, axis=0).reshape(-1)

        # set min/max of x to 0/1
        feat_min_x[0] = 0
        feat_max_x[0] = 1

        xi_set = np.concatenate(set_features_xi)
        feat_min_xi = np.min(xi_set, axis=0).reshape(-1)
        feat_max_xi = np.max(xi_set, axis=0).reshape(-1)

        # set min of xi to 0
        feat_min_xi[0] = 0
        feat_min_xi[1] = 0

        feat_scaler = {'x': (feat_min_x, feat_max_x), 'xi': (feat_min_xi, feat_max_xi)}

        # set feature scaler
        self.feat_scaler =  feat_scaler


