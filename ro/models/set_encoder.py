import collections

import torch
import torch.nn as nn



class SetEncoder(nn.Module):
    """ Set-based encoder and predictor for problem with single source of uncertainty,
        i.e., KP/CB.
    """

    def __init__(self,

                 x_input_dim,       # dimension of [x, prob]
                 x_embed_dims,      # dimensions for [x, prob] embedding network
                 x_post_agg_dims,   # dimensions for [x, prob] post-agg network

                 xi_input_dim,      # dimension of [xi, prob]
                 xi_embed_dims,     # dimensions for [xi, prob] embedding network
                 xi_post_agg_dims,  # dimensions for [xi, prob] post-agg network

                 value_net_dims,     # dimensions of value network

                 output_dim,        # dimension of output

                 agg_type="mean",
                 bias_embed=0,
                 bias_value=1,
                 dropout=0):
        """
        """
        super(SetEncoder, self).__init__()

        self.output_dim = output_dim

        self.x_input_dim = x_input_dim
        self.x_embed_dims = x_embed_dims
        self.x_post_agg_dims = x_post_agg_dims

        self.xi_input_dim = xi_input_dim
        self.xi_embed_dims = xi_embed_dims
        self.xi_post_agg_dims = xi_post_agg_dims

        self.value_input_dim = x_post_agg_dims[-1] + xi_post_agg_dims[-1]
        self.value_net_dims = value_net_dims
        
        self.agg_type = agg_type
        self.dropout = dropout
        self.bias_embed = bias_embed
        self.bias_value = bias_value

        self.x_embed_layers, self.x_embed_net = self.get_feed_forward_nn(
            input_dim = self.x_input_dim, 
            hidden_dims = self.x_embed_dims[:-1], 
            output_dim = self.x_embed_dims[-1], 
            bias = False, # must be false for taking sum/mean efficiently
            name = "x_embed")

        self.x_post_agg_layers, self.x_post_agg_net = self.get_feed_forward_nn(
            input_dim = self.x_embed_dims[-1], 
            hidden_dims = self.x_post_agg_dims[:-1], 
            output_dim = self.x_post_agg_dims[-1], 
            bias = self.bias_embed, 
            # bias = False, 
            name = "x_post")

        self.xi_embed_layers, self.xi_embed_net = self.get_feed_forward_nn(
            input_dim = self.xi_input_dim, 
            hidden_dims = self.xi_embed_dims[:-1], 
            output_dim = self.xi_embed_dims[-1], 
            bias = False, # must be false for taking sum/mean efficiently
            name = "xi_embed")

        self.xi_post_agg_layers, self.xi_post_agg_net = self.get_feed_forward_nn(
            input_dim = self.xi_embed_dims[-1], 
            hidden_dims = self.xi_post_agg_dims[:-1], 
            output_dim = self.xi_post_agg_dims[-1], 
            bias = self.bias_embed,  
            # bias = False, 
            name = "xi_post")

        self.value_layers, self.value_net = self.get_feed_forward_nn(
            input_dim = self.value_input_dim, 
            hidden_dims = self.value_net_dims, 
            output_dim = self.output_dim, 
            bias = self.bias_value, 
            # bias = True, 
            name = "value")

            
    def forward(self, x, xi, x_n_items, xi_n_items):
        """ Forward pass. """

        x_embed = self.x_embed_net(x)
        x_agg = self.agg_tensor(x_embed, x_n_items)
        x_agg = self.x_post_agg_net(x_agg)

        xi_embed = self.xi_embed_net(xi)
        xi_agg = self.agg_tensor(xi_embed, xi_n_items)
        xi_agg = self.xi_post_agg_net(xi_agg)

        value_net_input = torch.cat([x_agg, xi_agg], axis=1)
        y = self.value_net(value_net_input)

        return y

    def get_net_as_sequential(self, net_as_dict):
        """ Returns nn.sequential object of network passed as an ordered dict. """
        return torch.nn.Sequential(net_as_dict)


    def set_scalers(self, label_scaler, feat_scaler):
        self.label_scaler = label_scaler
        self.feat_scaler = feat_scaler


    def agg_tensor(self, x, n_items=None, axis=1):
        """ Aggregation function for tensors. """
        if self.agg_type == "sum":
            x_agg = torch.sum(x, axis=axis)

        elif self.agg_type == "mean":
            if n_items == None: # set default to being size of tensor
                n_items = x.shape[1]

            x_agg = torch.sum(x, axis=axis)
            x_agg = torch.div(x_agg, n_items.unsqueeze(1))

        return x_agg

    def get_feed_forward_nn(self, input_dim, hidden_dims, output_dim, bias, name):
        """ Gets feed-forward for specified dimensions. """
        layers = collections.OrderedDict()
        
        layers[f"{name}_in"] = nn.Linear(input_dim, hidden_dims[0], bias=bias)
        layers[f"{name}_act_in"] = nn.ReLU()
        
        if len(hidden_dims) == 1:
            if self.dropout:
               layers[f"{name}_drop_in"] = nn.Dropout(self.dropout)
            
        else:
            for i in range(hidden_dims - 1):
                layers[f"{name}_{i}"] = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                layers[f"{name}_act_{i}"] = nn.ReLU()
                if self.dropout:
                    layers[f"relu_drop_in"] = nn.Dropout(self.dropout)

        layers[f"{name}_out"] = nn.Linear(hidden_dims[0], output_dim, bias=bias)
        net = self.get_net_as_sequential(layers)

        return layers, net


    def get_grb_compatible_nn(self, net_as_dict):
        """ Gets gurobi compatible nn.sequential.  Specifically,
                - Remove dropout
                - Adds bias of 0's if no bias in layer
        """
        grb_layers = collections.OrderedDict()
        
        for name, layer in net_as_dict.items():

            # add ReLU layers to list
            if type(layer) == torch.nn.modules.activation.ReLU:
                grb_layers[name] = layer

            # remove dropout layer by skipping
            elif type(layer) == torch.nn.modules.dropout.Dropout:
                continue

            # linear layer
            if type(layer) == torch.nn.modules.linear.Linear:

                # if layer does not have bias, then copy weights and set bias to zero
                if layer.bias is None:
                    layer_cp = nn.Linear(layer.in_features, layer.out_features)
                    layer_cp.weight = layer.weight
                    layer_cp.bias = nn.Parameter(torch.zeros(layer.out_features))

                    grb_layers[name] = layer_cp
                    
                 # if layer has bias, then do nothing
                else:
                    grb_layers[name] = layer
            
        return torch.nn.Sequential(grb_layers)





class SetEncoderMultipleUncertainty(nn.Module):
    """ Set-based encoder and predictor for problem with multiple sources of uncertinaty,
        i.e., CFLP.
    """

    def __init__(self,

                 x_input_dim,         # dimension of [x, prob]
                 x_embed_dims,        # dimensions for [x, prob] embedding network
                 x_post_agg_dims,     # dimensions for [x, prob] post-agg network

                 xi_1_input_dim,      # dimension of [xi, prob]
                 xi_1_embed_dims,     # dimensions for [xi, prob] embedding network
                 xi_1_post_agg_dims,  # dimensions for [xi, prob] post-agg network

                 xi_2_input_dim,      # dimension of [xi, prob]
                 xi_2_embed_dims,     # dimensions for [xi, prob] embedding network
                 xi_2_post_agg_dims,  # dimensions for [xi, prob] post-agg network

                 value_net_dims,      # dimensions of value network

                 output_dim,          # dimension of output, 1 if no constraint uncertianty, 2 if constraint uncertainty

                 agg_type="sum",
                 bias_embed=0,
                 bias_value=1,
                 dropout=0):
        """ Constructor for SetEncoderMultipleUncertainty """
        super(SetEncoderMultipleUncertainty, self).__init__()

        self.output_dim = output_dim

        self.x_input_dim = x_input_dim
        self.x_embed_dims = x_embed_dims
        self.x_post_agg_dims = x_post_agg_dims

        self.xi_1_input_dim = xi_demand_input_dim
        self.xi_1_embed_dims = xi_demand_embed_dims
        self.xi_1_post_agg_dims = xi_demand_post_agg_dims

        self.xi_2_input_dim = xi_disrupt_input_dim
        self.xi_2_embed_dims = xi_disrupt_embed_dims
        self.xi_2_post_agg_dims = xi_disrupt_post_agg_dims

        self.value_input_dim = x_post_agg_dims[-1] + xi_1_post_agg_dims[-1] + xi_2_post_agg_dims[-1]
        self.value_net_dims = value_net_dims
        
        self.agg_type = agg_type
        self.dropout = dropout
        self.bias_embed = bias_embed
        self.bias_value = bias_value

        self.x_embed_layers, self.x_embed_net = self.get_feed_forward_nn(
            input_dim = self.x_input_dim, 
            hidden_dims = self.x_embed_dims[:-1], 
            output_dim = self.x_embed_dims[-1], 
            bias = False, # must be false for taking sum/mean efficiently
            name = "x_embed")

        self.x_post_agg_layers, self.x_post_agg_net = self.get_feed_forward_nn(
            input_dim = self.x_embed_dims[-1], 
            hidden_dims = self.x_post_agg_dims[:-1], 
            output_dim = self.x_post_agg_dims[-1], 
            bias = self.bias_embed, 
            name = "x_post")

        self.xi_1_embed_layers, self.xi_1_embed_net = self.get_feed_forward_nn(
            input_dim = self.xi_1_input_dim, 
            hidden_dims = self.xi_1_embed_dims[:-1], 
            output_dim = self.xi_1_embed_dims[-1], 
            bias =  False, # must be false for taking sum/mean efficiently
            name = "xi_1_embed")

        self.xi_1_post_agg_layers, self.xi_1_post_agg_net = self.get_feed_forward_nn(
            input_dim = self.xi_1_embed_dims[-1], 
            hidden_dims = self.xi_1_post_agg_dims[:-1], 
            output_dim = self.xi_1_post_agg_dims[-1], 
            bias = self.bias_embed, 
            name = "xi_1_post")

        self.xi_2_embed_layers, self.xi_2_embed_net = self.get_feed_forward_nn(
            input_dim = self.xi_2_input_dim, 
            hidden_dims = self.xi_2_embed_dims[:-1], 
            output_dim = self.xi_2_embed_dims[-1], 
            bias =  False, # must be false for taking sum/mean efficiently
            name = "xi_2_embed")

        self.xi_2_post_agg_layers, self.xi_2_post_agg_net = self.get_feed_forward_nn(
            input_dim = self.xi_2_embed_dims[-1], 
            hidden_dims = self.xi_2_post_agg_dims[:-1], 
            output_dim = self.xi_2_post_agg_dims[-1], 
            bias = self.bias_embed, 
            name = "xi_2_post")

        self.value_layers, self.value_net = self.get_feed_forward_nn(
            input_dim = self.value_input_dim, 
            hidden_dims = self.value_net_dims, 
            output_dim = self.output_dim, 
            bias = self.bias_value, 
            name = "value")

            
    def forward(self, x, xi_1, xi_2, x_n_items, xi_1_n_items, xi_2_n_items):
        """ Forward pass. """
        x_embed = self.x_embed_net(x)
        x_agg = self.agg_tensor(x_embed, x_n_items)
        x_agg = self.x_post_agg_net(x_agg)

        xi_1_embed = self.xi_1_embed_net(xi_1)
        xi_1_agg = self.agg_tensor(xi_1_embed, xi_1_n_items)
        xi_1_agg = self.xi_1_post_agg_net(xi_1_agg)

        xi_2_embed = self.xi_2_embed_net(xi_2)
        xi_2_agg = self.agg_tensor(xi_2_embed, xi_2_n_items)
        xi_2_agg = self.xi_2_post_agg_net(xi_2_agg)

        value_net_input = torch.cat([x_agg, xi_1_agg, xi_2_agg], axis=1)
        y = self.value_net(value_net_input)

        return y

    def get_net_as_sequential(self, net_as_dict):
        """ Returns nn.sequential object of network passed as an ordered dict. """
        return torch.nn.Sequential(net_as_dict)


    def set_scalers(self, label_scaler, feat_scaler):
        self.label_scaler = label_scaler
        self.feat_scaler = feat_scaler


    def agg_tensor(self, x, n_items=None, axis=1):
        """ Aggregation function for tensors. """
        if self.agg_type == "sum":
            x_agg = torch.sum(x, axis=axis)

        elif self.agg_type == "mean":
            if n_items == None: # set default to being size of tensor
                n_items = x.shape[1]

            x_agg = torch.sum(x, axis=axis)
            x_agg = torch.div(x_agg, n_items.unsqueeze(1))

        return x_agg

    def get_feed_forward_nn(self, input_dim, hidden_dims, output_dim, bias, name):
        """ Gets feed-forward for specified dimensions. """
        layers = collections.OrderedDict()
        
        layers[f"{name}_in"] = nn.Linear(input_dim, hidden_dims[0], bias=bias)
        layers[f"{name}_act_in"] = nn.ReLU()
        
        if len(hidden_dims) == 1:
            if self.dropout:
               layers[f"{name}_drop_in"] = nn.Dropout(self.dropout)
            
        else:
            for i in range(hidden_dims - 1):
                layers[f"{name}_{i}"] = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                layers[f"{name}_act_{i}"] = nn.ReLU()
                if self.dropout:
                    layers[f"relu_drop_in"] = nn.Dropout(self.dropout)

        layers[f"{name}_out"] = nn.Linear(hidden_dims[0], output_dim, bias=bias)
        net = self.get_net_as_sequential(layers)

        return layers, net


    def get_grb_compatible_nn(self, net_as_dict):
        """ Gets gurobi compatible nn.sequential.  Specifically,
                - Remove dropout
                - Adds bias of 0's if no bias in layer
        """
        grb_layers = collections.OrderedDict()
        
        for name, layer in net_as_dict.items():

            # add ReLU layers to list
            if type(layer) == torch.nn.modules.activation.ReLU:
                grb_layers[name] = layer

            # remove dropout layer by skipping
            elif type(layer) == torch.nn.modules.dropout.Dropout:
                continue

            # linear layer
            if type(layer) == torch.nn.modules.linear.Linear:

                # if layer does not have bias, then copy weights and set bias to zero
                if layer.bias is None:
                    layer_cp = nn.Linear(layer.in_features, layer.out_features)
                    layer_cp.weight = layer.weight
                    layer_cp.bias = nn.Parameter(torch.zeros(layer.out_features))

                    grb_layers[name] = layer_cp
                    
                 # if layer has bias, then do nothing
                else:
                    grb_layers[name] = layer
            
        return torch.nn.Sequential(grb_layers)