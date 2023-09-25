import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import time

import numpy as np
import timeit
"""
    3WLGNN / ThreeWLGNN
    Provably Powerful Graph Networks (Maron et al., 2019)
    https://papers.nips.cc/paper/8488-provably-powerful-graph-networks.pdf
    
    CODE adapted from https://github.com/hadarser/ProvablyPowerfulGraphNetworks_torch/
"""

from layers.three_wl_gnn_layers import RegularBlock, MlpBlock, SkipConnection, FullyConnected, diag_offdiag_maxpool
from layers.mlp_readout_layer import MLPReadout

class ThreeWLGNNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.in_dim_node = net_params['in_dim']
        depth_of_mlp = net_params['depth_of_mlp']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.layer_norm = net_params['layer_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']
        self.diag_pool_readout = True                     # if True, uses the new_suffix readout from original code
        
        block_features = [hidden_dim] * n_layers  # L here is the block number
        
        original_features_num = self.in_dim_node + 1  # Number of features of the input

        # sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = RegularBlock(depth_of_mlp, last_layer_features, next_layer_features, self.residual)
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features
        
        if self.diag_pool_readout:
            self.fc_layers = nn.ModuleList()
            for output_features in block_features:
                # each block's output will be pooled (thus have 2*output_features), and pass through a fully connected
                fc = FullyConnected(2*output_features, n_classes, activation_fn=None)
                self.fc_layers.append(fc)
        else:
            self.mlp_prediction = MLPReadout(sum(block_features)+original_features_num, n_classes)
        
        # dir for saved data(/param)
        self.out_dir = net_params['out_dir']

    def forward(self, x):                
        if self.diag_pool_readout:
            scores = torch.tensor(0, device=self.device, dtype=x.dtype)
        else:
            x_list = [x]
        
        for i, block in enumerate(self.reg_blocks):

            x = block(x)
            if self.diag_pool_readout:
                scores = self.fc_layers[i](diag_offdiag_maxpool(x)) + scores
            else:
                x_list.append(x)
        
        if self.diag_pool_readout:
            return scores
        else:
            # readout like RingGNN
            x_list = [torch.sum(torch.sum(x, dim=3), dim=2) for x in x_list]
            x_list = torch.cat(x_list, dim=1)
            
            x_out = self.mlp_prediction(x_list)
            return x_out
    
    def inference(self, x):                
        if self.diag_pool_readout:
            scores = torch.tensor(0, device=self.device, dtype=x.dtype)
        else:
            x_list = [x]
        
        assert(len(self.reg_blocks) == 1)
        print("num of 3WLGNN layers:", len(self.reg_blocks))
        
        # for i, block in enumerate(self.reg_blocks):
        i = 0
        # save input features of the network
        print("features info:", x.dtype, x.size())
        if(x.is_cuda):
            print("transfering tensor from GPU to CPU...")
            torch.cuda.synchronize()

            y = x.cpu().unsqueeze(-1).transpose(1, -1).squeeze(1)
            np.savetxt(self.out_dir + "data/features.txt", y.contiguous().view(-1, y.size(-1)).numpy(), delimiter=',', fmt="%.6f")

            print("the tensor is still on GPU after transferring:", x.is_cuda)
            torch.cuda.synchronize()
        else:
            y = x.unsqueeze(-1).transpose(1, -1).squeeze(1)
            np.savetxt(self.out_dir + "data/features.txt", y.contiguous().view(-1, y.size(-1)).numpy(), delimiter=',', fmt="%.6f")
        
        if(x.is_cuda):
            print("synchronizing CPU and GPU...")
            torch.cuda.synchronize()
        
        print("Running GNN model...")
        start = timeit.default_timer()
        
        block = self.reg_blocks[0]
        x = block.inference(x)

        if(x.is_cuda):
            torch.cuda.synchronize()
        end = timeit.default_timer()
        print("Inference time: %s Seconds" % (end-start))

        # save the inference time
        with open(self.out_dir + 'data/infer_time.log', 'w') as f:
            if(x.is_cuda):
                f.write("GPU Inference time: %s Seconds" % (end-start))
            else:
                f.write("CPU Inference time: %s Seconds" % (end-start))

        # save results of first layer of the network
        print("h2_l0 info:", x.dtype, x.size())

        if(x.is_cuda):
            print("transfering tensor from GPU to CPU...")
            torch.cuda.synchronize()
            y = x.cpu().unsqueeze(-1).transpose(1, -1).squeeze(1)
            np.savetxt(self.out_dir + "data/h2_l0.txt", y.contiguous().view(-1, y.size(-1)).numpy(), delimiter=',', fmt="%.6f")
            print("the tensor is still on GPU after transferring:", x.is_cuda)
            torch.cuda.synchronize()
        else:
            y = x.unsqueeze(-1).transpose(1, -1).squeeze(1)
            np.savetxt(self.out_dir + "data/h2_l0.txt", y.contiguous().view(-1, y.size(-1)).numpy(), delimiter=',', fmt="%.6f")

        if self.diag_pool_readout:
            scores = self.fc_layers[i](diag_offdiag_maxpool(x)) + scores
        else:
            x_list.append(x)
        
        if self.diag_pool_readout:
            return scores
        else:
            # readout like RingGNN
            x_list = [torch.sum(torch.sum(x, dim=3), dim=2) for x in x_list]
            x_list = torch.cat(x_list, dim=1)
            
            x_out = self.mlp_prediction(x_list)
            return x_out

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
    