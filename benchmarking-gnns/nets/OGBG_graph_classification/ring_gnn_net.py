import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import time

import numpy as np
import timeit

"""
    Ring-GNN
    On the equivalence between graph isomorphism testing and function approximation with GNNs (Chen et al, 2019)
    https://arxiv.org/pdf/1905.12560v1.pdf
"""
from layers.ring_gnn_equiv_layer import RingGNNEquivLayer
from layers.mlp_readout_layer import MLPReadout

class RingGNNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.in_dim_node = net_params['in_dim']
        avg_node_num = net_params['avg_node_num'] 
        radius = net_params['radius'] 
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.layer_norm = net_params['layer_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']
        
        self.depth = [torch.LongTensor([1+self.in_dim_node])] + [torch.LongTensor([hidden_dim])] * n_layers
            
        self.equi_modulelist = nn.ModuleList([RingGNNEquivLayer(self.device, m, n,
                                                                 layer_norm=self.layer_norm,
                                                                 residual=self.residual,
                                                                 dropout=dropout,
                                                                 radius=radius,
                                                                 k2_init=0.5/avg_node_num) for m, n in zip(self.depth[:-1], self.depth[1:])])
        
        self.prediction = MLPReadout(torch.sum(torch.stack(self.depth)).item(), n_classes)

        self.out_dir = net_params['out_dir']

    def forward(self, x):
        """
            CODE ADPATED FROM https://github.com/leichen2018/Ring-GNN/
        """

        # this x is the tensor with all info available => adj, node feat

        x_list = [x]
        for layer in self.equi_modulelist:    
            x = layer(x)
            x_list.append(x)
        
        # # readout
        x_list = [torch.sum(torch.sum(x, dim=3), dim=2) for x in x_list]
        x_list = torch.cat(x_list, dim=1)
        
        x_out = self.prediction(x_list)

        return x_out

    def inference(self, x):
        """
            CODE ADPATED FROM https://github.com/leichen2018/Ring-GNN/
        """

        # this x is the tensor with all info available => adj, node feat

        x_list = [x]

        assert(len(self.equi_modulelist) == 1)
        print("num of RingGNN layers:", len(self.equi_modulelist))

        # save input features of the network
        print("features info:", x.dtype, x.size())
        if(x.is_cuda):
            print("transfering tensor from GPU to CPU...")
            torch.cuda.synchronize()

            np.savetxt(self.out_dir + "data/features.txt", x.cpu().view(-1, x.size(-1)).numpy(), delimiter=',')

            print("the tensor is still on GPU after transferring:", x.is_cuda)
            torch.cuda.synchronize()

        else:
            np.savetxt(self.out_dir + "data/features.txt", x.view(-1, x.size(-1)).numpy(), delimiter=',')
        
        if(x.is_cuda):
            print("synchronizing CPU and GPU...")
            torch.cuda.synchronize()
        
        print("Running GNN model...")
        start = timeit.default_timer()

        layer = self.equi_modulelist[0]
        x = layer.inference(x)

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
            np.savetxt(self.out_dir + "data/h2_l0.txt", x.cpu().contiguous().view(-1, x.size(-1)).numpy(), delimiter=',')
            print("the tensor is still on GPU after transferring:", x.is_cuda)
            torch.cuda.synchronize()
        else:
            np.savetxt(self.out_dir + "data/h2_l0.txt", x.contiguous().view(-1, x.size(-1)).numpy(), delimiter=',')

        x_list.append(x)
        
        # # readout
        x_list = [torch.sum(torch.sum(x, dim=3), dim=2) for x in x_list]
        x_list = torch.cat(x_list, dim=1)
        
        x_out = self.prediction(x_list)

        return x_out
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss

