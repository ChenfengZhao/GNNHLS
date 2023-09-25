import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling


import numpy as np
import timeit

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from layers.gin_layer import GINLayer, ApplyNodeFunc, MLP

class GINNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp_GIN']               # GIN
        learn_eps = net_params['learn_eps_GIN']              # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN'] # GIN
        readout = net_params['readout']                      # this is graph_pooling_type
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        else:
            in_dim = net_params['in_dim']
            # self.embedding_h = nn.Embedding(in_dim, hidden_dim)
            self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        for layer in range(self.n_layers):
            mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, batch_norm, residual, 0, learn_eps))

        # Linear function for graph poolings (readout) of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(self.n_layers+1):
            self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))
        
        if readout == 'sum':
            self.pool = SumPooling()
        elif readout == 'mean':
            self.pool = AvgPooling()
        elif readout == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError
    
        # dir for saved data(/param)
        self.out_dir = net_params['out_dir']

        
    def forward(self, g, h, e, pos_enc=None):
        
        if self.pos_enc:
            h = self.embedding_pos_enc(pos_enc) 
        else:
            h = self.embedding_h(h.float())
        
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h)
            hidden_rep.append(h)

        # score_over_layer = 0

        # # perform pooling over all nodes in each graph in every layer
        # for i, h in enumerate(hidden_rep):
        #     pooled_h = self.pool(g, h)
        #     score_over_layer += self.linears_prediction[i](pooled_h)

        # return score_over_layer
        return h

    def inference(self, g, h, e, pos_enc=None):
        
        if self.pos_enc:
            h = self.embedding_pos_enc(pos_enc) 
        else:
            h = self.embedding_h(h.float())
        
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        # save input features of the network
        print("features info:", h.dtype, h.size())
        if(h.is_cuda):
            print("transfering tensor from GPU to CPU...")
            torch.cuda.synchronize()
            # np.savetxt(self.out_dir + "data/features.txt", h.cpu().numpy(), delimiter=',', fmt="%.7f")
            np.savetxt(self.out_dir + "data/features.txt", h.cpu().numpy(), delimiter=',')
            print("the tensor is still on GPU after transferring:", h.is_cuda)
            torch.cuda.synchronize()
        else:
            # np.savetxt(self.out_dir + "data/features.txt", h.numpy(), delimiter=',', fmt="%.7f")
            np.savetxt(self.out_dir + "data/features.txt", h.numpy(), delimiter=',')

        if(h.is_cuda):
            print("Start synchronizing GPU...")
            torch.cuda.synchronize()
        
        print("Running GNN model...")
        start = timeit.default_timer()

        print(len(self.ginlayers))
        conv_0 = self.ginlayers[0]
        h = conv_0.inference(g, h)

        if(h.is_cuda):
            print("Start synchronizing GPU...")
            torch.cuda.synchronize()
        end = timeit.default_timer()
        print("Inference time: %s Seconds" % (end-start))

        # save the inference time
        with open(self.out_dir + 'data/infer_time.log', 'w') as f:
            if(h.is_cuda):
                f.write("GPU Inference time: %s Seconds" % (end-start))
            else:
                f.write("CPU Inference time: %s Seconds" % (end-start))
        
        # save results of first layer of the network
        print("h2_l0 info:", h.dtype, h.size())
        # print("h2_l0", h)
        if(h.is_cuda):
            print("transfering tensor from GPU to CPU...")
            torch.cuda.synchronize()
            # np.savetxt(self.out_dir + "data/h2_l0.txt", h.cpu().numpy(), delimiter=',', fmt="%.7f")
            np.savetxt(self.out_dir + "data/h2_l0.txt", h.cpu().numpy(), delimiter=',')
            print("the tensor is still on GPU after transferring:", h.is_cuda)
            torch.cuda.synchronize()
        else:
            # np.savetxt(self.out_dir + "data/h2_l0.txt", h.numpy(), delimiter=',', fmt="%.7f")
            np.savetxt(self.out_dir + "data/h2_l0.txt", h.numpy(), delimiter=',')

        hidden_rep.append(h)
        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.linears_prediction[i](pooled_h)

        return score_over_layer
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss