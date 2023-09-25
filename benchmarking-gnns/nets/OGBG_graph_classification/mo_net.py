import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

import numpy as np
import timeit

"""
    GMM: Gaussian Mixture Model Convolution layer
    Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs (Federico Monti et al., CVPR 2017)
    https://arxiv.org/pdf/1611.08402.pdf
"""

from layers.gmm_layer import GMMLayer
from layers.mlp_readout_layer import MLPReadout

class MoNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        kernel = net_params['kernel']                       # for MoNet
        dim = net_params['pseudo_dim_MoNet']                # for MoNet
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']      
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']  
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        else:
            in_dim = net_params['in_dim']
            # self.embedding_h = nn.Embedding(in_dim, hidden_dim)
            self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        aggr_type = "sum"                                    # default for MoNet
        
        
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Hidden layer
        for _ in range(n_layers-1):
            self.layers.append(GMMLayer(hidden_dim, hidden_dim, dim, kernel, aggr_type,
                                        dropout, batch_norm, residual))
            self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
            
        # Output layer
        self.layers.append(GMMLayer(hidden_dim, out_dim, dim, kernel, aggr_type,
                                    dropout, batch_norm, residual))
        self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        
        self.MLP_layer = MLPReadout(out_dim, n_classes) 
        
        # dir for saved data(/param)
        self.out_dir = net_params['out_dir']

    def forward(self, g, h, e, pos_enc=None):

        # input embedding
        if self.pos_enc:
            h = self.embedding_pos_enc(pos_enc) 
        else:
            h = self.embedding_h(h.float())
        
        # computing the 'pseudo' named tensor which depends on node degrees
        g.ndata['deg'] = g.in_degrees()
        g.apply_edges(self.compute_pseudo)
        pseudo = g.edata['pseudo'].to(self.device).float()
        
        for i in range(len(self.layers)):
            h = self.layers[i](g, h, self.pseudo_proj[i](pseudo))
        g.ndata['h'] = h
            
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return self.MLP_layer(hg)


    def inference(self, g, h, e, pos_enc=None):

        # input embedding
        if self.pos_enc:
            h = self.embedding_pos_enc(pos_enc) 
        else:
            h = self.embedding_h(h.float())
        
        # computing the 'pseudo' named tensor which depends on node degrees
        g.ndata['deg'] = g.in_degrees()
        g.apply_edges(self.compute_pseudo)
        pseudo = g.edata['pseudo'].to(self.device).float()
        
        # save input features of the network
        print("features info:", h.dtype, h.size())
        if(h.is_cuda):
            print("transfering tensor from GPU to CPU...")
            torch.cuda.synchronize()
            np.savetxt(self.out_dir + "data/features.txt", h.cpu().numpy(), delimiter=',', fmt="%.7f")
            print("the tensor is still on GPU after transferring:", h.is_cuda)
            torch.cuda.synchronize()
        else:
            np.savetxt(self.out_dir + "data/features.txt", h.numpy(), delimiter=',', fmt="%.7f")
        
        # save pseudo of the network
        print("pseudo info:", pseudo.dtype, pseudo.size())
        if(pseudo.is_cuda):
            np.savetxt(self.out_dir + "data/pseudo.txt", pseudo.cpu().numpy(), delimiter=',', fmt="%.7f")
        else:
            np.savetxt(self.out_dir + "data/pseudo.txt", pseudo.numpy(), delimiter=',', fmt="%.7f")
        
        if(h.is_cuda):
            print("synchronizing CPU and GPU...")
            torch.cuda.synchronize()

        print("Running GNN model...")
        start = timeit.default_timer()

        h = self.layers[0](g, h, self.pseudo_proj[0](pseudo))

        if(h.is_cuda):
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
        if(h.is_cuda):
            print("transfering tensor from GPU to CPU...")
            torch.cuda.synchronize()
            np.savetxt(self.out_dir + "data/h2_l0.txt", h.cpu().numpy(), delimiter=',', fmt="%.7f")
            print("the tensor is still on GPU after transferring:", h.is_cuda)
            torch.cuda.synchronize()
        else:
            np.savetxt(self.out_dir + "data/h2_l0.txt", h.numpy(), delimiter=',', fmt="%.7f")
        
        g.ndata['h'] = h
            
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return self.MLP_layer(hg)

    def compute_pseudo(self, edges):
        # compute pseudo edge features for MoNet
        # to avoid zero division in case in_degree is 0, we add constant '1' in all node degrees denoting self-loop
        srcs = 1/np.sqrt(edges.src['deg']+1)
        dsts = 1/np.sqrt(edges.dst['deg']+1)
        pseudo = torch.cat((srcs.unsqueeze(-1), dsts.unsqueeze(-1)), dim=1)
        return {'pseudo': pseudo}

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
