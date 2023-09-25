import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

import numpy as np
import timeit

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer
from layers.mlp_readout_layer import MLPReadout

class GatedGCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_edge_type = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        else:
            in_dim = net_params['in_dim']
            # self.embedding_h = nn.Embedding(in_dim, hidden_dim)
            self.embedding_h = nn.Linear(in_dim, hidden_dim)

        if self.edge_feat:
            # self.embedding_e = nn.Embedding(num_edge_type, hidden_dim)
            self.embedding_e = nn.Linear(num_edge_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                       self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

        # dir for saved data(/param)
        self.out_dir = net_params['out_dir']

        
    def forward(self, g, h, e, pos_enc=None):

        # input embedding
        if self.pos_enc:
            h = self.embedding_pos_enc(pos_enc) 
        else:
            h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)   
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        
        # if self.readout == "sum":
        #     hg = dgl.sum_nodes(g, 'h')
        # elif self.readout == "max":
        #     hg = dgl.max_nodes(g, 'h')
        # elif self.readout == "mean":
        #     hg = dgl.mean_nodes(g, 'h')
        # else:
        #     hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(h)

    def inference(self, g, h, e, pos_enc=None):

        # input embedding
        if self.pos_enc:
            h = self.embedding_pos_enc(pos_enc) 
        else:
            h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)   
        
        # convnets
        # for conv in self.layers:
        #     h, e = conv(g, h, e)

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
        

        # save input edge feature
        print("edge_features info:", e.dtype, e.size())
        if(e.is_cuda):
            np.savetxt(self.out_dir + "data/edge_features.txt", e.cpu().numpy(), delimiter=',', fmt="%.7f")
        else:
            np.savetxt(self.out_dir + "data/edge_features.txt", e.numpy(), delimiter=',', fmt="%.7f")
        
        if(h.is_cuda):
            print("synchronizing CPU and GPU...")
            torch.cuda.synchronize()
        
        print("Running GNN model...")
        start = timeit.default_timer()

        h, e = self.layers[0](g, h, e)

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
    
        # save results of edge features
        print("e2_l0 info:", e.dtype, e.size())
        if(e.is_cuda):
            print("transfering tensor from GPU to CPU...")
            torch.cuda.synchronize()
            np.savetxt(self.out_dir + "data/e2_l0.txt", e.cpu().numpy(), delimiter=',', fmt="%.6f")
            torch.cuda.synchronize()
        else:
            np.savetxt(self.out_dir + "data/e2_l0.txt", e.numpy(), delimiter=',', fmt="%.6f")


        g.ndata['h'] = h
        
        # if self.readout == "sum":
        #     hg = dgl.sum_nodes(g, 'h')
        # elif self.readout == "max":
        #     hg = dgl.max_nodes(g, 'h')
        # elif self.readout == "mean":
        #     hg = dgl.mean_nodes(g, 'h')
        # else:
        #     hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(h)
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
