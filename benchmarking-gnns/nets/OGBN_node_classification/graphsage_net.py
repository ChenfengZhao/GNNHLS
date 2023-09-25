import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

import numpy as np
import timeit

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.graphsage_layer import GraphSageLayer
from layers.mlp_readout_layer import MLPReadout

class GraphSageNet(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """
    def __init__(self, net_params):
        super().__init__()
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']    
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.readout = net_params['readout']
        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        else:
            in_dim = net_params['in_dim']
            # self.embedding_h = nn.Embedding(in_dim, hidden_dim)
            self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type, batch_norm, residual) for _ in range(n_layers-1)])
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual))
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
        for conv in self.layers:
            h = conv(g, h)
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
            # np.savetxt(self.out_dir + "data/inter_rst/features_bfemb.txt", h.float().numpy(), delimiter=',', fmt="%.7f")

            h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)

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

        start = timeit.default_timer()

        for conv in self.layers:
            h = conv.inference(g, h)

        if(h.is_cuda):
            torch.cuda.synchronize()
        end = timeit.default_timer()
        print("Inference time: %s Seconds" % (end-start))

        # save the inference time
        with open(self.out_dir + 'data/infer_time.log', 'w') as f:
            f.write("Inference time: %s Seconds" % (end-start))

        g.ndata['h'] = h
        
        # save results of first layer of the network
        print("h2_l0 info:", h.dtype, h.size())
        # print("h2_l0", h)
        if(h.is_cuda):
            print("transfering tensor from GPU to CPU...")
            # torch.cuda.synchronize()
            # np.savetxt(self.out_dir + "data/h2_l0.txt", h.cpu().numpy(), delimiter=',', fmt="%.7f")
            np.savetxt(self.out_dir + "data/h2_l0.txt", h.cpu().numpy(), delimiter=',')
            print("the tensor is still on GPU after transferring:", h.is_cuda)
            # torch.cuda.synchronize()
        else:
            # np.savetxt(self.out_dir + "data/h2_l0.txt", h.numpy(), delimiter=',', fmt="%.7f")
            np.savetxt(self.out_dir + "data/h2_l0.txt", h.numpy(), delimiter=',')
        
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
        