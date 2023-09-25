import numpy as np, time, pickle, random, csv
import torch
from torch.utils.data import DataLoader, Dataset

import os
import pickle
import numpy as np

import dgl

from sklearn.model_selection import StratifiedKFold, train_test_split

random.seed(42)

from scipy import sparse as sp

# OGB Lib
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])

def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in TUsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g

def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    n = g.number_of_nodes()
    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(n) - N * A * N
    # Eigenvectors
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()
    return g

def creat_path(path):
    """Creat path+folder if not exist. Do nothing if path exists

    Parameters
    ----------
    path : str
        path + folder_name
    """
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print("path generated:", path)
    else:
        print("path exists:", path)

class OGBNDataset(torch.utils.data.Dataset):
    def __init__(self, name='ogbn-arxiv'):
        t0 = time.time()
        self.name = name
        print("[!] Dataset: ", self.name)
        # creat path+folder to store dataset if not exist
        dataset_path = "./data/ogbn/" + self.name
        creat_path(dataset_path)

        self.dataset = DglNodePropPredDataset(name = self.name, root=dataset_path)

        self.all = self.dataset

        self.graph, self.labels = self.all[0]  # single DGL graph

        # self.graph.edata['feat'] = torch.zeros(self.graph.number_of_edges(), 1, dtype=torch.long)
        self.graph.edata['feat'] = torch.randint(0, 2, (self.graph.number_of_edges(), 1))
        self.graph.ndata['feat'] = torch.randint(0, 7, (self.graph.number_of_nodes(), 9))

        # this function splits data into train/val/test and returns the indices
        self.all_idx = self.all.get_idx_split()
        
        # self.train = self.all[0][self.all_idx["train"]]
        # self.val = self.all[0][self.all_idx["valid"]]
        # self.test = self.all[0][self.all_idx["test"]]

        self.train_idx = self.all_idx["train"]
        self.valid_idx = self.all_idx["valid"]
        self.test_idx = self.all_idx["test"]

        self.evaluator = Evaluator(name=self.name)
        
        print("Time taken: {:.4f}s".format(time.time()-t0))
    
    
    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # # The input samples is a list of pairs (graph, label).
        # graphs, labels = map(list, zip(*samples))
        # # print("labels1", labels)
        # labels = torch.tensor(np.array(labels))
        # # print("labels2", labels)
        # batched_graph = dgl.batch(graphs)
        # return batched_graph, labels
        graphs, labels = map(list, zip(*samples))
        labels = torch.stack(labels)
        labels = torch.where(torch.isnan(labels), torch.zeros_like(labels), labels)
        labels = torch.mean(labels, dim=1, dtype=torch.float64).long()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    # only considering 'feat' of nodes and edges for MoNet and GatedGCN
    def collate_feat_only(self, samples):
        # The input samples is a list of pairs (graph, label).
        # graphs, labels = map(list, zip(*samples))
        # labels = torch.tensor(np.array(labels))
        # batched_graph = dgl.batch(graphs, node_attrs=['feat'], edge_attrs=['feat'])
        # return batched_graph, labels
        graphs, labels = map(list, zip(*samples))
        labels = torch.stack(labels)
        labels = torch.where(torch.isnan(labels), torch.zeros_like(labels), labels)
        labels = torch.mean(labels, dim=1, dtype=torch.float64).long()
        batched_graph = dgl.batch(graphs, node_attrs=['feat'], edge_attrs=['feat'])
        return batched_graph, labels
    

    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense_gnn(self, samples, pos_enc):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        g = graphs[0]
        adj = self._sym_normalize_adj(g.adjacency_matrix().to_dense())        
        """
            Adapted from https://github.com/leichen2018/Ring-GNN/
            Assigning node and edge feats::
            we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
            Then we build a zero-initialized tensor, say T, in R^{(1 + d_n + d_e) x n x n}. T[0, :, :] is the adjacency matrix.
            The diagonal T[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i. 
            The off diagonal T[1+d_n:, i, j] store edge features of edge(i, j).
        """
        zero_adj = torch.zeros_like(adj)
        if pos_enc:
            in_dim = g.ndata['pos_enc'].shape[1]        
            # use node feats to prepare adj
            adj_node_feat = torch.stack([zero_adj for j in range(in_dim)])
            adj_node_feat = torch.cat([adj.unsqueeze(0), adj_node_feat], dim=0)
            for node, node_feat in enumerate(g.ndata['pos_enc']):
                adj_node_feat[1:, node, node] = node_feat
            x_node_feat = adj_node_feat.unsqueeze(0)
            return x_node_feat, labels
        else: # no node features here
            in_dim = g.ndata['feat'].size(1)
            # use node feats to prepare adj
            adj_node_feat = torch.stack([zero_adj for j in range(in_dim)])
            adj_node_feat = torch.cat([adj.unsqueeze(0), adj_node_feat], dim=0)
            for node, node_feat in enumerate(g.ndata['feat']):
                adj_node_feat[1:, node, node] = node_feat
            x_no_node_feat = adj_node_feat.unsqueeze(0)
            return x_no_node_feat, labels
    
    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim = 0)#.squeeze()
        deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))
    
    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True
        for split_num in range(5):
            self.train[split_num].graph_lists = [self_loop(g) for g in self.train[split_num].graph_lists]
            self.val[split_num].graph_lists = [self_loop(g) for g in self.val[split_num].graph_lists]
            self.test[split_num].graph_lists = [self_loop(g) for g in self.test[split_num].graph_lists]
            
        for split_num in range(5):
            self.train[split_num] = DGLFormDataset(self.train[split_num].graph_lists, self.train[split_num].graph_labels)
            self.val[split_num] = DGLFormDataset(self.val[split_num].graph_lists, self.val[split_num].graph_labels)
            self.test[split_num] = DGLFormDataset(self.test[split_num].graph_lists, self.test[split_num].graph_labels)

    def _add_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        for split_num in range(5):
            self.train[split_num].graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.train[split_num].graph_lists]
            self.val[split_num].graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.val[split_num].graph_lists]
            self.test[split_num].graph_lists = [positional_encoding(g, pos_enc_dim) for g in self.test[split_num].graph_lists]





