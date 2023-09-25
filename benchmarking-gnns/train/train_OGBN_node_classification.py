"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
import dgl

from train.metrics import accuracy_SBM as accuracy

import torch.nn.functional as F

"""
    For GCNs
"""
def train_epoch_sparse(model, optimizer, device, data_loader, epoch):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        try:
            batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
        except:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    
    return epoch_loss, epoch_train_acc, optimizer

def train_epoch_sparse_ogbn(model, optimizer, device, graph, labels, train_idx, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    # since there is only a single graph, there is no iteration here
    graph = graph.to(device)
    batch_x = graph.ndata['feat'].to(device)  # num x feat
    batch_e = graph.edata['feat'].to(device)
    batch_labels = labels.to(device)
    optimizer.zero_grad()

    # try:
    #     batch_pos_enc = graph.ndata['pos_enc'].to(device)
    #     sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
    #     sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
    #     batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
    #     batch_scores = model.forward(graph, batch_x, batch_e, batch_pos_enc)[train_idx]
    # except:
    batch_scores = model.forward(graph, batch_x, batch_e)[train_idx]
    
    loss = model.loss(batch_scores, batch_labels.squeeze(1)[train_idx])
    # print("batch_scores.size()", batch_scores.size())
    # print(batch_labels.squeeze(1)[train_idx].size())
    # print(batch_labels[train_idx].size())
    # loss = F.nll_loss(batch_scores, batch_labels.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    epoch_loss += loss.detach().item()
    print("backwork success!")
    epoch_train_acc += accuracy(batch_scores, batch_labels.squeeze(1)[train_idx])
    # epoch_loss /= (iter + 1)
    # epoch_train_acc /= (iter + 1)
    
    return epoch_loss, epoch_train_acc, optimizer




def evaluate_network_sparse(model, device, data_loader, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            except:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        
    return epoch_test_loss, epoch_test_acc



def evaluate_network_sparse_ogbn(model, device, graph, labels, split_idx, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        batch_x = graph.ndata['feat'].to(device)
        batch_e = graph.edata['feat'].to(device)
        batch_labels = labels.to(device)

        try:
            batch_pos_enc = graph.ndata['pos_enc'].to(device)
            batch_scores = model.forward(graph, batch_x, batch_e, batch_pos_enc)[split_idx]
        except:
            batch_scores = model.forward(graph, batch_x, batch_e)[split_idx]
        # loss = model.loss(batch_scores, batch_labels[split_idx]) 
        loss = F.nll_loss(batch_scores, batch_labels.squeeze(1)[split_idx])
        epoch_test_loss += loss.detach().item()
        epoch_test_acc += accuracy(batch_scores, batch_labels[split_idx])

        # epoch_test_loss /= (iter + 1)
        # epoch_test_acc /= (iter + 1)
    
    return epoch_test_loss, epoch_test_acc





"""
    For WL-GNNs
"""
def train_epoch_dense(model, optimizer, device, data_loader, epoch, batch_size):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_with_node_feat, labels) in enumerate(data_loader):
        x_with_node_feat = x_with_node_feat.to(device)
        labels = labels.to(device)

        scores = model.forward(x_with_node_feat)
        loss = model.loss(scores, labels)
        loss.backward()
        
        if not (iter%batch_size):
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(scores, labels)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= (iter + 1)
    
    return epoch_loss, epoch_train_acc, optimizer



def evaluate_network_dense(model, device, data_loader, epoch):
    
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_with_node_feat, labels) in enumerate(data_loader):
            x_with_node_feat = x_with_node_feat.to(device)
            labels = labels.to(device)
            
            scores = model.forward(x_with_node_feat)
            loss = model.loss(scores, labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(scores, labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)
        
    return epoch_test_loss, epoch_test_acc
