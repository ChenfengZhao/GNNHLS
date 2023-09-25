




"""
    IMPORTING LIBS
"""
from data.OGBG import creat_path
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from data.infer_data_gen import *

from tensorboardX import SummaryWriter
from tqdm import tqdm

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self
        


"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from nets.OGBG_graph_classification.load_net import gnn_model # import GNNs
from data.data import LoadData # import dataset




"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device




"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param

"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs):
    avg_test_acc = []
    avg_train_acc = []
    avg_epochs = []

    t0 = time.time()
    per_epoch_time = []

    dataset = LoadData(DATASET_NAME)
    
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    if net_params['pos_enc']:
        print("[!] Adding graph positional encoding.")
        dataset._add_positional_encodings(net_params['pos_enc_dim'])
        
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
            
        t0_split = time.time()
        log_dir = os.path.join(root_log_dir, "RUN_0")
        writer = SummaryWriter(log_dir=log_dir)

        # setting seeds
        random.seed(params['seed'])
        np.random.seed(params['seed'])
        torch.manual_seed(params['seed'])
        if device.type == 'cuda':
            torch.cuda.manual_seed(params['seed'])

        print("RUN NUMBER: ", 0)
        print("trainset", trainset, type(trainset))
        print("Training Graphs: ", len(trainset))
        print("Validation Graphs: ", len(valset))
        print("Test Graphs: ", len(testset))
        print("Number of Classes: ", net_params['n_classes'])

        model = gnn_model(MODEL_NAME, net_params)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=params['lr_reduce_factor'],
                                                            patience=params['lr_schedule_patience'],
                                                            verbose=True)

        epoch_train_losses, epoch_val_losses = [], []
        epoch_train_accs, epoch_val_accs = [], [] 

        # batching exception for Diffpool
        drop_last = True if MODEL_NAME == 'DiffPool' else False
        # drop_last = False

        
        if MODEL_NAME in ['RingGNN', '3WLGNN']:
            # import train functions specific for WL-GNNs
            from train.train_OGBG_graph_classification import train_epoch_dense as train_epoch, evaluate_network_dense as evaluate_network
            from functools import partial # util function to pass pos_enc flag to collate function

            train_loader = DataLoader(trainset, shuffle=True, collate_fn=partial(dataset.collate_dense_gnn, pos_enc=net_params['pos_enc']))
            val_loader = DataLoader(valset, shuffle=False, collate_fn=partial(dataset.collate_dense_gnn, pos_enc=net_params['pos_enc']))
            test_loader = DataLoader(testset, shuffle=False, collate_fn=partial(dataset.collate_dense_gnn, pos_enc=net_params['pos_enc']))
        
        # use different collate func for Monet and GatedGCN to avoid the error:
        # ValueError: Expect graph 0 and 4 to have the same node attributes when node_attrs=ALL, got {'feat'} and {'h', 'deg', 'feat'}
        elif MODEL_NAME in ['MoNet', 'GatedGCN']:
            # import train functions for Monet and GatedGCN
            from train.train_OGBG_graph_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network
            # collate is changed to collate_feat_only
            train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate_feat_only)
            val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate_feat_only)
            test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate_feat_only)

        else:
            # import train functions for all other GCNs
            from train.train_OGBG_graph_classification import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

            train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
            val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
            test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)

    
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)    

                start = time.time()
                
                if MODEL_NAME in ['RingGNN', '3WLGNN']: # since different batch training function for dense GNNs
                    epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, params['batch_size'])
                else:   # for all other models common train function
                    epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                
                #epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch)

                _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch)
                
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_accs.append(epoch_train_acc)
                epoch_val_accs.append(epoch_val_acc)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                
                epoch_train_acc = 100.* epoch_train_acc
                epoch_test_acc = 100.* epoch_test_acc
                
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                                train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                                test_acc=epoch_test_acc)  

                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_0")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                    
                # Stop training after params['max_time'] hours
                if time.time()-t0_split > params['max_time']*3600/10:       # Dividing max_time by 10, since there are 10 runs in TUs
                    print('-' * 89)
                    print("Max_time for one train-val-test split experiment elapsed {:.3f} hours, so stopping".format(params['max_time']/10))
                    break

        _, test_acc = evaluate_network(model, device, test_loader, epoch)   
        _, train_acc = evaluate_network(model, device, train_loader, epoch)    
        avg_test_acc.append(test_acc)   
        avg_train_acc.append(train_acc)
        avg_epochs.append(epoch)

        print("Test Accuracy [LAST EPOCH]: {:.4f}".format(test_acc))
        print("Train Accuracy [LAST EPOCH]: {:.4f}".format(train_acc))
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
        
    
    print("TOTAL TIME TAKEN: {:.4f}hrs".format((time.time()-t0)/3600))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    # Final test accuracy value averaged over 5-fold
    print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}"""          .format(np.mean(np.array(avg_test_acc))*100, np.std(avg_test_acc)*100))
    print("\nAll splits Test Accuracies:\n", avg_test_acc)
    print("""\n\n\nFINAL RESULTS\n\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}"""          .format(np.mean(np.array(avg_train_acc))*100, np.std(avg_train_acc)*100))
    print("\nAll splits Train Accuracies:\n", avg_train_acc)

    writer.close()

    """
        Write the results in out/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY averaged: {:.3f}\n with test acc s.d. {:.3f}\nTRAIN ACCURACY averaged: {:.3f}\n with train s.d. {:.3f}\n\n
    Convergence Time (Epochs): {:.3f}\nTotal Time Taken: {:.3f} hrs\nAverage Time Per Epoch: {:.3f} s\n\n\nAll Splits Test Accuracies: {}\n\nAll Splits Train Accuracies: {}"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  np.mean(np.array(avg_test_acc))*100, np.std(avg_test_acc)*100,
                  np.mean(np.array(avg_train_acc))*100, np.std(avg_train_acc)*100, np.mean(np.array(avg_epochs)),
               (time.time()-t0)/3600, np.mean(per_epoch_time), avg_test_acc, avg_train_acc))
        
    # save parameters into external file
    print("Saving param of %s network..." % MODEL_NAME)
    creat_path(net_params['out_dir']+'data')
    creat_path(net_params['out_dir']+'data/inter_rst')
    torch.save(model.state_dict(), net_params['out_dir']+'data/param_%s' % MODEL_NAME)



def main():    
    """
        USER CONTROLS
    """
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--graph_norm', help="Please give a value for graph_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")

    parser.add_argument('--infer_only', action='store_true', default=False, help="indicate only inference without train if used; otherwise inference with training")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.graph_norm is not None:
        net_params['graph_norm'] = True if args.graph_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = int(args.pos_enc_dim)
        
        
      
    
    # OGBG
    # print("dataset.all", dataset.all)
    # print("dataset.all[0]", dataset.all[0])
    # exit()
    net_params['in_dim'] = dataset.all[0][0].ndata['feat'].size(-1)
    net_params['in_dim_edge'] = dataset.all[0][0].edata['feat'].size(-1)
    net_params['n_classes'] = int(dataset.all.num_classes)
    net_params['out_dir'] = out_dir

    # print("net_params['n_classes']", net_params['n_classes'], type(net_params['n_classes']))
    # exit()
    
    # RingGNN
    if MODEL_NAME == 'RingGNN':
        num_nodes_train = [dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))]
        num_nodes_test = [dataset.test[i][0].number_of_nodes() for i in range(len(dataset.test))]
        num_nodes = num_nodes_train + num_nodes_test
        net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))
        
    # RingGNN, 3WLGNN
    if MODEL_NAME in ['RingGNN', '3WLGNN']:
        if net_params['pos_enc']:
            net_params['in_dim'] = net_params['pos_enc_dim']
            
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)

    if not args.infer_only:
        print("Start Training %s..." % MODEL_NAME)
        train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs)


    # ************** Setup datset and GNN model for inference **************
    # setup dataset
    # preprocessing dataset in the same way as train_val_pipeline
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    if net_params['pos_enc']:
        print("[!] Adding graph positional encoding.")
        dataset._add_positional_encodings(net_params['pos_enc_dim'])

    # setup GNN model
    model = gnn_model(MODEL_NAME, net_params)
    print("device", device)
    model = model.to(device)

    # load saved parameters to model
    pretrained_dict = load_module_state(model, net_params['out_dir']+'data/param_%s' % MODEL_NAME)

    # load saved parameters to model which is trained on GPU and inferenced on CPU
    # pretrained_dict = load_module_state_tocpu(model, net_params['out_dir']+'data/param_%s' % MODEL_NAME)

    print(pretrained_dict)
    print("Printing info of all pretrained parameters")
    for par_name, par in pretrained_dict.items():
        print("par_name:", par_name, "|", "par_size:", par.shape)


    # ************** start inference **************
    model.eval()

    creat_path(net_params['out_dir']+'data')
    creat_path(net_params['out_dir']+'data/inter_rst')

    # generate inferfence data for MP-GNN
    if MODEL_NAME not in ['RingGNN', '3WLGNN']:
        print("generating data for MP-GNN")

        # Convert a list of tuples to two lists
        graph_list, label_list = map(list, zip(*dataset.all))
        test_bg = dgl.batch(graph_list)

        # batched node features and edge features
        batch_x = test_bg.ndata['feat'].to(device)
        batch_e = test_bg.edata['feat'].to(device)

        # print info of the batched graph
        print("total vertex # of bg:", test_bg.number_of_nodes())
        print("total edge # of bg:", test_bg.number_of_edges())

        # start inferencing
        print("starting inference")
        with torch.no_grad():
            model.inference(test_bg, batch_x, batch_e)
        
        
        # convert bg to csr format
        if dgl.__version__ < "0.5":
            g_csr = test_bg.adjacency_matrix_scipy(transpose=False, fmt="csr", return_edge_ids=True)
        else:
            raise NotImplementedError("edge ids hasn't been tested for dgl.version >= 0.5")
            g_csr = test_bg.adj(transpose=False, scipy_fmt="csr")
        
        # Transform csr indptr to machsuite format
        a = g_csr.indptr[0: g_csr.indptr.size-1]
        b = g_csr.indptr[1: g_csr.indptr.size]
        c = np.empty(a.size+b.size, dtype=g_csr.indptr.dtype)
        c[0::2] = a
        c[1::2] = b

        # src node index pointers
        # print("g_csr.indptr_trans", c, type(c))
        np.savetxt(out_dir +"data/csr_indptr_trans.txt", c, fmt='%u', delimiter=",")

        # src node indices
        np.savetxt(out_dir +"data/csr_indices.txt", g_csr.indices, fmt='%u', delimiter=",")
        # edge indices
        np.savetxt(out_dir +"data/csr_data.txt", g_csr.data, fmt='%u', delimiter=",")


        # save parameters of the model
        if MODEL_NAME == 'GCN':
            print("generating data for GCN")

            # indegree for each node
            in_deg = test_bg.in_degrees().clamp(min=1)
            np.savetxt(out_dir +"data/in_deg.txt", in_deg, fmt='%u', delimiter=",")
            
            # save weight of the model
            print("pretrained_dict['layers.0.conv.weight']", type(pretrained_dict['layers.0.conv.weight']), pretrained_dict['layers.0.conv.weight'].size())
            if pretrained_dict["layers.0.conv.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_l0.txt", pretrained_dict["layers.0.conv.weight"].cpu().numpy(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_l0.txt", pretrained_dict["layers.0.conv.weight"].numpy(), delimiter=',')

            # generate header file for each dataset and model
            dataset_header_gen(out_dir + "data/defines_gcn.h", DATASET_NAME, test_bg, net_params['hidden_dim'], net_params['out_dim'])

        elif MODEL_NAME == 'GraphSage':
            print("generating data for GraphSage")

            if net_params['sage_aggregator'] == "mean":
                print("Saving parameters for GraphSage_mean (not built-in)")

                # indegree for each node
                in_deg = test_bg.in_degrees().clamp(min=1)
                np.savetxt(out_dir +"data/in_deg.txt", in_deg, fmt='%u', delimiter=",")

                # save weight of the model
                if pretrained_dict["layers.0.nodeapply.linear.weight"].is_cuda:
                    np.savetxt(out_dir +"data/weight_mlp.txt", pretrained_dict["layers.0.nodeapply.linear.weight"].cpu().numpy().transpose(), delimiter=',')
                    print("weight_mlp.dtype", pretrained_dict["layers.0.nodeapply.linear.weight"].cpu().numpy().transpose().dtype)
                    print("weight_mlp.shape", pretrained_dict["layers.0.nodeapply.linear.weight"].cpu().numpy().transpose().shape)
                else:
                    np.savetxt(out_dir +"data/weight_mlp.txt", pretrained_dict["layers.0.nodeapply.linear.weight"].numpy().transpose(), delimiter=',')
                    print("weight_mlp.dtype", pretrained_dict["layers.0.nodeapply.linear.weight"].numpy().transpose().dtype)
                    print("weight_mlp.shape", pretrained_dict["layers.0.nodeapply.linear.weight"].numpy().transpose().shape)

                # generate the header file for dataset
                dataset_header_gen(out_dir +"data/defines_graphsage_mean.h", DATASET_NAME, test_bg, net_params['hidden_dim'], net_params['out_dim'])
            
            elif net_params['sage_aggregator'] == "maxpool":
                print("Saving parameters for GraphSage_maxpool (not built-in)")

                # save weight of the model
                if pretrained_dict["layers.0.nodeapply.linear.weight"].is_cuda:
                    np.savetxt(out_dir +"data/weight_mlp.txt", pretrained_dict["layers.0.nodeapply.linear.weight"].cpu().numpy().transpose(), delimiter=',')
                    print("weight_mlp.dtype", pretrained_dict["layers.0.nodeapply.linear.weight"].cpu().numpy().transpose().dtype)
                    print("weight_mlp.shape", pretrained_dict["layers.0.nodeapply.linear.weight"].cpu().numpy().transpose().shape)
                else:
                    np.savetxt(out_dir +"data/weight_mlp.txt", pretrained_dict["layers.0.nodeapply.linear.weight"].numpy().transpose(), delimiter=',')
                    print("weight_mlp.dtype", pretrained_dict["layers.0.nodeapply.linear.weight"].numpy().transpose().dtype)
                    print("weight_mlp.shape", pretrained_dict["layers.0.nodeapply.linear.weight"].numpy().transpose().shape)

                if pretrained_dict["layers.0.aggregator.linear.weight"].is_cuda:
                    np.savetxt(out_dir +"data/weight_pool.txt", pretrained_dict["layers.0.aggregator.linear.weight"].cpu().numpy().transpose(), delimiter=',')
                    print("weight_pool.dtype", pretrained_dict["layers.0.aggregator.linear.weight"].cpu().numpy().transpose().dtype)
                    print("weight_pool.shape", pretrained_dict["layers.0.aggregator.linear.weight"].cpu().numpy().transpose().shape)
                else:
                    np.savetxt(out_dir +"data/weight_pool.txt", pretrained_dict["layers.0.aggregator.linear.weight"].numpy().transpose(), delimiter=',')
                    print("weight_pool.dtype", pretrained_dict["layers.0.aggregator.linear.weight"].numpy().transpose().dtype)
                    print("weight_pool.shape", pretrained_dict["layers.0.aggregator.linear.weight"].numpy().transpose().shape)

                # generate the header file for dataset
                dataset_header_gen(out_dir + "data/defines_graphsage_maxpool.h", DATASET_NAME, test_bg, net_params['hidden_dim'], net_params['out_dim'])
            else:
                raise NotImplementedError("Cannot recognize sage aggregator: %s." % net_params['sage_aggregator'])
        
        elif MODEL_NAME == 'GAT':
            print("generating data for GAT")

            # save weight of the model
            if pretrained_dict["layers.0.gatconv.fc.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_fc.txt", pretrained_dict["layers.0.gatconv.fc.weight"].cpu().numpy().transpose(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_fc.txt", pretrained_dict["layers.0.gatconv.fc.weight"].numpy().transpose(), delimiter=',')
                print("weight_fc.dtype", pretrained_dict["layers.0.gatconv.fc.weight"].numpy().transpose().dtype)
                print("weight_fc.shape", pretrained_dict["layers.0.gatconv.fc.weight"].numpy().transpose().shape)

            if pretrained_dict["layers.0.gatconv.attn_l"].is_cuda:
                np.savetxt(out_dir +"data/attn_l.txt", pretrained_dict["layers.0.gatconv.attn_l"].cpu().squeeze(0).numpy(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/attn_l.txt", pretrained_dict["layers.0.gatconv.attn_l"].squeeze(0).numpy(), delimiter=',')
                print("attn_l.dtype", pretrained_dict["layers.0.gatconv.attn_l"].squeeze(0).numpy().dtype)
                print("attn_l.shape", pretrained_dict["layers.0.gatconv.attn_l"].squeeze(0).numpy().shape)

            if pretrained_dict["layers.0.gatconv.attn_r"].is_cuda:
                np.savetxt(out_dir +"data/attn_r.txt", pretrained_dict["layers.0.gatconv.attn_r"].cpu().squeeze(0).numpy(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/attn_r.txt", pretrained_dict["layers.0.gatconv.attn_r"].squeeze(0).numpy(), delimiter=',')
                print("attn_r.dtype", pretrained_dict["layers.0.gatconv.attn_r"].squeeze(0).numpy().dtype)
                print("attn_r.shape", pretrained_dict["layers.0.gatconv.attn_r"].squeeze(0).numpy().shape)

            dataset_header_gen_mhead(out_dir +"data/defines_gat.h", DATASET_NAME, test_bg, net_params['hidden_dim'] * net_params['n_heads'], net_params['n_heads'], net_params['hidden_dim'])
            # dataset_header_gen_mhead(out_dir +"data/defines_gat.h", DATASET_NAME, test_bg, net_params['hidden_dim'] * net_params['n_heads'], 1, net_params['out_dim'])
        
        elif MODEL_NAME == 'MoNet':
            print("generating data for MoNet")
            # save weight of the model

            # weight of fc layer
            if pretrained_dict["layers.0.fc.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_fc.txt", pretrained_dict["layers.0.fc.weight"].cpu().numpy().transpose(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_fc.txt", pretrained_dict["layers.0.fc.weight"].numpy().transpose(), delimiter=',')
                # print info
                print("weight_fc.dtype", pretrained_dict["layers.0.fc.weight"].numpy().transpose().dtype)
                print("weight_fc.shape", pretrained_dict["layers.0.fc.weight"].numpy().transpose().shape)
            
            # weight of pseudo_proj
            if pretrained_dict["pseudo_proj.0.0.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_pseudo_proj.txt", pretrained_dict["pseudo_proj.0.0.weight"].cpu().numpy().transpose(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_pseudo_proj.txt", pretrained_dict["pseudo_proj.0.0.weight"].numpy().transpose(), delimiter=',')
                # print info
                print("weight_pseudo_proj.dtype", pretrained_dict["pseudo_proj.0.0.weight"].numpy().transpose().dtype)
                print("weight_pseudo_proj.shape", pretrained_dict["pseudo_proj.0.0.weight"].numpy().transpose().shape)
            
            # bias of pseudo_proj
            if pretrained_dict["pseudo_proj.0.0.bias"].is_cuda:
                np.savetxt(out_dir +"data/bias_pseudo_proj.txt", pretrained_dict["pseudo_proj.0.0.bias"].cpu().numpy(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/bias_pseudo_proj.txt", pretrained_dict["pseudo_proj.0.0.bias"].numpy(), delimiter=',')
                # print info
                print("bias_pseudo_proj.dtype", pretrained_dict["pseudo_proj.0.0.bias"].numpy().dtype)
                print("bias_pseudo_proj.shape", pretrained_dict["pseudo_proj.0.0.bias"].numpy().shape)

            # mu
            if pretrained_dict["layers.0.mu"].is_cuda:
                np.savetxt(out_dir +"data/mu.txt", pretrained_dict["layers.0.mu"].cpu().numpy(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/mu.txt", pretrained_dict["layers.0.mu"].numpy(), delimiter=',')
                # print info
                print("mu.dtype", pretrained_dict["layers.0.mu"].numpy().dtype)
                print("mu.shape", pretrained_dict["layers.0.mu"].numpy().shape)
            
            # inv_sigma
            if pretrained_dict["layers.0.inv_sigma"].is_cuda:
                np.savetxt(out_dir +"data/inv_sigma.txt", pretrained_dict["layers.0.inv_sigma"].cpu().numpy(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/inv_sigma.txt", pretrained_dict["layers.0.inv_sigma"].numpy(), delimiter=',')
                # print info
                print("inv_sigma.dtype", pretrained_dict["layers.0.inv_sigma"].numpy().dtype)
                print("inv_sigma.shape", pretrained_dict["layers.0.inv_sigma"].numpy().shape)
            
            # generate the header file for dataset
            dataset_header_gen_mkernel(out_dir +"data/defines_monet.h", DATASET_NAME, test_bg, net_params['hidden_dim'], net_params['kernel'], net_params['out_dim'], net_params['pseudo_dim_MoNet'])
        
        elif MODEL_NAME == 'GatedGCN':
            print("generating data for GatedGCN")

            # save weight of the model
            # weight of layer A
            if pretrained_dict["layers.0.A.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_a.txt", pretrained_dict["layers.0.A.weight"].cpu().numpy().transpose(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_a.txt", pretrained_dict["layers.0.A.weight"].numpy().transpose(), delimiter=',')
                # print info
                print("weight_a.dtype", pretrained_dict["layers.0.A.weight"].numpy().transpose().dtype)
                print("weight_a.shape", pretrained_dict["layers.0.A.weight"].numpy().transpose().shape)
            
            # weight of layer B
            if pretrained_dict["layers.0.B.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_b.txt", pretrained_dict["layers.0.B.weight"].cpu().numpy().transpose(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_b.txt", pretrained_dict["layers.0.B.weight"].numpy().transpose(), delimiter=',')
                print("weight_b.dtype", pretrained_dict["layers.0.B.weight"].numpy().transpose().dtype)
                print("weight_b.shape", pretrained_dict["layers.0.B.weight"].numpy().transpose().shape)
            
            # weight of layer C
            if pretrained_dict["layers.0.C.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_c.txt", pretrained_dict["layers.0.C.weight"].cpu().numpy().transpose(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_c.txt", pretrained_dict["layers.0.C.weight"].numpy().transpose(), delimiter=',')
                print("weight_c.dtype", pretrained_dict["layers.0.C.weight"].numpy().transpose().dtype)
                print("weight_c.shape", pretrained_dict["layers.0.C.weight"].numpy().transpose().shape)
            
            # weight of layer D
            if pretrained_dict["layers.0.D.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_d.txt", pretrained_dict["layers.0.D.weight"].cpu().numpy().transpose(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_d.txt", pretrained_dict["layers.0.D.weight"].numpy().transpose(), delimiter=',')
                print("weight_d.dtype", pretrained_dict["layers.0.D.weight"].numpy().transpose().dtype)
                print("weight_d.shape", pretrained_dict["layers.0.D.weight"].numpy().transpose().shape)
            
            # weight of layer E
            if pretrained_dict["layers.0.E.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_e.txt", pretrained_dict["layers.0.E.weight"].cpu().numpy().transpose(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_e.txt", pretrained_dict["layers.0.E.weight"].numpy().transpose(), delimiter=',')
                print("weight_e.dtype", pretrained_dict["layers.0.E.weight"].numpy().transpose().dtype)
                print("weight_e.shape", pretrained_dict["layers.0.E.weight"].numpy().transpose().shape)
            
            # weight and bias of BN for h and e respectively
            if net_params['batch_norm']:
                if pretrained_dict["layers.0.bn_node_h.weight"].is_cuda:
                    wbmv_bn_he = torch.cat((pretrained_dict["layers.0.bn_node_h.weight"].cpu().view(1,-1),
                        pretrained_dict["layers.0.bn_node_h.bias"].cpu().view(1,-1),
                        pretrained_dict["layers.0.bn_node_h.running_mean"].cpu().view(1,-1),
                        pretrained_dict["layers.0.bn_node_h.running_var"].cpu().view(1,-1),
                        pretrained_dict["layers.0.bn_node_e.weight"].cpu().view(1,-1),
                        pretrained_dict["layers.0.bn_node_e.bias"].cpu().view(1,-1),
                        pretrained_dict["layers.0.bn_node_e.running_mean"].cpu().view(1,-1),
                        pretrained_dict["layers.0.bn_node_e.running_var"].cpu().view(1,-1)),
                        dim=0)
                else:
                    wbmv_bn_he = torch.cat((pretrained_dict["layers.0.bn_node_h.weight"].view(1,-1),
                        pretrained_dict["layers.0.bn_node_h.bias"].view(1,-1),
                        pretrained_dict["layers.0.bn_node_h.running_mean"].view(1,-1),
                        pretrained_dict["layers.0.bn_node_h.running_var"].view(1,-1),
                        pretrained_dict["layers.0.bn_node_e.weight"].view(1,-1),
                        pretrained_dict["layers.0.bn_node_e.bias"].view(1,-1),
                        pretrained_dict["layers.0.bn_node_e.running_mean"].view(1,-1),
                        pretrained_dict["layers.0.bn_node_e.running_var"].view(1,-1)),
                        dim=0)
                np.savetxt(out_dir +"data/wbmv_bn_he.txt", wbmv_bn_he.numpy(), delimiter=',')
                print("wbmv_bn_he.dtype", wbmv_bn_he.numpy().dtype)
                print("wbmv_bn_he.shape", wbmv_bn_he.numpy().shape)
            
            # generate the header file for dataset
            dataset_header_gen(out_dir +"data/defines_gatedgcn.h", DATASET_NAME, test_bg, net_params['hidden_dim'], net_params['out_dim'])


        elif MODEL_NAME == 'GIN':
            print("generating data for GIN")

            # save weight of the model
            # weight of MLP layer 0
            if pretrained_dict["ginlayers.0.apply_func.mlp.linears.0.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_mlp_0.txt", pretrained_dict["ginlayers.0.apply_func.mlp.linears.0.weight"].cpu().numpy().transpose(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_mlp_0.txt", pretrained_dict["ginlayers.0.apply_func.mlp.linears.0.weight"].numpy().transpose(), delimiter=',')
                # print info
                print("weight_mlp_0.dtype", pretrained_dict["ginlayers.0.apply_func.mlp.linears.0.weight"].numpy().transpose().dtype)
                print("weight_mlp_0.shape", pretrained_dict["ginlayers.0.apply_func.mlp.linears.0.weight"].numpy().transpose().shape)

            # weight of MLP layer 1
            if pretrained_dict["ginlayers.0.apply_func.mlp.linears.1.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_mlp_1.txt", pretrained_dict["ginlayers.0.apply_func.mlp.linears.1.weight"].cpu().numpy().transpose(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_mlp_1.txt", pretrained_dict["ginlayers.0.apply_func.mlp.linears.1.weight"].numpy().transpose(), delimiter=',')
                # print info
                print("weight_mlp_1.dtype", pretrained_dict["ginlayers.0.apply_func.mlp.linears.1.weight"].numpy().transpose().dtype)
                print("weight_mlp_1.shape", pretrained_dict["ginlayers.0.apply_func.mlp.linears.1.weight"].numpy().transpose().shape)

            # generate the header file for dataset
            dataset_header_gen_gin(out_dir +"data/defines_gin.h", DATASET_NAME, test_bg, net_params['hidden_dim'], net_params['hidden_dim'], net_params['hidden_dim'], pretrained_dict["ginlayers.0.eps"])
        
        else:
            raise NotImplementedError("Cannot recognize current model: %s" % MODEL_NAME)
    # generate inference data for WL-GNN
    else:
        print("generating data for WL-GNN")

        # Convert a list of tuples to two lists
        print("converting a list of tuples to two lists")
        graph_list, label_list = map(list, zip(*dataset.all))

        # since the memory capcity is not large enough for all graphs, here we only inference the first 500 graphs if there are more graphs in the dataset
        if(len(graph_list) >= 500):
            graph_list = graph_list[:500]

        # calculating model parameters
        print("calculating model parameters")
        graph_sizes = [g.number_of_nodes() for g in graph_list]
        max_n = np.max(graph_sizes)
        batch_size = len(graph_list)
        dn = net_params['in_dim']
        in_feats = dn + 1

        # print info of the input
        print("max vertex # in all graphs:", max_n)
        print("total # of graphs in the dataset:", batch_size)
        print("dimension of each vertex (node) feature:", dn)
        print("input feature size of the model:", in_feats)

        # create batched node features (for layer0)
        
        print("generating input features")
        feature = torch.zeros(batch_size, in_feats, max_n, max_n)

        for i in range(batch_size):
            adj = sym_normalize_adj(graph_list[i].adjacency_matrix().to_dense())
            feature[i, 1, :graph_sizes[i], :graph_sizes[i]] = adj

            # processing node features
            for node, node_feat in enumerate(graph_list[i].ndata['feat']):
                feature[i, 1:1+dn, node, node] = node_feat
            
            # # processing edge feature
            # for edge, edge_feat in enumerate(graph_list[i].edata['edge']):
            #     feature[i, 1+dn:, edge[0], edge[1]] = edge_feat
        
        print("Copying input features to the device")
        feature = feature.to(device)

        # start inferencing
        print("starting inference")
        with torch.no_grad():
            model.inference(feature)

        if MODEL_NAME == 'RingGNN':
            print("generating data for RingGNN")
            # save weight of the model
            # coeffs_list
            if pretrained_dict["equi_modulelist.0.coeffs_list.0"].is_cuda:
                coeffs_list_all = torch.cat([pretrained_dict["equi_modulelist.0.coeffs_list.0"].cpu(), pretrained_dict["equi_modulelist.0.coeffs_list.1"].cpu(), pretrained_dict["equi_modulelist.0.coeffs_list.2"].cpu()])
                np.savetxt(out_dir +"data/coeffs_list_all.txt", coeffs_list_all.view(-1, coeffs_list_all.size(-1)).numpy(), delimiter=',')
            else:
                coeffs_list_all = torch.cat([pretrained_dict["equi_modulelist.0.coeffs_list.0"], pretrained_dict["equi_modulelist.0.coeffs_list.1"], pretrained_dict["equi_modulelist.0.coeffs_list.2"]])
                np.savetxt(out_dir +"data/coeffs_list_all.txt", coeffs_list_all.view(-1, coeffs_list_all.size(-1)).numpy(), delimiter=',')
            # print info
            print("coeffs_list_all.dtype", coeffs_list_all.numpy().dtype)
            print("coeffs_list_all.shape", coeffs_list_all.numpy().shape)

            # diag_bias_list
            if pretrained_dict["equi_modulelist.0.diag_bias_list.0"].is_cuda:
                diag_bias_list_all = torch.cat([pretrained_dict["equi_modulelist.0.diag_bias_list.0"].cpu(), pretrained_dict["equi_modulelist.0.diag_bias_list.1"].cpu(), pretrained_dict["equi_modulelist.0.diag_bias_list.2"].cpu()])
                np.savetxt(out_dir +"data/diag_bias_list_all.txt", diag_bias_list_all.view(-1, diag_bias_list_all.size(1)).numpy(), delimiter=',')
            else:
                diag_bias_list_all = torch.cat([pretrained_dict["equi_modulelist.0.diag_bias_list.0"], pretrained_dict["equi_modulelist.0.diag_bias_list.1"], pretrained_dict["equi_modulelist.0.diag_bias_list.2"]])
                np.savetxt(out_dir +"data/diag_bias_list_all.txt", diag_bias_list_all.view(-1, diag_bias_list_all.size(1)).numpy(), delimiter=',')
            # print info
            print("diag_bias_list_all.dtype", diag_bias_list_all.numpy().dtype)
            print("diag_bias_list_all.shape", diag_bias_list_all.numpy().shape)

            # switch
            if pretrained_dict["equi_modulelist.0.switch.0"].is_cuda:
                switch_weight = torch.cat([pretrained_dict["equi_modulelist.0.switch.0"].cpu(), pretrained_dict["equi_modulelist.0.switch.1"].cpu()])
                np.savetxt(out_dir +"data/switch_weight.txt", switch_weight.numpy(), delimiter=',')
            else:
                switch_weight = torch.cat([pretrained_dict["equi_modulelist.0.switch.0"], pretrained_dict["equi_modulelist.0.switch.1"]])
                np.savetxt(out_dir +"data/switch_weight.txt", switch_weight.numpy(), delimiter=',')
            # print info
            print("switch_weight.dtype", switch_weight.numpy().dtype)
            print("switch_weight.shape", switch_weight.numpy().shape)

            # all_bias
            if pretrained_dict["equi_modulelist.0.all_bias"].is_cuda:
                np.savetxt(out_dir +"data/all_bias.txt", pretrained_dict["equi_modulelist.0.all_bias"].cpu().view(-1, pretrained_dict["equi_modulelist.0.all_bias"].size(1)).numpy(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/all_bias.txt", pretrained_dict["equi_modulelist.0.all_bias"].view(-1, pretrained_dict["equi_modulelist.0.all_bias"].size(1)).numpy(), delimiter=',')
            
            # generate the header file for dataset
            dataset_header_gen_ringgnn(out_dir +"data/defines_ringgnn.h", DATASET_NAME, batch_size, in_feats, net_params['hidden_dim'], max_n)

            
        elif MODEL_NAME == '3WLGNN':
            print("generating data for 3WLGNN")
            # save weight of the model

            # weight of mlp1 layer 0
            if pretrained_dict["reg_blocks.0.mlp1.convs.0.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_mlp1_l0.txt", pretrained_dict["reg_blocks.0.mlp1.convs.0.weight"].cpu().view(-1, pretrained_dict["reg_blocks.0.mlp1.convs.0.weight"].size(-1)).numpy(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_mlp1_l0.txt", pretrained_dict["reg_blocks.0.mlp1.convs.0.weight"].view(-1, pretrained_dict["reg_blocks.0.mlp1.convs.0.weight"].size(-1)).numpy(), delimiter=',')
            # print info
            print("weight_mlp1_l0.dtype", pretrained_dict["reg_blocks.0.mlp1.convs.0.weight"].dtype)
            print("weight_mlp1_l0.shape", pretrained_dict["reg_blocks.0.mlp1.convs.0.weight"].shape)
            
            # weight of mlp1 layer 1
            if pretrained_dict["reg_blocks.0.mlp1.convs.1.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_mlp1_l1.txt", pretrained_dict["reg_blocks.0.mlp1.convs.1.weight"].cpu().view(-1, pretrained_dict["reg_blocks.0.mlp1.convs.1.weight"].size(-1)).numpy(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_mlp1_l1.txt", pretrained_dict["reg_blocks.0.mlp1.convs.1.weight"].view(-1, pretrained_dict["reg_blocks.0.mlp1.convs.1.weight"].size(-1)).numpy(), delimiter=',')
            # print info
            print("weight_mlp1_l1.dtype", pretrained_dict["reg_blocks.0.mlp1.convs.1.weight"].dtype)
            print("weight_mlp1_l1.shape", pretrained_dict["reg_blocks.0.mlp1.convs.1.weight"].shape)

            # weight of mlp2 layer 0
            if pretrained_dict["reg_blocks.0.mlp2.convs.0.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_mlp2_l0.txt", pretrained_dict["reg_blocks.0.mlp2.convs.0.weight"].cpu().view(-1, pretrained_dict["reg_blocks.0.mlp2.convs.0.weight"].size(-1)).numpy(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_mlp2_l0.txt", pretrained_dict["reg_blocks.0.mlp2.convs.0.weight"].view(-1, pretrained_dict["reg_blocks.0.mlp2.convs.0.weight"].size(-1)).numpy(), delimiter=',')
            # print info
            print("weight_mlp2_l0.dtype", pretrained_dict["reg_blocks.0.mlp2.convs.0.weight"].dtype)
            print("weight_mlp2_l0.shape", pretrained_dict["reg_blocks.0.mlp2.convs.0.weight"].shape)

            # weight of mlp2 layer 1
            if pretrained_dict["reg_blocks.0.mlp2.convs.1.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_mlp2_l1.txt", pretrained_dict["reg_blocks.0.mlp2.convs.1.weight"].cpu().view(-1, pretrained_dict["reg_blocks.0.mlp2.convs.1.weight"].size(-1)).numpy(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_mlp2_l1.txt", pretrained_dict["reg_blocks.0.mlp2.convs.1.weight"].view(-1, pretrained_dict["reg_blocks.0.mlp2.convs.1.weight"].size(-1)).numpy(), delimiter=',')
            # print info
            print("weight_mlp2_l1.dtype", pretrained_dict["reg_blocks.0.mlp2.convs.1.weight"].dtype)
            print("weight_mlp2_l1.shape", pretrained_dict["reg_blocks.0.mlp2.convs.1.weight"].shape)

            # weight of skip conv layer
            if pretrained_dict["reg_blocks.0.skip.conv.weight"].is_cuda:
                np.savetxt(out_dir +"data/weight_skip.txt", pretrained_dict["reg_blocks.0.skip.conv.weight"].cpu().view(-1, pretrained_dict["reg_blocks.0.skip.conv.weight"].size(-1)).numpy(), delimiter=',')
            else:
                np.savetxt(out_dir +"data/weight_skip.txt", pretrained_dict["reg_blocks.0.skip.conv.weight"].view(-1, pretrained_dict["reg_blocks.0.skip.conv.weight"].size(-1)).numpy(), delimiter=',')
            # print info
            print("weight_skip.dtype", pretrained_dict["reg_blocks.0.skip.conv.weight"].dtype)
            print("weight_skip.shape", pretrained_dict["reg_blocks.0.skip.conv.weight"].shape)

            dataset_header_gen_threewlgnn(out_dir +"data/defines_threewlgnn.h", DATASET_NAME, batch_size, net_params['depth_of_mlp'], in_feats, net_params['hidden_dim'], max_n)

        else:
            raise NotImplementedError("Cannot recognize current model: %s" % MODEL_NAME)

    
    
    
    
main()    
















