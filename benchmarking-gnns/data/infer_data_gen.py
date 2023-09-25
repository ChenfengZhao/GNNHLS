# This script is used to generate the dataset header file and save model parameter for each model

import torch

def dataset_header_gen(fn, G_name, G, in_feats, out_feats):
    """generate dataset header file for GCN, GraphSage, GatedGCN

    Parameters
    ----------
    fn : str
        path of the header file (<relerative path> + defines_ + <model name>+.h)
    G_name : str
        Graph dataset name
    G : DGL graph class
        DGL input graph (batched or single)
    in_feats : int
        input feature size
    out_feats : int
        output feature size
    """

    with open(fn, 'w') as f:
        header_path = fn.split('/')
        header_name = "_".join(header_path[-1].split('.'))
        header_name_cap = header_name.upper()

        model_name = header_name.split('_')[1]

        f.write("// Graph dataset: %s\n" % G_name.upper())
        f.write("// GNN model name: %s\n" % model_name.upper())
        f.write("\n")

        f.write("#ifndef %s\n\n" % header_name_cap)
        f.write("#define %s\n\n" % header_name_cap)

        f.write("//Graph Info\n")
        f.write("#define N_NODES %d\n" % G.number_of_nodes())
        f.write("#define N_EDGES %d\n" % G.number_of_edges())
        f.write("\n")
        f.write("#define MAX_IN_DEG %d\n" % torch.max(G.in_degrees()).numpy())
        f.write("#define MIN_IN_DEG %d\n" % torch.min(G.in_degrees()).numpy())
        f.write("\n")

        f.write("// GNN Layer Model Info\n")
        f.write("#define FEATS_IN %d\n" % in_feats)
        f.write("#define FEATS_OUT %d\n" % out_feats)
        f.write("\n")


        f.write("#endif")

def dataset_header_gen_mhead(fn, G_name, G, in_feats, num_heads, out_feats):
    """generate dataset header file for GAT

    Parameters
    ----------
    fn : str
        path of the header file (<relerative path> + defines_ + <model name>+.h)
    G_name : str
        Graph dataset name
    G : DGL graph class
        DGL input graph (batched or single)
    in_feats : int
        input feature size
    num_heads: int
        number of heads
    out_feats : int
        output feature size
    """

    with open(fn, 'w') as f:
        header_path = fn.split('/')
        header_name = "_".join(header_path[-1].split('.'))
        header_name_cap = header_name.upper()

        model_name = header_name.split('_')[1]

        f.write("// Graph dataset: %s\n" % G_name.upper())
        f.write("// GNN model name: %s\n" % model_name.upper())
        f.write("\n")

        f.write("#ifndef %s\n\n" % header_name_cap)
        f.write("#define %s\n\n" % header_name_cap)

        f.write("//Graph Info\n")
        f.write("#define N_NODES %d\n" % G.number_of_nodes())
        f.write("#define N_EDGES %d\n" % G.number_of_edges())
        f.write("\n")
        f.write("#define MAX_IN_DEG %d\n" % torch.max(G.in_degrees()).numpy())
        f.write("#define MIN_IN_DEG %d\n" % torch.min(G.in_degrees()).numpy())
        f.write("\n")

        f.write("// GNN Layer Model Info\n")
        f.write("#define FEATS_IN %d\n" % in_feats)
        f.write("#define HEADS_NUM %d\n"% num_heads)
        f.write("#define FEATS_OUT %d\n" % out_feats)
        f.write("\n")


        f.write("#endif")

def dataset_header_gen_mkernel(fn, G_name, G, in_feats, num_kernels, out_feats, dim_pseudo):
    """generate dataset header file for MoNET

    Parameters
    ----------
    fn : str
        path of the header file (<relerative path> + defines_ + <model name>+.h)
    G_name : str
        Graph dataset name
    G : DGL graph class
        DGL input graph (batched or single)
    in_feats : int
        input feature size
    num_kernels: int
        number of kernels
    out_feats : int
        output feature size
    dim_pseudo : int
        dim of pseudo
    """

    with open(fn, 'w') as f:
        header_path = fn.split('/')
        header_name = "_".join(header_path[-1].split('.'))
        header_name_cap = header_name.upper()

        model_name = header_name.split('_')[1]

        f.write("// Graph dataset: %s\n" % G_name.upper())
        f.write("// GNN model name: %s\n" % model_name.upper())
        f.write("\n")

        f.write("#ifndef %s\n\n" % header_name_cap)
        f.write("#define %s\n\n" % header_name_cap)

        f.write("//Graph Info\n")
        f.write("#define N_NODES %d\n" % G.number_of_nodes())
        f.write("#define N_EDGES %d\n" % G.number_of_edges())
        f.write("\n")
        f.write("#define MAX_IN_DEG %d\n" % torch.max(G.in_degrees()).numpy())
        f.write("#define MIN_IN_DEG %d\n" % torch.min(G.in_degrees()).numpy())
        f.write("\n")

        f.write("// GNN Layer Model Info\n")
        f.write("#define FEATS_IN %d\n" % in_feats)
        f.write("#define KERNELS_NUM %d\n"% num_kernels)
        f.write("#define FEATS_OUT %d\n" % out_feats)
        f.write("#define PSEUDO_DIM %d\n" % dim_pseudo)
        f.write("\n")


        f.write("#endif")

def dataset_header_gen_gin(fn, G_name, G, in_feats, hidden_feats, out_feats, gin_eps):
    """generate dataset header file for GIN

    Parameters
    ----------
    fn : str
        path of the header file (<relerative path> + defines_ + <model name>+.h)
    G_name : str
        Graph dataset name
    G : DGL graph class
        DGL input graph (batched or single)
    in_feats : int
        input feature size
    hidden_feats : int
        hidden feature size
    out_feats : int
        output feature size
    gin_eps : float
        eps of the gin layer
    """

    with open(fn, 'w') as f:
        header_path = fn.split('/')
        header_name = "_".join(header_path[-1].split('.'))
        header_name_cap = header_name.upper()

        model_name = header_name.split('_')[1]

        f.write("// Graph dataset: %s\n" % G_name.upper())
        f.write("// GNN model name: %s\n" % model_name.upper())
        f.write("\n")

        f.write("#ifndef %s\n\n" % header_name_cap)
        f.write("#define %s\n\n" % header_name_cap)

        f.write("//Graph Info\n")
        f.write("#define N_NODES %d\n" % G.number_of_nodes())
        f.write("#define N_EDGES %d\n" % G.number_of_edges())
        f.write("\n")
        f.write("#define MAX_IN_DEG %d\n" % torch.max(G.in_degrees()).numpy())
        f.write("#define MIN_IN_DEG %d\n" % torch.min(G.in_degrees()).numpy())
        f.write("\n")

        f.write("// GNN Layer Model Info\n")
        f.write("#define FEATS_IN %d\n" % in_feats)
        f.write("#define FEATS_HIDDEN %d\n" % hidden_feats)
        f.write("#define FEATS_OUT %d\n" % out_feats)
        f.write("#define GIN_EPS %e\n" % gin_eps)
        f.write("\n")


        f.write("#endif")

def dataset_header_gen_threewlgnn(fn, G_name, G_num, depth_of_mlp, in_feats, out_feats, n, conv_kern_size = 1):
    """generate dataset header file for 3WLGNN

    Parameters
    ----------
    fn : str
        path of the header file (<relerative path> + defines_ + <model name>+.h)
    G_name : str
        Graph dataset name
    G_num : int
        number of graph in the batch (i.e. Batch size)
    depth_of_mlp : int
        depth of mlp
    in_feats : int
        input feature size
    out_feats : int
        output feature size
    n : int
        max graph size in the batch (graph size is the num of nodes)
    
    Notes
    ----------
    The kernel size of all the 2DConv is set to 1 by default.
    """

    with open(fn, 'w') as f:
        header_path = fn.split('/')
        header_name = "_".join(header_path[-1].split('.'))
        header_name_cap = header_name.upper()

        model_name = header_name.split('_')[1]

        f.write("// Graph dataset: %s\n" % G_name.upper())
        f.write("// GNN model name: %s\n" % model_name.upper())
        f.write("\n")

        f.write("#ifndef %s\n\n" % header_name_cap)
        f.write("#define %s\n\n" % header_name_cap)

        f.write("//Graph Info\n")
        f.write("//BATCH_SIZE is the number of graphs in the batch\n")
        f.write("#define BATCH_SIZE %d\n" % G_num)
        f.write("#define MAX_N %d\n" % n)
        f.write("\n")

        f.write("// GNN Layer Model Info\n")
        f.write("#define MLP_DEPTH %d\n" % depth_of_mlp)
        f.write("#define FEATS_IN %d\n" % in_feats)
        f.write("#define FEATS_OUT %d\n" % out_feats)
        f.write("#define CONV_KERN_SIZE %d\n" % conv_kern_size)
        f.write("\n")


        f.write("#endif")

def dataset_header_gen_ringgnn(fn, G_name, G_num, in_feats, out_feats, n, radius=2):
    """generate  dataset header file for RingGNN

    Parameters
    ----------
    fn : str
        path of the header file (<relerative path> + defines_ + <model name>+.h)
    G_name : str
        Graph dataset name
    G_num : int
        number of graph in the batch (i.e. Batch size)
    in_feats : int
        input feature size
    out_feats : int
        output feature size
    n : int
        max graph size in the batch (graph size is the num of nodes)
    radius : int
        used to control the number of loop inside ringgnn
    
    Notes
    ----------
    The kernel size of all the 2DConv is set to 1 by default.
    """

    with open(fn, 'w') as f:
        header_path = fn.split('/')
        header_name = "_".join(header_path[-1].split('.'))
        header_name_cap = header_name.upper()

        model_name = header_name.split('_')[1]

        f.write("// Graph dataset: %s\n" % G_name.upper())
        f.write("// GNN model name: %s\n" % model_name.upper())
        f.write("\n")

        f.write("#ifndef %s\n\n" % header_name_cap)
        f.write("#define %s\n\n" % header_name_cap)

        f.write("//Graph Info\n")
        f.write("//BATCH_SIZE is the number of graphs in the batch\n")
        f.write("#define BATCH_SIZE %d\n" % G_num)
        f.write("#define MAX_N %d\n" % n)
        f.write("\n")

        f.write("// GNN Layer Model Info\n")
        f.write("#define FEATS_IN %d\n" % in_feats)
        f.write("#define FEATS_OUT %d\n" % out_feats)
        f.write("#define RADIUS %d\n" % radius)
        f.write("#define W_MAT_NUM %d\n" % (radius * (radius+1) / 2))
        f.write("\n")


        f.write("#endif")

def load_module_state(model, state_name):
    """Load parameters to the target gnn model on the same device as training 
        from an external file

    Parameters
    ----------
    model : class
        The target GNN model
    state_name : str
        The path + fn of the external file

    Returns
    -------
    pretrained_dict : dict
        dict of parameters (param_name: param)
    
    Notes
    -------
    train on CPU -> inference on CPU
    train on GPU -> inference on GPU
    """    
    pretrained_dict = torch.load(state_name)
    model_dict = model.state_dict()

    # to delete, to correct grud names
    '''
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('grud_forward'):
            new_dict['grud'+k[12:]] = v
        else:
            new_dict[k] = v
    pretrained_dict = new_dict
    '''

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return pretrained_dict


def load_module_state_tocpu(model, state_name):
    """Load parameters from an external file generated from GPU to the target gnn model on CPU
        

    Parameters
    ----------
    model : class
        The target GNN model
    state_name : str
        The path + fn of the external file

    Returns
    -------
    pretrained_dict : dict
        dict of parameters (param_name: param)
    
    Notes
    -------
    train on GPU -> inference on CPU
    """
    device = torch.device('cpu')
    pretrained_dict = torch.load(state_name, map_location=device)
    model_dict = model.state_dict()

    # to delete, to correct grud names
    '''
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('grud_forward'):
            new_dict['grud'+k[12:]] = v
        else:
            new_dict[k] = v
    pretrained_dict = new_dict
    '''

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return pretrained_dict

def sym_normalize_adj(adj):
    deg = torch.sum(adj, dim = 0)#.squeeze()
    deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
    deg_inv = torch.diag(deg_inv)
    return torch.mm(deg_inv, torch.mm(adj, deg_inv))