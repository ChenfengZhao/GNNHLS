#!/bin/bash
set -e

DATASET="ogbn-arxiv"
# DATASET="ogbn-proteins"
# DATASET="ogbn-products"

# ONLY_INFER="--infer_only" # comment it to train and inference; other only inference
# GPU_CONFIG="--gpu_id=1"

# ********** GCN **********
# GNN_NAME="GCN"

# # input feature size = output feature size = 128
# python main_infer_OGBN_node_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbn/OGBN_node_clustering_${GNN_NAME}_${DATASET}_100k_128.json" ${ONLY_INFER}

# ********** GraphSage_mean **********
# GNN_NAME="GraphSage_mean"

# # input feature siize = output feature size = 128
# python main_infer_OGBN_node_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbn/OGBN_node_clustering_${GNN_NAME}_${DATASET}_100k_128.json" ${ONLY_INFER}

# ********** GraphSage_maxpool **********
# GNN_NAME="GraphSage_maxpool"
# python main_infer_OGBN_node_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbn/OGBN_node_clustering_${GNN_NAME}_${DATASET}_100k.json"

# 

# ********** GAT **********
# GNN_NAME="GAT"

# # input feature = head * out feature = 128
# python main_infer_OGBN_node_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbn/OGBN_node_clustering_${GNN_NAME}_${DATASET}_100k_128.json" ${ONLY_INFER}

# ********** GIN **********
# GNN_NAME="GIN_sum"

# # input feature siize = output feature size = 128
# python main_infer_OGBN_node_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbn/OGBN_node_clustering_${GNN_NAME}_${DATASET}_100k_128.json" ${ONLY_INFER}


# ********** MoNet **********
# GNN_NAME="MoNet"

# # input feature siize = output feature size = 64
# python main_infer_OGBN_node_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbn/OGBN_node_clustering_${GNN_NAME}_${DATASET}_100k_64.json" ${ONLY_INFER}

# ********** GatedGCN **********
GNN_NAME="GatedGCN"

# # input feature siize = output feature size = 64
# python main_infer_OGBN_node_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbn/OGBN_node_clustering_${GNN_NAME}_${DATASET}_100k_64.json" ${ONLY_INFER}

# input feature siize = output feature size = 32
python main_infer_OGBN_node_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbn/OGBN_node_clustering_${GNN_NAME}_${DATASET}_100k_32.json" ${ONLY_INFER}

# input feature size = output feature size = 16 for ogbn-protein
# python main_infer_OGBN_node_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbn/OGBN_node_clustering_${GNN_NAME}_${DATASET}_100k_16.json" ${ONLY_INFER}
