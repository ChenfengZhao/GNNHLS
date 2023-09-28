#!/bin/bash
set -e

# DATASET="ogbg-molhiv"
DATASET="ogbg-moltox21"

# ONLY_INFER="--infer_only" # comment it to train and inference; other only inference

# ********** GCN **********
GNN_NAME="GCN"

# input feature size = output feature size = 128
python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k_128.json" ${ONLY_INFER}

# ********** GraphSage_mean **********
# GNN_NAME="GraphSage_mean"

# # input feature siize = output feature size = 128
# python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k_128.json" ${ONLY_INFER}

# ********** GAT **********
# GNN_NAME="GAT"

# # input feature = head * out feature = 128
# python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k_128.json" ${ONLY_INFER}

# ********** GIN **********
# GNN_NAME="GIN_sum"

# # input feature siize = output feature size = 128
# python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k_128.json" ${ONLY_INFER}

# ********** MoNet **********
# GNN_NAME="MoNet"

# # input feature siize = output feature size = 64
# python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k_64.json" ${ONLY_INFER}

# ********** GatedGCN **********
# GNN_NAME="GatedGCN"

# # input feature siize = output feature size = 32
# python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k_32.json" ${ONLY_INFER}
