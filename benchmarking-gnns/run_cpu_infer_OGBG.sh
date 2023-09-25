#!/bin/bash
set -e

# DATASET="ogbg-molhiv"
DATASET="ogbg-moltox21"

# ONLY_INFER="--infer_only" # comment it to train and inference; other only inference
# GPU_CONFIG="--gpu_id=0"

# ********** GCN **********
# GNN_NAME="GCN"

# # input feature size = output feature size = 146
# # python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k.json" ${ONLY_INFER}

# # input feature size = output feature size = 144
# # python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k_144.json" ${ONLY_INFER}

# # input feature size = output feature size = 128
# python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k_128.json" ${ONLY_INFER}

# # input feature size = output feature size = 16
# # python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k_16.json" ${ONLY_INFER}

# ********** GraphSage_mean **********
# GNN_NAME="GraphSage_mean"

# # input feature siize = output feature size = 128
# python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k_128.json" ${ONLY_INFER}

# ********** GraphSage_maxpool **********
# GNN_NAME="GraphSage_maxpool"

# python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k.json"

# ********** GAT **********
# GNN_NAME="GAT"

# # # input feature = head * out feature = 144
# # python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k.json" ${ONLY_INFER}

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
GNN_NAME="GatedGCN"

# # input feature siize = output feature size = 64
# python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k_64.json" ${ONLY_INFER}

# input feature siize = output feature size = 32
python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k_32.json" ${ONLY_INFER}


# ********** 3WLGNN **********
# GNN_NAME="3WLGNN"

# # output feature = 16
# python main_infer_OGBG_graph_classification.py --dataset=${DATASET} ${GPU_CONFIG} --config="configs_infer_ogbg/OGBG_graph_classification_${GNN_NAME}_${DATASET}_100k_16.json" ${ONLY_INFER}