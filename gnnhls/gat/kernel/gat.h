#ifndef GAT_H
#define GAT_H

#include "../defines.h"
#include "../data/defines_gat.h"
#include <math.h>
#include <hls_stream.h>
#include <hls_vector.h>

extern "C" float expf(float);
// #include <cmath>
// #include <omp.h>

typedef struct edge_t_struct {
  // These fields are common in practice, but we elect not to use them.
  //weight_t weight;
  //node_index_t src;
  node_index_t dst;
} edge_dst_t;

typedef struct edge_gas_t_struct {
  // These fields are common in practice, but we elect not to use them.
  // weight_t weight;
  node_index_t src;
  // node_index_t dst;
} edge_src_t;

typedef struct node_t_struct {
  edge_index_t edge_begin;
  edge_index_t edge_end;
} node_t;

// define vector types
#define W_FT_IN 64

# define D 16 // D <= W_FT_IN

typedef hls::vector<TYPE, W_FT_IN> vec_ft_in_t;
typedef hls::vector<TYPE, HEADS_NUM> vec_head_full_t;
typedef hls::vector<TYPE, FEATS_OUT> vec_ft_out_full_t;
typedef hls::vector<TYPE, HEADS_NUM*FEATS_OUT> vec_head_ft_out_full_t;

// extern "C"{
// int gat_hls(node_t nod_src[N_NODES],
//             edge_src_t edge_src[N_EDGES],
//             // edge_index_t edge_idx[N_EDGES],
//             TYPE ft_in_mat[N_NODES*FEATS_IN],
//             TYPE w_fc_mat[FEATS_IN*HEADS_NUM*FEATS_OUT],
//             TYPE ft_fc_mat[N_NODES*HEADS_NUM*FEATS_OUT],
//             TYPE attn_l_mat[HEADS_NUM*FEATS_OUT],
//             TYPE attn_r_mat[HEADS_NUM*FEATS_OUT],
//             TYPE el_mat[N_NODES*HEADS_NUM],
//             TYPE er_mat[N_NODES*HEADS_NUM],
//             TYPE e_mat[N_EDGES*HEADS_NUM],
//             // TYPE a_mat[N_EDGES*HEADS_NUM],
//             TYPE rst_mat[N_NODES*HEADS_NUM*FEATS_OUT]);
// }

// extern "C"{
// int gat_hls(node_t nod_src[N_NODES],
//             edge_src_t edge_src[N_EDGES],
//             // edge_index_t edge_idx[N_EDGES],
//             TYPE ft_in_mat[N_NODES*FEATS_IN],
//             TYPE w_fc_mat[FEATS_IN*HEADS_NUM*FEATS_OUT],
//             TYPE ft_fc_mat[N_NODES*HEADS_NUM*FEATS_OUT],
//             TYPE attn_l_mat[HEADS_NUM*FEATS_OUT],
//             TYPE attn_r_mat[HEADS_NUM*FEATS_OUT],
//             TYPE el_mat[N_NODES*HEADS_NUM],
//             TYPE er_mat[N_NODES*HEADS_NUM],
//             TYPE e_mat[N_EDGES*HEADS_NUM],
//             // TYPE a_mat[N_EDGES*HEADS_NUM],
//             node_index_t nidx_begin,
//             node_index_t nidx_end,
//             TYPE rst_mat[N_NODES*HEADS_NUM*FEATS_OUT]);
// }

extern "C"{
int gat_hls_kern1(// TYPE ft_in_mat[N_NODES*FEATS_IN],
            vec_ft_in_t ft_in_mat[N_NODES*FEATS_IN/W_FT_IN],
            TYPE w_fc_mat[FEATS_IN*HEADS_NUM*FEATS_OUT],
            // TYPE ft_fc_mat[N_NODES*HEADS_NUM*FEATS_OUT],
            vec_ft_out_full_t ft_fc_mat[N_NODES*HEADS_NUM],
            TYPE attn_l_mat[HEADS_NUM*FEATS_OUT],
            TYPE attn_r_mat[HEADS_NUM*FEATS_OUT],
            TYPE el_mat[N_NODES*HEADS_NUM],
            TYPE er_mat[N_NODES*HEADS_NUM],
            node_index_t nidx_begin,
            node_index_t nidx_end);
}

// extern "C"{
// int gat_hls_kern2(node_t nod_src[N_NODES],
//             edge_src_t edge_src[N_EDGES],
//             vec_ft_out_full_t ft_fc_mat[N_NODES*HEADS_NUM],
//             vec_head_full_t el_mat[N_NODES],
//             vec_head_full_t er_mat[N_NODES],
//             TYPE e_mat[N_EDGES*HEADS_NUM],
//             node_index_t nidx_begin,
//             node_index_t nidx_end,
//             vec_ft_out_full_t rst_mat[N_NODES*HEADS_NUM]);
// }

extern "C"{
int gat_hls_kern2(node_t nod_src[N_NODES],
            edge_src_t edge_src[N_EDGES],
            edge_src_t edge_src2[N_EDGES],
            // TYPE ft_fc_mat[N_NODES*HEADS_NUM*FEATS_OUT],
            vec_ft_out_full_t ft_fc_mat[N_NODES*HEADS_NUM],
            TYPE el_mat[N_NODES*HEADS_NUM],
            TYPE er_mat[N_NODES*HEADS_NUM],
            TYPE el_mat2[N_NODES*HEADS_NUM],
            TYPE er_mat2[N_NODES*HEADS_NUM],
            node_index_t nidx_begin,
            node_index_t nidx_end,
            vec_ft_out_full_t rst_mat[N_NODES*HEADS_NUM]);
}

#endif