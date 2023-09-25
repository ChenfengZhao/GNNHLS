#ifndef GCN_H
#define GCN_H

#include "../defines.h"
#include "../data/defines_gcn.h"
#include <hls_stream.h>
#include <hls_vector.h>

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

// extern "C"{
// int gcn_hls(node_t nod_src[N_NODES],
//             edge_src_t edge_src[N_EDGES],
//             TYPE ft_in_mat[N_NODES*FEATS_IN],
//             // TYPE ft_h_mat[N_NODES*FEATS_IN],
//             TYPE w_mat[FEATS_IN*FEATS_OUT],
//             // int deg[N_NODES],
//             TYPE rst_mat[N_NODES*FEATS_OUT]);
// }

#define W_FT_IN 64
#define W_FT_OUT 64

# define D 16 // D <= W_FT_IN

typedef hls::vector<TYPE, W_FT_IN> vec_ft_in_t;
typedef hls::vector<TYPE, FEATS_IN> vec_ft_in_full_t;
typedef hls::vector<TYPE, W_FT_OUT> vec_ft_out_t;
typedef hls::vector<TYPE, FEATS_OUT> vec_ft_out_full_t;

// extern "C"{
// int gcn_hls(node_t nod_src[N_NODES],
//             edge_src_t edge_src[N_EDGES],
//             TYPE ft_in_mat[N_NODES*FEATS_IN],
//             // TYPE ft_h_mat[N_NODES*FEATS_IN],
//             TYPE w_mat[FEATS_IN*FEATS_OUT],
//             // int deg[N_NODES],
//             node_index_t nidx_begin,
//             node_index_t nidx_end,
//             TYPE rst_mat[N_NODES*FEATS_OUT]);
// }

extern "C"{
int gcn_hls(node_t nod_src[N_NODES],
            edge_src_t edge_src[N_EDGES],
            vec_ft_in_t ft_in_mat[N_NODES*FEATS_IN/W_FT_IN],
            // TYPE ft_h_mat[N_NODES*FEATS_IN],
            TYPE w_mat[FEATS_IN*FEATS_OUT],
            // int deg[N_NODES],
            node_index_t nidx_begin,
            node_index_t nidx_end,
            vec_ft_out_t rst_mat[N_NODES*FEATS_OUT/W_FT_OUT]);
}

#endif