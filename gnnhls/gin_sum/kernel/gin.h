#ifndef GIN_H
#define GIN_H

#include "../defines.h"
#include "../data/defines_gin.h"
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

// define vector types
// typedef hls::vector<TYPE, FEATS_IN> vec_ft_in_t;
#define W_FT_IN 64
#define W_FT_OUT 64
#define W_FT_HIDDEN 64
// #define W_FT_HIDDEN 16
# define D 16 // D <= W_FT_IN
// # define D 1

typedef hls::vector<TYPE, W_FT_IN> vec_ft_in_t;
typedef hls::vector<TYPE, W_FT_OUT> vec_ft_out_t;
typedef hls::vector<TYPE, W_FT_HIDDEN> vec_ft_hidden_t;

typedef hls::vector<TYPE, FEATS_OUT> vec_ft_out_full_t;
typedef hls::vector<TYPE, FEATS_HIDDEN> vec_ft_hidden_full_t;

typedef hls::vector<TYPE, D> vec_d_t;
// typedef hls::vector<vec_d_t, W_FT_IN/D> vec_ft_in_d_t;
// typedef hls::vector<TYPE, 16> vec_type16_t;
// typedef hls::vector<TYPE, 32> vec_type32_t;

extern "C"{
int gin_hls(node_t nod_src[N_NODES],
            edge_src_t edge_src[N_EDGES],
            vec_ft_in_t ft_in_agg_mat[N_NODES*FEATS_IN/W_FT_IN],
            vec_ft_in_t ft_in_tar_mat[N_NODES*FEATS_IN/W_FT_IN],
            // TYPE ft_in_mat[N_NODES*FEATS_IN],
            // TYPE ft_h_mat[N_NODES*FEATS_IN],
            TYPE w_mlp0_mat[FEATS_IN*FEATS_HIDDEN],
            // TYPE rst_mlp0_mat[N_NODES*FEATS_HIDDEN],
            TYPE w_mlp1_mat[FEATS_HIDDEN*FEATS_OUT],
            node_index_t nidx_begin,
            node_index_t nidx_end,
            // TYPE rst_mat[N_NODES*FEATS_OUT]
            vec_ft_out_t rst_mat[N_NODES*FEATS_OUT/W_FT_OUT]);
}

#endif