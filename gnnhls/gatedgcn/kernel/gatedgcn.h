#ifndef GATEDGCN_H
#define GATEDGCN_H

#include "../defines.h"
#include "../data/defines_gatedgcn.h"
#include <math.h>
#include <hls_stream.h>
#include <hls_vector.h>

extern "C" float expf(float);

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

// define vector types
// #define W_FT_IN 32
// #define W_FT_OUT 32

// # define D 8 // D <= W_FT_IN

#define W_FT_IN 16
#define W_FT_OUT 16

# define D 8 // D <= W_FT_IN

typedef hls::vector<TYPE, W_FT_IN> vec_ft_in_t;
typedef hls::vector<TYPE, W_FT_OUT> vec_ft_out_t;

typedef hls::vector<TYPE, FEATS_IN> vec_ft_in_full_t;
typedef hls::vector<TYPE, FEATS_OUT> vec_ft_out_full_t;

typedef struct node_t_struct {
  edge_index_t edge_begin;
  edge_index_t edge_end;
} node_t;

extern "C"{
int gatedgcn_hls(node_t nod_src[N_NODES],
            edge_src_t edge_src[N_EDGES],
            edge_index_t edge_idx[N_EDGES],
            // TYPE ft_in_mat[N_NODES*FEATS_IN],
            // TYPE ft_in_agg_mat[N_NODES*FEATS_IN],
            vec_ft_in_t ft_in_agg_mat[N_NODES*FEATS_IN/W_FT_IN],
            vec_ft_in_t ft_in_tar_mat[N_NODES*FEATS_IN/W_FT_IN],
            vec_ft_in_t ft_in_e_mat[N_EDGES*FEATS_IN/W_FT_IN],

            TYPE w_a_mat[FEATS_IN*FEATS_OUT],
            TYPE w_b_mat[FEATS_IN*FEATS_OUT],
            TYPE w_c_mat[FEATS_IN*FEATS_OUT],
            TYPE w_d_mat[FEATS_IN*FEATS_OUT],
            TYPE w_e_mat[FEATS_IN*FEATS_OUT],
            // TYPE wbmv_bn_he_mat[8*FEATS_OUT],

            // TYPE ft_ah_mat[N_NODES*FEATS_OUT],
            // TYPE ft_bh_mat[N_NODES*FEATS_OUT],
            // TYPE ft_ce_mat[N_EDGES*FEATS_OUT],
            // TYPE ft_dh_mat[N_NODES*FEATS_OUT],
            // TYPE ft_eh_mat[N_NODES*FEATS_OUT],

            // TYPE ft_sigma_mat[N_EDGES*FEATS_OUT],
            // TYPE ft_sigmasum_mat[N_NODES*FEATS_OUT],
            // TYPE ft_sigmasum_h_mat[N_NODES*FEATS_OUT],

            node_index_t nidx_begin,
            node_index_t nidx_end,
            
            // TYPE rst_e_mat[N_EDGES*FEATS_OUT],
            vec_ft_out_t rst_e_mat[N_EDGES*FEATS_OUT/W_FT_OUT],
            // TYPE rst_h_mat[N_NODES*FEATS_OUT]
            vec_ft_out_t rst_h_mat[N_NODES*FEATS_OUT/W_FT_OUT]);
}

#endif