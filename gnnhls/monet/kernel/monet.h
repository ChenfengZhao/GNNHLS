#ifndef MONET_H
#define MONET_H

#include "../defines.h"
#include "../data/defines_monet.h"
#include <math.h>
#include <hls_stream.h>
#include <hls_vector.h>

extern "C" float tanhf(float);
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


typedef struct node_t_struct {
  edge_index_t edge_begin;
  edge_index_t edge_end;
} node_t;

#define W_FT_IN 32

# define D 8 // D <= W_FT_IN

typedef hls::vector<TYPE, 2> vec_type2_t;
typedef hls::vector<TYPE, W_FT_IN> vec_ft_in_t;
typedef hls::vector<TYPE, FEATS_IN> vec_ft_in_full_t;
typedef hls::vector<TYPE, KERNELS_NUM> vec_kern_num_full_t;
typedef hls::vector<TYPE, PSEUDO_DIM> vec_pseudo_dim_full_t;
typedef hls::vector<TYPE, FEATS_OUT> vec_ft_out_full_t;
typedef hls::vector<TYPE, KERNELS_NUM*FEATS_IN> vec_kern_num_ft_in_full_t;
typedef hls::vector<TYPE, KERNELS_NUM*FEATS_OUT> vec_kern_num_ft_out_full_t;

extern "C"{
int monet_hls(node_t nod_src[N_NODES],
            edge_src_t edge_src[N_EDGES],
            edge_index_t edge_idx[N_EDGES],
            // TYPE ft_in_mat[N_NODES*FEATS_IN],
            vec_ft_in_full_t ft_in_mat[N_NODES],
            TYPE w_fc_mat[FEATS_IN*KERNELS_NUM*FEATS_OUT],
            // TYPE ft_fc_mat[N_NODES*KERNELS_NUM*FEATS_OUT],
            // TYPE pseudo_in_mat[N_EDGES*2],
            vec_type2_t pseudo_in_mat[N_EDGES],
            TYPE w_pp_mat[2*PSEUDO_DIM],
            TYPE bias_pp_mat[PSEUDO_DIM],
            // TYPE pseudo_pp_mat[N_EDGES*PSEUDO_DIM],
            TYPE mu_mat[KERNELS_NUM*PSEUDO_DIM],
            TYPE inv_sigma_mat[KERNELS_NUM*PSEUDO_DIM],
            // TYPE gaussian_mat[N_EDGES*KERNELS_NUM],
            // TYPE rst_agg_mat[N_NODES*KERNELS_NUM*FEATS_OUT],
            node_index_t nidx_begin,
            node_index_t nidx_end,
            // TYPE rst_mat[N_NODES*FEATS_OUT]
            vec_ft_out_full_t rst_mat[N_NODES]);
}

#endif