/*
HLS kernel of GMM (Gaussian Mixture Model Convolution) layer

Reference
[1] Monti, Federico, et al. "Geometric deep learning on graphs and manifolds using mixture model cnns." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
*/

#include "monet.h"

TYPE relu(TYPE ft_h){

    if(ft_h < 0){
        ft_h = 0;
    }

    return ft_h;
}

// read weight of pseudo_proj from memory
void read_weight_pp(TYPE w_pp_mat[2*PSEUDO_DIM],
                    TYPE w_pp[2][PSEUDO_DIM]){

    rd_w_pp_l0: for (int i = 0; i < 2; i++){
        rd_w_pp_l1: for (int j = 0; j < PSEUDO_DIM; j++){
            #pragma HLS pipeline II=1

            w_pp[i][j] = w_pp_mat[i*PSEUDO_DIM + j];
        }
    }
}

// read bias of pseudo_proj from memory
void read_bias_pp(TYPE bias_pp_mat[PSEUDO_DIM],
                TYPE bias_pp[PSEUDO_DIM]){

    rd_b_pp: for (int j = 0; j < PSEUDO_DIM; j++){
        #pragma HLS pipeline II=1

        bias_pp[j] = bias_pp_mat[j];
    }
}

// read mu_mat from memory (a weight for gaussian computation)
void read_weight_mu(TYPE mu_mat[KERNELS_NUM*PSEUDO_DIM],
                    TYPE mu[KERNELS_NUM][PSEUDO_DIM]){

    rd_w_mu_l0: for (int i = 0; i < KERNELS_NUM; i++){
        rd_w_mu_l1: for (int j = 0; j < PSEUDO_DIM; j++){
            #pragma HLS pipeline II=1

            mu[i][j] = mu_mat[i*PSEUDO_DIM + j];
        }
    }
}

// read inv_sigma from memory
void read_weight_inv_sigma(TYPE inv_sigma_mat[KERNELS_NUM*PSEUDO_DIM],
                            TYPE inv_sigma[KERNELS_NUM][PSEUDO_DIM]){
    
    rd_w_inv_sig_l0: for(int i = 0; i < KERNELS_NUM; i++){
        rd_w_inv_sig_l1: for (int j = 0; j < PSEUDO_DIM; j++){
            #pragma HLS pipeline II=1

            inv_sigma[i][j] = inv_sigma_mat[i*PSEUDO_DIM + j];
        }
    }
}

// read the weight of fc layer from memory
void read_weight_fc(TYPE w_fc_mat[FEATS_IN*KERNELS_NUM*FEATS_OUT],
                    TYPE w_fc[FEATS_IN][KERNELS_NUM*FEATS_OUT]){

    rd_w_fc_l0: for (int i = 0; i < FEATS_IN; i++){

        rd_w_fc_l1: for (int j = 0; j < KERNELS_NUM; j++){

            rd_w_fc_l2: for (int k = 0; k < FEATS_OUT; k++){
                #pragma HLS pipeline II=1
                
                w_fc[i][j*FEATS_OUT + k] = w_fc_mat[i*KERNELS_NUM*FEATS_OUT + j*FEATS_OUT + k];
            }
        }
    }
}

// read nod_src from memory
void read_nod_src(node_t nod_src[N_NODES],
                // node_index_t n,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<node_t> nod_src_stream[10]){
    
    rd_nod_src_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        #pragma HLS pipeline II=1

        node_t nod_src_n = nod_src[n];

        for (int i = 0; i < 10; i++){
            #pragma HLS UNROLL

            nod_src_stream[i] << nod_src_n;
        }
    }
}

// read edge_src from memory
void read_edge_src(edge_src_t edge_src[N_EDGES],
                hls::stream<node_t>& nod_src_stream0,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<node_index_t>& tmp_src_stream){

    rd_edge_src_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        
        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        rd_edge_src_mem_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){
            #pragma HLS pipeline II=1

            tmp_src_stream << edge_src[e].src;
        }
    }
}

// read pseudo_in_mat from memory
void read_pseudo_in_mem(// TYPE pseudo_in_mat[N_EDGES*2],
                        vec_type2_t pseudo_in_mat[N_EDGES],
                        edge_index_t edge_idx[N_EDGES],
                        hls::stream<node_t>& nod_src_stream0,
                        // edge_index_t tmp_begin,
                        // edge_index_t tmp_end,
                        node_index_t nidx_begin,
                        node_index_t nidx_end,
                        // TYPE pseudo_in_stream[2]
                        // hls::stream<TYPE>& pseudo_in_stream
                        hls::stream<vec_type2_t>& pseudo_in_stream)
{
    
    rd_p_mem_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        rd_p_mem_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){
            #pragma HLS pipeline II=1

            edge_index_t e_idx = edge_idx[e];

            // rd_p_mem_l0: for(int k = 0; k < 2; k++){
            //     #pragma HLS pipeline II=1
            //     // pseudo_in_stream[k] = pseudo_in_mat[e_idx*2 + k];
            //     pseudo_in_stream << pseudo_in_mat[e_idx*2 + k];
            // }

            pseudo_in_stream << pseudo_in_mat[e_idx];
        }
    }
}

// compute pseudo_pp by pseudo_proj (matrix multiplication)
void update_pseudo_pp(// hls::stream<TYPE>& pseudo_in_stream,
                     hls::stream<vec_type2_t>& pseudo_in_stream,
                    hls::stream<node_t>& nod_src_stream0,
                    TYPE w_pp[2][PSEUDO_DIM],
                    TYPE bias_pp[PSEUDO_DIM],
                    // edge_index_t tmp_begin,
                    // edge_index_t tmp_end,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    // TYPE pseudo_pp_stream[PSEUDO_DIM]
                    // hls::stream<TYPE>& pseudo_pp_stream
                    hls::stream<vec_pseudo_dim_full_t>& pseudo_pp_stream
                    ){
    
    update_pseudo_pp_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        update_pseudo_pp_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){
            
            // read pseudo_in from stream
            vec_type2_t pseudo_in_vec = pseudo_in_stream.read();

            // update pseudo_pp
            TYPE rst_pp[PSEUDO_DIM];
            #pragma HLS array_partition variable=rst_pp complete dim=1
            update_pseudo_pp_l0: for(int k = 0; k < 2; k++){

                TYPE pseudo_in_temp = pseudo_in_vec[k];

                update_pseudo_pp_l1: for (int j = 0; j < PSEUDO_DIM; j++){
                    // #pragma HLS pipeline II=1

                    TYPE rst_pp_temp = (k == 0) ? 0 : rst_pp[j];

                    // rst_pp[j] = rst_pp_temp + pseudo_in_vec[k] * w_pp[k][j];
                    rst_pp[j] = rst_pp_temp + pseudo_in_temp * w_pp[k][j];
                }
            }

            // write pseudo_pp to stream (bias)
            vec_pseudo_dim_full_t pseudo_pp_vec;
            wr_ppsedu_pp_stm: for (int j = 0; j < PSEUDO_DIM; j++){
                // #pragma HLS pipeline II=1
                #ifdef FLOAT32
                    pseudo_pp_vec[j] = tanhf(rst_pp[j] + bias_pp[j]);
                #else
                    pseudo_pp_vec[j] = tanh(rst_pp[j] + bias_pp[j]);
                #endif
            }

            pseudo_pp_stream << pseudo_pp_vec;
        }
    }
}

// compute gaussian weight
void comp_weight_gaussian(// hls::stream<TYPE>& pseudo_pp_stream,
                        hls::stream<vec_pseudo_dim_full_t>& pseudo_pp_stream,
                        hls::stream<node_t>& nod_src_stream0,
                        TYPE mu[KERNELS_NUM][PSEUDO_DIM],
                        TYPE inv_sigma[KERNELS_NUM][PSEUDO_DIM],
                        // edge_index_t tmp_begin,
                        // edge_index_t tmp_end,
                        node_index_t nidx_begin,
                        node_index_t nidx_end,
                        // hls::stream<TYPE>& gaussian_stream
                        hls::stream<vec_kern_num_full_t>& gaussian_stream)
{

    comp_w_gaussian_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        comp_w_gaussian_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

            // read pseudo_pp from stream
            vec_pseudo_dim_full_t pseudo_pp_vec = pseudo_pp_stream.read();

            // compute gaussian weight
            TYPE ga_rst[KERNELS_NUM];
            #pragma HLS array_partition variable=ga_rst complete dim=1
            comp_w_gaussian_l0: for(int k = 0; k < PSEUDO_DIM; k++){
   
                comp_w_gaussian_l1: for(int j = 0; j < KERNELS_NUM; j++){

                    TYPE ga_rst_temp = (k == 0) ? 0 : ga_rst[j];

                    TYPE ga_mu_temp = pseudo_pp_vec[k] - mu[j][k];

                    TYPE ga_temp = ga_mu_temp * inv_sigma[j][k];

                    ga_rst[j] = ga_rst_temp + (-0.5) * ga_temp * ga_temp;
                }
            }


            // write gaussian weight to stream
            vec_kern_num_full_t ga_rst_vec;
            wr_w_gaussian: for(int j = 0; j < KERNELS_NUM; j++){
                #ifdef FLOAT32
                    // gaussian_stream[j] = expf(ga_rst[j]);
                    // gaussian_stream << expf(ga_rst[j]);
                    ga_rst_vec[j] = expf(ga_rst[j]);
                #else
                    // gaussian_stream[j] = exp(ga_rst[j]);
                    // gaussian_stream << exp(ga_rst[j]);
                    ga_rst_vec[j] = exp(ga_rst[j]);
                #endif
            }

            gaussian_stream << ga_rst_vec;
        }
    }
}

// read input features to be aggregated from memory
void read_feat_in_agg(// TYPE ft_in_mat[N_NODES*FEATS_IN],
                    vec_ft_in_full_t ft_in_mat[N_NODES],
                    hls::stream<node_t>& nod_src_stream0,
                    hls::stream<node_index_t>& tmp_src_stream,
                    // edge_src_t edge_src[N_EDGES],
                    // edge_index_t tmp_begin,
                    // edge_index_t tmp_end,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    // hls::stream<TYPE>& ft_in_agg_stream
                    hls::stream<vec_ft_in_full_t>& ft_in_agg_stream)
{
    
    for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        for(edge_index_t e=tmp_begin; e<tmp_end; e++){

            // node_index_t tmp_src = edge_src[e].src;
            node_index_t tmp_src = tmp_src_stream.read();

            // for(int j = 0; j < FEATS_IN; j++){
            //     // ft_in_agg[j] = ft_in_mat[tmp_src*FEATS_IN + j];
            //     ft_in_agg_stream << ft_in_mat[tmp_src*FEATS_IN + j];
            // }
            ft_in_agg_stream << ft_in_mat[tmp_src];
        }
    }
}

// aggregate features while multiply with e_gaussian 
// ft_in_1 * e_gaussian_1 + ft_in_2 * e_gaussian_2 + ....
void agg_feat_in(// hls::stream<TYPE>& gaussian_stream,
                hls::stream<vec_kern_num_full_t>& gaussian_stream,
                hls::stream<node_t>& nod_src_stream0,
                // hls::stream<TYPE>& ft_in_agg_stream,
                hls::stream<vec_ft_in_full_t>& ft_in_agg_stream,
                // edge_index_t tmp_begin,
                // edge_index_t tmp_end,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                // TYPE rst_agg_stream[KERNELS_NUM][FEATS_IN]
                // hls::stream<TYPE>& rst_agg_stream
                // hls::stream<vec_ft_in_full_t>& rst_agg_stream
                hls::stream<vec_kern_num_ft_in_full_t>& rst_agg_stream
                )
{

    agg_ft_in_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;


        // TYPE ft_in_agg_buf[FEATS_IN];

        if(tmp_begin != tmp_end){

            TYPE rst_agg[KERNELS_NUM][FEATS_IN];
            #pragma HLS array_partition variable=rst_agg complete dim=1
            #pragma HLS array_partition variable=rst_agg complete dim=2

            // vec_ft_in_full_t rst_agg[KERNELS_NUM];
            // #pragma HLS array_partition variable=rst_agg complete dim=1

            // vec_kern_num_ft_in_full_t rst_agg_vec;
            agg_ft_in_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){
                #pragma HLS pipeline II=1

                // read from  ft_in_agg (stream)
                // for(int j = 0; j < FEATS_IN; j++){
                //     // ft_in_agg_buf[j] = ft_in_agg[j];
                //     ft_in_agg_buf[j] = ft_in_agg_stream.read();
                // }

                // read features to be aggregated from memory
                vec_ft_in_full_t ft_in_agg_vec = ft_in_agg_stream.read();

                // read gaussian result from stream
                vec_kern_num_full_t gaussian_vec = gaussian_stream.read();

                // aggregate features
                agg_ft_in_l0: for(int i = 0; i < KERNELS_NUM; i++){
                    #pragma HLS UNROLL

                    // TYPE gaussian_temp = gaussian_stream[i];
                    // TYPE gaussian_temp = gaussian_stream.read();
                    TYPE gaussian_temp = gaussian_vec[i];

                    agg_ft_in_l1: for(int j = 0; j < FEATS_IN; j++){
                        #pragma HLS UNROLL

                        TYPE rst_agg_temp = (e == tmp_begin) ? 0 : rst_agg[i][j];

                        rst_agg[i][j]  = rst_agg_temp + ft_in_agg_vec[j] * gaussian_temp;

                        // TYPE rst_agg_temp = (e == tmp_begin) ? 0 : rst_agg_vec[i*FEATS_IN + j];

                        // rst_agg_vec[i*FEATS_IN + j]  = rst_agg_temp + ft_in_agg_vec[j] * gaussian_temp;
                    }
                }
            }

            // // write aggretated results to stream
            // for(int j = 0; j < FEATS_IN; j++){
            //     for(int i = 0; i < KERNELS_NUM; i++){
                
            //         rst_agg_stream << rst_agg[i][j];
            //         // rst_agg_stream <<  rst_agg_vec[i*FEATS_IN + j];
            //     }
            // }

            // write aggretated results to stream
            // wr_agg_rst_stm: for(int i = 0; i < KERNELS_NUM; i++){
            //     #pragma HLS pipeline II=1

            //     rst_agg_stream << rst_agg[i];
            // }
            // rst_agg_stream << rst_agg_vec;

            // write aggretated results to stream
            vec_kern_num_ft_in_full_t rst_agg_vec;
            wr_rst_agg_stm_l0: for(int i = 0; i < KERNELS_NUM; i++){
                #pragma HLS UNROLL
                
                wr_rst_agg_stm_l1: for(int j = 0; j < FEATS_IN; j++){
                    #pragma HLS UNROLL
                    
                    rst_agg_vec[i*FEATS_IN + j] = rst_agg[i][j];
                }
            }

            rst_agg_stream << rst_agg_vec;
        }
    }
}

/*
// update aggregated results using fc weights
void update_agg_fc(// TYPE rst_agg_stream[KERNELS_NUM][FEATS_IN],
                    // hls::stream<TYPE>& rst_agg_stream,
                    // hls::stream<vec_ft_in_full_t>& rst_agg_stream,
                    hls::stream<vec_kern_num_ft_in_full_t>& rst_agg_stream,
                    hls::stream<node_t>& nod_src_stream0,
                   TYPE w_fc[FEATS_IN][KERNELS_NUM*FEATS_OUT],
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    // TYPE rst_update_stream[KERNELS_NUM][FEATS_OUT]
                    hls::stream<TYPE>& rst_update_stream)
{

    update_agg_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;
        if(tmp_begin != tmp_end){

            // receive the aggregated features and convert it from a vector to scalars
            vec_kern_num_ft_in_full_t rst_agg_vec = rst_agg_stream.read();
            TYPE rst_agg_buf[KERNELS_NUM][FEATS_IN];
            #pragma HLS array_partition variable=rst_agg_buf complete dim=1
            #pragma HLS array_partition variable=rst_agg_buf complete dim=2
            rd_rst_agg_stm_l0: for(int i = 0; i < KERNELS_NUM; i++){
                #pragma HLS pipeline II=1
                
                rd_rst_agg_stm_l1: for(int j = 0; j < FEATS_IN; j++){
                    #pragma HLS UNROLL
                    
                    rst_agg_buf[i][j] = rst_agg_vec[i*FEATS_IN + j];
                }
            }
            
            // update results
            TYPE rst_update[KERNELS_NUM][FEATS_OUT];
            #pragma HLS array_partition variable=rst_update complete dim=1
            #pragma HLS array_partition variable=rst_update complete dim=2
            update_agg_l0: for (int k = 0; k < FEATS_IN; k++){
                #pragma HLS pipeline II=1

                update_agg_l1: for(int i = 0; i < KERNELS_NUM; i++){
                    #pragma HLS UNROLL

                    // TYPE rst_agg_temp = rst_agg_stream[i][k];
                    // TYPE rst_agg_temp = rst_agg_stream.read();
                    
                    update_agg_l2: for (int j = 0; j < FEATS_OUT; j++){
                        #pragma HLS UNROLL

                        TYPE rst_update_temp = (k == 0) ? 0 : rst_update[i][j];


                        rst_update[i][j] = rst_update_temp + rst_agg_buf[i][k] * w_fc[k][i][j];
                    }
                }
            }

            // write rst_update to stream
            wr_rst_update_stm: for(int i = 0; i < KERNELS_NUM; i++){
                for (int j = 0; j < FEATS_OUT; j++){

                    // rst_update_stream[i][j] =  rst_update[i][j];
                    rst_update_stream << rst_update[i][j];
                }
            }
        }
    }
}
*/



// update aggregated results using fc weights
void update_agg_fc(// TYPE rst_agg_stream[KERNELS_NUM][FEATS_IN],
                    // hls::stream<TYPE>& rst_agg_stream,
                    // hls::stream<vec_ft_in_full_t>& rst_agg_stream,
                    hls::stream<vec_kern_num_ft_in_full_t>& rst_agg_stream,
                    hls::stream<node_t>& nod_src_stream0,
                   TYPE w_fc[FEATS_IN][KERNELS_NUM*FEATS_OUT],
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    // TYPE rst_update_stream[KERNELS_NUM][FEATS_OUT]
                    // hls::stream<TYPE>& rst_update_stream
                    hls::stream<vec_kern_num_ft_out_full_t>& rst_update_p1_stream)
{

    update_agg_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;
        if(tmp_begin != tmp_end){

            // receive the aggregated features and convert it from a vector to scalars
            vec_kern_num_ft_in_full_t rst_agg_vec = rst_agg_stream.read();
            TYPE rst_agg_buf[KERNELS_NUM][FEATS_IN];
            #pragma HLS array_partition variable=rst_agg_buf complete dim=1
            #pragma HLS array_partition variable=rst_agg_buf complete dim=2
            rd_rst_agg_stm_l0: for(int i = 0; i < KERNELS_NUM; i++){
                #pragma HLS pipeline II=1
                
                rd_rst_agg_stm_l1: for(int j = 0; j < FEATS_IN; j++){
                    #pragma HLS UNROLL
                    
                    rst_agg_buf[i][j] = rst_agg_vec[i*FEATS_IN + j];
                }
            }
            
            // update results
            TYPE rst_update[D][KERNELS_NUM*FEATS_OUT];
            // #pragma HLS array_partition variable=rst_update complete dim=1
            #pragma HLS array_partition variable=rst_update complete dim=2
            // #pragma HLS array_partition variable=rst_update complete dim=3
            update_agg_l0: for (int k = 0; k < FEATS_IN/D; k++){

                update_agg_l1: for (int kd = 0; kd < D; kd++){
                    #pragma HLS pipeline II=1

                        
                    update_agg_l3: for (int j = 0; j < KERNELS_NUM*FEATS_OUT; j++){
                        #pragma HLS UNROLL

                        TYPE rst_update_temp = (k == 0) ? 0 : rst_update[kd][j];
                        
                        rst_update[kd][j] = rst_update_temp + rst_agg_buf[j/FEATS_OUT][k*D + kd] * w_fc[k*D + kd][j];
                    }
                }
            }


            // write rst_update to stream
            wr_rst_update_stm_l0: for (int kd = 0; kd < D; kd++){

                vec_kern_num_ft_out_full_t rst_update_vec_temp;

                wr_rst_update_stm_l2: for (int j = 0; j < KERNELS_NUM*FEATS_OUT; j++){
                    #pragma HLS UNROLL

                    // rst_update_stream[i][j] =  rst_update[i][j];
                    // rst_update_stream << rst_update[kd][i][j];
                    // rst_update_stream << rst_update[kd][j];
                    rst_update_vec_temp[j] = rst_update[kd][j];
                }

                rst_update_p1_stream << rst_update_vec_temp;
            }
        }
    }
}

void update_agg_fc_sum(hls::stream<vec_kern_num_ft_out_full_t>& rst_update_p1_stream,
                    hls::stream<node_t>& nod_src_stream0,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    // hls::stream<TYPE>& rst_update_stream
                    hls::stream<vec_ft_out_full_t>& rst_update_stream
                    )
{
    
    update_agg_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;
        if(tmp_begin != tmp_end){

            TYPE rst_update[D][KERNELS_NUM*FEATS_OUT];
            #pragma HLS array_partition variable=rst_update complete dim=2
            rd_agg_p1_stm_l0: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1

                vec_kern_num_ft_out_full_t rst_agg_vec_temp = rst_update_p1_stream.read();

                rd_agg_p1_stm_l1: for(int j=0; j<KERNELS_NUM*FEATS_OUT; j++){
                    #pragma HLS UNROLL

                    rst_update[kd][j] = rst_agg_vec_temp[j];
                }
            }

            // sum the D rows of results
            TYPE rst_update_sum[KERNELS_NUM*FEATS_OUT];
            #pragma HLS array_partition variable=rst_update_sum cyclic factor=16 dim=1
            update_agg_sum_l0: for (int kd = 0; kd < D; kd++){

                update_agg_sum_l1: for(int j=0; j<KERNELS_NUM*FEATS_OUT; j++){
                    #pragma HLS pipeline II=1 rewind
                    #pragma HLS UNROLL factor=16

                    TYPE rst_sum_temp = (kd == 0) ? 0 : rst_update_sum[j];

                    rst_update_sum[j] = rst_sum_temp + rst_update[kd][j];
                }
            }

            // // write rst_update to stream
            // for (int i = 0; i < KERNELS_NUM*FEATS_OUT; i++){
                
            //     rst_update_stream << rst_update_sum[i];
            // }

            // write rst_update to stream
            wr_rst_update_stm_l0: for (int i = 0; i < KERNELS_NUM; i++){
                #pragma HLS pipeline II=1
                
                vec_ft_out_full_t rst_update_sum_vec;
                wr_rst_update_stm_l1: for (int j = 0; j < FEATS_OUT; j++){
                    // #pragma HLS pipeline II=1 rewind
                    // #pragma HLS UNROLL factor=16
                    #pragma HLS UNROLL

                    rst_update_sum_vec[j] = rst_update_sum[i*FEATS_OUT + j];
                }

                rst_update_stream << rst_update_sum_vec;
            }
        }
    }
}



// sum rst_update in the order of kernel
void ksum_rst_update(// TYPE rst_update_stream[KERNELS_NUM][FEATS_OUT],
                    // hls::stream<TYPE>& rst_update_stream,
                    hls::stream<vec_ft_out_full_t>& rst_update_stream,
                    hls::stream<node_t>& nod_src_stream0,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    // TYPE rst_ksum_stream[FEATS_OUT]
                    // hls::stream<TYPE>& rst_ksum_stream
                    hls::stream<vec_ft_out_full_t>& rst_ksum_stream
                    )
{

    ksum_rst_update_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if(tmp_begin != tmp_end){
        
            // sum in the order of kernel_num
            vec_ft_out_full_t rst_ksum;
            ksum_rst_update_l0: for (int i = 0; i < KERNELS_NUM; i++){
                #pragma HLS pipeline II=1

                vec_ft_out_full_t rst_ksum_temp = (i == 0) ? 0 : rst_ksum;

                rst_ksum = rst_ksum_temp + rst_update_stream.read();
            }

            // write result of kernel-wise sum (rst_ksum) to stream
            vec_ft_out_full_t rst_ksum_vec;
            wr_ksum_rst_stm: for (int j = 0; j < FEATS_OUT; j++){
                #pragma HLS UNROLL

                // rst_ksum_stream[j] = rst_ksum[j];
                // rst_ksum_stream << rst_ksum[j];
                #ifdef ACTIVATION
                    rst_ksum_vec[j] = relu(rst_ksum[j]);
                #else
                    rst_ksum_vec[j] = rst_ksum[j];
                #endif
            }

            rst_ksum_stream << rst_ksum_vec;
        }
    }
}

// write ksum results to memory
void write_ksum_rst_mem(// TYPE rst_ksum_stream[FEATS_OUT],
                        // hls::stream<TYPE>& rst_ksum_stream,
                        hls::stream<vec_ft_out_full_t>& rst_ksum_stream,
                        hls::stream<node_t>& nod_src_stream0,
                        // node_index_t n,
                        node_index_t nidx_begin,
                        node_index_t nidx_end,
                        // TYPE rst_mat[N_NODES*FEATS_OUT]
                        vec_ft_out_full_t rst_mat[N_NODES])
{
    for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if(tmp_begin != tmp_end){

            rst_mat[n] = rst_ksum_stream.read();
        }
    }
}

// Compute results for one node
void compute_one_node(node_t nod_src[N_NODES],
                    edge_src_t edge_src[N_EDGES],
                    edge_index_t edge_idx[N_EDGES],
                    // TYPE ft_in_mat[N_NODES*FEATS_IN],
                    vec_ft_in_full_t ft_in_mat[N_NODES],
                    // TYPE pseudo_in_mat[N_EDGES*2],
                    vec_type2_t pseudo_in_mat[N_EDGES],
                    TYPE w_pp[2][PSEUDO_DIM],
                    TYPE bias_pp[PSEUDO_DIM],
                    TYPE mu[KERNELS_NUM][PSEUDO_DIM],
                    TYPE inv_sigma[KERNELS_NUM][PSEUDO_DIM],
                   TYPE w_fc[FEATS_IN][KERNELS_NUM*FEATS_OUT],
                    // node_index_t n,
                    // edge_index_t tmp_begin,
                    // edge_index_t tmp_end,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    // TYPE rst_mat[N_NODES*FEATS_OUT]
                    vec_ft_out_full_t rst_mat[N_NODES]
                    )
{
    const int feats_in = FEATS_IN;
    const int feats_out = FEATS_OUT;
    const int pseudo_dim = PSEUDO_DIM;
    const int kernels_num = KERNELS_NUM;

    #pragma HLS dataflow

    // read nod_src from memory
    hls::stream<node_t> nod_src_stream[10];
    #pragma HLS stream variable=nod_src_stream[0] depth=10
    #pragma HLS stream variable=nod_src_stream[1] depth=11
    #pragma HLS stream variable=nod_src_stream[2] depth=12
    #pragma HLS stream variable=nod_src_stream[3] depth=13
    #pragma HLS stream variable=nod_src_stream[4] depth=14
    #pragma HLS stream variable=nod_src_stream[5] depth=15
    #pragma HLS stream variable=nod_src_stream[6] depth=16
    #pragma HLS stream variable=nod_src_stream[7] depth=17
    #pragma HLS stream variable=nod_src_stream[8] depth=18
    #pragma HLS stream variable=nod_src_stream[9] depth=19
    read_nod_src(nod_src, nidx_begin, nidx_end, nod_src_stream);

    // read edge_src from memory
    hls::stream<node_index_t> tmp_src_stream;
    #pragma HLS stream variable=tmp_src_stream depth=14
    read_edge_src(edge_src, nod_src_stream[0], nidx_begin, nidx_end, tmp_src_stream);

    // compute e_gaussian and weighted aggregate features (ft_in * e_gaussian)
    // @ToDo: in HLS kernel, put functions in this loop to dataflow scope and connect them with stream channel
    // TYPE pseudo_in_stream[2];
    // hls::stream<TYPE> pseudo_in_stream;
    // #pragma HLS stream variable=pseudo_in_stream depth=6
    // read_pseudo_in_mem(pseudo_in_mat, edge_idx, nod_src_stream[1], nidx_begin, nidx_end, pseudo_in_stream);

    hls::stream<vec_type2_t> pseudo_in_stream;
    #pragma HLS stream variable=pseudo_in_stream depth=10
    read_pseudo_in_mem(pseudo_in_mat, edge_idx, nod_src_stream[1], nidx_begin, nidx_end, pseudo_in_stream);

    // compute pseudo_pp by pseudo_proj (matrix multiplication)
    // hls::stream<TYPE> pseudo_pp_stream;
    // #pragma HLS stream variable=pseudo_pp_stream depth=pseudo_dim*2
    hls::stream<vec_pseudo_dim_full_t> pseudo_pp_stream;
    #pragma HLS stream variable=pseudo_pp_stream depth=10
    update_pseudo_pp(pseudo_in_stream, nod_src_stream[2], w_pp, bias_pp, nidx_begin, nidx_end, pseudo_pp_stream);

    // compute gaussian weight
    // hls::stream<TYPE> gaussian_stream;
    // #pragma HLS stream variable=gaussian_stream depth=kernels_num*2
    hls::stream<vec_kern_num_full_t> gaussian_stream;
    #pragma HLS stream variable=gaussian_stream depth=11
    // comp_weight_gaussian(pseudo_pp_stream, mu, inv_sigma, tmp_begin, tmp_end, gaussian_stream);
    comp_weight_gaussian(pseudo_pp_stream, nod_src_stream[3], mu, inv_sigma, nidx_begin, nidx_end, gaussian_stream);

    // read input features to be aggregated from memory
    // hls::stream<TYPE> ft_in_agg_stream;
    // #pragma HLS stream variable=ft_in_agg_stream depth=feats_in*2
    hls::stream<vec_ft_in_full_t> ft_in_agg_stream;
    #pragma HLS stream variable=ft_in_agg_stream depth=10
    read_feat_in_agg(ft_in_mat, nod_src_stream[4], tmp_src_stream, nidx_begin, nidx_end, ft_in_agg_stream);

    // aggregate features while multiply with e_gaussian 
    // ft_in_1 * e_gaussian_1 + ft_in_2 * e_gaussian_2 + ....
    // hls::stream<TYPE> rst_agg_stream;
    // #pragma HLS stream variable=rst_agg_stream depth=kernels_num*feats_in*2
    // hls::stream<vec_ft_in_full_t> rst_agg_stream;
    // #pragma HLS stream variable=rst_agg_stream depth=kernels_num*10
    hls::stream<vec_kern_num_ft_in_full_t> rst_agg_stream;
    #pragma HLS stream variable=rst_agg_stream depth=10
    agg_feat_in(gaussian_stream, nod_src_stream[5], ft_in_agg_stream, nidx_begin, nidx_end, rst_agg_stream);
    
    // update aggregated results
    // TYPE rst_update_stream[KERNELS_NUM][FEATS_OUT];
    // hls::stream<TYPE> rst_update_stream;
    // #pragma HLS stream variable=rst_update_stream depth=kernels_num*feats_out*2
    hls::stream<vec_kern_num_ft_out_full_t> rst_update_p1_stream;
    #pragma HLS stream variable=rst_update_p1_stream depth=32
    update_agg_fc(rst_agg_stream, nod_src_stream[6], w_fc, nidx_begin, nidx_end, rst_update_p1_stream);

    // hls::stream<TYPE> rst_update_stream;
    // #pragma HLS stream variable=rst_update_stream depth=kernels_num*feats_out*2
    hls::stream<vec_ft_out_full_t> rst_update_stream;
    #pragma HLS stream variable=rst_update_stream depth=10
    update_agg_fc_sum(rst_update_p1_stream, nod_src_stream[7], nidx_begin, nidx_end, rst_update_stream);

    // sum in the order of kernel
    // TYPE rst_ksum_stream[FEATS_OUT];
    // hls::stream<TYPE> rst_ksum_stream;
    // #pragma HLS stream variable=rst_ksum_stream depth=feats_out*2
    hls::stream<vec_ft_out_full_t> rst_ksum_stream;
    #pragma HLS stream variable=rst_ksum_stream depth=10
    ksum_rst_update(rst_update_stream, nod_src_stream[8], nidx_begin, nidx_end, rst_ksum_stream);

    // write ksum results to memory
    write_ksum_rst_mem(rst_ksum_stream, nod_src_stream[9], nidx_begin, nidx_end, rst_mat);
}

// Compute results for all nodes
void compute_all_node(node_t nod_src[N_NODES],
                    edge_src_t edge_src[N_EDGES],
                    edge_index_t edge_idx[N_EDGES],
                    // TYPE ft_in_mat[N_NODES*FEATS_IN],
                    vec_ft_in_full_t ft_in_mat[N_NODES],
                    // TYPE pseudo_in_mat[N_EDGES*2],
                    vec_type2_t pseudo_in_mat[N_EDGES],
                    TYPE w_pp[2][PSEUDO_DIM],
                    TYPE bias_pp[PSEUDO_DIM],
                    TYPE mu[KERNELS_NUM][PSEUDO_DIM],
                    TYPE inv_sigma[KERNELS_NUM][PSEUDO_DIM],
                   TYPE w_fc[FEATS_IN][KERNELS_NUM*FEATS_OUT],
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    // TYPE rst_mat[N_NODES*FEATS_OUT]
                    vec_ft_out_full_t rst_mat[N_NODES]
                    )
{

    // message passing (aggregate (ft_in * e_gaussian) and update (* w_fc) and sum)
    // traverse all nodes and edges stored in CSR format
    // n is the idx of target node,
    // i is the idx of kernel,
    // tmp_src is the idx of input neighbors
    // for(node_index_t n = nidx_begin; n < nidx_end; n++){

    //     edge_index_t tmp_begin = nod_src[n].edge_begin;
    //     edge_index_t tmp_end = nod_src[n].edge_end;

    //     if(tmp_begin != tmp_end){

    //         // Compute results for one node
    //         compute_one_node(edge_src, edge_idx, ft_in_mat, pseudo_in_mat, w_pp, bias_pp, mu, inv_sigma, w_fc, n, tmp_begin, tmp_end, rst_mat);
    //     }
    // }

    compute_one_node(nod_src, edge_src, edge_idx, ft_in_mat, pseudo_in_mat, w_pp, bias_pp, mu, inv_sigma, w_fc, nidx_begin, nidx_end, rst_mat);
}


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
            vec_ft_out_full_t rst_mat[N_NODES])
{

    #pragma HLS INTERFACE m_axi port=nod_src bundle=aximm1
    #pragma HLS INTERFACE m_axi port=edge_src bundle=aximm2
    #pragma HLS INTERFACE m_axi port=edge_idx bundle=aximm3

    #pragma HLS INTERFACE m_axi port=ft_in_mat bundle=aximm4

    #pragma HLS INTERFACE m_axi port=w_fc_mat bundle=aximm5

    #pragma HLS INTERFACE m_axi port=pseudo_in_mat bundle=aximm6

    #pragma HLS INTERFACE m_axi port=w_pp_mat bundle=aximm5
    #pragma HLS INTERFACE m_axi port=bias_pp_mat bundle=aximm5
    #pragma HLS INTERFACE m_axi port=mu_mat bundle=aximm5
    #pragma HLS INTERFACE m_axi port=inv_sigma_mat bundle=aximm5

    #pragma HLS INTERFACE s_axilite port=nidx_begin
    #pragma HLS INTERFACE s_axilite port=nidx_end

    #pragma HLS INTERFACE m_axi port=rst_mat bundle=aximm7
    
    // read weight of pseudo_proj from memory
    TYPE w_pp[2][PSEUDO_DIM];
    #pragma HLS array_partition variable=w_pp complete dim=1
    #pragma HLS array_partition variable=w_pp complete dim=2
    read_weight_pp(w_pp_mat, w_pp);

    // read bias of pseudo_proj from memory
    TYPE bias_pp[PSEUDO_DIM];
    #pragma HLS array_partition variable=bias_pp complete dim=1
    read_bias_pp(bias_pp_mat, bias_pp);

    // read mu_mat from memory
    TYPE mu[KERNELS_NUM][PSEUDO_DIM];
    #pragma HLS array_partition variable=mu complete dim=1
    #pragma HLS array_partition variable=mu complete dim=2
    read_weight_mu(mu_mat, mu);
    
    // read inv_sigma from memory
    TYPE inv_sigma[KERNELS_NUM][PSEUDO_DIM];
    #pragma HLS array_partition variable=inv_sigma complete dim=1
    #pragma HLS array_partition variable=inv_sigma complete dim=2
    read_weight_inv_sigma(inv_sigma_mat, inv_sigma);

    // read the weight of fc layer from memory
    TYPE w_fc[FEATS_IN][KERNELS_NUM*FEATS_OUT];
    #pragma HLS array_partition variable=w_fc complete dim=2
    // #pragma HLS array_partition variable=w_fc complete dim=3
    read_weight_fc(w_fc_mat, w_fc);

    // Compute results for all nodes
    compute_all_node(nod_src, edge_src, edge_idx, ft_in_mat, pseudo_in_mat, w_pp, bias_pp, mu, inv_sigma, w_fc, nidx_begin, nidx_end, rst_mat);
    return 0;
}