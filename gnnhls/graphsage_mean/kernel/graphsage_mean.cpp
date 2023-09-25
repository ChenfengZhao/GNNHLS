/*
HLS kernel of GraphSage (mean) layer

Reference
[1] Hamilton, William L., Rex Ying, and Jure Leskovec. "Inductive representation learning on 
    large graphs." Proceedings of the 31st International Conference on Neural Information 
    Processing Systems. 2017.
*/

#include "graphsage_mean.h"

TYPE relu(TYPE ft_h){

    if(ft_h < 0){
        ft_h = 0;
    }

    return ft_h;
}

// Read weights for aggreted feature update/apply phase 
void read_weight_agg(TYPE w_mlp_mat[2*FEATS_IN*FEATS_OUT],
                     TYPE w_update_agg[FEATS_IN][FEATS_OUT]){
    
    rd_w_agg_l0: for (int i = 0; i < FEATS_IN; i++){
        rd_w_agg_l1: for (int j = 0; j < FEATS_OUT; j++){
            #pragma HLS pipeline II=1

            w_update_agg[i][j] = w_mlp_mat[(i+FEATS_IN)*FEATS_OUT + j];
        }
    }
}

// Read weights for target feature update/apply phase
void read_weight_tar(TYPE w_mlp_mat[2*FEATS_IN*FEATS_OUT],
                    TYPE w_update_tar[FEATS_IN][FEATS_OUT]){

    rd_w_tar_l0: for (int i = 0; i < FEATS_IN; i++){
        rd_w_tar_l1: for (int j = 0; j < FEATS_OUT; j++){
            #pragma HLS pipeline II=1

            w_update_tar[i][j] = w_mlp_mat[i*FEATS_OUT + j];
        }
    }
}

// // read input features to be aggregated from memory
// // (@ToDo: try to put read memory outside loop and inside dataflow scope)
// void read_feat_in_agg(edge_src_t edge_src[N_EDGES],
//                       TYPE ft_in_agg_mat[N_NODES*FEATS_IN],
//                       edge_index_t tmp_begin,
//                       edge_index_t tmp_end,
//                     //   TYPE ft_in_agg[FEATS_IN]
//                       hls::stream<TYPE>& ft_in_agg_stream){

//     rd_ft_in_mem_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){
//         node_index_t tmp_src = edge_src[e].src;

//         rd_ft_in_mem: for(int i=0; i<FEATS_IN; i++){
//             #pragma HLS pipeline II=1

//             // ft_in_agg[i] = ft_in_agg_mat[tmp_src*FEATS_IN + i];
//             ft_in_agg_stream << ft_in_agg_mat[tmp_src*FEATS_IN + i];
//         }
//     }
// }

// read the begin and end indices of source nodes for each target node
// read nod_src from memory
void read_nod_src(node_t nod_src[N_NODES],
                    // node_index_t n,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    hls::stream<node_t>& nod_src_stream){
    
    rd_nod_src_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        #pragma HLS pipeline II=1

        nod_src_stream << nod_src[n];
    }
}

// split the nod_src_stream
void split_nod_stream(hls::stream<node_t>& nod_src_stream,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    hls::stream<node_t>& nod_src_stream1,
                    hls::stream<node_t>& nod_src_stream2,
                    hls::stream<node_t>& nod_src_stream3,
                    hls::stream<node_t>& nod_src_stream4,
                    hls::stream<node_t>& nod_src_stream6,
                    hls::stream<node_t>& nod_src_stream5){

    split_nod_stm_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        #pragma HLS pipeline II=1

        node_t nod_src_temp = nod_src_stream.read();
        nod_src_stream1 << nod_src_temp;
        nod_src_stream2 << nod_src_temp;
        nod_src_stream3 << nod_src_temp;
        nod_src_stream4 << nod_src_temp;
        nod_src_stream6 << nod_src_temp;
        nod_src_stream5 << nod_src_temp;
    }
}

// read source node idx for each target node
// read edge_src from memory
void read_edge_src(edge_src_t edge_src[N_EDGES],
                // edge_index_t tmp_begin,
                // edge_index_t tmp_end,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<node_t>& nod_src_stream1,
                hls::stream<node_index_t>& tmp_src_stream){
    
    rd_edge_src_mem_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        node_t nod_src_temp = nod_src_stream1.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        rd_edge_src_mem_eloop_l0: for(edge_index_t e=tmp_begin; e<tmp_end; e++){
            #pragma HLS pipeline II=1

            tmp_src_stream << edge_src[e].src;
        }
    }
}

// read input features to be aggregated from memory
// (@ToDo: try to put read memory outside loop and inside dataflow scope)
void read_feat_in_agg(// edge_src_t edge_src[N_EDGES],
                      hls::stream<node_index_t>& tmp_src_stream,
                      vec_ft_in_t ft_in_agg_mat[N_NODES*FEATS_IN/W_FT_IN],
                    //   edge_index_t tmp_begin,
                    //   edge_index_t tmp_end,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    hls::stream<node_t>& nod_src_stream2,
                    //   TYPE ft_in_agg[FEATS_IN]
                      hls::stream<vec_ft_in_t>& ft_in_agg_stream){
    
    rd_ft_in_mem_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream2.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;
        
        rd_ft_in_mem_eloop_l0: for (int k = 0; k < (tmp_end - tmp_begin); k++){
            node_index_t tmp_src = tmp_src_stream.read();

            rd_ft_in_mem_eloop_l1: for (int i = 0; i < FEATS_IN/W_FT_IN; i++){
                #pragma HLS pipeline II=1

                ft_in_agg_stream << ft_in_agg_mat[tmp_src*FEATS_IN/W_FT_IN + i];
            }
        }
    }
}

// aggregate features
void agg_feat_in(// TYPE ft_in_agg[FEATS_IN],
                hls::stream<vec_ft_in_t>& ft_in_agg_stream,
                // edge_index_t tmp_begin,
                // edge_index_t tmp_end,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<node_t>& nod_src_stream3,
                // TYPE ft_h_agg[FEATS_IN]
                // TYPE ft_h_agg_stream[FEATS_IN]
                hls::stream<vec_ft_in_t>& ft_h_agg_stream){
    
    agg_feat_in_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream3.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;
        
        // TYPE ft_h_agg[FEATS_IN];
        // #pragma HLS array_partition variable=ft_h_agg cyclic factor=32 dim=1

        vec_ft_in_t ft_h_agg[FEATS_IN/W_FT_IN];

        if(tmp_end != tmp_begin){

            // vec_ft_in_t ft_h_agg = 0;
            agg_ft_in_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

                // read from ft_in_agg_stream
                vec_ft_in_t ft_in_agg[FEATS_IN/W_FT_IN];
                rd_ft_in_stm: for(int i = 0; i < FEATS_IN/W_FT_IN; i++){
                    #pragma HLS pipeline II=1

                    ft_in_agg[i] = ft_in_agg_stream.read();
                }

                // aggregrate features
                agg_ft_in: for(int i=0; i<FEATS_IN/W_FT_IN; i++){
                    #pragma HLS pipeline II=1

                    vec_ft_in_t ft_h_temp = (e == tmp_begin) ? 0 : ft_h_agg[i];
                    ft_h_agg[i] = ft_h_temp + ft_in_agg[i];
                }
            }
            
            // calculate in_degree
            edge_index_t in_deg_n = tmp_end - tmp_begin;

            wr_ft_h_agg_stm: for(int i=0; i<FEATS_IN/W_FT_IN; i++){
                // for (int j = 0; j < W_FT_IN; j++){
                //     #pragma HLS pipeline II=1

                //     ft_h_agg_stream[i*W_FT_IN+j] = ft_h_agg[i][j] / in_deg_n;

                // }
                ft_h_agg_stream << ft_h_agg[i] / in_deg_n;
            }
        }
    }
}


// update/apply phase for the aggrated results
void update_agg(TYPE w_update_agg[FEATS_IN][FEATS_OUT],
                // TYPE ft_h_agg_stream[FEATS_IN],
                hls::stream<vec_ft_in_t>& ft_h_agg_stream,
                // edge_index_t tmp_begin,
                // edge_index_t tmp_end,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<node_t>& nod_src_stream4,
                // TYPE rst_agg_stream[FEATS_OUT]
                // hls::stream<vec_ft_out_t>& rst_agg_stream
                // hls::stream<vec_ft_out_t>& rst_agg_p1_stream
                hls::stream<vec_ft_out_full_t>& rst_agg_p1_stream){

    update_agg_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        node_t nod_src_temp = nod_src_stream4.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;
        
        if(tmp_end != tmp_begin){

            // receive the aggregated features and convert it from a vector of width W_FT_IN to an array of scalar
            TYPE ft_h_agg[FEATS_IN];
            const int w_ft_in = W_FT_IN;
            #pragma HLS array_partition variable=ft_h_agg cyclic factor=w_ft_in dim=1

            rd_ft_h_agg_stm_l0: for(int i=0; i<FEATS_IN/W_FT_IN; i++){
                #pragma HLS pipeline II=1

                vec_ft_in_t ft_h_temp = ft_h_agg_stream.read();

                rd_ft_h_agg_stm_l1: for (int j = 0; j < W_FT_IN; j++){
                    #pragma HLS UNROLL

                    ft_h_agg[i*W_FT_IN + j] = ft_h_temp[j];
                }
            }

            // update/apply phase for the aggrated results
            TYPE rst_agg[D][FEATS_OUT];
            #pragma HLS array_partition variable=rst_agg cyclic factor=2 dim=1
            #pragma HLS array_partition variable=rst_agg cyclic factor=128 dim=2
            update_agg_l0: for(int k=0; k<FEATS_IN/D; k++){

                update_agg_l1: for (int kd = 0; kd < D; kd++){
                    #pragma HLS pipeline II=1 rewind
                    #pragma HLS UNROLL factor=2

                    TYPE ft_h_agg_temp = ft_h_agg[k*D + kd];

                    update_agg_l2: for(int j=0; j<FEATS_OUT; j++){
                        #pragma HLS UNROLL

                        TYPE rst_agg_temp = (k == 0) ? 0 : rst_agg[kd][j];

                        // rst_agg[kd][j] = rst_agg_temp + ft_h_agg[k*D + kd] * w_update_agg[k*D + kd][j];
                        rst_agg[kd][j] = rst_agg_temp + ft_h_agg_temp * w_update_agg[k*D + kd][j];
                    }
                }
            }

            // // write the result of agg update p1 to stream
            // wr_rst_agg_p1_stm_l0: for (int kd = 0; kd < D; kd++){
            //     wr_rst_agg_p1_stm_l1: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
            //         #pragma HLS pipeline II=1

            //         vec_ft_out_t rst_agg_vec_temp;

            //         wr_rst_agg_p1_stm_l2: for (int j = 0; j < W_FT_OUT; j++){
            //             #pragma HLS UNROLL

            //             rst_agg_vec_temp[j] = rst_agg[kd][i*W_FT_OUT + j];
            //         }
                    
            //         rst_agg_p1_stream << rst_agg_vec_temp;
            //     }
            // }

            // write the result of agg update p1 to stream
            wr_rst_agg_p1_stm_l0: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1

                vec_ft_out_full_t rst_agg_vec_temp;

                wr_rst_agg_p1_stm_l1: for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS UNROLL

                    rst_agg_vec_temp[j] = rst_agg[kd][j];
                }

                rst_agg_p1_stream << rst_agg_vec_temp;
            }
        }
    }
}

void update_agg_sum(// hls::stream<vec_ft_out_t>& rst_agg_p1_stream,
                    hls::stream<vec_ft_out_full_t>& rst_agg_p1_stream,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    hls::stream<node_t>& nod_src_stream6,
                    hls::stream<vec_ft_out_t>& rst_agg_stream){
    
    update_agg_sum_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        node_t nod_src_temp = nod_src_stream6.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if(tmp_end != tmp_begin){

            TYPE rst_agg[D][FEATS_OUT];
            #pragma HLS array_partition variable=rst_agg complete dim=2
            rd_agg_p1_stm_l0: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1

                vec_ft_out_full_t rst_agg_vec_temp = rst_agg_p1_stream.read();

                rd_agg_p1_stm_l1: for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS UNROLL
                    rst_agg[kd][j] = rst_agg_vec_temp[j];
                }
            }

            // sum the D rows of results
            TYPE rst_agg_sum[FEATS_OUT];
            #pragma HLS array_partition variable=rst_agg_sum cyclic factor=16 dim=1
            update_agg_sum_l0: for (int kd = 0; kd < D; kd++){

                update_agg_sum_l1: for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS pipeline II=1 rewind
                    #pragma HLS UNROLL factor=16

                    TYPE rst_sum_temp = (kd == 0) ? 0 : rst_agg_sum[j];

                    rst_agg_sum[j] = rst_sum_temp + rst_agg[kd][j];
                }
            }

            // write rst_agg to stream and convert it from an array of scalar to vector of width W_FT_OUT
            wr_rst_agg_stm_l0: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
                #pragma HLS pipeline II=1

                vec_ft_out_t rst_agg_sum_temp;

                wr_rst_agg_stm_l1: for (int j = 0; j < W_FT_OUT; j++){
                    #pragma HLS UNROLL
                    
                    rst_agg_sum_temp[j] = rst_agg_sum[i*W_FT_OUT + j];
                }

                rst_agg_stream << rst_agg_sum_temp;
            }
        }
    }
}

// read the target feature from memory
void read_feat_in_tar(vec_ft_in_t ft_in_tar_mat[N_NODES*FEATS_IN/W_FT_IN],
                    //  node_index_t n,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                      hls::stream<vec_ft_in_t>& ft_in_tar_stream){
    
    rd_ft_in_tar_mem_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        rd_ft_in_tar_mem_l0: for (int i = 0; i < FEATS_IN/W_FT_IN; i++){
            #pragma HLS pipeline II=1

            ft_in_tar_stream << ft_in_tar_mat[n*FEATS_IN/W_FT_IN + i];
        }
    }
}

// update/apply phase for the target feature
void update_tar(// TYPE ft_in_tar_stream[FEATS_IN],
                hls::stream<vec_ft_in_t>& ft_in_tar_stream,
                TYPE w_update_tar[FEATS_IN][FEATS_OUT],
                node_index_t nidx_begin,
                node_index_t nidx_end,
                // TYPE rst_tar_stream[FEATS_OUT]
                // hls::stream<vec_ft_out_t>& rst_tar_stream
                // hls::stream<vec_ft_out_t>& rst_tar_p1_stream
                hls::stream<vec_ft_out_full_t>& rst_tar_p1_stream){

    update_tar_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        // receive the aggregated features and convert it from a vector of width W_FT_IN to an array of scalar
        TYPE ft_in_tar[FEATS_IN];
        const int w_ft_in = W_FT_IN;
        #pragma HLS array_partition variable=ft_in_tar cyclic factor=w_ft_in dim=1

        rd_ft_in_tar_stm_l0: for(int i=0; i<FEATS_IN/W_FT_IN; i++){
            #pragma HLS pipeline II=1

            vec_ft_in_t ft_in_tar_temp = ft_in_tar_stream.read();

            rd_ft_in_tar_stm_l1: for (int j = 0; j < W_FT_IN; j++){
                #pragma HLS UNROLL

                ft_in_tar[i*W_FT_IN + j] = ft_in_tar_temp[j];
            }
        }

        // update the target feature
        TYPE rst_tar[D][FEATS_OUT];
        #pragma HLS array_partition variable=rst_tar cyclic factor=2 dim=1
        #pragma HLS array_partition variable=rst_tar cyclic factor=128 dim=2
        update_tar_l0: for(int k=0; k<FEATS_IN/D; k++){

            update_tar_l1: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1 rewind
                #pragma HLS UNROLL factor=2

                TYPE ft_in_tar_temp = ft_in_tar[k*D + kd];

                update_tar_l2: for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS UNROLL

                    TYPE rst_tar_temp = (k == 0) ? 0 : rst_tar[kd][j];

                    rst_tar[kd][j] = rst_tar_temp + ft_in_tar_temp * w_update_tar[k*D + kd][j];
                }
            }
        }

        // // write the result of target update p1 to stream
        // wr_rst_tar_p1_stm_l0: for (int kd = 0; kd < D; kd++){
        //     wr_rst_tar_p1_stm_l1: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
        //         #pragma HLS pipeline II=1

        //         vec_ft_out_t rst_tar_vec_temp;

        //         wr_rst_tar_p1_stm_l2: for (int j = 0; j < W_FT_OUT; j++){
        //             #pragma HLS UNROLL

        //             rst_tar_vec_temp[j] = rst_tar[kd][i*W_FT_OUT + j];
        //         }
                
        //         rst_tar_p1_stream << rst_tar_vec_temp;
        //     }
        // }

        // write the result of target update p1 to stream
        wr_rst_tar_p1_stm_l0: for (int kd = 0; kd < D; kd++){
            #pragma HLS pipeline II=1

            vec_ft_out_full_t rst_tar_vec_temp;

            for(int j=0; j<FEATS_OUT; j++){
                #pragma HLS UNROLL

                rst_tar_vec_temp[j] = rst_tar[kd][j];
            }

            rst_tar_p1_stream << rst_tar_vec_temp;
        }
    }
}

void update_tar_sum(// hls::stream<vec_ft_out_t>& rst_tar_p1_stream,
                    hls::stream<vec_ft_out_full_t>& rst_tar_p1_stream,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    hls::stream<vec_ft_out_t>& rst_tar_stream){

    update_tar_sum_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        // TYPE rst_tar[D][FEATS_OUT];
        // const int w_ft_out = W_FT_OUT;
        // #pragma HLS array_partition variable=rst_tar cyclic factor=w_ft_out dim=2
        // rd_tar_p1_stm_l0: for (int kd = 0; kd < D; kd++){
        //     rd_tar_p1_stm_l1: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
        //         #pragma HLS pipeline II=1

        //         vec_ft_out_t rst_tar_vec_temp = rst_tar_p1_stream.read();

        //         rd_tar_p1_stm_l2: for (int j = 0; j < W_FT_OUT; j++){
        //             #pragma HLS UNROLL

        //             rst_tar[kd][i*W_FT_OUT + j] = rst_tar_vec_temp[j];
        //         }
        //     }
        // }

        TYPE rst_tar[D][FEATS_OUT];
        #pragma HLS array_partition variable=rst_tar complete dim=2
        rd_tar_p1_stm_l0: for (int kd = 0; kd < D; kd++){
            #pragma HLS pipeline II=1

            vec_ft_out_full_t rst_tar_vec_temp = rst_tar_p1_stream.read();

            rd_tar_p1_stm_l1: for(int j=0; j<FEATS_OUT; j++){
                #pragma HLS UNROLL
                rst_tar[kd][j] = rst_tar_vec_temp[j];
            }
        }

        // sum the D rows of results
        TYPE rst_tar_sum[FEATS_OUT];
        #pragma HLS array_partition variable=rst_tar_sum cyclic factor=16 dim=1
        update_tar_sum_l0: for (int kd = 0; kd < D; kd++){

            update_tar_sum_l1: for(int j=0; j<FEATS_OUT; j++){
                #pragma HLS pipeline II=1 rewind
                #pragma HLS UNROLL factor=16

                TYPE rst_sum_temp = (kd == 0) ? 0 : rst_tar_sum[j];

                rst_tar_sum[j] = rst_sum_temp + rst_tar[kd][j];
            }
        }

        // write rst_tar to stream and convert it from an array of scalar to vector of width W_FT_OUT
        wr_rst_tar_stm_l0: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
            #pragma HLS pipeline II=1

            vec_ft_out_t rst_tar_sum_temp;

            wr_rst_tar_stm_l1: for (int j = 0; j < W_FT_OUT; j++){
                #pragma HLS UNROLL
                
                rst_tar_sum_temp[j] = rst_tar_sum[i*W_FT_OUT + j];
            }

            rst_tar_stream << rst_tar_sum_temp;
        }
    }
}

// concat the update/apply results of aggregated features and the target feature
void concat_update_rst( // TYPE rst_agg_stream[FEATS_OUT],
                hls::stream<vec_ft_out_t>& rst_agg_stream,
                //   TYPE rst_tar_stream[FEATS_OUT],
                hls::stream<vec_ft_out_t>& rst_tar_stream,
                //   edge_index_t tmp_begin,
                //   edge_index_t tmp_end,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<node_t>& nod_src_stream5,
                //   TYPE rst_cat_stream[FEATS_OUT]
                hls::stream<vec_ft_out_t>& rst_cat_stream){
    
    cat_update_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream5.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;
        
        concat_upate_rst_l0: for(int i = 0; i < FEATS_OUT/W_FT_OUT; i++){

            #pragma HLS pipeline II=1

            if(tmp_end != tmp_begin){
                rst_cat_stream << rst_agg_stream.read() + rst_tar_stream.read();
            }
            else{
                rst_cat_stream << rst_tar_stream.read();
            }
        }
    }
}


// normalization for each vertex (cal base + norm div)
void norm_reduce_sum(// TYPE rst_cat_stream[FEATS_OUT],
        hls::stream<vec_ft_out_t>& rst_cat_stream1,
        node_index_t nidx_begin,
        node_index_t nidx_end,
        // TYPE rst_norm_stream[FEATS_OUT]
        hls::stream<vec_ft_out_t>& rst_cat_stream2,
        hls::stream<TYPE>& rst_norm_reduce_sum_stream){

    norm_reduce_sum_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        // #pragma HLS expression_balance

        norm_reduce_sum_l0: for(int i = 0; i < FEATS_OUT/W_FT_OUT; i++){
            #pragma HLS pipeline II=1
            // #pragma HLS expression_balance

            // read concat rst from stream
            vec_ft_out_t rst_cat_temp = rst_cat_stream1.read();
            rst_cat_stream2 << rst_cat_temp;

            vec_ft_out_t base_vec_temp = rst_cat_temp * rst_cat_temp;

            // base_squre += base_vec_temp.reduce_add();
            rst_norm_reduce_sum_stream << base_vec_temp.reduce_add();
        }
    }
}

// accumulate norm reduced sum results of all FIFO segments
void norm_acc_rst(hls::stream<TYPE>& rst_norm_reduce_sum_stream,
            node_index_t nidx_begin,
            node_index_t nidx_end,
             hls::stream<TYPE>& rst_norm_acc_stream){
    
    norm_acc_rst_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        TYPE base_squre = 0;

        norm_acc_rst_l0: for(int i = 0; i < FEATS_OUT/W_FT_OUT; i++){
            #pragma HLS pipeline II=1
            base_squre += rst_norm_reduce_sum_stream.read();
        }

        #ifdef FLOAT32
            TYPE base_norm = sqrtf(base_squre);
        #else
            TYPE base_norm = sqrt(base_squre);
        #endif

        if(base_norm < 1e-12){
            base_norm = 1e-12;
        }

        rst_norm_acc_stream << base_norm;
    }
}

// divide concat results by norm accumulate results
void norm_div(hls::stream<vec_ft_out_t>& rst_cat_stream2,
            hls::stream<TYPE>& rst_norm_acc_stream,
            node_index_t nidx_begin,
            node_index_t nidx_end,
             hls::stream<vec_ft_out_t>& rst_norm_stream
            ){

    for(node_index_t n=nidx_begin; n<nidx_end; n++){

        norm_div_l0: for (int i = 0; i < FEATS_OUT/W_FT_OUT; i++){
            #pragma HLS pipeline II=1

            TYPE base_norm;

            if(i == 0){
                base_norm = rst_norm_acc_stream.read();
            }

            vec_ft_out_t rst_cat_vec = rst_cat_stream2.read();

            vec_ft_out_t rst_norm_div_vec;

            // write norm rst to stream and convert an array of scalar to a vector
            norm_div_l1: for (int j = 0; j < W_FT_OUT; j++){
                #pragma HLS UNROLL

                #ifdef ACTIVATION
                    rst_norm_div_vec[j] = relu(rst_cat_vec[j] / base_norm);
                #else
                    rst_norm_div_vec[j] = rst_cat_vec[j] / base_norm;
                #endif
            }

            rst_norm_stream << rst_norm_div_vec;
        }
    }
}

// // write norm results to memory
// void write_norm_rst_mem(// TYPE rst_norm_stream[FEATS_OUT],
//                         hls::stream<vec_ft_out_t>& rst_norm_stream,
//                         node_index_t n,
//                         // TYPE rst_mat[N_NODES*FEATS_OUT]
//                         vec_ft_out_t rst_mat[N_NODES*FEATS_OUT/W_FT_OUT]){

//     wr_norm_rst_mem_l0: for(int i = 0; i < FEATS_OUT/W_FT_OUT; i++){
//         #pragma HLS pipeline II=1

//         vec_ft_out_t rst_norm = rst_norm_stream.read();

//         // activation or not
//         vec_ft_out_t rst_act;
//         #ifdef ACTIVATION
//             wr_norm_rst_mem_l1: for (int j = 0; j < W_FT_OUT; j++){
//                 #pragma HLS UNROLL
//                 rst_act[j] = relu(rst_norm[j]);
//             }
//         #else
//             wr_norm_rst_mem_l1: for (int j = 0; j < W_FT_OUT; j++){
//                 #pragma HLS UNROLL
//                 rst_act[j] = rst_norm[j];
//             }
//         #endif

//         rst_mat[n*FEATS_OUT/W_FT_OUT + i] = rst_act;
//     }
// }

// write norm results to memory
void write_norm_rst_mem(// TYPE rst_norm_stream[FEATS_OUT],
                        hls::stream<vec_ft_out_t>& rst_norm_stream,
                        // node_index_t n,
                        node_index_t nidx_begin,
                        node_index_t nidx_end,
                        // TYPE rst_mat[N_NODES*FEATS_OUT]
                        vec_ft_out_t rst_mat[N_NODES*FEATS_OUT/W_FT_OUT]){

    wr_norm_rst_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        vec_ft_out_t rst_norm_buf[FEATS_OUT/W_FT_OUT];
        rd_norm_rst_stm: for(int i = 0; i < FEATS_OUT/W_FT_OUT; i++){
            #pragma HLS pipeline II=1
            rst_norm_buf[i] = rst_norm_stream.read();
        }

        wr_norm_rst_mem: for(int i = 0; i < FEATS_OUT/W_FT_OUT; i++){
            #pragma HLS pipeline II=1
            rst_mat[n*FEATS_OUT/W_FT_OUT + i] = rst_norm_buf[i];
        }
    }
}

// Compute results for one node
void compute_one_node(node_t nod_src[N_NODES],
                    edge_src_t edge_src[N_EDGES],
                    vec_ft_in_t ft_in_agg_mat[N_NODES*FEATS_IN/W_FT_IN],
                    vec_ft_in_t ft_in_tar_mat[N_NODES*FEATS_IN/W_FT_IN],
                    // node_index_t n,
                    // edge_index_t tmp_begin,
                    // edge_index_t tmp_end,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    TYPE w_update_agg[FEATS_IN][FEATS_OUT],
                    TYPE w_update_tar[FEATS_IN][FEATS_OUT],
                    // TYPE rst_mat[N_NODES*FEATS_OUT]
                    vec_ft_out_t rst_mat[N_NODES*FEATS_OUT/W_FT_OUT]){
    const int feats_in = FEATS_IN;
    const int feats_out = FEATS_OUT;

    #pragma HLS dataflow

    // TYPE ft_h_agg_stream[FEATS_IN];
    hls::stream<vec_ft_in_t> ft_h_agg_stream;
    #pragma HLS stream variable=ft_h_agg_stream depth=10

    // TYPE rst_agg_stream[FEATS_OUT];
    hls::stream<vec_ft_out_t> rst_agg_stream;
    #pragma HLS stream variable=rst_agg_stream depth=10

    hls::stream<vec_ft_in_t> ft_in_tar_stream; // stream channel
    #pragma HLS stream variable=ft_in_tar_stream depth=10

    // TYPE rst_tar_stream[FEATS_OUT];
    hls::stream<vec_ft_out_t> rst_tar_stream;
    #pragma HLS stream variable=rst_tar_stream depth=10

    // TYPE rst_cat_stream[FEATS_OUT];
    hls::stream<vec_ft_out_t> rst_cat_stream;
    #pragma HLS stream variable=rst_cat_stream depth=10

    hls::stream<vec_ft_out_t> rst_norm_stream;
    #pragma HLS stream variable=rst_norm_stream depth=10

    // read nod_src from memory
    hls::stream<node_t> nod_src_stream;
    #pragma HLS stream variable=nod_src_stream depth=10
    read_nod_src(nod_src, nidx_begin, nidx_end, nod_src_stream);

    // split the nod_src_stream
    hls::stream<node_t> nod_src_stream1;
    #pragma HLS stream variable=nod_src_stream1 depth=11
    hls::stream<node_t> nod_src_stream2;
    #pragma HLS stream variable=nod_src_stream2 depth=12
    hls::stream<node_t> nod_src_stream3;
    #pragma HLS stream variable=nod_src_stream3 depth=13
    hls::stream<node_t> nod_src_stream4;
    #pragma HLS stream variable=nod_src_stream4 depth=14
    hls::stream<node_t> nod_src_stream6;
    #pragma HLS stream variable=nod_src_stream6 depth=15
    hls::stream<node_t> nod_src_stream5;
    #pragma HLS stream variable=nod_src_stream5 depth=17
    split_nod_stream(nod_src_stream, nidx_begin, nidx_end, nod_src_stream1, nod_src_stream2, nod_src_stream3, nod_src_stream4, nod_src_stream6, nod_src_stream5);

    // read edge_src from memory
    hls::stream<node_index_t> tmp_src_stream;
    #pragma HLS stream variable=tmp_src_stream depth=10
    // read_edge_src(edge_src, tmp_begin, tmp_end, tmp_src_stream);
    read_edge_src(edge_src, nidx_begin, nidx_end, nod_src_stream1, tmp_src_stream);

    // // read and aggregate features (traverse all the src nodes)
    // read_agg_feat_in(edge_src, ft_in_agg_mat, tmp_begin, tmp_end, ft_h_agg_stream);

    // read features for aggregation (traverse all the src nodes)
    hls::stream<vec_ft_in_t> ft_in_agg_stream;
    #pragma HLS stream variable=ft_in_agg_stream depth=10
    // read_feat_in_agg(edge_src, ft_in_agg_mat, tmp_begin, tmp_end, ft_in_agg_stream);
    // read_feat_in_agg(tmp_src_stream, ft_in_agg_mat, tmp_begin, tmp_end, ft_in_agg_stream);
    read_feat_in_agg(tmp_src_stream, ft_in_agg_mat, nidx_begin, nidx_end, nod_src_stream2, ft_in_agg_stream);

    // aggregate features (traverse all the src nodes)
    // agg_feat_in(ft_in_agg_stream, tmp_begin, tmp_end, ft_h_agg_stream);
    agg_feat_in(ft_in_agg_stream, nidx_begin, nidx_end, nod_src_stream3, ft_h_agg_stream);

    // update/apply phase for the aggrated results
    hls::stream<vec_ft_out_full_t> rst_agg_p1_stream;
    #pragma HLS stream variable=rst_agg_p1_stream depth=16
    update_agg(w_update_agg, ft_h_agg_stream, nidx_begin, nidx_end, nod_src_stream4, rst_agg_p1_stream);

    update_agg_sum(rst_agg_p1_stream, nidx_begin, nidx_end, nod_src_stream6, rst_agg_stream);


    // read the target feature from memory
    read_feat_in_tar(ft_in_tar_mat, nidx_begin, nidx_end, ft_in_tar_stream);

    // update/apply phase for the target feature
    hls::stream<vec_ft_out_full_t> rst_tar_p1_stream;
    #pragma HLS stream variable=rst_tar_p1_stream depth=16
    update_tar(ft_in_tar_stream, w_update_tar, nidx_begin, nidx_end, rst_tar_p1_stream);

    update_tar_sum(rst_tar_p1_stream, nidx_begin, nidx_end, rst_tar_stream);


    // concat the update/apply results of aggregated features and the target feature
    // concat_update_rst(rst_agg_stream, rst_tar_stream, tmp_begin, tmp_end, rst_cat_stream);
    concat_update_rst(rst_agg_stream, rst_tar_stream, nidx_begin, nidx_end, nod_src_stream5, rst_cat_stream);

    // normalization for each vertex (cal base + norm div)
    // norm(rst_cat_stream, nidx_begin, nidx_end, rst_norm_stream);
    hls::stream<vec_ft_out_t> rst_cat_stream2;
    #pragma HLS stream variable=rst_cat_stream2 depth=10
    hls::stream<TYPE> rst_norm_reduce_sum_stream;
    #pragma HLS stream variable=rst_norm_reduce_sum_stream depth=16
    norm_reduce_sum(rst_cat_stream, nidx_begin, nidx_end, rst_cat_stream2, rst_norm_reduce_sum_stream);

    hls::stream<TYPE> rst_norm_acc_stream;
    #pragma HLS stream variable=rst_norm_acc_stream depth=10
    norm_acc_rst(rst_norm_reduce_sum_stream, nidx_begin, nidx_end, rst_norm_acc_stream);

    norm_div(rst_cat_stream2, rst_norm_acc_stream, nidx_begin, nidx_end, rst_norm_stream);

    // write norm results to memory
    write_norm_rst_mem(rst_norm_stream, nidx_begin, nidx_end, rst_mat);
}

// Compute results for all nodes
void compute_all_node(node_t nod_src[N_NODES],
                    edge_src_t edge_src[N_EDGES],
                    vec_ft_in_t ft_in_agg_mat[N_NODES*FEATS_IN/W_FT_IN],
                    vec_ft_in_t ft_in_tar_mat[N_NODES*FEATS_IN/W_FT_IN],
                    TYPE w_update_agg[FEATS_IN][FEATS_OUT],
                    TYPE w_update_tar[FEATS_IN][FEATS_OUT],
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    // TYPE rst_mat[N_NODES*FEATS_OUT]
                    vec_ft_out_t rst_mat[N_NODES*FEATS_OUT/W_FT_OUT]){
    
    // traverse all nodes stored in CSR format
    // n is the idx of target node,
    // i is the idx of the feature element
    // tmp_src is the idx of input neighbors
    // for(node_index_t n=0; n<N_NODES; n++){
    // loop_nodes: for(node_index_t n=nidx_begin; n<nidx_end; n++){

    //     // aggregate phase (mean) = massage + reduce (mean)
    //     edge_index_t tmp_begin = nod_src[n].edge_begin;
    //     edge_index_t tmp_end = nod_src[n].edge_end;

    //     // calculate in_degree
    //     // edge_index_t in_deg_n = tmp_end - tmp_begin;
    //     // in_deg_n = (in_deg_n == 0)? 1: in_deg_n;

    //     // Compute results for one node
    //     compute_one_node(edge_src, ft_in_agg_mat, ft_in_tar_mat, n, tmp_begin, tmp_end, w_update_agg, w_update_tar, rst_mat);
    // }

    // loop_nodes: for(node_index_t n=nidx_begin; n<nidx_end; n++){

    //     // aggregate phase (mean) = massage + reduce (mean)
    //     // Compute results for one node
    //     compute_one_node(nod_src, edge_src, ft_in_agg_mat, ft_in_tar_mat, n, nidx_begin, nidx_end, w_update_agg, w_update_tar, rst_mat);
    // }

    compute_one_node(nod_src, edge_src, ft_in_agg_mat, ft_in_tar_mat, nidx_begin, nidx_end, w_update_agg, w_update_tar, rst_mat);
}

int graphsage_mean_hls(node_t nod_src[N_NODES],
            edge_src_t edge_src[N_EDGES],
            vec_ft_in_t ft_in_agg_mat[N_NODES*FEATS_IN/W_FT_IN],
            vec_ft_in_t ft_in_tar_mat[N_NODES*FEATS_IN/W_FT_IN],
            // int in_deg[N_NODES],
            // TYPE ft_h_mat[N_NODES*2*FEATS_IN],
            TYPE w_mlp_mat[2*FEATS_IN*FEATS_OUT],
            node_index_t nidx_begin,
            node_index_t nidx_end,
            // TYPE rst_mat[N_NODES*FEATS_OUT]
            vec_ft_out_t rst_mat[N_NODES*FEATS_OUT/W_FT_OUT]){
    #pragma HLS INTERFACE m_axi port=nod_src bundle=aximm1
    #pragma HLS INTERFACE m_axi port=edge_src bundle=aximm6
    // #pragma HLS INTERFACE m_axi port=ft_in_mat bundle=aximm2
    #pragma HLS INTERFACE m_axi port=ft_in_agg_mat bundle=aximm2
    #pragma HLS INTERFACE m_axi port=ft_in_tar_mat bundle=aximm3
    #pragma HLS INTERFACE m_axi port=w_mlp_mat bundle=aximm4

    #pragma HLS INTERFACE s_axilite port=nidx_begin
    #pragma HLS INTERFACE s_axilite port=nidx_end

    #pragma HLS INTERFACE m_axi port=rst_mat bundle=aximm5
    
    // Read weights for aggreted feature update/apply phase 
    TYPE w_update_agg[FEATS_IN][FEATS_OUT];
    #pragma HLS array_partition variable=w_update_agg cyclic factor=2 dim=1
    #pragma HLS array_partition variable=w_update_agg cyclic factor=128 dim=2
    
    // #pragma HLS array_partition variable=w_update_agg complete dim=1
    read_weight_agg(w_mlp_mat, w_update_agg);

    // Read weights for target feature update/apply phase
    TYPE w_update_tar[FEATS_IN][FEATS_OUT];
    #pragma HLS array_partition variable=w_update_tar cyclic factor=2 dim=1
    #pragma HLS array_partition variable=w_update_tar cyclic factor=128 dim=2
    read_weight_tar(w_mlp_mat, w_update_tar);
    
    // Compute results for all nodes
    compute_all_node(nod_src, edge_src, ft_in_agg_mat, ft_in_tar_mat, w_update_agg, w_update_tar, nidx_begin, nidx_end, rst_mat);

    return 0;
}