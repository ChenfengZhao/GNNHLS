/*
HLS kernel of GIN (Graph Isomorphism Networks) layer

Reference
[1] HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
*/

#include "gin.h"
// #include <hls_stream.h>

// # define D 16

TYPE relu(TYPE ft_h){

    if(ft_h < 0){
        ft_h = 0;
    }

    return ft_h;
}

// Read weights for MLP layer 0
void read_weight_mlp0(TYPE w_mlp0_mat[FEATS_IN*FEATS_HIDDEN],
                    TYPE w_mlp0[FEATS_IN][FEATS_HIDDEN]){
    
    rd_w_mlp0_l0: for (int i = 0; i < FEATS_IN; i++){
        rd_w_mlp0_l1: for (int j = 0; j < FEATS_HIDDEN; j++){
            // #pragma HLS pipeline II=1 rewind
            // #pragma HLS UNROLL factor=16
            #pragma HLS pipeline II=1

            w_mlp0[i][j] = w_mlp0_mat[i*FEATS_HIDDEN + j];
        }
    }
}

// Read weights for MLP layer 1
void read_weight_mlp1(TYPE w_mlp1_mat[FEATS_HIDDEN*FEATS_OUT],
                TYPE w_mlp1[FEATS_HIDDEN][FEATS_OUT]){

    rd_w_mlp1_l0: for (int i = 0; i < FEATS_HIDDEN; i++){
        rd_w_mlp1_l1: for (int j = 0; j < FEATS_OUT; j++){
            // #pragma HLS pipeline II=1
            // #pragma HLS UNROLL factor=16
            #pragma HLS pipeline II=1

            w_mlp1[i][j] = w_mlp1_mat[i*FEATS_OUT + j];
        }
    }
}

// // read input features to be aggregated 
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

// read nod_src from memory
void read_nod_src(node_t nod_src[N_NODES],
                // node_index_t n,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<node_t>& nod_src_stream0,
                hls::stream<node_t>& nod_src_stream1,
                hls::stream<node_t>& nod_src_stream2,
                hls::stream<node_t>& nod_src_stream3){
    
    rd_nod_src_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        #pragma HLS pipeline II=1
        node_t nod_src_n = nod_src[n];
        nod_src_stream0 << nod_src_n;
        nod_src_stream1 << nod_src_n;
        nod_src_stream2 << nod_src_n;
        nod_src_stream3 << nod_src_n;
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

// read input features to be aggregated 
// (@ToDo: try to put read memory outside loop and inside dataflow scope)
void read_feat_in_agg(// edge_src_t edge_src[N_EDGES],
                      hls::stream<node_index_t>& tmp_src_stream,
                      vec_ft_in_t ft_in_agg_mat[N_NODES*FEATS_IN/W_FT_IN],
                    //   edge_index_t tmp_begin,
                    //   edge_index_t tmp_end,
                    hls::stream<node_t>& nod_src_stream1,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    //   TYPE ft_in_agg[FEATS_IN]
                      hls::stream<vec_ft_in_t>& ft_in_agg_stream){
    
    rd_ft_in_agg_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        node_t nod_src_temp = nod_src_stream1.read();
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
void agg_feat_in(hls::stream<vec_ft_in_t>& ft_in_agg_stream,
                    // edge_index_t tmp_begin,
                    // edge_index_t tmp_end,
                    hls::stream<node_t>& nod_src_stream2,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    // TYPE ft_h_agg_stream[FEATS_IN]
                    hls::stream<vec_ft_in_t>& ft_h_agg_stream){

    for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream2.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if(tmp_end != tmp_begin){

            vec_ft_in_t ft_h_agg[FEATS_IN/W_FT_IN];
            // traverse all the src nodes
            agg_ft_in_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){
                
                // // aggregrate features
                // agg_ft_in: for(int i=0; i<FEATS_IN/W_FT_IN; i++){
                //     #pragma HLS pipeline II=1

                //     vec_ft_in_t ft_h_temp = (e == tmp_begin) ? 0 : ft_h_agg[i];
                //     // vec_ft_in_t ft_in_agg = ft_in_agg_stream.read();
                //     // ft_h_agg[i] = ft_h_temp + ft_in_agg;
                //     ft_h_agg[i] = ft_h_temp + ft_in_agg_stream.read();
                // }

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

            // write the result of aggregated results to stream
            wr_ft_h_agg_stm: for(int i=0; i<FEATS_IN/W_FT_IN; i++){
                #pragma HLS pipeline II=1

                // for (int j = 0; j < W_FT_IN; j++){

                //     ft_h_agg_stream[i*W_FT_IN+j] = ft_h_agg[i][j];
                // }

                ft_h_agg_stream << ft_h_agg[i];
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

        rd_ft_in_tar_mem: for (int i = 0; i < FEATS_IN/W_FT_IN; i++){
            #pragma HLS pipeline II=1

            ft_in_tar_stream << ft_in_tar_mat[n*FEATS_IN/W_FT_IN + i];
        }
    }
}

// combine/concat the aggregated results ft_h_agg and target features ft_in_tar_stream: (1 + eps)*ft_in_tar_stream[i] + ft_h_agg[i]
void concat_rst(// TYPE ft_h_agg_stream[FEATS_IN],
                hls::stream<vec_ft_in_t>& ft_h_agg_stream,
                hls::stream<vec_ft_in_t>& ft_in_tar_stream,
                // edge_index_t tmp_begin,
                // edge_index_t tmp_end,
                hls::stream<node_t>& nod_src_stream3,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<vec_ft_in_t>& rst_cat_stream){

    for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream3.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        concat_rst: for(int i = 0; i < FEATS_IN/W_FT_IN; i++){
            #pragma HLS pipeline II=1

            if(tmp_end != tmp_begin){
                // rst_cat_stream[i] = (1 + GIN_EPS) * ft_in_tar_stream[i] + ft_h_agg_stream[i];
                rst_cat_stream << (1 + GIN_EPS) * ft_in_tar_stream.read() + ft_h_agg_stream.read();
            }
            else{
                // rst_cat_stream[i] = (1 + GIN_EPS) * ft_in_tar_stream[i];
                rst_cat_stream << (1 + GIN_EPS) * ft_in_tar_stream.read();
            }
        }
    }
}

// MLP layer 0 (update phase)
void mlp0(hls::stream<vec_ft_in_t>& rst_cat_stream,
        TYPE w_mlp0[FEATS_IN][FEATS_HIDDEN],
        node_index_t nidx_begin,
        node_index_t nidx_end,
        // TYPE rst_mlp0_stream[FEATS_HIDDEN]
        // hls::stream<vec_ft_hidden_t>& rst_mlp0_stream
        hls::stream<vec_ft_hidden_full_t>& rst_mlp0_p1_stream){

    for(node_index_t n=nidx_begin; n<nidx_end; n++){

        // read rst_cat from stream to local buffer
        TYPE rst_cat_buf[FEATS_IN];
        const int w_ft_in = W_FT_IN;
        #pragma HLS array_partition variable=rst_cat_buf cyclic factor=w_ft_in dim=1
        rd_rst_cat_stm_l0: for(int i=0; i<FEATS_IN/W_FT_IN; i++){
            #pragma HLS pipeline II=1

            vec_ft_in_t rst_cat_temp = rst_cat_stream.read();

            rd_rst_cat_stm_l1: for (int j = 0; j < W_FT_IN; j++){
                #pragma HLS UNROLL

                rst_cat_buf[i*W_FT_IN + j] = rst_cat_temp[j];
            }
        }

        // MLP layer 0
        TYPE rst_mlp0_tmp[D][FEATS_HIDDEN];
        #pragma HLS array_partition variable=rst_mlp0_tmp cyclic factor=2 dim=1
        #pragma HLS array_partition variable=rst_mlp0_tmp cyclic factor=128 dim=2
        mlp0_l0: for(int k=0; k<FEATS_IN/D; k++){

            mlp0_l1: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1 rewind
                #pragma HLS UNROLL factor=2

                TYPE rst_cat_temp = rst_cat_buf[k*D + kd];

                mlp0_l2: for(int j=0; j<FEATS_HIDDEN; j++){
                    // #pragma HLS pipeline II=1 rewind
                    // #pragma HLS UNROLL factor=128
                    #pragma HLS UNROLL
                    
                    TYPE rst_mlp0_temp = (k == 0) ? 0 : rst_mlp0_tmp[kd][j];

                    // rst_mlp0_tmp[kd][j] = rst_mlp0_temp + rst_cat_buf[k*D + kd] * w_mlp0[k*D + kd][j];
                    rst_mlp0_tmp[kd][j] = rst_mlp0_temp + rst_cat_temp * w_mlp0[k*D + kd][j];
                }
            }
        }

        // write the result of mlp0 p1 to stream
        wr_rst_mlp0_p1_stm_l0: for (int kd = 0; kd < D; kd++){
            #pragma HLS pipeline II=1

            vec_ft_hidden_full_t rst_mlp0_vec_temp;

            for(int j=0; j<FEATS_HIDDEN; j++){
                #pragma HLS UNROLL

                rst_mlp0_vec_temp[j] = rst_mlp0_tmp[kd][j];
            }

            rst_mlp0_p1_stream << rst_mlp0_vec_temp;
        }
    }
}

void mlp0_sum(hls::stream<vec_ft_hidden_full_t>& rst_mlp0_p1_stream,
            node_index_t nidx_begin,
            node_index_t nidx_end,
            hls::stream<vec_ft_hidden_t>& rst_mlp0_stream){
    
    update_mlp0_sum_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        TYPE rst_mlp0_tmp[D][FEATS_HIDDEN];
        #pragma HLS array_partition variable=rst_mlp0_tmp complete dim=2
        rd_mlp0_p1_stm_l0: for (int kd = 0; kd < D; kd++){
            #pragma HLS pipeline II=1

            vec_ft_hidden_full_t rst_mlp0_vec_temp = rst_mlp0_p1_stream.read();

            rd_mlp0_p1_stm_l1: for(int j=0; j<FEATS_HIDDEN; j++){
                #pragma HLS UNROLL
                rst_mlp0_tmp[kd][j] = rst_mlp0_vec_temp[j];
            }
        }

        // sum the D rows of results
        TYPE rst_mlp0[FEATS_HIDDEN];
        #pragma HLS array_partition variable=rst_mlp0 cyclic factor=16 dim=1
        mlp0_sum_l0: for (int kd = 0; kd < D; kd++){
            mlp0_sum_l1: for(int j=0; j<FEATS_HIDDEN; j++){
                #pragma HLS pipeline II=1 rewind
                #pragma HLS UNROLL factor=16

                TYPE rst_mlp0_temp2 = (kd == 0) ? 0 : rst_mlp0[j];

                rst_mlp0[j] = rst_mlp0_temp2 + rst_mlp0_tmp[kd][j];
            }
        }

        // write results of MLP layer 0 to stream
        wr_rst_mlp0_stm_l0: for (int i = 0; i < FEATS_HIDDEN/W_FT_HIDDEN; i++){
            #pragma HLS pipeline II=1

            vec_ft_hidden_t rst_mlp0_temp;

            wr_rst_mlp0_stm_l1: for (int j = 0; j < W_FT_HIDDEN; j++){
                #pragma HLS UNROLL

                #ifdef ACTIVATION
                    rst_mlp0_temp[j] = relu(rst_mlp0[i*W_FT_HIDDEN + j]);
                #else
                    rst_mlp0_temp[j] = rst_mlp0[i*W_FT_HIDDEN + j];
                #endif
            }

            rst_mlp0_stream << rst_mlp0_temp;
        }
    }
}

// MLP layer 1 (update phase)
void mlp1(// TYPE rst_mlp1_stream[FEATS_HIDDEN],
        hls::stream<vec_ft_hidden_t>& rst_mlp1_stream,
        TYPE w_mlp1[FEATS_HIDDEN][FEATS_OUT],
        node_index_t nidx_begin,
        node_index_t nidx_end,
        // TYPE rst_mlp1_stream[FEATS_OUT]
        hls::stream<vec_ft_out_full_t>& rst_mlp1_p1_stream){

    for(node_index_t n=nidx_begin; n<nidx_end; n++){
        TYPE rst_mlp1_buf[FEATS_HIDDEN];
        const int w_ft_hidden = W_FT_HIDDEN;
        #pragma HLS array_partition variable=rst_mlp1_buf cyclic factor=w_ft_hidden dim=1
        rd_rst_mlp1_stm_l0: for(int i=0; i<FEATS_HIDDEN/W_FT_HIDDEN; i++){
            #pragma HLS pipeline II=1

            // rst_mlp1_buf[j] = rst_mlp1_stream[j];
            vec_ft_hidden_t rst_mlp1_temp = rst_mlp1_stream.read();

            rd_rst_mlp1_stm_l1: for (int j = 0; j < W_FT_HIDDEN; j++){
                #pragma HLS UNROLL
                
                rst_mlp1_buf[i*W_FT_HIDDEN + j] = rst_mlp1_temp[j];
            }
        }

        // MLP layer 1
        TYPE rst_mlp1_tmp[D][FEATS_OUT];
        #pragma HLS array_partition variable=rst_mlp1_tmp cyclic factor=2 dim=1
        #pragma HLS array_partition variable=rst_mlp1_tmp cyclic factor=128 dim=2
        mlp1_l0: for(int k=0; k<FEATS_HIDDEN/D; k++){
            // TYPE rst_mlp1_temp = rst_mlp1_stream[k];
            mlp1_l1: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1 rewind
                #pragma HLS UNROLL factor=2

                TYPE rst_mlp1_temp = rst_mlp1_buf[k*D + kd];

                mlp1_l2: for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS UNROLL

                    TYPE rst_temp = (k == 0) ? 0 : rst_mlp1_tmp[kd][j];

                    // rst_mlp1_tmp[kd][j] = rst_temp + rst_mlp1_buf[k*D + kd] * w_mlp1[k*D + kd][j];
                    rst_mlp1_tmp[kd][j] = rst_temp + rst_mlp1_temp * w_mlp1[k*D + kd][j];
                }
            }
        }

        // write the result of mlp1 p1 to stream
        wr_rst_mlp1_p1_stm_l0: for (int kd = 0; kd < D; kd++){
            #pragma HLS pipeline II=1

            vec_ft_out_full_t rst_mlp1_vec_temp;

            for(int j=0; j<FEATS_OUT; j++){
                #pragma HLS UNROLL

                rst_mlp1_vec_temp[j] = rst_mlp1_tmp[kd][j];
            }

            rst_mlp1_p1_stream << rst_mlp1_vec_temp;
        }
    }
}

void mlp1_sum(hls::stream<vec_ft_out_full_t>& rst_mlp1_p1_stream,
            node_index_t nidx_begin,
            node_index_t nidx_end,
            hls::stream<vec_ft_out_t>& rst_mlp1_stream){
        
    for(node_index_t n=nidx_begin; n<nidx_end; n++){

        TYPE rst_mlp1_tmp[D][FEATS_OUT];
        #pragma HLS array_partition variable=rst_mlp1_tmp complete dim=2
        rd_mlp1_p1_stm_l0: for (int kd = 0; kd < D; kd++){
            #pragma HLS pipeline II=1

            vec_ft_out_full_t rst_mlp1_vec_temp = rst_mlp1_p1_stream.read();

            rd_mlp1_p1_stm_l1: for(int j=0; j<FEATS_OUT; j++){
                #pragma HLS UNROLL
                rst_mlp1_tmp[kd][j] = rst_mlp1_vec_temp[j];
            }
        }

        // sum the D rows of results
        TYPE rst_mlp1[FEATS_OUT];
        #pragma HLS array_partition variable=rst_mlp1 cyclic factor=16 dim=1
        mlp1_sum_l0: for (int kd = 0; kd < D; kd++){
            mlp1_sum_l1: for(int j=0; j<FEATS_OUT; j++){
                #pragma HLS pipeline II=1 rewind
                #pragma HLS UNROLL factor=16

                TYPE rst_mlp1_temp1 = (kd == 0) ? 0 : rst_mlp1[j];

                rst_mlp1[j] = rst_mlp1_temp1 + rst_mlp1_tmp[kd][j];
            }
        }

        // // write results of MLP layer 1 to stream
        // wr_rst_mlp1_stm: for(int j=0; j<FEATS_OUT; j++){
        //     #pragma HLS pipeline II=1

        //     rst_mlp1_stream[j] = rst_mlp1[j];
        // }

        // write results of MLP layer 1 to stream
        wr_rst_mlp1_stm_l0: for (int i = 0; i < FEATS_OUT/W_FT_OUT; i++){
            #pragma HLS pipeline II=1

            vec_ft_out_t rst_mlp1_temp;

            wr_rst_mlp1_stm_l1: for (int j = 0; j < W_FT_OUT; j++){
                #pragma HLS UNROLL

                #ifdef ACTIVATION
                    rst_mlp1_temp[j] = relu(rst_mlp1[i*W_FT_OUT + j]);
                #else
                    rst_mlp1_temp[j] = rst_mlp1[i*W_FT_OUT + j];
                #endif
            }

            rst_mlp1_stream << rst_mlp1_temp;
        }
    }
}
// write results of MLP layer 1 to memory
void write_mlp1_rst_mem(// TYPE rst_mlp1_stream[FEATS_OUT],
                         hls::stream<vec_ft_out_t>& rst_mlp1_stream,
                        // node_index_t n,
                        node_index_t nidx_begin,
                        node_index_t nidx_end,
                        // TYPE rst_mat[N_NODES*FEATS_OUT]
                        vec_ft_out_t rst_mat[N_NODES*FEATS_OUT/W_FT_OUT]){
    
    // wr_mlp1_rst_mem: for(int j=0; j<FEATS_OUT; j++){
    //     #pragma HLS pipeline II=1

    //     #ifdef ACTIVATION
    //         rst_mat[n*FEATS_OUT +j] = relu(rst_mlp1_stream[j]);
    //     #else
    //         rst_mat[n*FEATS_OUT +j] = rst_mlp1_stream[j];
    //     #endif
    // }

    for(node_index_t n=nidx_begin; n<nidx_end; n++){
        wr_mlp1_rst_mem: for (int i = 0; i < FEATS_OUT/W_FT_OUT; i++){
            #pragma HLS pipeline II=1
            
            rst_mat[n*FEATS_OUT/W_FT_OUT +i] = rst_mlp1_stream.read();
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
                    TYPE w_mlp0[FEATS_IN][FEATS_HIDDEN],
                    TYPE w_mlp1[FEATS_HIDDEN][FEATS_OUT],
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    vec_ft_out_t rst_mat[N_NODES*FEATS_OUT/W_FT_OUT]){
    
    const int feats_in = FEATS_IN;
    const int feats_hidden = FEATS_HIDDEN;
    const int feats_out = FEATS_OUT;

    #pragma HLS dataflow

    // TYPE ft_h_agg_stream[FEATS_IN];
    hls::stream<vec_ft_in_t> ft_h_agg_stream;
    #pragma HLS stream variable=ft_h_agg_stream depth=10

    // TYPE ft_in_tar_stream[FEATS_IN];
    hls::stream<vec_ft_in_t> ft_in_tar_stream;
    #pragma HLS stream variable=ft_in_tar_stream depth=10

    // TYPE rst_cat_stream[FEATS_OUT];
    hls::stream<vec_ft_in_t> rst_cat_stream;
    #pragma HLS stream variable=rst_cat_stream depth=10

    // TYPE rst_mlp0_stream[FEATS_HIDDEN];
    hls::stream<vec_ft_hidden_t> rst_mlp0_stream;
    #pragma HLS stream variable=rst_mlp0_stream depth=10

    // TYPE rst_mlp1_stream[FEATS_OUT];
    hls::stream<vec_ft_out_t> rst_mlp1_stream;
    #pragma HLS stream variable=rst_mlp1_stream depth=10

    // read nod_src from memory
    hls::stream<node_t> nod_src_stream0;
    #pragma HLS stream variable=nod_src_stream0 depth=4
    hls::stream<node_t> nod_src_stream1;
    #pragma HLS stream variable=nod_src_stream1 depth=5
    hls::stream<node_t> nod_src_stream2;
    #pragma HLS stream variable=nod_src_stream2 depth=6
    hls::stream<node_t> nod_src_stream3;
    #pragma HLS stream variable=nod_src_stream3 depth=8
    read_nod_src(nod_src, nidx_begin, nidx_end, nod_src_stream0, nod_src_stream1, nod_src_stream2, nod_src_stream3);

    // read edge_src from memory
    hls::stream<node_index_t> tmp_src_stream;
    #pragma HLS stream variable=tmp_src_stream depth=10
    read_edge_src(edge_src, nod_src_stream0, nidx_begin, nidx_end, tmp_src_stream);

    // // read and aggregate features (traverse all the src nodes)
    // TYPE ft_h_agg_stream[FEATS_IN];
    // read_agg_feat_in(edge_src, ft_in_agg_mat, tmp_begin, tmp_end, ft_h_agg_stream);

    // read features to be aggregated
    hls::stream<vec_ft_in_t> ft_in_agg_stream;
    #pragma HLS stream variable=ft_in_agg_stream depth=10
    // read_feat_in_agg(edge_src, ft_in_agg_mat, tmp_begin, tmp_end, ft_in_agg_stream);
    read_feat_in_agg(tmp_src_stream, ft_in_agg_mat, nod_src_stream1, nidx_begin, nidx_end, ft_in_agg_stream);

    // aggregate features
    agg_feat_in(ft_in_agg_stream, nod_src_stream2, nidx_begin, nidx_end, ft_h_agg_stream);

    // read the target feature from memory
    read_feat_in_tar(ft_in_tar_mat, nidx_begin, nidx_end, ft_in_tar_stream);

    // combine/concat the aggregated results ft_h_agg and target features ft_in_tar_stream: (1 + eps)*ft_in_tar_stream[i] + ft_h_agg[i]
    concat_rst(ft_h_agg_stream, ft_in_tar_stream, nod_src_stream3, nidx_begin, nidx_end, rst_cat_stream);

    // MLP layer 0 (update phase)
    hls::stream<vec_ft_hidden_full_t> rst_mlp0_p1_stream;
    #pragma HLS stream variable=rst_mlp0_p1_stream depth=16
    mlp0(rst_cat_stream, w_mlp0, nidx_begin, nidx_end, rst_mlp0_p1_stream);
    mlp0_sum(rst_mlp0_p1_stream, nidx_begin, nidx_end, rst_mlp0_stream);

    // MLP layer 1
    hls::stream<vec_ft_out_full_t> rst_mlp1_p1_stream;
    #pragma HLS stream variable=rst_mlp1_p1_stream depth=16
    mlp1(rst_mlp0_stream, w_mlp1, nidx_begin, nidx_end, rst_mlp1_p1_stream);
    mlp1_sum(rst_mlp1_p1_stream, nidx_begin, nidx_end, rst_mlp1_stream);

    // write results of MLP layer 1 to memory
    write_mlp1_rst_mem(rst_mlp1_stream, nidx_begin, nidx_end, rst_mat);
}

// Compute results for all nodes
void compute_all_node(node_t nod_src[N_NODES],
                    edge_src_t edge_src[N_EDGES],
                    vec_ft_in_t ft_in_agg_mat[N_NODES*FEATS_IN/W_FT_IN],
                    vec_ft_in_t ft_in_tar_mat[N_NODES*FEATS_IN/W_FT_IN],
                    TYPE w_mlp0[FEATS_IN][FEATS_HIDDEN],
                    TYPE w_mlp1[FEATS_HIDDEN][FEATS_OUT],
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    vec_ft_out_t rst_mat[N_NODES*FEATS_OUT/W_FT_OUT]){

    // message passing (aggregate + sum)
    // traverse all nodes stored in CSR format
    // n is the idx of target node,
    // i is the idx of the feature element
    // tmp_src is the idx of input neighbors
    // loop_nodes: for(node_index_t n=nidx_begin; n<nidx_end; n++){

    //     // aggregate phase (sum) = massage + reduce
    //     // edge_index_t tmp_begin = nod_src[n].edge_begin;
    //     // edge_index_t tmp_end = nod_src[n].edge_end;

    //     // Compute results for one node
    //     compute_one_node(nod_src, edge_src, ft_in_agg_mat, ft_in_tar_mat, n, w_mlp0, w_mlp1, rst_mat);
    // }

    compute_one_node(nod_src, edge_src, ft_in_agg_mat, ft_in_tar_mat, w_mlp0, w_mlp1, nidx_begin, nidx_end, rst_mat);
}

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
            vec_ft_out_t rst_mat[N_NODES*FEATS_OUT/W_FT_OUT]){

    #pragma HLS INTERFACE m_axi port=nod_src bundle=aximm1
    #pragma HLS INTERFACE m_axi port=edge_src bundle=aximm6
    // #pragma HLS INTERFACE m_axi port=ft_in_mat bundle=aximm2
    #pragma HLS INTERFACE m_axi port=ft_in_agg_mat bundle=aximm2
    #pragma HLS INTERFACE m_axi port=ft_in_tar_mat bundle=aximm3
    #pragma HLS INTERFACE m_axi port=w_mlp0_mat bundle=aximm4
    #pragma HLS INTERFACE m_axi port=w_mlp1_mat bundle=aximm4

    #pragma HLS INTERFACE s_axilite port=nidx_begin
    #pragma HLS INTERFACE s_axilite port=nidx_end

    #pragma HLS INTERFACE m_axi port=rst_mat bundle=aximm5
    // Read weights for MLP layer 0
    TYPE w_mlp0[FEATS_IN][FEATS_HIDDEN];
    #pragma HLS array_partition variable=w_mlp0 cyclic factor=2 dim=1
    #pragma HLS array_partition variable=w_mlp0 cyclic factor=128 dim=2
    read_weight_mlp0(w_mlp0_mat, w_mlp0);
    
    // Read weights for MLP layer 1
    TYPE w_mlp1[FEATS_HIDDEN][FEATS_OUT];
    #pragma HLS array_partition variable=w_mlp0 cyclic factor=2 dim=1
    #pragma HLS array_partition variable=w_mlp1 cyclic factor=128 dim=2
    read_weight_mlp1(w_mlp1_mat, w_mlp1);
    
    compute_all_node(nod_src, edge_src, ft_in_agg_mat, ft_in_tar_mat, w_mlp0, w_mlp1, nidx_begin, nidx_end, rst_mat);
    return 0;
}