/*
HLS kernel of GCN (graph converlution network) layer

Reference
[1] Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional 
    networks." arXiv preprint arXiv:1609.02907 (2016).
[2] Dwivedi, Vijay Prakash, et al. "Benchmarking graph neural networks." arXiv preprint arXiv:2003.
    00982 (2020).
*/

#include "gcn.h"

TYPE relu(TYPE ft_h){

    if(ft_h < 0){
        ft_h = 0;
    }

    return ft_h;
}

// read w_mat form memory to local memory
void read_weight(TYPE w_mat[FEATS_IN*FEATS_OUT],
                TYPE w_array[FEATS_IN][FEATS_OUT]){
    
    for(int i_w = 0; i_w < FEATS_IN; i_w++){

        for(int j_w = 0; j_w < FEATS_OUT; j_w++){
            #pragma HLS pipeline II=1

            w_array[i_w][j_w] = w_mat[i_w*FEATS_OUT + j_w];
        }
    }
}

/*
// initialize array[FEATS_IN] to a constant (e.g., 0 in this case)
void init_array_zero1(TYPE array[FEATS_IN]){
    
    int i;
    for (int i = 0; i < FEATS_IN; i++)
    {
        #pragma HLS unroll
        array[i] = 0;
    }
}

// initialize array[FEATS_OUT] to a constant (e.g., 0 in this case)
void init_array_zero2(TYPE array[FEATS_OUT]){
    
    int i;
    for (int i = 0; i < FEATS_OUT; i++)
    {
        array[i] = 0;
    }
}

// read w_mat form memory to local memory
void read_weight(TYPE w_mat[FEATS_IN*FEATS_OUT],
                TYPE w_array[FEATS_IN*FEATS_OUT]){
    int i_w;
    for(i_w = 0; i_w < FEATS_IN; i_w++){
        #pragma HLS pipeline rewind
        int j_w;
        for(j_w = 0; j_w < FEATS_OUT; j_w++){

            w_array[i_w*FEATS_OUT + j_w] = w_mat[i_w*FEATS_OUT + j_w];
        }
    }
}

// read the input feature of a source node
void read_feat_in(TYPE ft_in_mat[N_NODES*FEATS_IN],
                node_index_t tmp_src,
                TYPE ft_in_array[FEATS_IN]){

    int i;
    rd_ft_in: for(i=0; i<FEATS_IN; i++){
        // #pragma HLS unroll
        #pragma HLS pipeline II=1
        ft_in_array[i] = ft_in_mat[tmp_src*FEATS_IN + i];
    }
}

// reduce/aggregate features from source nodes
void aggregate_feat_in(TYPE ft_in_array[FEATS_IN],
                    edge_index_t e,
                    edge_index_t tmp_begin,
                    TYPE ft_bf_array[FEATS_IN]){

    int i;
    agg_ft_in: for(i=0; i<FEATS_IN; i++){
        #pragma HLS unroll
        #pragma HLS pipeline rewind
        TYPE ft_h_temp = (e == tmp_begin) ? 0 : ft_bf_array[i];
        ft_h_temp += ft_in_array[i];
        ft_bf_array[i] = ft_h_temp;
    }
}

// write data from local buffer ft_bf_array to the stream ft_h_array
void wr_to_stream1(TYPE ft_bf_array[FEATS_IN],
                   TYPE ft_h_array[FEATS_IN]){
        int i;
        for(i=0; i<FEATS_IN; i++){
            // #pragma HLS pipeline rewind
            #pragma HLS unroll
            ft_h_array[i] = ft_bf_array[i];
        }
}

// compute results by aggregating = message passing + reduce
void comp_rst_feat_in(TYPE ft_in_mat[N_NODES*FEATS_IN], 
                    // node_t nod_src[N_NODES],
                    edge_index_t tmp_begin,
                    edge_index_t tmp_end,
                    edge_src_t edge_src[N_EDGES],
                    node_index_t n,
                    TYPE ft_h_array[FEATS_IN]){

    // edge_index_t e;
    // edge_index_t tmp_begin = nod_src[n].edge_begin;
    // edge_index_t tmp_end = nod_src[n].edge_end;

    // initialize ft_h_array 
    // if(tmp_begin == tmp_end){
    //     init_array_zero1(ft_h_array);
    // }
    // else{

        TYPE ft_bf_array[FEATS_IN];
        #pragma HLS ARRAY_PARTITION variable=ft_bf_array dim=1 complete

        comp_rst: for(edge_index_t e=tmp_begin; e<tmp_end; e++){
            #pragma HLS pipeline rewind
            // #pragma HLS dataflow

            TYPE ft_in_array[FEATS_IN];
            #pragma HLS ARRAY_PARTITION variable=ft_in_array dim=1 complete
            // #pragma HLS stream off variable=ft_in_array 

            node_index_t tmp_src = edge_src[e].src;

            // read the input feature of the source node tmp_src
            read_feat_in(ft_in_mat, tmp_src, ft_in_array);

            // rd_ft_in: for(int i=0; i<FEATS_IN; i++){
            //     ft_in_array[i] = ft_in_mat[tmp_src*FEATS_IN + i];
            // }
            
            // reduce/aggregate features from source nodes
            aggregate_feat_in(ft_in_array, e, tmp_begin, ft_bf_array);

            // agg_ft_in: for(int i=0; i<FEATS_IN; i++){
            //     TYPE ft_h_temp = (e == tmp_begin) ? 0 : ft_bf_array[i];
            //     ft_h_temp += ft_in_array[i];
            //     ft_bf_array[i] = ft_h_temp;
            // }
        }

        // write data from local buffer ft_bf_array to the stream ft_h_array
        wr_to_stream1(ft_bf_array, ft_h_array);
        // for(int i=0; i<FEATS_IN; i++){
        //     ft_h_array[i] = ft_bf_array[i];
        // }
    // }
}

// read data from stream ft_h_array to loacl buffer ft_bf2_array
void rd_stream_h(TYPE ft_h_array[FEATS_IN],
            TYPE ft_bf2_array[FEATS_IN]){
    
    int i_u;
    rd_stream_h: for(i_u = 0; i_u < FEATS_IN; i_u++){
        // #pragma HLS pipeline rewind
        #pragma HLS unroll
        ft_bf2_array[i_u] = ft_h_array[i_u];
    }
}

// // multiply the reduced feature with weight
// void update_ft_in(TYPE ft_bf2_array[FEATS_IN],
//             TYPE w_array[FEATS_IN*FEATS_OUT],
//             TYPE rst_bf_array[FEATS_OUT]){
    
//     int i_u;
//     update_ft_in: for (i_u = 0; i_u < FEATS_IN; i_u++){
//         #pragma HLS pipeline rewind
//         int j;
//         TYPE ft_bf2_temp = ft_bf2_array[i_u];
//         for(j=0; j<FEATS_OUT; j++){
//             #pragma HLS unroll
//             TYPE rst_temp = (i_u == 0) ? 0 : rst_bf_array[j];
//             rst_temp += ft_bf2_temp * w_array[i_u*FEATS_OUT + j];
//             rst_bf_array[j] = rst_temp;
//         }
//     }
// }

// multiply the reduced feature with weight
void update_ft_in(TYPE ft_bf2_array[FEATS_IN],
            TYPE w_array[FEATS_IN*FEATS_OUT],
            TYPE rst_bf_array[FEATS_OUT]){
    
    int i_u;
    update_ft_in: for (i_u = 0; i_u < FEATS_IN; i_u++){
        #pragma HLS pipeline rewind
        int j;
        for(j=0; j<FEATS_OUT; j++){
            TYPE rst_temp = (i_u == 0) ? 0 : rst_bf_array[j];
            rst_temp += ft_bf2_array[i_u] * w_array[i_u*FEATS_OUT + j];
            rst_bf_array[j] = rst_temp;
        }
    }
}

// write data from local buffer rst_bf_array to stream rst_temp_arry
void wr_stream_rst(TYPE rst_bf_array[FEATS_OUT],
             TYPE rst_temp_arry[FEATS_OUT]){
    int i_u;
    wr_stream_rst: for(i_u = 0; i_u < FEATS_OUT; i_u++){
        // #pragma HLS pipeline rewind
        #pragma HLS unroll
        rst_temp_arry[i_u] = rst_bf_array[i_u];
    }
}

// update/apply phase (multiply with weight)
void update_rst(TYPE ft_h_array[FEATS_IN],
                TYPE w_array[FEATS_IN*FEATS_OUT],
                TYPE rst_temp_arry[FEATS_OUT]){
    
    TYPE ft_bf2_array[FEATS_IN];
    #pragma HLS ARRAY_PARTITION variable=ft_bf2_array dim=1 complete

    TYPE rst_bf_array[FEATS_OUT];
    #pragma HLS ARRAY_PARTITION variable=rst_bf_array dim=1 complete

    // read data from stream ft_h_array to loacl buffer ft_bf2_array
    rd_stream_h(ft_h_array, ft_bf2_array);
    // rd_stream_h: for(int i_u = 0; i_u < FEATS_IN; i_u++){
    //     ft_bf2_array[i_u] = ft_h_array[i_u];
    // }

    // multiply with weight
    update_ft_in(ft_bf2_array, w_array, rst_bf_array);
    // update_ft_in: for (int i_u = 0; i_u < FEATS_IN; i_u++){
    //     int j;
    //     for(j=0; j<FEATS_OUT; j++){
    //         TYPE rst_temp = (i_u == 0) ? 0 : rst_bf_array[j];
    //         rst_temp += ft_bf2_array[i_u] * w_array[i_u*FEATS_OUT + j];
    //         rst_bf_array[j] = rst_temp;
    //     }
    // }

    // write data from local buffer rst_bf_array to stream rst_temp_arry
    wr_stream_rst(rst_bf_array, rst_temp_arry);
    // wr_stream_rst: for(int i_u = 0; i_u < FEATS_OUT; i_u++){
    //     rst_temp_arry[i_u] = rst_bf_array[i_u];
    // }
}

// write rst from buffer to rst_mat
void wr_output(TYPE rst_temp_arry[FEATS_OUT],
               node_index_t n,
               TYPE rst_mat[N_NODES*FEATS_OUT]){
    
    int k;
    loop_output: for (k = 0; k < FEATS_OUT; k++){
        // #pragma HLS pipeline rewind
        #pragma HLS unroll

        #ifdef ACTIVATION
            rst_mat[n*FEATS_OUT +k] = relu(rst_temp_arry[k]);
        #else
            rst_mat[n*FEATS_OUT +k] = rst_temp_arry[k];
        #endif
    }
}
*/

// read nod_src from memory
void read_nod_src(node_t nod_src[N_NODES],
                // node_index_t n,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<node_t> nod_src_stream[6]){
    rd_nod_src_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        #pragma HLS pipeline II=1
        node_t nod_src_n = nod_src[n];

        // nod_src_stream[0] << nod_src_n;
        // nod_src_stream[1] << nod_src_n;
        // nod_src_stream[2] << nod_src_n;
        // nod_src_stream[3] << nod_src_n;

        for (int i = 0; i < 6; i++){
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

// read input features to be aggregated 
// (@ToDo: try to put read memory outside loop and inside dataflow scope)
void read_feat_in_agg(// edge_src_t edge_src[N_EDGES],
                      hls::stream<node_index_t>& tmp_src_stream,
                      vec_ft_in_t ft_in_mat[N_NODES*FEATS_IN/W_FT_IN],
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

                ft_in_agg_stream << ft_in_mat[tmp_src*FEATS_IN/W_FT_IN + i];
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
                    hls::stream<vec_ft_in_t>& ft_h_agg_stream)
{

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

// update/apply phase for the aggrated results
void update_agg(TYPE w_array[FEATS_IN][FEATS_OUT],
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
                hls::stream<vec_ft_out_full_t>& rst_agg_p1_stream)
{

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

                        // rst_agg[kd][j] = rst_agg_temp + ft_h_agg[k*D + kd] * w_array[k*D + kd][j];
                        rst_agg[kd][j] = rst_agg_temp + ft_h_agg_temp * w_array[k*D + kd][j];
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
                    hls::stream<vec_ft_out_t>& rst_agg_stream)
{
    
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
                    
                    #ifdef ACTIVATION
                        rst_agg_sum_temp[j] = relu(rst_agg_sum[i*W_FT_OUT + j]);
                    #else
                        rst_agg_sum_temp[j] = rst_agg_sum[i*W_FT_OUT + j];
                    #endif
                }

                rst_agg_stream << rst_agg_sum_temp;
            }
        }
    }
}

// write results memory
void write_rst_mem(// TYPE rst_agg_stream[FEATS_OUT],
                         hls::stream<vec_ft_out_t>& rst_agg_stream,
                        // node_index_t n,
                        node_index_t nidx_begin,
                        node_index_t nidx_end,
                        hls::stream<node_t>& nod_src_stream,
                        // TYPE rst_mat[N_NODES*FEATS_OUT]
                        vec_ft_out_t rst_mat[N_NODES*FEATS_OUT/W_FT_OUT])
{

    for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if(tmp_end != tmp_begin){
            wr_rst_mem: for (int i = 0; i < FEATS_OUT/W_FT_OUT; i++){
                #pragma HLS pipeline II=1
                // vec_ft_out_t rst_agg_vec = rst_agg_stream.read();

                rst_mat[n*FEATS_OUT/W_FT_OUT +i] = rst_agg_stream.read();
            }
        }
    }
}

// Compute results for one node
void compute_one_node(vec_ft_in_t ft_in_mat[N_NODES*FEATS_IN/W_FT_IN],
                    node_t nod_src[N_NODES],
                    edge_src_t edge_src[N_EDGES],
                    TYPE w_array[FEATS_IN][FEATS_OUT],
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    vec_ft_out_t rst_mat[N_NODES*FEATS_OUT/W_FT_OUT]){
    #pragma HLS dataflow


    // read nod_src from memory
    hls::stream<node_t> nod_src_stream[6];
    #pragma HLS stream variable=nod_src_stream[0] depth=5
    #pragma HLS stream variable=nod_src_stream[1] depth=6
    #pragma HLS stream variable=nod_src_stream[2] depth=7
    #pragma HLS stream variable=nod_src_stream[3] depth=8
    #pragma HLS stream variable=nod_src_stream[4] depth=9
    #pragma HLS stream variable=nod_src_stream[5] depth=10
    read_nod_src(nod_src, nidx_begin, nidx_end, nod_src_stream);

    // read edge_src from memory
    hls::stream<node_index_t> tmp_src_stream;
    #pragma HLS stream variable=tmp_src_stream depth=10
    read_edge_src(edge_src, nod_src_stream[0], nidx_begin, nidx_end, tmp_src_stream);

    // read features to be aggregated
    hls::stream<vec_ft_in_t> ft_in_agg_stream;
    #pragma HLS stream variable=ft_in_agg_stream depth=10
    read_feat_in_agg(tmp_src_stream, ft_in_mat, nod_src_stream[1], nidx_begin, nidx_end, ft_in_agg_stream);

    // aggregate features
    hls::stream<vec_ft_in_t> ft_h_agg_stream;
    #pragma HLS stream variable=ft_h_agg_stream depth=10
    agg_feat_in(ft_in_agg_stream, nod_src_stream[2], nidx_begin, nidx_end, ft_h_agg_stream);

    // update/apply phase for the aggrated results
    hls::stream<vec_ft_out_full_t> rst_agg_p1_stream;
    #pragma HLS stream variable=rst_agg_p1_stream depth=16
    update_agg(w_array, ft_h_agg_stream, nidx_begin, nidx_end, nod_src_stream[3], rst_agg_p1_stream);

    hls::stream<vec_ft_out_t> rst_agg_stream;
    #pragma HLS stream variable=rst_agg_stream depth=10
    update_agg_sum(rst_agg_p1_stream, nidx_begin, nidx_end, nod_src_stream[4], rst_agg_stream);

    // write results to memory
    write_rst_mem(rst_agg_stream, nidx_begin, nidx_end, nod_src_stream[5], rst_mat);
}

// Compute results for all nodes
void compute_all_node(vec_ft_in_t ft_in_mat[N_NODES*FEATS_IN/W_FT_IN], 
                    node_t nod_src[N_NODES],
                    edge_src_t edge_src[N_EDGES],
                    TYPE w_array[FEATS_IN][FEATS_OUT],
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    vec_ft_out_t rst_mat[N_NODES*FEATS_OUT/W_FT_OUT]){
    
    // node_index_t n;

    // // traverse all nodes stored in CSR format
    // // n is the idx of target node,
    // // i is the idx of the feature element
    // // tmp_src is the idx of input neighbors
    // // #pragma omp parallel for
    // loop_nodes: for(n=nidx_begin; n<nidx_end; n++){
    // // loop_nodes: for(n=0; n<N_NODES; n=n+2){
    //     // #pragma HLS pipeline rewind
    //     // #pragma HLS unroll factor=8

    //     edge_index_t tmp_begin = nod_src[n].edge_begin;
    //     edge_index_t tmp_end = nod_src[n].edge_end;

    //     if(tmp_begin != tmp_end){
    //         compute_one_node(ft_in_mat, tmp_begin, tmp_end, edge_src, w_array, n, rst_mat);
    //     }
    // }

    compute_one_node(ft_in_mat, nod_src, edge_src, w_array, nidx_begin, nidx_end, rst_mat);
}


extern "C"{
int gcn_hls(node_t nod_src[N_NODES],
            edge_src_t edge_src[N_EDGES],
            // TYPE ft_in_mat[N_NODES*FEATS_IN],
            vec_ft_in_t ft_in_mat[N_NODES*FEATS_IN/W_FT_IN],
            TYPE w_mat[FEATS_IN*FEATS_OUT],
            node_index_t nidx_begin,
            node_index_t nidx_end,
            // TYPE rst_mat[N_NODES*FEATS_OUT]
            vec_ft_out_t rst_mat[N_NODES*FEATS_OUT/W_FT_OUT]
            ){

    #pragma HLS INTERFACE m_axi port=nod_src bundle=aximm1
    #pragma HLS INTERFACE m_axi port=edge_src bundle=aximm2
    #pragma HLS INTERFACE m_axi port=ft_in_mat bundle=aximm3
    #pragma HLS INTERFACE m_axi port=w_mat bundle=aximm4
    #pragma HLS INTERFACE s_axilite port=nidx_begin
    #pragma HLS INTERFACE s_axilite port=nidx_end
    #pragma HLS INTERFACE m_axi port=rst_mat bundle=aximm5

    // array partition identifier
    const int feats_out = FEATS_OUT;

    // edge_index_t e;
    // int i,j,k;
    TYPE w_array[FEATS_IN][FEATS_OUT];
    #pragma HLS array_partition variable=w_array cyclic factor=2 dim=1
    #pragma HLS array_partition variable=w_array complete dim=2
    
    // #pragma HLS dataflow
    
    // read w_mat form memory to local memory
    read_weight(w_mat, w_array);
    
    // Compute results for all nodes
    compute_all_node(ft_in_mat, nod_src, edge_src, w_array, nidx_begin, nidx_end, rst_mat);

    return 0;
}
} // end extern "C"