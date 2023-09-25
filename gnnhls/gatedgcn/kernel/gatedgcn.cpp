/*
HLS kernel of GMM (Gaussian Mixture Model Convolution) layer

Reference
[1] Monti, Federico, et al. "Geometric deep learning on graphs and manifolds using mixture model cnns." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
*/

#include "gatedgcn.h"

TYPE sigmoid(TYPE a){

    #ifdef FLOAT32
        return 1 / (1 + expf(-a));
    #else
        return 1 / (1 + exp(-a));
    #endif
}

TYPE relu(TYPE ft_h){

    if(ft_h < 0){
        ft_h = 0;
    }

    return ft_h;
}

// Read weights for layer A
void read_weight_a(TYPE w_a_mat[FEATS_IN*FEATS_OUT],
                TYPE w_a[FEATS_IN][FEATS_OUT]){

    rd_w_a_l0: for (int i = 0; i < FEATS_IN; i++){

        rd_w_a_l1: for (int j = 0; j < FEATS_OUT; j++){
            #pragma HLS pipeline II=1
            
            w_a[i][j] = w_a_mat[i*FEATS_OUT + j];
        }
    }
}

// Read weights for layer B
void read_weight_b(TYPE w_b_mat[FEATS_IN*FEATS_OUT],
                TYPE w_b[FEATS_IN][FEATS_OUT]){

    rd_w_b_l0: for (int i = 0; i < FEATS_IN; i++){

        rd_w_b_l1: for (int j = 0; j < FEATS_OUT; j++){
            #pragma HLS pipeline II=1

            w_b[i][j] = w_b_mat[i*FEATS_OUT + j];
        }
    }
}


// Read weights for layer C
void read_weight_c(TYPE w_c_mat[FEATS_IN*FEATS_OUT],
                TYPE w_c[FEATS_IN][FEATS_OUT]){
    rd_w_c_l0: for (int i = 0; i < FEATS_IN; i++){

        rd_w_c_l1: for (int j = 0; j < FEATS_OUT; j++){
            #pragma HLS pipeline II=1

            w_c[i][j] = w_c_mat[i*FEATS_OUT + j];

        }
    }
}

// Read weights for layer D
void read_weight_d(TYPE w_d_mat[FEATS_IN*FEATS_OUT],
                TYPE w_d[FEATS_IN][FEATS_OUT]){

    rd_w_d_l0: for (int i = 0; i < FEATS_IN; i++){

        rd_w_d_l1: for (int j = 0; j < FEATS_OUT; j++){
            #pragma HLS pipeline II=1

            w_d[i][j] = w_d_mat[i*FEATS_OUT + j];
        }
    }
}

// Read weights for layer E
void read_weight_e(TYPE w_e_mat[FEATS_IN*FEATS_OUT],
                TYPE w_e[FEATS_IN][FEATS_OUT]){
    rd_w_e_l0: for (int i = 0; i < FEATS_IN; i++){

        rd_w_e_l1: for (int j = 0; j < FEATS_OUT; j++){
            #pragma HLS pipeline II=1

            w_e[i][j] = w_e_mat[i*FEATS_OUT + j];
        }
    }
}

// read nod_src from memory
void read_nod_src(node_t nod_src[N_NODES],
                // node_index_t n,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<node_t> nod_src_stream[15]){
    
    rd_nod_src_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        #pragma HLS pipeline II=1

        node_t nod_src_n = nod_src[n];

        rd_nod_src_l0: for (int i = 0; i < 15; i++){
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

// read ft_in for layer A and E from memory (target features)
void read_ft_in_tar(// TYPE ft_in_mat[N_NODES*FEATS_IN],
                    vec_ft_in_t ft_in_tar_mat[N_NODES*FEATS_IN/W_FT_IN],
                    hls::stream<node_t>& nod_src_stream0,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    hls::stream<vec_ft_in_t>& ft_in_ah_stream,
                    hls::stream<vec_ft_in_t>& ft_in_eh_stream)
{

    rd_ft_in_tar_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        rd_ft_in_tar_l0: for (int i = 0; i < FEATS_IN/W_FT_IN; i++){
            #pragma HLS pipeline II=1

            vec_ft_in_t ft_in_tar_temp = ft_in_tar_mat[n*FEATS_IN/W_FT_IN + i];

            ft_in_ah_stream << ft_in_tar_temp;

            if (tmp_begin != tmp_end){

                ft_in_eh_stream << ft_in_tar_temp;

                // for (int j = 0; j < W_FT_IN; j++){

                //     ft_in_eh_stream << ft_in_tar_temp[j];
                // }
            }
        }
    }
}

// execute linear A layer (linear projection: in_feats -> out_feats)
void update_tar_ah( // TYPE ft_in_ah_stream[FEATS_IN],
            // hls::stream<TYPE>& ft_in_ah_stream,
            hls::stream<vec_ft_in_t>& ft_in_ah_stream,
            // hls::stream<TYPE>& ft_in_ah_stream,
            TYPE w_a[FEATS_IN][FEATS_OUT],
            node_index_t nidx_begin,
            node_index_t nidx_end,
            // TYPE rst_ah_stream[FEATS_OUT]
            // hls::stream<TYPE>& rst_ah_stream
            hls::stream<vec_ft_out_full_t>& rst_ah_p1_stream
            )
{

    update_ah_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        // TYPE rst_ah[FEATS_OUT];
        // for (int k = 0; k < FEATS_IN; k++){
        //     // TYPE ft_in_ah_temp = ft_in_ah_stream[k];
        //     // TYPE ft_in_ah_temp = ft_in_ah_stream.read();
        //     TYPE ft_in_ah_temp = ft_in_ah_stream.read();

        //     for ( int j = 0; j < FEATS_OUT; j++){

        //         TYPE rst_ah_temp = (k == 0) ? 0 : rst_ah[j];
        //         // rst_ah_temp += ft_in_ah_temp * w_a_mat[k*FEATS_OUT + j];
        //         rst_ah_temp += ft_in_ah_temp * w_a[k][j];
        //         rst_ah[j] = rst_ah_temp;
        //     }
        // }

        // // write results of linear A layer to stream 
        // for ( int j = 0; j < FEATS_OUT; j++){
        //     // rst_ah_stream[j] = rst_ah[j];
        //     rst_ah_stream << rst_ah[j];
        // }

        // read ft_in_ah from stream to local buffer
        TYPE ft_in_ah_buf[FEATS_IN];
        const int w_ft_in = W_FT_IN;
        #pragma HLS array_partition variable=ft_in_ah_buf cyclic factor=w_ft_in dim=1
        rd_ft_in_ah_stm_l0: for(int i=0; i<FEATS_IN/W_FT_IN; i++){
            #pragma HLS pipeline II=1

            vec_ft_in_t ft_in_ah_vec = ft_in_ah_stream.read();

            rd_ft_in_ah_stm_l1: for (int j = 0; j < W_FT_IN; j++){
                #pragma HLS UNROLL

                ft_in_ah_buf[i*W_FT_IN + j] = ft_in_ah_vec[j];
            }
        }

        // update ft_in_ah
        TYPE rst_ah_tmp[D][FEATS_OUT];
        // #pragma HLS array_partition variable=rst_ah_tmp cyclic factor=2 dim=1
        #pragma HLS array_partition variable=rst_ah_tmp complete dim=2
        update_ah_l0: for(int k=0; k<FEATS_IN/D; k++){

            update_ah_l1: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1

                TYPE ft_in_ah_temp = ft_in_ah_buf[k*D + kd];

                update_ah_l2: for(int j=0; j<FEATS_OUT; j++){
                    // #pragma HLS pipeline II=1 rewind
                    // #pragma HLS UNROLL factor=128
                    #pragma HLS UNROLL
                    
                    TYPE rst_ah_temp = (k == 0) ? 0 : rst_ah_tmp[kd][j];

                    rst_ah_tmp[kd][j] = rst_ah_temp + ft_in_ah_temp * w_a[k*D + kd][j];
                }
            }
        }

        // write the result of ah p1 to stream
        wr_rst_ah_p1_stm_l0: for (int kd = 0; kd < D; kd++){
            #pragma HLS pipeline II=1

            vec_ft_out_full_t rst_ah_vec_temp;

            for(int j=0; j<FEATS_OUT; j++){
                #pragma HLS UNROLL

                rst_ah_vec_temp[j] = rst_ah_tmp[kd][j];
            }

            rst_ah_p1_stream << rst_ah_vec_temp;
        }
    }
}

void update_tar_ah_sum(hls::stream<vec_ft_out_full_t>& rst_ah_p1_stream,
            node_index_t nidx_begin,
            node_index_t nidx_end,
            hls::stream<vec_ft_out_t>& rst_ah_stream
            )
{
    
    update_ah_sum_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        TYPE rst_ah_tmp[D][FEATS_OUT];
        #pragma HLS array_partition variable=rst_ah_tmp complete dim=2
        rd_ah_p1_stm_l0: for (int kd = 0; kd < D; kd++){
            #pragma HLS pipeline II=1

            vec_ft_out_full_t rst_ah_vec_temp = rst_ah_p1_stream.read();

            rd_ah_p1_stm_l1: for(int j=0; j<FEATS_OUT; j++){
                #pragma HLS UNROLL
                rst_ah_tmp[kd][j] = rst_ah_vec_temp[j];
            }
        }

        // sum the D rows of results
        TYPE rst_ah[FEATS_OUT];
        #pragma HLS array_partition variable=rst_ah cyclic factor=8 dim=1
        ah_sum_l0: for (int kd = 0; kd < D; kd++){
            ah_sum_l1: for(int j=0; j<FEATS_OUT; j++){
                #pragma HLS pipeline II=1 rewind
                #pragma HLS UNROLL factor=8

                TYPE rst_ah_temp2 = (kd == 0) ? 0 : rst_ah[j];

                rst_ah[j] = rst_ah_temp2 + rst_ah_tmp[kd][j];
            }
        }

        // write results of ah update to stream
        wr_rst_ah_stm_l0: for (int i = 0; i < FEATS_OUT/W_FT_OUT; i++){
            #pragma HLS pipeline II=1

            vec_ft_out_t rst_ah_temp;

            wr_rst_ah_stm_l1: for (int j = 0; j < W_FT_OUT; j++){
                #pragma HLS UNROLL

                rst_ah_temp[j] = rst_ah[i*W_FT_OUT + j];
            }

            rst_ah_stream << rst_ah_temp;
        }

        // wr_rst_ah_stm_l0: for (int i = 0; i < FEATS_OUT; i++){
        //     #pragma HLS pipeline II=1

        //     rst_ah_stream << rst_ah[i];

        // }
    }
}

// execute linear E layer (linear projection: in_feats -> out_feats)
void update_tar_eh(hls::stream<vec_ft_in_t>& ft_in_eh_stream,
                hls::stream<node_t>& nod_src_stream0,
                TYPE w_e[FEATS_IN][FEATS_OUT],
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<vec_ft_out_full_t>& rst_eh_p1_stream
                ){
    for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if (tmp_begin != tmp_end){

            // read ft_in_eh from stream to local buffer
            TYPE ft_in_eh_buf[FEATS_IN];
            const int w_ft_in = W_FT_IN;
            #pragma HLS array_partition variable=ft_in_eh_buf cyclic factor=w_ft_in dim=1
            rd_ft_in_eh_stm_l0: for(int i=0; i<FEATS_IN/W_FT_IN; i++){
                #pragma HLS pipeline II=1

                vec_ft_in_t ft_in_eh_vec = ft_in_eh_stream.read();

                rd_ft_in_eh_stm_l1: for (int j = 0; j < W_FT_IN; j++){
                    #pragma HLS UNROLL

                    ft_in_eh_buf[i*W_FT_IN + j] = ft_in_eh_vec[j];
                }
            }

            // update ft_in_eh
            TYPE rst_eh_tmp[D][FEATS_OUT];
            // #pragma HLS array_partition variable=rst_eh_tmp cyclic factor=2 dim=1
            #pragma HLS array_partition variable=rst_eh_tmp complete dim=2
            update_eh_l0: for(int k=0; k<FEATS_IN/D; k++){

                update_eh_l1: for (int kd = 0; kd < D; kd++){
                    #pragma HLS pipeline II=1

                    TYPE ft_in_eh_temp = ft_in_eh_buf[k*D + kd];

                    update_eh_l2: for(int j=0; j<FEATS_OUT; j++){
                        #pragma HLS UNROLL
                        
                        TYPE rst_eh_temp = (k == 0) ? 0 : rst_eh_tmp[kd][j];

                        rst_eh_tmp[kd][j] = rst_eh_temp + ft_in_eh_temp * w_e[k*D + kd][j];
                    }
                }
            }

            // write the result of eh p1 to stream
            wr_rst_eh_p1_stm_l0: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1

                vec_ft_out_full_t rst_eh_vec_temp;

                for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS UNROLL

                    rst_eh_vec_temp[j] = rst_eh_tmp[kd][j];
                }

                rst_eh_p1_stream << rst_eh_vec_temp;
            }
        }
    }
}

void update_tar_eh_sum(hls::stream<vec_ft_out_full_t>& rst_eh_p1_stream,
            hls::stream<node_t>& nod_src_stream0,
            node_index_t nidx_begin,
            node_index_t nidx_end,
            hls::stream<vec_ft_out_t>& rst_eh_stream)
{
    
    update_eh_sum_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if (tmp_begin != tmp_end){

            TYPE rst_eh_tmp[D][FEATS_OUT];
            #pragma HLS array_partition variable=rst_eh_tmp complete dim=2
            rd_eh_p1_stm_l0: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1

                vec_ft_out_full_t rst_eh_vec_temp = rst_eh_p1_stream.read();

                rd_eh_p1_stm_l1: for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS UNROLL
                    rst_eh_tmp[kd][j] = rst_eh_vec_temp[j];
                }
            }

            // sum the D rows of results
            TYPE rst_eh[FEATS_OUT];
            #pragma HLS array_partition variable=rst_eh cyclic factor=8 dim=1
            eh_sum_l0: for (int kd = 0; kd < D; kd++){
                eh_sum_l1: for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS pipeline II=1 rewind
                    #pragma HLS UNROLL factor=8

                    TYPE rst_eh_temp2 = (kd == 0) ? 0 : rst_eh[j];

                    rst_eh[j] = rst_eh_temp2 + rst_eh_tmp[kd][j];
                }
            }

            // write results of eh update to stream
            wr_rst_eh_stm_l0: for (int i = 0; i < FEATS_OUT/W_FT_OUT; i++){
                #pragma HLS pipeline II=1

                vec_ft_out_t rst_eh_temp;

                wr_rst_eh_stm_l1: for (int j = 0; j < W_FT_OUT; j++){
                    #pragma HLS UNROLL

                    rst_eh_temp[j] = rst_eh[i*W_FT_OUT + j];
                }

                rst_eh_stream << rst_eh_temp;
            }

            // wr_rst_eh_stm_l0: for (int i = 0; i < FEATS_OUT; i++){
            //     #pragma HLS pipeline II=1

            //     rst_eh_stream << rst_eh[i];

            // }
        }
    }
}

// read aggregated features from memory
void read_feat_in_agg(vec_ft_in_t ft_in_agg_mat[N_NODES*FEATS_IN/W_FT_IN],
                    hls::stream<node_t>& nod_src_stream0,
                    hls::stream<node_index_t>& tmp_src_stream,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    hls::stream<vec_ft_in_t>& ft_in_dh_stream,
                    hls::stream<vec_ft_in_t>& ft_in_bh_stream)
{

    rd_ft_in_agg_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        rd_ft_in_agg_l0: for(edge_index_t e=tmp_begin; e<tmp_end; e++){
            // node_index_t tmp_src = edge_src[e].src;
            node_index_t tmp_src = tmp_src_stream.read();

            // for (int i = 0; i < FEATS_IN; i++){
            //     #pragma HLS pipeline II=1

            //     TYPE ft_in_agg_temp = ft_in_mat[tmp_src*FEATS_IN + i];
            //     // ft_in_dh[i] = ft_in_agg_temp;
            //     // ft_in_bh[i] = ft_in_agg_temp;
            //     ft_in_dh_stream << ft_in_agg_temp;
            //     ft_in_bh_stream << ft_in_agg_temp;
            // }

            rd_ft_in_agg_l1: for (int i = 0; i < FEATS_IN/W_FT_IN; i++){
                #pragma HLS pipeline II=1

                vec_ft_in_t ft_in_agg_temp = ft_in_agg_mat[tmp_src*FEATS_IN/W_FT_IN + i];

                ft_in_dh_stream << ft_in_agg_temp;
                ft_in_bh_stream << ft_in_agg_temp;
            }
        }
    }
}

// execute linear D layer (linear projection: in_feats -> out_feats)
void update_dh(// hls::stream<TYPE>& ft_in_dh_stream,
                hls::stream<vec_ft_in_t>& ft_in_dh_stream,
                hls::stream<node_t>& nod_src_stream0,
                TYPE w_d[FEATS_IN][FEATS_OUT],
                // edge_index_t tmp_begin,
                // edge_index_t tmp_end,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                // hls::stream<TYPE>& rst_dh_stream
                hls::stream<vec_ft_out_full_t>& rst_dh_p1_stream)
{

    // execute linear D layer
    update_dh_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        update_dh_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

            // TYPE rst_dh[FEATS_OUT];
            // for (int k = 0; k < FEATS_IN; k++){
            //     // TYPE ft_in_dh_temp = ft_in_dh[k];
            //     TYPE ft_in_dh_temp = ft_in_dh_stream.read();

            //     for (int j = 0; j < FEATS_OUT; j++){

            //         TYPE rst_dh_temp = (k == 0) ? 0 : rst_dh[j];
            //         // rst_dh_temp += ft_in_dh_temp * w_d_mat[k*FEATS_OUT + j];
            //         rst_dh_temp += ft_in_dh_temp * w_d[k][j];
            //         rst_dh[j] = rst_dh_temp;
            //     }
            // }

            // // write results of linear D layer to stream
            // for ( int j = 0; j < FEATS_OUT; j++){
            //     // rst_dh_stream[j] = rst_dh[j];
            //     rst_dh_stream << rst_dh[j];
            // }

            // read ft_in_dh from stream to local buffer
            TYPE ft_in_dh_buf[FEATS_IN];
            const int w_ft_in = W_FT_IN;
            #pragma HLS array_partition variable=ft_in_dh_buf cyclic factor=w_ft_in dim=1
            rd_ft_in_dh_stm_l0: for(int i=0; i<FEATS_IN/W_FT_IN; i++){
                #pragma HLS pipeline II=1

                vec_ft_in_t ft_in_dh_vec = ft_in_dh_stream.read();

                rd_ft_in_dh_stm_l1: for (int j = 0; j < W_FT_IN; j++){
                    #pragma HLS UNROLL

                    ft_in_dh_buf[i*W_FT_IN + j] = ft_in_dh_vec[j];
                }
            }

            // update ft_in_dh
            TYPE rst_dh_tmp[D][FEATS_OUT];
            // #pragma HLS array_partition variable=rst_dh_tmp cyclic factor=2 dim=1
            #pragma HLS array_partition variable=rst_dh_tmp complete dim=2
            update_dh_l0: for(int k=0; k<FEATS_IN/D; k++){

                update_dh_l1: for (int kd = 0; kd < D; kd++){
                    #pragma HLS pipeline II=1

                    TYPE ft_in_dh_temp = ft_in_dh_buf[k*D + kd];

                    update_dh_l2: for(int j=0; j<FEATS_OUT; j++){
                        // #pragma HLS pipeline II=1 rewind
                        // #pragma HLS UNROLL factor=128
                        #pragma HLS UNROLL
                        
                        TYPE rst_dh_temp = (k == 0) ? 0 : rst_dh_tmp[kd][j];

                        rst_dh_tmp[kd][j] = rst_dh_temp + ft_in_dh_temp * w_d[k*D + kd][j];
                    }
                }
            }

            // write the result of dh p1 to stream
            wr_rst_dh_p1_stm_l0: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1

                vec_ft_out_full_t rst_dh_vec_temp;

                for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS UNROLL

                    rst_dh_vec_temp[j] = rst_dh_tmp[kd][j];
                }

                rst_dh_p1_stream << rst_dh_vec_temp;
            }
        }
    }
}

void update_dh_sum(hls::stream<vec_ft_out_full_t>& rst_dh_p1_stream,
            hls::stream<node_t>& nod_src_stream0,
            node_index_t nidx_begin,
            node_index_t nidx_end,
            hls::stream<vec_ft_out_t>& rst_dh_stream
            // hls::stream<TYPE>& rst_dh_stream
            )
{
    
    update_dh_sum_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        update_dh_sum_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

            TYPE rst_dh_tmp[D][FEATS_OUT];
            #pragma HLS array_partition variable=rst_dh_tmp complete dim=2
            rd_dh_p1_stm_l0: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1

                vec_ft_out_full_t rst_dh_vec_temp = rst_dh_p1_stream.read();

                rd_dh_p1_stm_l1: for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS UNROLL
                    rst_dh_tmp[kd][j] = rst_dh_vec_temp[j];
                }
            }

            // sum the D rows of results
            TYPE rst_dh[FEATS_OUT];
            #pragma HLS array_partition variable=rst_dh cyclic factor=8 dim=1
            dh_sum_l0: for (int kd = 0; kd < D; kd++){
                dh_sum_l1: for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS pipeline II=1 rewind
                    #pragma HLS UNROLL factor=8

                    TYPE rst_dh_temp2 = (kd == 0) ? 0 : rst_dh[j];

                    rst_dh[j] = rst_dh_temp2 + rst_dh_tmp[kd][j];
                }
            }

            // write results of dh update to stream
            wr_rst_dh_stm_l0: for (int i = 0; i < FEATS_OUT/W_FT_OUT; i++){
                #pragma HLS pipeline II=1

                vec_ft_out_t rst_dh_temp;

                wr_rst_dh_stm_l1: for (int j = 0; j < W_FT_OUT; j++){
                    #pragma HLS UNROLL

                    rst_dh_temp[j] = rst_dh[i*W_FT_OUT + j];
                }

                rst_dh_stream << rst_dh_temp;
            }

            // wr_rst_dh_stm_l0: for (int i = 0; i < FEATS_OUT; i++){
            //     #pragma HLS pipeline II=1

            //     rst_dh_stream << rst_dh[i];

            // }
        }
    }
}


// // execute linear B layer (linear projection: in_feats -> out_feats)
// void update_bh(hls::stream<TYPE>& ft_in_bh_stream,
//                 hls::stream<node_t>& nod_src_stream0,
//                 TYPE w_b[FEATS_IN][FEATS_OUT],
//                 // edge_index_t tmp_begin,
//                 // edge_index_t tmp_end,
//                 node_index_t nidx_begin,
//                 node_index_t nidx_end,
//                 hls::stream<TYPE>& rst_bh_stream){
        
//     for(node_index_t n=nidx_begin; n<nidx_end; n++){

//         node_t nod_src_temp = nod_src_stream0.read();
//         edge_index_t tmp_begin = nod_src_temp.edge_begin;
//         edge_index_t tmp_end = nod_src_temp.edge_end;

//         for(edge_index_t e=tmp_begin; e<tmp_end; e++){
//             TYPE rst_bh[FEATS_OUT];
//             // TYPE rst_bh_stream[FEATS_OUT];
//             for (int k = 0; k < FEATS_IN; k++){
//                 // TYPE ft_in_bh_temp = ft_in_bh[k];
//                 TYPE ft_in_bh_temp = ft_in_bh_stream.read();

//                 for (int j = 0; j < FEATS_OUT; j++){

//                     TYPE rst_bh_temp = (k == 0) ? 0 : rst_bh[j];
//                     // rst_bh_temp += ft_in_bh_temp * w_b_mat[k*FEATS_OUT + j];
//                     rst_bh_temp += ft_in_bh_temp * w_b[k][j];
//                     rst_bh[j] = rst_bh_temp;
//                 }
//             }

//             // write results of linear B layer to stream
//             for ( int j = 0; j < FEATS_OUT; j++){
//                 // rst_bh_stream[j] = rst_bh[j];
//                 rst_bh_stream << rst_bh[j];
//             }
//         }
//     }
// }

// execute linear B layer (linear projection: in_feats -> out_feats)
void update_bh(// hls::stream<TYPE>& ft_in_bh_stream,
                hls::stream<vec_ft_in_t>& ft_in_bh_stream,
                hls::stream<node_t>& nod_src_stream0,
                TYPE w_b[FEATS_IN][FEATS_OUT],
                // edge_index_t tmp_begin,
                // edge_index_t tmp_end,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                // hls::stream<TYPE>& rst_bh_stream
                hls::stream<vec_ft_out_full_t>& rst_bh_p1_stream)
{

    // execute linear D layer
    update_bh_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        update_bh_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

            // TYPE rst_bh[FEATS_OUT];
            // for (int k = 0; k < FEATS_IN; k++){
            //     // TYPE ft_in_bh_temp = ft_in_bh[k];
            //     TYPE ft_in_bh_temp = ft_in_bh_stream.read();

            //     for (int j = 0; j < FEATS_OUT; j++){

            //         TYPE rst_bh_temp = (k == 0) ? 0 : rst_bh[j];
            //         // rst_bh_temp += ft_in_bh_temp * w_b_mat[k*FEATS_OUT + j];
            //         rst_bh_temp += ft_in_bh_temp * w_b[k][j];
            //         rst_bh[j] = rst_bh_temp;
            //     }
            // }

            // // write results of linear D layer to stream
            // for ( int j = 0; j < FEATS_OUT; j++){
            //     // rst_bh_stream[j] = rst_bh[j];
            //     rst_bh_stream << rst_bh[j];
            // }

            // read ft_in_bh from stream to local buffer
            TYPE ft_in_bh_buf[FEATS_IN];
            const int w_ft_in = W_FT_IN;
            #pragma HLS array_partition variable=ft_in_bh_buf cyclic factor=w_ft_in dim=1
            rd_ft_in_bh_stm_l0: for(int i=0; i<FEATS_IN/W_FT_IN; i++){
                #pragma HLS pipeline II=1

                vec_ft_in_t ft_in_bh_vec = ft_in_bh_stream.read();

                rd_ft_in_bh_stm_l1: for (int j = 0; j < W_FT_IN; j++){
                    #pragma HLS UNROLL

                    ft_in_bh_buf[i*W_FT_IN + j] = ft_in_bh_vec[j];
                }
            }

            // update ft_in_bh
            TYPE rst_bh_tmp[D][FEATS_OUT];
            // #pragma HLS array_partition variable=rst_bh_tmp cyclic factor=2 dim=1
            #pragma HLS array_partition variable=rst_bh_tmp complete dim=2
            update_bh_l0: for(int k=0; k<FEATS_IN/D; k++){

                update_bh_l1: for (int kd = 0; kd < D; kd++){
                    #pragma HLS pipeline II=1

                    TYPE ft_in_bh_temp = ft_in_bh_buf[k*D + kd];

                    update_bh_l2: for(int j=0; j<FEATS_OUT; j++){
                        // #pragma HLS pipeline II=1 rewind
                        // #pragma HLS UNROLL factor=128
                        #pragma HLS UNROLL
                        
                        TYPE rst_bh_temp = (k == 0) ? 0 : rst_bh_tmp[kd][j];

                        rst_bh_tmp[kd][j] = rst_bh_temp + ft_in_bh_temp * w_b[k*D + kd][j];
                    }
                }
            }

            // write the result of bh p1 to stream
            wr_rst_bh_p1_stm_l0: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1

                vec_ft_out_full_t rst_bh_vec_temp;

                for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS UNROLL

                    rst_bh_vec_temp[j] = rst_bh_tmp[kd][j];
                }

                rst_bh_p1_stream << rst_bh_vec_temp;
            }
        }
    }
}

void update_bh_sum(hls::stream<vec_ft_out_full_t>& rst_bh_p1_stream,
            hls::stream<node_t>& nod_src_stream0,
            node_index_t nidx_begin,
            node_index_t nidx_end,
            hls::stream<vec_ft_out_t>& rst_bh_stream
            // hls::stream<TYPE>& rst_bh_stream
            )
{
    
    update_bh_sum_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        update_bh_sum_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

            TYPE rst_bh_tmp[D][FEATS_OUT];
            #pragma HLS array_partition variable=rst_bh_tmp complete dim=2
            rd_bh_p1_stm_l0: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1

                vec_ft_out_full_t rst_bh_vec_temp = rst_bh_p1_stream.read();

                rd_bh_p1_stm_l1: for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS UNROLL
                    rst_bh_tmp[kd][j] = rst_bh_vec_temp[j];
                }
            }

            // sum the D rows of results
            TYPE rst_bh[FEATS_OUT];
            #pragma HLS array_partition variable=rst_bh cyclic factor=8 dim=1
            bh_sum_l0: for (int kd = 0; kd < D; kd++){
                bh_sum_l1: for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS pipeline II=1 rewind
                    #pragma HLS UNROLL factor=8

                    TYPE rst_bh_temp2 = (kd == 0) ? 0 : rst_bh[j];

                    rst_bh[j] = rst_bh_temp2 + rst_bh_tmp[kd][j];
                }
            }

            // write results of bh update to stream
            wr_rst_bh_stm_l0: for (int i = 0; i < FEATS_OUT/W_FT_OUT; i++){
                #pragma HLS pipeline II=1

                vec_ft_out_t rst_bh_temp;

                wr_rst_bh_stm_l1: for (int j = 0; j < W_FT_OUT; j++){
                    #pragma HLS UNROLL

                    rst_bh_temp[j] = rst_bh[i*W_FT_OUT + j];
                }

                rst_bh_stream << rst_bh_temp;
            }

            // wr_rst_bh_stm_l0: for (int i = 0; i < FEATS_OUT; i++){
            //     #pragma HLS pipeline II=1

            //     rst_bh_stream << rst_bh[i];

            // }
        }
    }
}

// read aggregated edge features from memory
void read_ft_in_edge_agg(edge_index_t edge_idx[N_EDGES],
                        hls::stream<node_t>& nod_src_stream0,
                        vec_ft_in_t ft_in_e_mat[N_EDGES*FEATS_IN/W_FT_IN],
                        // edge_index_t tmp_begin,
                        // edge_index_t tmp_end,
                        node_index_t nidx_begin,
                        node_index_t nidx_end,
                        hls::stream<edge_index_t>& e_idx_stream,
                        hls::stream<vec_ft_in_t>& ft_in_ce_stream){

    rd_ft_in_edge_agg_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        rd_ft_in_edge_agg_l0: for(edge_index_t e=tmp_begin; e<tmp_end; e++){
            edge_index_t e_idx = edge_idx[e];
            // write e_idx to stream
            e_idx_stream << e_idx;

            // write aggregated edge feature to stream
            rd_ft_in_edge_agg_l1: for (int i = 0; i < FEATS_IN/W_FT_IN; i++){
                #pragma HLS pipeline II=1

                // ft_in_ce[i] = ft_in_e_mat[e_idx*FEATS_IN + i];
                ft_in_ce_stream << ft_in_e_mat[e_idx*FEATS_IN/W_FT_IN + i];
            }
        }
    }
}

// execute linear C layer: edge (linear projection: in_feats -> out_feats)
void update_ce(hls::stream<vec_ft_in_t>& ft_in_ce_stream,
                hls::stream<node_t>& nod_src_stream0,
                TYPE w_c[FEATS_IN][FEATS_OUT],
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<vec_ft_out_full_t>& rst_ce_p1_stream)
{

    // execute linear D layer
    update_ce_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        update_ce_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

            // read ft_in_ce from stream to local buffer
            TYPE ft_in_ce_buf[FEATS_IN];
            const int w_ft_in = W_FT_IN;
            #pragma HLS array_partition variable=ft_in_ce_buf cyclic factor=w_ft_in dim=1
            rd_ft_in_ce_stm_l0: for(int i=0; i<FEATS_IN/W_FT_IN; i++){
                #pragma HLS pipeline II=1

                vec_ft_in_t ft_in_ce_vec = ft_in_ce_stream.read();

                rd_ft_in_ce_stm_l1: for (int j = 0; j < W_FT_IN; j++){
                    #pragma HLS UNROLL

                    ft_in_ce_buf[i*W_FT_IN + j] = ft_in_ce_vec[j];
                }
            }

            // update ft_in_ce
            TYPE rst_ce_tmp[D][FEATS_OUT];
            // #pragma HLS array_partition variable=rst_ce_tmp cyclic factor=2 dim=1
            #pragma HLS array_partition variable=rst_ce_tmp complete dim=2
            update_ce_l0: for(int k=0; k<FEATS_IN/D; k++){

                update_ce_l1: for (int kd = 0; kd < D; kd++){
                    #pragma HLS pipeline II=1

                    TYPE ft_in_ce_temp = ft_in_ce_buf[k*D + kd];

                    update_ce_l2: for(int j=0; j<FEATS_OUT; j++){
                        // #pragma HLS pipeline II=1 rewind
                        // #pragma HLS UNROLL factor=128
                        #pragma HLS UNROLL
                        
                        TYPE rst_ce_temp = (k == 0) ? 0 : rst_ce_tmp[kd][j];

                        rst_ce_tmp[kd][j] = rst_ce_temp + ft_in_ce_temp * w_c[k*D + kd][j];
                    }
                }
            }

            // write the result of ce p1 to stream
            wr_rst_ce_p1_stm_l0: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1

                vec_ft_out_full_t rst_ce_vec_temp;

                for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS UNROLL

                    rst_ce_vec_temp[j] = rst_ce_tmp[kd][j];
                }

                rst_ce_p1_stream << rst_ce_vec_temp;
            }
        }
    }
}

void update_ce_sum(hls::stream<vec_ft_out_full_t>& rst_ce_p1_stream,
            hls::stream<node_t>& nod_src_stream0,
            node_index_t nidx_begin,
            node_index_t nidx_end,
            hls::stream<vec_ft_out_t>& rst_ce_stream
            // hls::stream<TYPE>& rst_ce_stream
            )
{
    
    update_ce_sum_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        update_ce_sum_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

            TYPE rst_ce_tmp[D][FEATS_OUT];
            #pragma HLS array_partition variable=rst_ce_tmp complete dim=2
            rd_ce_p1_stm_l0: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1

                vec_ft_out_full_t rst_ce_vec_temp = rst_ce_p1_stream.read();

                rd_ce_p1_stm_l1: for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS UNROLL
                    rst_ce_tmp[kd][j] = rst_ce_vec_temp[j];
                }
            }

            // sum the D rows of results
            TYPE rst_ce[FEATS_OUT];
            #pragma HLS array_partition variable=rst_ce cyclic factor=8 dim=1
            ce_sum_l0: for (int kd = 0; kd < D; kd++){
                ce_sum_l1: for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS pipeline II=1 rewind
                    #pragma HLS UNROLL factor=8

                    TYPE rst_ce_temp2 = (kd == 0) ? 0 : rst_ce[j];

                    rst_ce[j] = rst_ce_temp2 + rst_ce_tmp[kd][j];
                }
            }

            // write results of ce update to stream
            wr_rst_ce_stm_l0: for (int i = 0; i < FEATS_OUT/W_FT_OUT; i++){
                #pragma HLS pipeline II=1

                vec_ft_out_t rst_ce_temp;

                wr_rst_ce_stm_l1: for (int j = 0; j < W_FT_OUT; j++){
                    #pragma HLS UNROLL

                    rst_ce_temp[j] = rst_ce[i*W_FT_OUT + j];
                }

                rst_ce_stream << rst_ce_temp;
            }

            // wr_rst_ce_stm_l0: for (int i = 0; i < FEATS_OUT; i++){
            //     #pragma HLS pipeline II=1

            //     rst_ce_stream << rst_ce[i];

            // }
        }
    }
}

// calculate rst_e, sum_sigma and sum_sigma_h
void comp_rst_e_sum_sigma(hls::stream<node_t>& nod_src_stream0,
                        hls::stream<vec_ft_out_t>& rst_bh_stream,
                        hls::stream<vec_ft_out_t>& rst_dh_stream,
                        hls::stream<vec_ft_out_t>& rst_eh_stream,
                        hls::stream<vec_ft_out_t>& rst_ce_stream,
                        node_index_t nidx_begin,
                        node_index_t nidx_end,
                        hls::stream<vec_ft_out_t>& rst_e_stream,
                        hls::stream<vec_ft_out_t>& rst_sum_sigma_stream,
                        hls::stream<vec_ft_out_t>& rst_sum_sigma_h_stream
                        )
{

    comp_rst_e_sum_sigma_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        // read rst_eh from stream
        if (tmp_begin != tmp_end){

            vec_ft_out_t rst_eh[FEATS_OUT/W_FT_OUT];

            for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
                #pragma HLS pipeline II=1

                // rst_eh[i] = rst_eh_stream[i];
                rst_eh[i] = rst_eh_stream.read();
            }
        
            vec_ft_out_t rst_sum_sigma[FEATS_OUT/W_FT_OUT]; // aggregated sigma
            vec_ft_out_t rst_sum_sigma_h[FEATS_OUT/W_FT_OUT]; // aggregated weighted sigma
            comp_rst_e_sum_sigma_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

                comp_rst_e_sum_sigma_l0: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
                    #pragma HLS pipeline II=1

                    // compute rst_e
                    vec_ft_out_t rst_e_vec = rst_dh_stream.read() + rst_eh[i] + rst_ce_stream.read();

                    // write rst_e to stream
                    rst_e_stream << rst_e_vec;

                    // compute ft_sigma
                    vec_ft_out_t ft_sigma_vec;
                    comp_ft_sigma_l1: for (int j = 0; j < W_FT_OUT; j++){
                        #pragma HLS UNROLL

                        ft_sigma_vec[j] = sigmoid(rst_e_vec[j]);
                    }

                    vec_ft_out_t ft_sigma_vec_temp1 = ft_sigma_vec;
                    vec_ft_out_t ft_sigma_vec_temp2 = ft_sigma_vec;

                    // compute aggregate ft_sigma
                    vec_ft_out_t sum_sigma_vec = (e == tmp_begin) ? 0 : rst_sum_sigma[i];

                    rst_sum_sigma[i] = sum_sigma_vec + ft_sigma_vec_temp1;

                    // weighted ft_sigma (ft_bh * ft_sigma)
                    vec_ft_out_t sum_sigma_h_vec = (e == tmp_begin) ? 0 : rst_sum_sigma_h[i];

                    vec_ft_out_t sum_sigma_h_mul = ft_sigma_vec_temp2 * rst_bh_stream.read();

                    rst_sum_sigma_h[i] = sum_sigma_h_vec + sum_sigma_h_mul;
                } 
            }

            // write rst_sum_sigma & rst_sum_sigma_h to stream
            wr_rst_sum_sigma_h_stm:for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
                #pragma HLS pipeline II=1

                // write rst_sum_sigma to stream
                rst_sum_sigma_stream << rst_sum_sigma[i];

                // write rst_sum_sigma_h to stream
                rst_sum_sigma_h_stream << rst_sum_sigma_h[i];
            }
        }
    }
}

/*
// calculate edge weights: rst_e_mat and ft_sigma
void comp_rst_e_sigma(// hls::stream<TYPE>& rst_dh_stream,
                    hls::stream<vec_ft_out_t>& rst_dh_stream,
                    hls::stream<node_t>& nod_src_stream0,
                    // TYPE rst_eh_stream[FEATS_OUT],
                    // hls::stream<TYPE>& rst_eh_stream,
                    hls::stream<vec_ft_out_t>& rst_eh_stream,
                    // hls::stream<TYPE>& rst_ce_stream,
                    hls::stream<vec_ft_out_t>& rst_ce_stream,
                    // edge_index_t tmp_begin,
                    // edge_index_t tmp_end,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    // hls::stream<TYPE>& rst_e_stream,
                    hls::stream<vec_ft_out_t>& rst_e_stream,
                    // hls::stream<TYPE>& ft_sigma_stream
                    // hls::stream<TYPE>& ft_sigma_stream1,
                    // hls::stream<TYPE>& ft_sigma_stream2
                    hls::stream<vec_ft_out_t>& ft_sigma_stream1,
                    hls::stream<vec_ft_out_t>& ft_sigma_stream2
                    )
{

    comp_rst_e_sigma_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        // read rst_eh from stream
        if (tmp_begin != tmp_end){
            // TYPE rst_eh[FEATS_OUT];
            // for(int i=0; i<FEATS_OUT; i++){

            //     // rst_eh[i] = rst_eh_stream[i];
            //     rst_eh[i] = rst_eh_stream.read();
            // }

            // TYPE rst_eh[FEATS_OUT];
            vec_ft_out_t rst_eh[FEATS_OUT/W_FT_OUT];
            for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
                #pragma HLS pipeline II=1

                // rst_eh[i] = rst_eh_stream[i];
                rst_eh[i] = rst_eh_stream.read();
            }

            // compute st_e_mat and ft_sigma
            comp_rst_e_sigma_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

                comp_rst_e_sigma_l0: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
                    #pragma HLS pipeline II=1

                    vec_ft_out_t rst_e_vec = rst_dh_stream.read() + rst_eh[i] + rst_ce_stream.read();

                    // compute rst_e
                    rst_e_stream << rst_e_vec;

                    // compute ft_sigma
                    vec_ft_out_t rst_e_sigma_vec;
                    comp_rst_e_sigma_l1: for (int j = 0; j < W_FT_OUT; j++){
                        #pragma HLS UNROLL

                        rst_e_sigma_vec[j] = sigmoid(rst_e_vec[j]);
                    }
                    
                    // TYPE rst_e_sigma_temp = sigmoid(rst_e_temp);
                    // ft_sigma_stream1 << rst_e_sigma_temp;
                    // ft_sigma_stream2 << rst_e_sigma_temp;
                    ft_sigma_stream1 << rst_e_sigma_vec;
                    ft_sigma_stream2 << rst_e_sigma_vec;
                }
            }
        }
    }
}
*/

// write rst_e to memory
void write_rst_e_mem(hls::stream<edge_index_t>& e_idx_stream,
                    hls::stream<node_t>& nod_src_stream0,
                    // hls::stream<TYPE>& rst_e_stream,
                    hls::stream<vec_ft_out_t>& rst_e_stream,
                    // edge_index_t tmp_begin,
                    // edge_index_t tmp_end,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    // TYPE rst_e_mat[N_EDGES*FEATS_OUT]
                    vec_ft_out_t rst_e_mat[N_EDGES*FEATS_OUT/W_FT_OUT]
                    ){

    wr_rst_e_mem_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        wr_rst_e_mem_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

            edge_index_t e_idx_temp = e_idx_stream.read();

            // for(int i=0; i<FEATS_OUT; i++){
            //     #ifdef ACTIVATION
            //         // rst_e_mat[e_idx*FEATS_OUT + i] = relu(rst_e_stream[i]);
            //         // rst_e_mat[e_idx_temp*FEATS_OUT + i] = relu(rst_e_stream[i]);
            //         rst_e_mat[e_idx_temp*FEATS_OUT + i] = relu(rst_e_stream.read());
            //     #else
            //         // rst_e_mat[e_idx*FEATS_OUT + i] = rst_e_stream[i];
            //         // rst_e_mat[e_idx_temp*FEATS_OUT + i] = rst_e_stream[i];
            //         rst_e_mat[e_idx_temp*FEATS_OUT + i] = rst_e_stream.read();
            //     #endif
            // }

            wr_rst_e_mem_l0: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
                #pragma HLS pipeline II=1

                vec_ft_out_t rst_e_vec = rst_e_stream.read();

                vec_ft_out_t rst_e_act_vec;
                for (int j = 0; j < W_FT_OUT; j++){
                    #pragma HLS UNROLL

                    #ifdef ACTIVATION

                        rst_e_act_vec[j] = relu(rst_e_vec[j]); 
                    #else

                        rst_e_act_vec[j] = rst_e_vec[j];
                    #endif
                }

                rst_e_mat[e_idx_temp*FEATS_OUT/W_FT_OUT + i] = rst_e_act_vec;
            }
        }
    }
}
/*
// aggregate ft_sigma
void agg_ft_sigma(hls::stream<vec_ft_out_t>& ft_sigma_stream1,
                hls::stream<node_t>& nod_src_stream0,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<vec_ft_out_t>& rst_sum_sigma_stream)
{

    agg_ft_sigma_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if (tmp_begin != tmp_end){

            vec_ft_out_t rst_sum_sigma[FEATS_OUT/W_FT_OUT];
            agg_ft_sigma_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

                
                agg_ft_sigma_l0: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
                    #pragma HLS pipeline II=1
                    

                    vec_ft_out_t sum_sigma_vec = (e == tmp_begin) ? 0 : rst_sum_sigma[i];

                    rst_sum_sigma[i] = sum_sigma_vec + ft_sigma_stream1.read();

                }
                
                // vec_ft_out_t ft_sigma_vec_buf[FEATS_OUT/W_FT_OUT];
                // rd_ft_sigma_stm: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
                //     #pragma HLS pipeline II=1

                //     ft_sigma_vec_buf[i] = ft_sigma_stream1.read();
                // }

                // agg_ft_sigma_l0: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
                //     #pragma HLS pipeline II=1

                //     vec_ft_out_t sum_sigma_vec = (e == tmp_begin) ? 0 : rst_sum_sigma[i];
                //     rst_sum_sigma[i] = sum_sigma_vec + ft_sigma_vec_buf[i];
                // }
            }

            // // write rst_sum_sigma to stream
            // for(int i=0; i<FEATS_OUT; i++){
            //     // rst_sum_sigma_stream[i] = rst_sum_sigma[i];
            //     rst_sum_sigma_stream << rst_sum_sigma[i];
            // }

            // write rst_sum_sigma to stream
            wr_rst_sum_sigma_stm: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
                #pragma HLS pipeline II=1

                rst_sum_sigma_stream << rst_sum_sigma[i];
            }
        }
    }
}

// aggregate weighted ft_sigma (ft_bh * ft_sigma)
void agg_ft_sigma_bh(hls::stream<vec_ft_out_t>& ft_sigma_stream2,
                    hls::stream<vec_ft_out_t>& rst_bh_stream,
                    hls::stream<node_t>& nod_src_stream0,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    hls::stream<vec_ft_out_t>& rst_sum_sigma_h_stream)
{
    agg_ft_sigma_bh_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if (tmp_begin != tmp_end){

            // TYPE rst_sum_sigma_h[FEATS_OUT];
            // for(edge_index_t e=tmp_begin; e<tmp_end; e++){

            //     for(int i=0; i<FEATS_OUT; i++){

            //         TYPE sum_sigma_h_temp = (e == tmp_begin) ? 0 : rst_sum_sigma_h[i];
            //         // sum_sigma_h_temp += ft_sigma_stream[i] * ft_bh_mat[tmp_src*FEATS_OUT + i];
            //         // sum_sigma_h_temp += ft_sigma_stream2[i] * rst_bh_stream[i];
            //         // sum_sigma_h_temp += ft_sigma_stream2[i] * rst_bh_stream.read();
            //         sum_sigma_h_temp += ft_sigma_stream2.read() * rst_bh_stream.read();
            //         rst_sum_sigma_h[i] = sum_sigma_h_temp;
            //     }
            // }

            vec_ft_out_t rst_sum_sigma_h[FEATS_OUT/W_FT_OUT];
            agg_ft_sigma_bh_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

                agg_ft_sigma_bh_l0: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
                    #pragma HLS pipeline II=1

                    vec_ft_out_t sum_sigma_h_vec = (e == tmp_begin) ? 0 : rst_sum_sigma_h[i];

                    vec_ft_out_t sum_sigma_h_mul = ft_sigma_stream2.read() * rst_bh_stream.read();

                    rst_sum_sigma_h[i] = sum_sigma_h_vec + sum_sigma_h_mul;
                }
            }

            // write rst_sum_sigma_h to stream
            wr_rst_sum_sigma_h_stm: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
                #pragma HLS pipeline II=1
                
                rst_sum_sigma_h_stream << rst_sum_sigma_h[i];
            }
        }
    }
}
*/

// compute results and write them to stream (rst_h = rst_ah + (rst_sum_sigma_h / (rst_sum_sigma + EPS)))
void comp_rst(// TYPE rst_ah_stream[FEATS_OUT],
            //  hls::stream<TYPE>& rst_ah_stream,
             hls::stream<vec_ft_out_t>& rst_ah_stream,
             hls::stream<node_t>& nod_src_stream0,
            // TYPE rst_sum_sigma_h_stream[FEATS_OUT],
            // TYPE rst_sum_sigma_stream[FEATS_OUT],
            // hls::stream<TYPE>& rst_sum_sigma_h_stream,
            hls::stream<vec_ft_out_t>& rst_sum_sigma_h_stream,
            // hls::stream<TYPE>& rst_sum_sigma_stream,
            hls::stream<vec_ft_out_t>& rst_sum_sigma_stream,
            // edge_index_t tmp_begin,
            // edge_index_t tmp_end,
            node_index_t nidx_begin,
            node_index_t nidx_end,
            // TYPE rst_h_stream[FEATS_OUT]
            // hls::stream<TYPE>& rst_h_stream
            hls::stream<vec_ft_out_t>& rst_h_stream){

    comp_rst_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        comp_rst_l0: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
            #pragma HLS pipeline II=1

            if (tmp_begin != tmp_end){

                rst_h_stream << rst_ah_stream.read() + (rst_sum_sigma_h_stream.read() / (rst_sum_sigma_stream.read() + EPS));
            }
            else{

                rst_h_stream << rst_ah_stream.read();
            }
        }
    }
}

// write results to memory
void write_rst_h_mem(hls::stream<vec_ft_out_t>& rst_h_stream,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    vec_ft_out_t rst_h_mat[N_NODES*FEATS_OUT/W_FT_OUT]){


    wr_rst_h_mem_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        wr_rst_h_mem_l0: for(int i=0; i<FEATS_OUT/W_FT_OUT; i++){
            #pragma HLS pipeline II=1

            vec_ft_out_t rst_h_vec = rst_h_stream.read();

            vec_ft_out_t rst_h_act_vec;

            wr_rst_h_mem_l1: for (int j = 0; j < W_FT_OUT; j++){
                #pragma HLS UNROLL

                #ifdef ACTIVATION

                    rst_h_act_vec[j] = relu(rst_h_vec[j]);
                #else
                    rst_h_act_vec[j] = rst_h_vec[j];
                #endif
            }

            rst_h_mat[n*FEATS_OUT/W_FT_OUT + i] = rst_h_act_vec;
        }
    }
}

// Compute results for one node
void compute_one_node(node_t nod_src[N_NODES],
                    edge_src_t edge_src[N_EDGES],
                    edge_index_t edge_idx[N_EDGES],
                    vec_ft_in_t ft_in_agg_mat[N_NODES*FEATS_IN/W_FT_IN],
                    vec_ft_in_t ft_in_tar_mat[N_NODES*FEATS_IN/W_FT_IN],
                    vec_ft_in_t ft_in_e_mat[N_EDGES*FEATS_IN/W_FT_IN],
                    TYPE w_a[FEATS_IN][FEATS_OUT],
                    TYPE w_b[FEATS_IN][FEATS_OUT],
                    TYPE w_c[FEATS_IN][FEATS_OUT],
                    TYPE w_d[FEATS_IN][FEATS_OUT],
                    TYPE w_e[FEATS_IN][FEATS_OUT],
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    vec_ft_out_t rst_e_mat[N_EDGES*FEATS_OUT/W_FT_OUT],
                    vec_ft_out_t rst_h_mat[N_NODES*FEATS_OUT/W_FT_OUT])
{
    
    const int feats_in = FEATS_IN;
    const int feats_out = FEATS_OUT;

    #pragma HLS dataflow

    // read nod_src from memory
    hls::stream<node_t> nod_src_stream[15];
    #pragma HLS stream variable=nod_src_stream[0] depth=10
    #pragma HLS stream variable=nod_src_stream[1] depth=11
    #pragma HLS stream variable=nod_src_stream[2] depth=14
    #pragma HLS stream variable=nod_src_stream[3] depth=15
    #pragma HLS stream variable=nod_src_stream[4] depth=16
    #pragma HLS stream variable=nod_src_stream[5] depth=17
    #pragma HLS stream variable=nod_src_stream[6] depth=18
    #pragma HLS stream variable=nod_src_stream[7] depth=19
    #pragma HLS stream variable=nod_src_stream[8] depth=20
    #pragma HLS stream variable=nod_src_stream[9] depth=21
    #pragma HLS stream variable=nod_src_stream[10] depth=22
    #pragma HLS stream variable=nod_src_stream[11] depth=23
    #pragma HLS stream variable=nod_src_stream[12] depth=24
    #pragma HLS stream variable=nod_src_stream[13] depth=25
    #pragma HLS stream variable=nod_src_stream[14] depth=26
    read_nod_src(nod_src, nidx_begin, nidx_end, nod_src_stream);

    // read edge_src from memory
    hls::stream<node_index_t> tmp_src_stream;
    #pragma HLS stream variable=tmp_src_stream depth=15
    read_edge_src(edge_src, nod_src_stream[0], nidx_begin, nidx_end, tmp_src_stream);

    // read ft_in for layer A and E from memory (target features)
    hls::stream<vec_ft_in_t> ft_in_ah_stream;
    #pragma HLS stream variable=ft_in_ah_stream depth=20
    hls::stream<vec_ft_in_t> ft_in_eh_stream;
    #pragma HLS stream variable=ft_in_eh_stream depth=26
    read_ft_in_tar(ft_in_tar_mat, nod_src_stream[1], nidx_begin, nidx_end, ft_in_ah_stream, ft_in_eh_stream);

    // execute linear A layer (linear projection: in_feats -> out_feats)
    // TYPE rst_ah_stream[FEATS_OUT];
    hls::stream<vec_ft_out_full_t> rst_ah_p1_stream;
    #pragma HLS stream variable=rst_ah_p1_stream depth=10
    // update_tar_ah(ft_in_ah_stream, w_a, rst_ah_stream);
    update_tar_ah(ft_in_ah_stream, w_a, nidx_begin, nidx_end, rst_ah_p1_stream);

    hls::stream<vec_ft_out_t> rst_ah_stream;
    #pragma HLS stream variable=rst_ah_stream depth=48
    update_tar_ah_sum(rst_ah_p1_stream, nidx_begin, nidx_end, rst_ah_stream);
    
    // execute linear E layer (linear projection: in_feats -> out_feats)
    // TYPE rst_eh_stream[FEATS_OUT];
    hls::stream<vec_ft_out_full_t> rst_eh_p1_stream;
    #pragma HLS stream variable=rst_eh_p1_stream depth=10
    update_tar_eh(ft_in_eh_stream, nod_src_stream[2], w_e, nidx_begin, nidx_end, rst_eh_p1_stream);

    hls::stream<vec_ft_out_t> rst_eh_stream;
    #pragma HLS stream variable=rst_eh_stream depth=36
    update_tar_eh_sum(rst_eh_p1_stream, nod_src_stream[3], nidx_begin, nidx_end, rst_eh_stream);


    // read aggregated features from memory
    hls::stream<vec_ft_in_t> ft_in_dh_stream;
    #pragma HLS stream variable=ft_in_dh_stream depth=20
    hls::stream<vec_ft_in_t> ft_in_bh_stream;
    #pragma HLS stream variable=ft_in_bh_stream depth=24
    // read_feat_in_agg(edge_src, ft_in_agg_mat, tmp_begin, tmp_end, ft_in_dh_stream, ft_in_bh_stream);
    read_feat_in_agg(ft_in_agg_mat, nod_src_stream[4], tmp_src_stream, nidx_begin, nidx_end, ft_in_dh_stream, ft_in_bh_stream);

    // execute linear D layer (linear projection: in_feats -> out_feats)
    hls::stream<vec_ft_out_full_t> rst_dh_p1_stream;
    #pragma HLS stream variable=rst_dh_p1_stream depth=10
    update_dh(ft_in_dh_stream, nod_src_stream[5], w_d, nidx_begin, nidx_end, rst_dh_p1_stream);

    hls::stream<vec_ft_out_t> rst_dh_stream;
    #pragma HLS stream variable=rst_dh_stream depth=30
    update_dh_sum(rst_dh_p1_stream, nod_src_stream[6], nidx_begin, nidx_end, rst_dh_stream);

    // execute linear B layer (linear projection: in_feats -> out_feats)
    hls::stream<vec_ft_out_full_t> rst_bh_p1_stream;
    #pragma HLS stream variable=rst_bh_p1_stream depth=5
    update_bh(ft_in_bh_stream, nod_src_stream[7], w_b, nidx_begin, nidx_end, rst_bh_p1_stream);

    hls::stream<vec_ft_out_t> rst_bh_stream;
    #pragma HLS stream variable=rst_bh_stream depth=32
    update_bh_sum(rst_bh_p1_stream, nod_src_stream[8], nidx_begin, nidx_end, rst_bh_stream);

    // read aggregated edge features from memory
    hls::stream<edge_index_t> e_idx_stream;
    #pragma HLS stream variable=e_idx_stream depth=13
    hls::stream<vec_ft_in_t> ft_in_ce_stream;
    #pragma HLS stream variable=ft_in_ce_stream depth=20
    read_ft_in_edge_agg(edge_idx, nod_src_stream[9], ft_in_e_mat, nidx_begin, nidx_end, e_idx_stream, ft_in_ce_stream);

    // execute linear C layer: edge (linear projection: in_feats -> out_feats)
    hls::stream<vec_ft_out_full_t> rst_ce_p1_stream;
    #pragma HLS stream variable=rst_ce_p1_stream depth=10
    update_ce(ft_in_ce_stream, nod_src_stream[10], w_c, nidx_begin, nidx_end, rst_ce_p1_stream);

    hls::stream<vec_ft_out_t> rst_ce_stream;
    #pragma HLS stream variable=rst_ce_stream depth=20
    update_ce_sum(rst_ce_p1_stream, nod_src_stream[11], nidx_begin, nidx_end, rst_ce_stream);

    // // calculate edge weights: rst_e_mat and ft_sigma
    // hls::stream<vec_ft_out_t> rst_e_stream;
    // #pragma HLS stream variable=rst_e_stream depth=10
    // hls::stream<vec_ft_out_t> ft_sigma_stream1;
    // #pragma HLS stream variable=ft_sigma_stream1 depth=22
    // hls::stream<vec_ft_out_t> ft_sigma_stream2;
    // #pragma HLS stream variable=ft_sigma_stream2 depth=24
    // comp_rst_e_sigma(rst_dh_stream, nod_src_stream[12], rst_eh_stream, rst_ce_stream, nidx_begin, nidx_end, rst_e_stream, ft_sigma_stream1, ft_sigma_stream2);

    // calculate rst_e, sum_sigma and sum_sigma_h
    hls::stream<vec_ft_out_t> rst_e_stream;
    #pragma HLS stream variable=rst_e_stream depth=20
    hls::stream<vec_ft_out_t> rst_sum_sigma_stream;
    #pragma HLS stream variable=rst_sum_sigma_stream depth=21
    hls::stream<vec_ft_out_t> rst_sum_sigma_h_stream;
    #pragma HLS stream variable=rst_sum_sigma_h_stream depth=21
    comp_rst_e_sum_sigma(nod_src_stream[12], rst_bh_stream, rst_dh_stream, rst_eh_stream, rst_ce_stream, nidx_begin, nidx_end, rst_e_stream, rst_sum_sigma_stream, rst_sum_sigma_h_stream);

    // write rst_e to memory
    write_rst_e_mem(e_idx_stream, nod_src_stream[13], rst_e_stream,  nidx_begin, nidx_end, rst_e_mat);

    // // aggregate ft_sigma
    // hls::stream<vec_ft_out_t> rst_sum_sigma_stream;
    // #pragma HLS stream variable=rst_sum_sigma_stream depth=22
    // agg_ft_sigma(ft_sigma_stream1, nod_src_stream[14],  nidx_begin, nidx_end, rst_sum_sigma_stream);

    // // aggregate weighted ft_sigma (ft_bh * ft_sigma)
    // hls::stream<vec_ft_out_t> rst_sum_sigma_h_stream;
    // #pragma HLS stream variable=rst_sum_sigma_h_stream depth=20
    // agg_ft_sigma_bh(ft_sigma_stream2, rst_bh_stream, nod_src_stream[15], nidx_begin, nidx_end, rst_sum_sigma_h_stream);

    // compute results and write them to stream (rst_h = rst_ah + (rst_sum_sigma_h / (rst_sum_sigma + EPS)))
    // TYPE rst_h_stream[FEATS_OUT];
    hls::stream<vec_ft_out_t> rst_h_stream;
    #pragma HLS stream variable=rst_h_stream depth=20
    comp_rst(rst_ah_stream, nod_src_stream[14], rst_sum_sigma_h_stream, rst_sum_sigma_stream, nidx_begin, nidx_end, rst_h_stream);

    // write results to memory
    write_rst_h_mem(rst_h_stream, nidx_begin, nidx_end, rst_h_mat);
}

// Compute results for all nodes
void compute_all_node(node_t nod_src[N_NODES],
                    edge_src_t edge_src[N_EDGES],
                    edge_index_t edge_idx[N_EDGES],
                    // TYPE ft_in_agg_mat[N_NODES*FEATS_IN],
                    vec_ft_in_t ft_in_agg_mat[N_NODES*FEATS_IN/W_FT_IN],
                    vec_ft_in_t ft_in_tar_mat[N_NODES*FEATS_IN/W_FT_IN],
                    vec_ft_in_t ft_in_e_mat[N_EDGES*FEATS_IN/W_FT_IN],
                    TYPE w_a[FEATS_IN][FEATS_OUT],
                    TYPE w_b[FEATS_IN][FEATS_OUT],
                    TYPE w_c[FEATS_IN][FEATS_OUT],
                    TYPE w_d[FEATS_IN][FEATS_OUT],
                    TYPE w_e[FEATS_IN][FEATS_OUT],
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    // TYPE rst_e_mat[N_EDGES*FEATS_OUT],
                    vec_ft_out_t rst_e_mat[N_EDGES*FEATS_OUT/W_FT_OUT],
                    // TYPE rst_h_mat[N_NODES*FEATS_OUT]
                    vec_ft_out_t rst_h_mat[N_NODES*FEATS_OUT/W_FT_OUT]
                    ){

    // calculate edge weights: rst_e_mat (not using DEh)
    // traverse all nodes stored in CSR format
    // n is the idx of target node,
    // i is the idx of the feature element
    // tmp_src is the idx of input neighbors
    compute_one_node(nod_src, edge_src, edge_idx, ft_in_agg_mat, ft_in_tar_mat, ft_in_e_mat, w_a, w_b, w_c, w_d, w_e, nidx_begin, nidx_end, rst_e_mat, rst_h_mat);

}

int gatedgcn_hls(node_t nod_src[N_NODES],
            edge_src_t edge_src[N_EDGES],
            edge_index_t edge_idx[N_EDGES],
            // TYPE ft_in_mat[N_NODES*FEATS_IN],
            // TYPE ft_in_agg_mat[N_NODES*FEATS_IN],
            vec_ft_in_t ft_in_agg_mat[N_NODES*FEATS_IN/W_FT_IN],
            vec_ft_in_t ft_in_tar_mat[N_NODES*FEATS_IN/W_FT_IN],
            // TYPE ft_in_e_mat[N_EDGES*FEATS_IN],
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
            vec_ft_out_t rst_h_mat[N_NODES*FEATS_OUT/W_FT_OUT]){

    #pragma HLS INTERFACE m_axi port=nod_src bundle=aximm1
    #pragma HLS INTERFACE m_axi port=edge_src bundle=aximm2
    #pragma HLS INTERFACE m_axi port=edge_idx bundle=aximm3
   
    #pragma HLS INTERFACE m_axi port=ft_in_agg_mat bundle=aximm4
    #pragma HLS INTERFACE m_axi port=ft_in_tar_mat bundle=aximm5
    #pragma HLS INTERFACE m_axi port=ft_in_e_mat bundle=aximm6

    #pragma HLS INTERFACE m_axi port=w_a_mat bundle=aximm7
    #pragma HLS INTERFACE m_axi port=w_b_mat bundle=aximm7
    #pragma HLS INTERFACE m_axi port=w_c_mat bundle=aximm7
    #pragma HLS INTERFACE m_axi port=w_d_mat bundle=aximm7
    #pragma HLS INTERFACE m_axi port=w_e_mat bundle=aximm7

    #pragma HLS INTERFACE s_axilite port=nidx_begin
    #pragma HLS INTERFACE s_axilite port=nidx_end

    #pragma HLS INTERFACE m_axi port=rst_e_mat bundle=aximm8
    #pragma HLS INTERFACE m_axi port=rst_h_mat bundle=aximm9
    
    // node_index_t n;
    // edge_index_t ie;

    // Read weights for layer A
    TYPE w_a[FEATS_IN][FEATS_OUT];
    #pragma HLS array_partition variable=w_a complete dim=2
    read_weight_a(w_a_mat, w_a);

    // Read weights for layer B
    TYPE w_b[FEATS_IN][FEATS_OUT];
    #pragma HLS array_partition variable=w_b complete dim=2
    read_weight_b(w_b_mat, w_b);

    // Read weights for layer C
    TYPE w_c[FEATS_IN][FEATS_OUT];
    #pragma HLS array_partition variable=w_c complete dim=2
    read_weight_c(w_c_mat, w_c);

    // Read weights for layer D
    TYPE w_d[FEATS_IN][FEATS_OUT];
    #pragma HLS array_partition variable=w_d complete dim=2
    read_weight_d(w_d_mat, w_d);

    // Read weights for layer E
    TYPE w_e[FEATS_IN][FEATS_OUT];
    #pragma HLS array_partition variable=w_e complete dim=2
    read_weight_e(w_e_mat, w_e);

    // Compute results for all nodes
    compute_all_node(nod_src, edge_src, edge_idx, ft_in_agg_mat, ft_in_tar_mat, ft_in_e_mat, w_a, w_b, w_c, w_d, w_e, nidx_begin, nidx_end, rst_e_mat, rst_h_mat);

    return 0;
}