/*
HLS kernel of GAT (graph converlution network) layer

Reference
[1] Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional 
    networks." arXiv preprint arXiv:1609.02907 (2016).
[2] Dwivedi, Vijay Prakash, et al. "Benchmarking graph neural networks." arXiv preprint arXiv:2003.
    00982 (2020).
*/

#include "gat.h"

// read ft_fc_mat to local array
void read_weight_fc(TYPE w_fc_mat[FEATS_IN*HEADS_NUM*FEATS_OUT],
                    TYPE w_fc[FEATS_IN][HEADS_NUM][FEATS_OUT]){

    rd_w_fc: for(int i_fc = 0; i_fc < FEATS_IN; i_fc++){

        for(int j_fc = 0; j_fc < HEADS_NUM; j_fc++){

            for(int k_fc = 0; k_fc < FEATS_OUT; k_fc++){
                #pragma HLS pipeline II=1
                
                w_fc[i_fc][j_fc][k_fc] = w_fc_mat[i_fc*HEADS_NUM*FEATS_OUT + j_fc*FEATS_OUT + k_fc];
            }
        }
    }
}

// read attn_l_mat to local array
void read_attn_l(TYPE attn_l_mat[HEADS_NUM*FEATS_OUT],
                TYPE attn_l[HEADS_NUM][FEATS_OUT]){
    
    rd_attn_l: for(int j_att = 0; j_att < FEATS_OUT; j_att++){
        for(int i_att = 0; i_att < HEADS_NUM; i_att++){
            #pragma HLS pipeline II=1

            attn_l[i_att][j_att] = attn_l_mat[i_att*FEATS_OUT + j_att];
        }
    }
}

// read attn_r_mat to local array
void read_attn_r(TYPE attn_r_mat[HEADS_NUM*FEATS_OUT],
                TYPE attn_r[HEADS_NUM][FEATS_OUT]){

    rd_attn_r: for(int j_att = 0; j_att < FEATS_OUT; j_att++){
        for(int i_att = 0; i_att < HEADS_NUM; i_att++){
            #pragma HLS pipeline II=1

            attn_r[i_att][j_att] = attn_r_mat[i_att*FEATS_OUT + j_att];
        }
    }
}

// read input feature from memory
void read_feat_in(// TYPE ft_in_mat[N_NODES*FEATS_IN],
                vec_ft_in_t ft_in_mat[N_NODES*FEATS_IN/W_FT_IN],
                // node_index_t n,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<vec_ft_in_t>& ft_in_tar_stream){
    rd_ft_in_mem_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        rd_ft_in: for(int i = 0; i < FEATS_IN/W_FT_IN; i++){
            #pragma HLS pipeline II=1

            ft_in_tar_stream << ft_in_mat[n*FEATS_IN/W_FT_IN + i];
        }
    }
}

// compute fc layer (multiplication)
// and write the calculated fc feature (ft_fc_array) to a stream channel (ft_fc_chan)
void update_tar_fc(// TYPE ft_in_temp_array[FEATS_IN],
                hls::stream<vec_ft_in_t>& ft_in_tar_stream,
                TYPE w_fc[FEATS_IN][HEADS_NUM][FEATS_OUT],
                node_index_t nidx_begin,
                node_index_t nidx_end,
                // TYPE ft_fc_chan[HEADS_NUM][FEATS_OUT]
                hls::stream<vec_head_ft_out_full_t>& rst_tar_p1_stream){

    // TYPE ft_fc_array[HEADS_NUM][FEATS_OUT]; // RAM, not stream channel
    // #pragma HLS ARRAY_PARTITION variable=ft_fc_array complete dim=0

    update_tar_fc_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        // read ft_in_tar from stream to local buffer
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
        TYPE rst_tar[D][HEADS_NUM][FEATS_OUT];
        #pragma HLS array_partition variable=rst_tar cyclic factor=2 dim=1
        #pragma HLS array_partition variable=rst_tar complete dim=2
        update_tar_l0: for(int k=0; k<FEATS_IN/D; k++){

            update_tar_l1: for (int kd = 0; kd < D; kd++){
                #pragma HLS pipeline II=1 rewind
                #pragma HLS UNROLL factor=2

                TYPE ft_in_tar_temp = ft_in_tar[k*D + kd];

                update_tar_l3: for(int i = 0; i < HEADS_NUM; i++){
                    #pragma HLS UNROLL

                    update_tar_l2: for(int j=0; j<FEATS_OUT; j++){
                        #pragma HLS UNROLL

                        TYPE rst_tar_temp = (k == 0) ? 0 : rst_tar[kd][i][j];

                        rst_tar[kd][i][j] = rst_tar_temp + ft_in_tar_temp * w_fc[k*D + kd][i][j];
                    }
                }
            }
        }

        // write the result of target update p1 to stream
        wr_rst_tar_p1_stm_l0: for (int kd = 0; kd < D; kd++){
            #pragma HLS pipeline II=1

            vec_head_ft_out_full_t rst_tar_vec_temp;

            for(int i = 0; i < HEADS_NUM; i++){
                #pragma HLS UNROLL
                
                for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS UNROLL

                    rst_tar_vec_temp[i*FEATS_OUT + j] = rst_tar[kd][i][j];
                }
            }

            rst_tar_p1_stream << rst_tar_vec_temp;
        }
    }
}

void update_tar_sum(// hls::stream<vec_ft_out_t>& rst_tar_p1_stream,
                    hls::stream<vec_head_ft_out_full_t>& rst_tar_p1_stream,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    hls::stream<vec_ft_out_full_t>& rst_fc_stream){

    update_tar_sum_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){


        TYPE rst_tar[D][HEADS_NUM*FEATS_OUT];
        #pragma HLS array_partition variable=rst_tar complete dim=2
        // #pragma HLS array_partition variable=rst_tar complete dim=3
        rd_tar_p1_stm_l0: for (int kd = 0; kd < D; kd++){
            #pragma HLS pipeline II=1

            vec_head_ft_out_full_t rst_tar_vec_temp = rst_tar_p1_stream.read();

            rd_tar_p1_stm_l1: for(int i = 0; i < HEADS_NUM; i++){
                #pragma HLS UNROLL

                rd_tar_p1_stm_l2: for(int j=0; j<FEATS_OUT; j++){
                    #pragma HLS UNROLL

                    // rst_tar[kd][i][j] = rst_tar_vec_temp[i*FEATS_OUT + j];
                    rst_tar[kd][i*FEATS_OUT + j] = rst_tar_vec_temp[i*FEATS_OUT + j];
                }
            }
        }

        // sum the D rows of results
        // TYPE rst_tar_sum[HEADS_NUM][FEATS_OUT];
        // #pragma HLS array_partition variable=rst_tar_sum cyclic factor=16 dim=2
        TYPE rst_tar_sum[HEADS_NUM*FEATS_OUT];
        #pragma HLS array_partition variable=rst_tar_sum cyclic factor=16 dim=1
        update_tar_sum_l0: for (int kd = 0; kd < D; kd++){

            update_tar_sum_l1: for(int i = 0; i < HEADS_NUM*FEATS_OUT; i++){

                // update_tar_sum_l2: for(int j=0; j<FEATS_OUT; j++){
                    // #pragma HLS pipeline II=1 rewind
                    // #pragma HLS UNROLL factor=16

                    // TYPE rst_sum_temp = (kd == 0) ? 0 : rst_tar_sum[i][j];

                    // rst_tar_sum[i][j] = rst_sum_temp + rst_tar[kd][i][j];
                // }

                #pragma HLS pipeline II=1 rewind
                #pragma HLS UNROLL factor=16

                TYPE rst_sum_temp = (kd == 0) ? 0 : rst_tar_sum[i];

                rst_tar_sum[i] = rst_sum_temp + rst_tar[kd][i];
            }
        }

        // write rst_tar to stream and convert it from an array of scalar to vector of width W_FT_OUT
        wr_rst_tar_stm_l0: for(int i=0; i<HEADS_NUM; i++){
            #pragma HLS pipeline II=1

            vec_ft_out_full_t rst_tar_sum_temp;

            wr_rst_tar_stm_l1: for (int j = 0; j < FEATS_OUT; j++){
                #pragma HLS UNROLL
                
                // rst_tar_sum_temp[j] = rst_tar_sum[i][j];
                rst_tar_sum_temp[j] = rst_tar_sum[i*FEATS_OUT + j];
            }

            rst_fc_stream << rst_tar_sum_temp;
        }
    }
}

// read stream channel ft_fc_chan to 3 other stream channels
void split1(hls::stream<vec_ft_out_full_t>& rst_fc_stream,
        //    TYPE ft_fc_array_1[HEADS_NUM][FEATS_OUT],
        //    TYPE ft_fc_array_2[HEADS_NUM][FEATS_OUT],
        //    TYPE ft_fc_array_3[HEADS_NUM][FEATS_OUT]
            node_index_t nidx_begin,
            node_index_t nidx_end,
           hls::stream<vec_ft_out_full_t> rst_fc_stream_branch[3]){

    split1_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        split1_l0: for(int i = 0; i < HEADS_NUM; i++){
            #pragma HLS pipeline II=1

            vec_ft_out_full_t rst_fc_vec = rst_fc_stream.read();
            rst_fc_stream_branch[0] << rst_fc_vec;
            rst_fc_stream_branch[1] << rst_fc_vec;
            rst_fc_stream_branch[2] << rst_fc_vec;
        }
    }
}

// write ft_fc_array to ft_fc_mat
void write_feat_fc(hls::stream<vec_ft_out_full_t>& rst_fc_stream_branch1,
            node_index_t nidx_begin,
            node_index_t nidx_end,
            vec_ft_out_full_t ft_fc_mat[N_NODES*HEADS_NUM]){
    
    wr_ft_fc_mem_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        wr_ft_fc_mem_l0: for(int i = 0; i < HEADS_NUM; i++){
            #pragma HLS pipeline II=1

            // vec_ft_out_full_t rst_fc_vec = rst_fc_stream_branch1.read();

            // ft_fc_mat[n*HEADS_NUM + i] = rst_fc_vec;

            ft_fc_mat[n*HEADS_NUM + i] = rst_fc_stream_branch1.read();
        }
    }
}

// compute el
// and write calculated el (el_bf_array) to the stream channel (el_array)
void comp_el(hls::stream<vec_ft_out_full_t>& rst_fc_stream_branch1,
            TYPE attn_l[HEADS_NUM][FEATS_OUT],
            node_index_t nidx_begin,
            node_index_t nidx_end,
            hls::stream<TYPE>& rst_el_stream){
    
    for(node_index_t n=nidx_begin; n<nidx_end; n++){

        for(int i = 0; i < HEADS_NUM; i++){
            #pragma HLS pipeline II=1

            vec_ft_out_full_t rst_fc_vec = rst_fc_stream_branch1.read();

            vec_ft_out_full_t rst_el_vec_temp;

            for(int j = 0; j < FEATS_OUT; j++){
                #pragma HLS UNROLL

                rst_el_vec_temp[j] = rst_fc_vec[j] * attn_l[i][j];
            }

            // rst_el_vec[i] = rst_el_vec_temp.reduce_add();
            rst_el_stream << rst_el_vec_temp.reduce_add();
        }
    }
}

// write el from stream channel to memory
// i.e., channel (el_array) consumer
void write_el_mem(hls::stream<TYPE>& rst_el_stream,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                TYPE el_mat[N_NODES*HEADS_NUM])
{
    wr_el_mem_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        wr_el_mem_l0: for(int i = 0; i < HEADS_NUM; i++){
            #pragma HLS pipeline II=1

            el_mat[n*HEADS_NUM + i] = rst_el_stream.read();
        }
    }
}

// compute er
void comp_er(hls::stream<vec_ft_out_full_t>& rst_fc_stream_branch1,
            TYPE attn_r[HEADS_NUM][FEATS_OUT],
            node_index_t nidx_begin,
            node_index_t nidx_end,
            hls::stream<TYPE>& rst_er_stream){
    
    for(node_index_t n=nidx_begin; n<nidx_end; n++){

        for(int i = 0; i < HEADS_NUM; i++){
            #pragma HLS pipeline II=1

            vec_ft_out_full_t rst_fc_vec = rst_fc_stream_branch1.read();

            vec_ft_out_full_t rst_er_vec_temp;

            for(int j = 0; j < FEATS_OUT; j++){
                #pragma HLS UNROLL

                rst_er_vec_temp[j] = rst_fc_vec[j] * attn_r[i][j];
            }

            // rst_er_vec[i] = rst_er_vec_temp.reduce_add();
            rst_er_stream << rst_er_vec_temp.reduce_add();
        }
    }
}

// write er from stream channer to memory
// i.e., channer (er_array) consumer
void write_er_mem(hls::stream<TYPE>& rst_er_stream,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                TYPE er_mat[N_NODES*HEADS_NUM])
{
    wr_er_mem_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        wr_er_mem_l0: for(int i = 0; i < HEADS_NUM; i++){
            #pragma HLS pipeline II=1

            er_mat[n*HEADS_NUM + i] = rst_er_stream.read();
        }
    }
}

// execute fc layer and compute attention score for each node (node-wise computation)
void comp_fc_attn_one_node(// TYPE ft_in_mat[N_NODES*FEATS_IN],
                            vec_ft_in_t ft_in_mat[N_NODES*FEATS_IN/W_FT_IN],
                            // node_index_t n,
                            TYPE w_fc[FEATS_IN][HEADS_NUM][FEATS_OUT],
                            TYPE attn_l[HEADS_NUM][FEATS_OUT],
                            TYPE attn_r[HEADS_NUM][FEATS_OUT],
                            vec_ft_out_full_t ft_fc_mat[N_NODES*HEADS_NUM],
                            node_index_t nidx_begin,
                            node_index_t nidx_end,
                            TYPE el_mat[N_NODES*HEADS_NUM],
                            // vec_head_full_t el_mat[N_NODES],
                            TYPE er_mat[N_NODES*HEADS_NUM]){
    
    #pragma HLS dataflow
    
    
    // read input feature from memory
    hls::stream<vec_ft_in_t> ft_in_tar_stream;
    #pragma HLS stream variable=ft_in_tar_stream depth=10
    read_feat_in(ft_in_mat, nidx_begin, nidx_end, ft_in_tar_stream);

    // compute fc layer (multiplication)
    // and write the calculated fc feature (ft_fc_array) to a stream channel (ft_fc_chan)
    hls::stream<vec_head_ft_out_full_t> rst_tar_p1_stream;
    #pragma HLS stream variable=rst_tar_p1_stream depth=16
    update_tar_fc(ft_in_tar_stream, w_fc, nidx_begin, nidx_end, rst_tar_p1_stream);

    hls::stream<vec_ft_out_full_t> rst_fc_stream;
    #pragma HLS stream variable=rst_fc_stream depth=10
    update_tar_sum(rst_tar_p1_stream, nidx_begin, nidx_end, rst_fc_stream);

    // read stream channel ft_fc_chan to 3 other stream channels
    hls::stream<vec_ft_out_full_t> rst_fc_stream_branch[3];
    #pragma HLS stream variable=rst_fc_stream_branch[0] depth=10
    #pragma HLS stream variable=rst_fc_stream_branch[1] depth=11
    #pragma HLS stream variable=rst_fc_stream_branch[2] depth=12
    split1(rst_fc_stream, nidx_begin, nidx_end, rst_fc_stream_branch);

    // write ft_fc_array to ft_fc_mat
    write_feat_fc(rst_fc_stream_branch[0], nidx_begin, nidx_end, ft_fc_mat);

    // compute attention scores (first projection then addition)
    // first compute el, er for each vertex
    // then compute e_k = el + er for each edge

    // compute el
    // and write calculated el (el_bf_array) to the stream channel (el_array)
    hls::stream<TYPE> rst_el_stream;
    #pragma HLS stream variable=rst_el_stream depth=16
    comp_el(rst_fc_stream_branch[1], attn_l, nidx_begin, nidx_end, rst_el_stream);

    // write el from stream channel to memory
    // i.e., channel (el_array) consumer
    write_el_mem(rst_el_stream, nidx_begin, nidx_end, el_mat);

    // compute er
    hls::stream<TYPE> rst_er_stream;
    #pragma HLS stream variable=rst_er_stream depth=16
    comp_er(rst_fc_stream_branch[2], attn_r, nidx_begin, nidx_end, rst_er_stream);

    // read el from stream channel to memory
    write_er_mem(rst_er_stream, nidx_begin, nidx_end, er_mat);
}

// execute fc layer and compute attention score for each node (node-wise computation) for all nodes
void comp_fc_attn_all_node(// TYPE ft_in_mat[N_NODES*FEATS_IN],
                           vec_ft_in_t ft_in_mat[N_NODES*FEATS_IN/W_FT_IN],
                           TYPE w_fc[FEATS_IN][HEADS_NUM][FEATS_OUT],
                           TYPE attn_l[HEADS_NUM][FEATS_OUT],
                           TYPE attn_r[HEADS_NUM][FEATS_OUT],
                           node_index_t nidx_begin,
                           node_index_t nidx_end,
                           vec_ft_out_full_t ft_fc_mat[N_NODES*HEADS_NUM],
                           TYPE el_mat[N_NODES*HEADS_NUM],
                           TYPE er_mat[N_NODES*HEADS_NUM]){
    
    // gat_fc_attn: for(node_index_t n=nidx_begin; n<nidx_end; n++){

    //     comp_fc_attn_one_node(ft_in_mat, n, w_fc, attn_l, attn_r, ft_fc_mat, el_mat, er_mat);
    // }
    comp_fc_attn_one_node(ft_in_mat, w_fc, attn_l, attn_r, ft_fc_mat, nidx_begin, nidx_end, el_mat, er_mat);
}


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
            node_index_t nidx_end){
    
    #pragma HLS INTERFACE m_axi port=ft_in_mat bundle=aximm1
    #pragma HLS INTERFACE m_axi port=w_fc_mat bundle=aximm2
    #pragma HLS INTERFACE m_axi port=ft_fc_mat bundle=aximm3

    #pragma HLS INTERFACE m_axi port=attn_l_mat bundle=aximm2
    #pragma HLS INTERFACE m_axi port=attn_r_mat bundle=aximm2

    #pragma HLS INTERFACE m_axi port=el_mat bundle=aximm4
    #pragma HLS INTERFACE m_axi port=er_mat bundle=aximm5


    #pragma HLS INTERFACE s_axilite port=nidx_begin
    #pragma HLS INTERFACE s_axilite port=nidx_end


    // read ft_fc_mat to local array
    TYPE w_fc[FEATS_IN][HEADS_NUM][FEATS_OUT];
    #pragma HLS array_partition variable=w_fc cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=w_fc complete dim=2
    #pragma HLS ARRAY_PARTITION variable=w_fc complete dim=3
    read_weight_fc(w_fc_mat, w_fc);

    // read attn_l_mat to local array
    TYPE attn_l[HEADS_NUM][FEATS_OUT];
    #pragma HLS ARRAY_PARTITION variable=attn_l complete dim=2
    
    read_attn_l(attn_l_mat, attn_l);

    // read attn_r_mat to local array
    TYPE attn_r[HEADS_NUM][FEATS_OUT];
    #pragma HLS ARRAY_PARTITION variable=attn_r complete dim=2

    read_attn_r(attn_r_mat, attn_r);
    
    // #pragma omp parallel for
    // execute fc layer (linear projection: in_feats -> num_heads * out_feats)
    // and compute attention scores (first projection then addition) for all nodes
    // first compute el, er for each vertex
    // then compute e_k = el + er for each edge
    comp_fc_attn_all_node(ft_in_mat, w_fc, attn_l, attn_r, nidx_begin, nidx_end, ft_fc_mat, el_mat, er_mat);

    return 0;
}