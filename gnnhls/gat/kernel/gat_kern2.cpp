/*
HLS kernel of GAT (graph converlution network) layer

Reference
[1] Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional 
    networks." arXiv preprint arXiv:1609.02907 (2016).
[2] Dwivedi, Vijay Prakash, et al. "Benchmarking graph neural networks." arXiv preprint arXiv:2003.
    00982 (2020).
*/

#include "gat.h"

TYPE leaky_relu(TYPE a, TYPE neg_slope){
    if(a < 0){
        return a * neg_slope;
    }
    return a;
}

TYPE elu(TYPE a, TYPE elu_slope){
    if(a < 0){
        #ifdef FLOAT32
            return elu_slope * (expf(a) - 1);
        #else
            return elu_slope * (exp(a) - 1);
        #endif
    }
    return a;
}


// read nod_src from memory
void read_nod_src(node_t nod_src[N_NODES],
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<node_t> nod_src_stream[12]){
    
    rd_nod_src_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        #pragma HLS pipeline II=1

        node_t nod_src_n = nod_src[n];

        for (int i = 0; i < 12; i++){
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
                hls::stream<node_index_t> tmp_src_stream[2]){

    rd_edge_src_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        
        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        rd_edge_src_mem_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){
            #pragma HLS pipeline II=1

            // tmp_src_stream << edge_src[e].src;
            node_index_t tmp_src = edge_src[e].src;
            for (int i = 0; i < 2; i++){
                #pragma HLS UNROLL

                tmp_src_stream[i] << tmp_src;
            }
        }
    }
}

// read er_mat from memory to local buffer (er_array)
void read_mem_er(TYPE er_mat[N_NODES*HEADS_NUM],
            hls::stream<node_t>& nod_src_stream0,
            node_index_t nidx_begin,
            node_index_t nidx_end,
            // TYPE er_array[HEADS_NUM]
            hls::stream<TYPE>& er_stream)
{

    rd_er_mem_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if(tmp_begin != tmp_end){

            rd_er: for(int i = 0; i < HEADS_NUM; i++){
                #pragma HLS pipeline II=1

                // er_array[i] = er_mat[n*HEADS_NUM + i];
                er_stream <<  er_mat[n*HEADS_NUM + i];
            }
        }
    }
}

// read el from memory to stream channel
void read_mem_el(TYPE el_mat[N_NODES*HEADS_NUM],
                // edge_src_t edge_src[N_EDGES],
                // edge_index_t e,
                // edge_index_t tmp_begin,
                // node_index_t src_n_cache[MAX_IN_DEG],
                // TYPE el_array[HEADS_NUM]
                hls::stream<node_t>& nod_src_stream0,
                hls::stream<node_index_t>& tmp_src_stream0,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<TYPE>& el_stream
                ){
    
    // node_index_t tmp_src = edge_src[e].src;
    // edge_index_t e_pos = e - tmp_begin;
    // src_n_cache[e_pos] = tmp_src;

    // rd_el: for(int i = 0; i < HEADS_NUM; i++){
    //     #pragma HLS pipeline II=1

    //     el_array[i] = el_mat[tmp_src*HEADS_NUM + i];
    // }

    rd_el_mem_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        rd_el_mem_loop_l0: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

            node_index_t tmp_src = tmp_src_stream0.read();

            rd_el: for(int i = 0; i < HEADS_NUM; i++){
                #pragma HLS pipeline II=1

                // el_array[i] = el_mat[tmp_src*HEADS_NUM + i];
                el_stream << el_mat[tmp_src*HEADS_NUM + i];
            }
        }
    }
}

// read edge_src from memory
void read_edge_src2(edge_src_t edge_src2[N_EDGES],
                hls::stream<node_t>& nod_src_stream0,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<node_index_t>& tmp_src_stream2){

    rd_edge_src2_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
        
        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        rd_edge_src2_mem_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){
            #pragma HLS pipeline II=1

            // tmp_src_stream << edge_src[e].src;
            node_index_t tmp_src = edge_src2[e].src;
            // for (int i = 0; i < 2; i++){
            //     #pragma HLS UNROLL

            //     tmp_src_stream[i] << tmp_src;
            // }

            tmp_src_stream2 << tmp_src;
        }
    }
}

// read er_mat from memory to local buffer (er_array) for sum
void read_mem_er_2(TYPE er_mat2[N_NODES*HEADS_NUM],
            hls::stream<node_t>& nod_src_stream0,
            node_index_t nidx_begin,
            node_index_t nidx_end,
            // TYPE er_array[HEADS_NUM]
            hls::stream<TYPE>& er_stream2)
{

    rd_er_2_mem_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if(tmp_begin != tmp_end){

            rd_er_2: for(int i = 0; i < HEADS_NUM; i++){
                #pragma HLS pipeline II=1

                er_stream2 <<  er_mat2[n*HEADS_NUM + i];
            }
        }
    }
}

// read el from memory to stream channel for sum
void read_mem_el_2(TYPE el_mat2[N_NODES*HEADS_NUM],
                // edge_src_t edge_src[N_EDGES],
                // edge_index_t e,
                // edge_index_t tmp_begin,
                // node_index_t src_n_cache[MAX_IN_DEG],
                // TYPE el_array[HEADS_NUM]
                hls::stream<node_t>& nod_src_stream0,
                hls::stream<node_index_t>& tmp_src_stream0,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<TYPE>& el_stream2
                ){
    
    rd_el_2_mem_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        rd_el_2_mem_loop_l0: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

            node_index_t tmp_src = tmp_src_stream0.read();

            rd_el: for(int i = 0; i < HEADS_NUM; i++){
                #pragma HLS pipeline II=1

                el_stream2 << el_mat2[tmp_src*HEADS_NUM + i];
            }
        }
    }
}

// compute e = leaky_relu(el + er) for all heads
// and write computed e to stream channel (e_array)
void comp_e(
        // TYPE el_array[HEADS_NUM],
        // TYPE er_array[HEADS_NUM],
        // TYPE e_array[HEADS_NUM]
        hls::stream<TYPE>& el_stream,
        hls::stream<TYPE>& er_stream,
        hls::stream<node_t>& nod_src_stream0,
        node_index_t nidx_begin,
        node_index_t nidx_end,
        hls::stream<TYPE>& e_stream){

    comp_e_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if(tmp_begin != tmp_end){
            
            // read er from stream
            TYPE er_array[HEADS_NUM];
            rd_er_stm: for(int i = 0; i < HEADS_NUM; i++){
                #pragma HLS pipeline II=1

                er_array[i] = er_stream.read();
            }

            comp_e_l0: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

                comp_e_l1: for(int i = 0; i < HEADS_NUM; i++){
                    #pragma HLS pipeline II=1

                    // calculate e_k = el + er for each edge
                    // TYPE e_k = el_array[i] + er_array[i];
                    TYPE e_k = el_stream.read() + er_array[i];

                    TYPE e_k_lrelu = leaky_relu(e_k, NEG_SLOPE);

                    // e_array[i] = e_k_lrelu;
                    // perform expf computation here at once for the following softmax
                    #ifdef FLOAT32
                        // e_array[i] = expf(e_k_lrelu);
                        e_stream << expf(e_k_lrelu);
                    #else
                        // e_array[i] = exp(e_k_lrelu);
                        e_stream <<  exp(e_k_lrelu);
                    #endif
                }
            }
        }
    }
}

// compute e = leaky_relu(el + er) for all heads
// and write computed e to stream channel (e_array)
void comp_e_sum(
        // TYPE el_array[HEADS_NUM],
        // TYPE er_array[HEADS_NUM],
        // TYPE e_array[HEADS_NUM]
        hls::stream<TYPE>& el_stream2,
        hls::stream<TYPE>& er_stream2,
        hls::stream<node_t>& nod_src_stream0,
        node_index_t nidx_begin,
        node_index_t nidx_end,
        hls::stream<TYPE>& ek_sum_stream){

    comp_e_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if(tmp_begin != tmp_end){
            
            // read er from stream
            TYPE er_array[HEADS_NUM];
            rd_er_stm: for(int i = 0; i < HEADS_NUM; i++){
                #pragma HLS pipeline II=1

                er_array[i] = er_stream2.read();
            }

            TYPE ek_sum_array[HEADS_NUM];
            comp_e_sum_l0: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

                comp_e_sum_l1: for(int i = 0; i < HEADS_NUM; i++){
                    #pragma HLS pipeline II=1

                    // Get previous sum
                    TYPE last = (e == tmp_begin) ? 0 : ek_sum_array[i];

                    TYPE e_k = el_stream2.read() + er_array[i];

                    TYPE e_k_lrelu = leaky_relu(e_k, NEG_SLOPE);

                    // e_array[i] = e_k_lrelu;
                    // perform expf computation here at once for the following softmax
                    #ifdef FLOAT32;
                        TYPE e_temp = expf(e_k_lrelu);
                    #else
                        TYPE e_temp =  exp(e_k_lrelu);
                    #endif

                    TYPE e_k_sum = last + e_temp;

                    // Write back results
                    ek_sum_array[i] = e_k_sum;
                }
            }

            // write ek_sum_array to stream
            wr_ek_sum_stm: for(int i = 0; i < HEADS_NUM; i++){
                #pragma HLS pipeline II=1

                ek_sum_stream << ek_sum_array[i];
            }
        }
    }
}

/*
// split e_stream to 2 other streams
void split2(
            // TYPE e_array[HEADS_NUM],
            // TYPE e_array_1[HEADS_NUM],
            // TYPE e_array_2[HEADS_NUM]
            hls::stream<TYPE>& e_stream,
            hls::stream<node_t>& nod_src_stream0,
            node_index_t nidx_begin,
            node_index_t nidx_end,
            hls::stream<TYPE>& e_stream_1,
            hls::stream<TYPE>& e_stream_2
            )
{

    split2_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        split2_l0: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

            split2_l1: for(int i = 0; i < HEADS_NUM; i++){
                #pragma HLS pipeline II=1

                // TYPE e_tmp = e_array[i];
                // e_array_1[i] = e_tmp;
                // e_array_2[i] = e_tmp;

                TYPE e_tmp = e_stream.read();
                e_stream_1 << e_tmp;
                e_stream_2 << e_tmp;
            }
        }
    }
}

// write e to cache (e_cache)
void write_cache_e(
                //   TYPE e_array_1[HEADS_NUM],
                //   edge_index_t e,
                //   edge_index_t tmp_begin,
                hls::stream<TYPE>& e_stream_1,
                hls::stream<node_t>& nod_src_stream0,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                TYPE e_cache[MAX_IN_DEG][HEADS_NUM]
                ){

    // edge_index_t e_pos = e - tmp_begin;
    wr_cache_e_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        wr_cache_e_l0: for(edge_index_t e=0; e<(tmp_end - tmp_begin); e++){

            wr_cache_e_l1: for(int i = 0; i < HEADS_NUM; i++){
                #pragma HLS pipeline II=1

                // e_cache[e_pos][i] = e_array_1[i];
                e_cache[e][i] = e_stream_1.read();
            }
        }
    }
}

// calculate sum of e_k
void comp_e_sum(
                // TYPE e_array_2[HEADS_NUM],
                // edge_index_t e,
                // edge_index_t tmp_begin,
                // TYPE ek_sum_array[HEADS_NUM]
                hls::stream<TYPE>& e_stream_2,
                hls::stream<node_t>& nod_src_stream0,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<TYPE>& ek_sum_stream
                ){

    comp_e_sum_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if(tmp_begin != tmp_end){
            TYPE ek_sum_array[HEADS_NUM];

            comp_e_sum_l0: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

                comp_e_sum_l1: for(int i = 0; i < HEADS_NUM; i++){
                    #pragma HLS pipeline II=1

                    // Get previous sum
                    TYPE last = (e == tmp_begin) ? 0 : ek_sum_array[i];

                    // Update current sum
                    // #ifdef FLOAT32
                    //     e_k_sum += expf(e_array_2[i]);
                    // #else
                    //     e_k_sum += exp(e_array_2[i]);
                    // #endif
                    // TYPE e_temp = e_array_2[i];
                    TYPE e_temp = e_stream_2.read();
                    TYPE e_k_sum = last + e_temp;

                    // Write back results
                    ek_sum_array[i] = e_k_sum;
                }
            }

            // write ek_sum_array to stream
            wr_ek_sum_stm: for(int i = 0; i < HEADS_NUM; i++){
                #pragma HLS pipeline II=1

                ek_sum_stream << ek_sum_array[i];
            }
        }
    }
}



// compute softmax
void comp_softmax(
                // TYPE e_cache[MAX_IN_DEG][HEADS_NUM],
                // hls::stream<TYPE>& e_cache,
                hls::stream<TYPE>& e_stream_1,
                hls::stream<TYPE>& ek_sum_stream,
                hls::stream<node_t>& nod_src_stream0,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<TYPE>& a_stream
                )
{

    comp_softmax_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if(tmp_begin != tmp_end){

            TYPE e_cache[MAX_IN_DEG][HEADS_NUM];
            wr_cache_e_l0: for(edge_index_t e=0; e<(tmp_end - tmp_begin); e++){

                wr_cache_e_l1: for(int i = 0; i < HEADS_NUM; i++){

                    e_cache[e][i] = e_stream_1.read();
                }
            }

            TYPE ek_sum_array[HEADS_NUM];

            rd_ek_sum_stm: for(int i = 0; i < HEADS_NUM; i++){
                #pragma HLS pipeline II=1

                ek_sum_array[i] = ek_sum_stream.read();
            }

            comp_softmax_l0: for(edge_index_t e=0; e<(tmp_end - tmp_begin); e++){

                comp_softmax_l1: for(int i = 0; i < HEADS_NUM; i++){
                    #pragma HLS pipeline II=1

                    a_stream << e_cache[e][i] / ek_sum_array[i];
                }
            }
        }
    }
}
*/

// compute softmax
void comp_softmax(
                // TYPE e_cache[MAX_IN_DEG][HEADS_NUM],
                // hls::stream<TYPE>& e_cache,
                hls::stream<TYPE>& e_stream,
                hls::stream<TYPE>& ek_sum_stream,
                hls::stream<node_t>& nod_src_stream0,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                hls::stream<TYPE>& a_stream
                )
{

    comp_softmax_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if(tmp_begin != tmp_end){

            TYPE ek_sum_array[HEADS_NUM];

            rd_ek_sum_stm: for(int i = 0; i < HEADS_NUM; i++){
                #pragma HLS pipeline II=1

                ek_sum_array[i] = ek_sum_stream.read();
            }

            comp_softmax_l0: for(edge_index_t e=0; e<(tmp_end - tmp_begin); e++){

                comp_softmax_l1: for(int i = 0; i < HEADS_NUM; i++){
                    #pragma HLS pipeline II=1

                    TYPE e_temp = e_stream.read();
                    a_stream << e_temp / ek_sum_array[i];
                }
            }
        }
    }
}



// read neighbor input feature from memory
void read_mem_fc_nbr(// TYPE ft_fc_mat[N_NODES*HEADS_NUM*FEATS_OUT],
                    vec_ft_out_full_t ft_fc_mat[N_NODES*HEADS_NUM],
                    // edge_src_t edge_src[N_EDGES],
                    // edge_index_t e,
                    // edge_index_t tmp_begin,
                    // node_index_t src_n_cache[MAX_IN_DEG],
                    // TYPE ft_fc_nbr_array[HEADS_NUM][FEATS_OUT]
                    hls::stream<node_t>& nod_src_stream0,
                    hls::stream<node_index_t>& tmp_src_stream0,
                    node_index_t nidx_begin,
                    node_index_t nidx_end,
                    // hls::stream<TYPE>& ft_fc_nbr_stream
                    hls::stream<vec_ft_out_full_t>& ft_fc_nbr_stream
                    )
{
    
    // // node_index_t tmp_src = edge_src[e].src;
    // edge_index_t e_pos = e - tmp_begin;
    // node_index_t tmp_src = src_n_cache[e_pos];

    // rd_fc_2: for(int i = 0; i < HEADS_NUM; i++){
    //     for(int j = 0; j < FEATS_OUT; j++){
    //         #pragma HLS pipeline II=1
    //         ft_fc_nbr_array[i][j] = ft_fc_mat[tmp_src*HEADS_NUM*FEATS_OUT + i*FEATS_OUT + j];
    //     }
    // }

    rd_mem_fc_nbr_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        rd_mem_fc_nbr_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

            node_index_t tmp_src = tmp_src_stream0.read();

            rd_mem_fc_nbr_l0: for(int i = 0; i < HEADS_NUM; i++){
                #pragma HLS pipeline II=1

                ft_fc_nbr_stream << ft_fc_mat[tmp_src*HEADS_NUM + i];
            }
        }
    }
}

// compute final results
// message passing (aggregate (ft_fc * a) and sum)
void comp_rst(
            // TYPE ft_fc_nbr_array[HEADS_NUM][FEATS_OUT],
            // TYPE a_array[HEADS_NUM],
            // edge_index_t e,
            // edge_index_t tmp_begin,
            // TYPE rst_temp_array[HEADS_NUM][FEATS_OUT]
            // hls::stream<TYPE>& ft_fc_nbr_stream,
            hls::stream<vec_ft_out_full_t>& ft_fc_nbr_stream,
            hls::stream<TYPE>& a_stream,
            hls::stream<node_t>& nod_src_stream0,
            node_index_t nidx_begin,
            node_index_t nidx_end,
            // hls::stream<TYPE>& rst_stream
            hls::stream<vec_ft_out_full_t>& rst_stream
            ){

    comp_rst_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){
    
        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if(tmp_begin != tmp_end){

            // TYPE rst_temp_array[HEADS_NUM][FEATS_OUT];
            vec_ft_out_full_t rst_temp_array[HEADS_NUM];

            comp_rst_eloop: for(edge_index_t e=tmp_begin; e<tmp_end; e++){

                comp_rst_l0: for(int i = 0; i < HEADS_NUM; i++){
                    #pragma HLS pipeline II=1

                    // TYPE a_temp = a_array[i];
                    TYPE a_temp = a_stream.read();

                    vec_ft_out_full_t ft_fc_nbr_vec = ft_fc_nbr_stream.read();

                    // comp_rst_l1: for(int j = 0; j < FEATS_OUT; j++){
                    //     #pragma HLS UNROLL

                    //     // Get previous sum
                    //     TYPE last = (e == tmp_begin) ? 0 : rst_temp_array[i][j];

                    //     // Update current sum
                    //     // TYPE ft_fc_nbr_temp = ft_fc_nbr_array[i][j];
                    //     TYPE ft_fc_nbr_temp = ft_fc_nbr_stream.read();
                    //     // TYPE a_temp = a_array[i];
                    //     TYPE rst_temp = ft_fc_nbr_temp * a_temp + last;

                    //     // write back results
                    //     rst_temp_array[i][j] = rst_temp;
                    // }

                    // Get previous sum
                    vec_ft_out_full_t last_vec = (e == tmp_begin) ? 0 : rst_temp_array[i];

                    rst_temp_array[i] = last_vec + a_temp * ft_fc_nbr_vec;
                }
            }

            // write the results to stream
            wr_rst_stm_l0: for(int i = 0; i < HEADS_NUM; i++){
                
                // wr_rst_stm_l1: for(int j = 0; j < FEATS_OUT; j++){
                //     #pragma HLS pipeline II=1

                //     rst_stream << rst_temp_array[i][j];

                // }

                #pragma HLS pipeline II=1
                rst_stream << rst_temp_array[i];
            }
        }
    }
}

// // write rst from local buffer(rst_temp_array) to stream channel (rst_temp_chan)
// void write_steam_rst(TYPE rst_temp_array[HEADS_NUM][FEATS_OUT],
//                     // edge_index_t tmp_begin,
//                     // edge_index_t tmp_end,
//                      TYPE rst_temp_chan[HEADS_NUM][FEATS_OUT]){
    
//     // if(tmp_begin == tmp_end){
//     //     wr_stream_rst0: for(int i_rst = 0; i_rst < HEADS_NUM; i_rst++){
//     //         for(int j = 0; j < FEATS_OUT; j++){

//     //             rst_temp_chan[i_rst][j] = 0;
//     //         }
//     //     }
//     // }
//     // else{
//     // if(tmp_begin != tmp_end){
//         wr_stream_rst: for(int i_rst = 0; i_rst < HEADS_NUM; i_rst++){
//             #pragma HLS pipeline II=1

//             for(int j = 0; j < FEATS_OUT; j++){
//                 #pragma HLS pipeline II=1

//                 rst_temp_chan[i_rst][j] = rst_temp_array[i_rst][j];
//             }
//         }
//     // }
// }

// // activate (elu) rst
// // write final results to memory
// void wirte_mem_rst(TYPE rst_temp_chan[HEADS_NUM][FEATS_OUT],
//                 // edge_index_t tmp_begin,
//                 // edge_index_t tmp_end,
//                 node_index_t n,
//                 TYPE rst_mat[N_NODES*HEADS_NUM*FEATS_OUT]){
//     // if(tmp_begin != tmp_end){
//         wr_elu_rst: for(int i_rst = 0; i_rst < HEADS_NUM; i_rst++){
//             for(int j = 0; j < FEATS_OUT; j++){
//                 #pragma HLS pipeline II=1

//                 #ifdef ACTIVATION
//                     rst_mat[n*HEADS_NUM*FEATS_OUT + i_rst*FEATS_OUT + j] = elu(rst_temp_chan[i_rst][j], ELU_SLOP);
//                 #else
//                 rst_mat[n*HEADS_NUM*FEATS_OUT + i_rst*FEATS_OUT + j] = rst_temp_chan[i_rst][j];
//                 #endif
//             }
//         }
//     // }
// }

// activate (elu) rst
// write final results to memory
void wirte_mem_rst(
                // hls::stream<TYPE>& rst_stream,
                hls::stream<vec_ft_out_full_t>& rst_stream,
                hls::stream<node_t>& nod_src_stream0,
                node_index_t nidx_begin,
                node_index_t nidx_end,
                // TYPE rst_mat[N_NODES*HEADS_NUM*FEATS_OUT]
                vec_ft_out_full_t rst_mat[N_NODES*HEADS_NUM])
{
    wr_rst_mem_nloop: for(node_index_t n=nidx_begin; n<nidx_end; n++){

        node_t nod_src_temp = nod_src_stream0.read();
        edge_index_t tmp_begin = nod_src_temp.edge_begin;
        edge_index_t tmp_end = nod_src_temp.edge_end;

        if(tmp_begin != tmp_end){

            wr_elu_rst: for(int i_rst = 0; i_rst < HEADS_NUM; i_rst++){
                #pragma HLS pipeline II=1

                // for(int j = 0; j < FEATS_OUT; j++){
                //     #pragma HLS pipeline II=1

                //     #ifdef ACTIVATION
                //         rst_mat[n*HEADS_NUM*FEATS_OUT + i_rst*FEATS_OUT + j] = elu(rst_stream.read(), ELU_SLOP);
                //     #else
                //         rst_mat[n*HEADS_NUM*FEATS_OUT + i_rst*FEATS_OUT + j] = rst_stream.read();
                //     #endif
                // }

                vec_ft_out_full_t rst_vec = rst_stream.read();

                vec_ft_out_full_t rst_act_vec;

                for(int j = 0; j < FEATS_OUT; j++){
                    #pragma HLS UNROLL

                    #ifdef ACTIVATION
                        rst_act_vec[j] = elu(rst_vec[j], ELU_SLOP);
                    #else
                        rst_act_vec[j] = rst_vec[j];
                    #endif
                }

                rst_mat[n*HEADS_NUM + i_rst] = rst_act_vec;
            }
        }
    }
}


// calcaulate the sum of e_k for the following softmax and finally calculate the final resluts by massage passing (edge-wise computation)
void comp_ek_sum_rst_one_node(node_t nod_src[N_NODES],
                        edge_src_t edge_src[N_EDGES],
                        edge_src_t edge_src2[N_EDGES],
                        // node_index_t n,
                        // edge_index_t tmp_begin,
                        // edge_index_t tmp_end,
                        node_index_t nidx_begin,
                        node_index_t nidx_end,
                        TYPE el_mat[N_NODES*HEADS_NUM],
                        TYPE er_mat[N_NODES*HEADS_NUM],
                        TYPE el_mat2[N_NODES*HEADS_NUM],
                        TYPE er_mat2[N_NODES*HEADS_NUM],
                        vec_ft_out_full_t ft_fc_mat[N_NODES*HEADS_NUM],
                        vec_ft_out_full_t rst_mat[N_NODES*HEADS_NUM]){
    #pragma HLS dataflow
    const int head_num = HEADS_NUM;
    const int feats_out = FEATS_OUT;
    const int max_in_deg = MAX_IN_DEG;

    // read nod_src from memory
    hls::stream<node_t> nod_src_stream[12];
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
    #pragma HLS stream variable=nod_src_stream[10] depth=20
    #pragma HLS stream variable=nod_src_stream[11] depth=21
    read_nod_src(nod_src, nidx_begin, nidx_end, nod_src_stream);

    // read edge_src from memory
    hls::stream<node_index_t> tmp_src_stream[2];
    #pragma HLS stream variable=tmp_src_stream[0] depth=10
    #pragma HLS stream variable=tmp_src_stream[1] depth=17
    read_edge_src(edge_src, nod_src_stream[0], nidx_begin, nidx_end, tmp_src_stream);

    // read er_mat from memory to local buffer (er_array)
    hls::stream<TYPE> er_stream;
    #pragma HLS stream variable=er_stream depth=head_num*10
    read_mem_er(er_mat, nod_src_stream[1], nidx_begin, nidx_end, er_stream);

    // read el from memory to stream channel
    hls::stream<TYPE> el_stream;
    #pragma HLS stream variable=el_stream depth=head_num*10
    read_mem_el(el_mat, nod_src_stream[2], tmp_src_stream[0], nidx_begin, nidx_end, el_stream);

    // compute e = leaky_relu(el + er) for all heads
    // and write computed e to stream channel (e_array)
    hls::stream<TYPE> e_stream;
    #pragma HLS stream variable=e_stream depth=head_num*10
    comp_e(el_stream, er_stream, nod_src_stream[3], nidx_begin, nidx_end, e_stream);


    // read edge_src2 from memory for sum
    hls::stream<node_index_t> tmp_src_stream2;
    #pragma HLS stream variable=tmp_src_stream2 depth=11
    read_edge_src2(edge_src2, nod_src_stream[4], nidx_begin, nidx_end, tmp_src_stream2);

    // read er_mat2 from memory to local buffer for sum
    hls::stream<TYPE> er_stream2;
    #pragma HLS stream variable=er_stream depth=head_num*10
    read_mem_er_2(er_mat2, nod_src_stream[5], nidx_begin, nidx_end, er_stream2);

    // read el_mat2 from memory to local buffer for sum
    hls::stream<TYPE> el_stream2;
    #pragma HLS stream variable=el_stream depth=head_num*10
    read_mem_el_2(el_mat2, nod_src_stream[6], tmp_src_stream2, nidx_begin, nidx_end, el_stream2);

    // compute e_sum
    hls::stream<TYPE> ek_sum_stream;
    #pragma HLS stream variable=ek_sum_stream depth=head_num*10
    comp_e_sum(el_stream2, er_stream2, nod_src_stream[7], nidx_begin, nidx_end, ek_sum_stream);

    // compute softmax
    hls::stream<TYPE> a_stream;
    #pragma HLS stream variable=a_stream depth=head_num*10
    comp_softmax(e_stream, ek_sum_stream, nod_src_stream[8], nidx_begin, nidx_end, a_stream);

    // read neighbor input feature from memory
    // hls::stream<TYPE> ft_fc_nbr_stream;
    // #pragma HLS stream variable=ft_fc_nbr_stream depth=head_num*feats_out*10
    hls::stream<vec_ft_out_full_t> ft_fc_nbr_stream;
    #pragma HLS stream variable=ft_fc_nbr_stream depth=head_num*10
    read_mem_fc_nbr(ft_fc_mat, nod_src_stream[9], tmp_src_stream[1], nidx_begin, nidx_end, ft_fc_nbr_stream);

    // compute final results
    // message passing (aggregate (ft_fc * a) and sum)
    // hls::stream<TYPE> rst_stream;
    // #pragma HLS stream variable=rst_stream depth=head_num*feats_out*10
    hls::stream<vec_ft_out_full_t> rst_stream;
    #pragma HLS stream variable=rst_stream depth=head_num*10
    comp_rst(ft_fc_nbr_stream, a_stream, nod_src_stream[10], nidx_begin, nidx_end, rst_stream);

    // activate (elu) rst
    // write final results to memory
    wirte_mem_rst(rst_stream, nod_src_stream[11], nidx_begin, nidx_end, rst_mat);
}


// calcaulate the sum of e_k for the following softmax and finally calculate the final resluts by massage passing (edge-wise computation) for all nodes
void comp_ek_sum_rst_all_node(node_t nod_src[N_NODES],
                              edge_src_t edge_src[N_EDGES],
                              edge_src_t edge_src2[N_EDGES],
                              TYPE el_mat[N_NODES*HEADS_NUM],
                              TYPE er_mat[N_NODES*HEADS_NUM],
                              TYPE el_mat2[N_NODES*HEADS_NUM],
                              TYPE er_mat2[N_NODES*HEADS_NUM],
                              vec_ft_out_full_t ft_fc_mat[N_NODES*HEADS_NUM],
                              node_index_t nidx_begin,
                              node_index_t nidx_end,
                              vec_ft_out_full_t rst_mat[N_NODES*HEADS_NUM]){

    comp_ek_sum_rst_one_node(nod_src, edge_src, edge_src2, nidx_begin, nidx_end, el_mat, er_mat, el_mat2, er_mat2, ft_fc_mat, rst_mat);
}

int gat_hls_kern2(node_t nod_src[N_NODES],
            edge_src_t edge_src[N_EDGES],
            edge_src_t edge_src2[N_EDGES],
            vec_ft_out_full_t ft_fc_mat[N_NODES*HEADS_NUM],
            TYPE el_mat[N_NODES*HEADS_NUM],
            TYPE er_mat[N_NODES*HEADS_NUM],
            TYPE el_mat2[N_NODES*HEADS_NUM],
            TYPE er_mat2[N_NODES*HEADS_NUM],
            node_index_t nidx_begin,
            node_index_t nidx_end,
            vec_ft_out_full_t rst_mat[N_NODES*HEADS_NUM]){
    
    #pragma HLS INTERFACE m_axi port=nod_src bundle=aximm1
    #pragma HLS INTERFACE m_axi port=edge_src bundle=aximm2
    #pragma HLS INTERFACE m_axi port=edge_src2 bundle=aximm3

    #pragma HLS INTERFACE m_axi port=ft_fc_mat bundle=aximm4

    #pragma HLS INTERFACE m_axi port=el_mat bundle=aximm5
    #pragma HLS INTERFACE m_axi port=er_mat bundle=aximm6
    #pragma HLS INTERFACE m_axi port=el_mat2 bundle=aximm7
    #pragma HLS INTERFACE m_axi port=er_mat2 bundle=aximm8

    #pragma HLS INTERFACE s_axilite port=nidx_begin
    #pragma HLS INTERFACE s_axilite port=nidx_end

    #pragma HLS INTERFACE m_axi port=rst_mat bundle=aximm9


    // calculate e_k for each edge
    // calcaulate the sum of e_k for the following softmax
    // Finally calculate the final resluts by massage passing
    // traverse all nodes stored in CSR format
    // n is the idx of target node
    // i is the idx of the head
    // j is the idx of out_feats
    // #pragma omp parallel for
    comp_ek_sum_rst_all_node(nod_src, edge_src, edge_src2, el_mat, er_mat, el_mat2, er_mat2, ft_fc_mat, nidx_begin, nidx_end, rst_mat);

    return 0;
}