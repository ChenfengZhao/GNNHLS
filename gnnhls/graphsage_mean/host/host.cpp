/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

// #define DATA_SIZE 4096

#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <CL/cl2.hpp>

#include "util.h"

// Forward declaration of utility functions included at the end of this file
std::vector<cl::Device> get_xilinx_devices();
char *read_binary_file(const std::string &xclbin_file_name, unsigned &nb);

// define number of CUs
int num_cu = 1;

// ------------------------------------------------------------------------------------
// Main program
// ------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    // ------------------------------------------------------------------------------------
    // Step 1: Initialize the OpenCL environment
    // ------------------------------------------------------------------------------------
    cl_int err;
    std::string binaryFile = (argc != 2) ? "graphsage_mean.xclbin" : argv[1];
    unsigned fileBufSize;
    std::vector<cl::Device> devices = get_xilinx_devices();
    devices.resize(1);
    cl::Device device = devices[0];
    cl::Context context(device, NULL, NULL, NULL, &err);
    char *fileBuf = read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};
    cl::Program program(context, devices, bins, NULL, &err);
    // cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);

    // cl::Kernel krnl_vector_add(program, "graphsage_mean_hls", &err);
    // cl::Kernel krnl_gnn(program, "graphsage_mean_hls", &err);

    // creating multiple kernel objects
    std::vector<cl::Kernel> krnls_gnn(num_cu);
    for (int i = 0; i < num_cu; i++){
        krnls_gnn[i] = cl::Kernel(program, "graphsage_mean_hls", &err);
    }

    double kernel_time_in_sec = 0;
    std::chrono::duration<double> kernel_time(0);

    // ------------------------------------------------------------------------------------
    // Step 2: Create buffers and initialize test values
    // ------------------------------------------------------------------------------------
    // Create the buffers and allocate memory

    // buffer for graph info
    cl::Buffer indptr_buf(context, CL_MEM_READ_ONLY, 2*N_NODES * sizeof(uint64_t), NULL, &err);
    cl::Buffer indices_buf(context, CL_MEM_READ_ONLY, N_EDGES * sizeof(uint64_t), NULL, &err);
    // buffer for input feature
    cl::Buffer feats_buf(context, CL_MEM_READ_ONLY, N_NODES*FEATS_IN * sizeof(TYPE), NULL, &err);
    // buffer for hidden feature
    // cl::Buffer feats_h_buf(context, CL_MEM_READ_WRITE, N_NODES*2*FEATS_IN * sizeof(TYPE), NULL, &err);
    // buffer for weight matrix
    cl::Buffer weight_mlp_buf(context, CL_MEM_READ_ONLY, 2*FEATS_IN*FEATS_OUT * sizeof(TYPE), NULL, &err);
    // buffer for degree
    // cl::Buffer deg_buf(context, CL_MEM_READ_ONLY, N_NODES * sizeof(int), NULL, &err);
    // buffer for final results
    cl::Buffer rst_mat_buf(context, CL_MEM_WRITE_ONLY, N_NODES*FEATS_OUT * sizeof(TYPE), NULL, &err);


    // Graph parameter setting
    // node_index_t nidx_begin = 0;
    // node_index_t nidx_end = N_NODES;

    node_index_t nidx_seg_size =  N_NODES / num_cu;
    std::cout << "nid segment size:" << nidx_seg_size << std::endl;

    node_index_t nidx_seg_begin = 0;
    node_index_t nidx_seg_end;

    std::vector<node_index_t> nidx_begins(num_cu);
    std::vector<node_index_t> nidx_ends(num_cu);

    for(int i=0; i<num_cu; i++){
        nidx_begins[i] = nidx_seg_begin;
        
        if(i == (num_cu-1)){
            nidx_ends[i] = N_NODES;
        }
        else{
            nidx_ends[i] = nidx_seg_begin + nidx_seg_size;
        }
        nidx_seg_begin = nidx_ends[i];
    }

    std::cout << "printing segment begin and end nidx" << std::endl;
    for(int i=0; i<num_cu; i++){
        std::cout << "nidx_begin_" << i << ": " << nidx_begins[i] << std::endl;
        std::cout << "nidx_ends_" << i << ": " << nidx_ends[i] << std::endl;
    }

    // Map buffers to kernel arguments, thereby assigning them to specific device memory banks
    for (int i = 0; i < num_cu; i++){
        int narg = 0;

        krnls_gnn[i].setArg(narg++, indptr_buf);
        krnls_gnn[i].setArg(narg++, indices_buf);
        krnls_gnn[i].setArg(narg++, feats_buf);
        krnls_gnn[i].setArg(narg++, feats_buf);
        krnls_gnn[i].setArg(narg++, weight_mlp_buf);
        krnls_gnn[i].setArg(narg++, nidx_begins[i]);
        krnls_gnn[i].setArg(narg++, nidx_ends[i]);
        krnls_gnn[i].setArg(narg++, rst_mat_buf);
    }

    // Map host-side buffer memory to user-space pointers
    // buffer for graph info
    uint64_t* indptr = (uint64_t *)q.enqueueMapBuffer(indptr_buf, CL_TRUE, CL_MAP_WRITE, 0, 2*N_NODES * sizeof(uint64_t));
    uint64_t* indices = (uint64_t*)q.enqueueMapBuffer(indices_buf, CL_TRUE, CL_MAP_WRITE, 0, N_EDGES * sizeof(uint64_t));
    // buffer for input feature
    TYPE* feats = (TYPE*)q.enqueueMapBuffer(feats_buf, CL_TRUE, CL_MAP_WRITE, 0, N_NODES*FEATS_IN * sizeof(TYPE));
    // buffer for hidden feature
    // TYPE* feats_h = (TYPE*)q.enqueueMapBuffer(feats_h_buf, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, N_NODES*FEATS_IN * sizeof(TYPE));
    // buffer for weight matrix
    TYPE* weight_mlp = (TYPE*)q.enqueueMapBuffer(weight_mlp_buf, CL_TRUE, CL_MAP_WRITE, 0, 2*FEATS_IN*FEATS_OUT * sizeof(TYPE));
    // buffer for degree
    // int* deg = (int*)q.enqueueMapBuffer(deg_buf, CL_TRUE, CL_MAP_WRITE, 0, N_NODES * sizeof(int));
    // buffer for final results
    TYPE* rst_mat = (TYPE*)q.enqueueMapBuffer(rst_mat_buf, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, N_NODES*FEATS_OUT * sizeof(TYPE));


    // Initialize the vectors used in the test
    read_2d_mat_ui64("../data/csr_indptr_trans.txt", 2*N_NODES, 1, 1024, indptr);
    read_2d_mat_ui64("../data/csr_indices.txt", N_EDGES, 1, 1024, indices);
    read_2d_mat("../data/features.txt", N_NODES, FEATS_IN, FEATS_IN*32, feats);
    read_2d_mat("../data/weight_mlp.txt", 2*FEATS_IN, FEATS_OUT, FEATS_OUT*32, weight_mlp);

    // initialize the result mem
    // init_2d_mat(N_NODES, FEATS_OUT, 0, rst_mat);


    // ------------------------------------------------------------------------------------
    // Step 3: Run the kernel
    // ------------------------------------------------------------------------------------
    // Set kernel arguments

    for (int i = 0; i < num_cu; i++){
        int narg = 0;

        krnls_gnn[i].setArg(narg++, indptr_buf);
        krnls_gnn[i].setArg(narg++, indices_buf);
        krnls_gnn[i].setArg(narg++, feats_buf);
        krnls_gnn[i].setArg(narg++, feats_buf);
        krnls_gnn[i].setArg(narg++, weight_mlp_buf);
        krnls_gnn[i].setArg(narg++, nidx_begins[i]);
        krnls_gnn[i].setArg(narg++, nidx_ends[i]);
        krnls_gnn[i].setArg(narg++, rst_mat_buf);
    }

    // Schedule transfer of inputs to device memory, execution of kernel, and transfer of outputs back to host memory

    // q.enqueueMigrateMemObjects({indptr_buf, indices_buf, feats_buf, feats_h_buf, weight_buf, deg_buf}, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({indptr_buf, indices_buf, feats_buf, weight_mlp_buf}, 0 /* 0 means from host*/);
    // running HLS kernel and measure the execution time
    q.finish();
    auto kernel_start = std::chrono::high_resolution_clock::now();

    // q.enqueueTask(krnl_gnn);
    for (int i = 0; i < num_cu; i++) {
        // Launch the kernel
        q.enqueueTask(krnls_gnn[i]);
    }
    q.finish();

    auto kernel_end = std::chrono::high_resolution_clock::now();
    kernel_time += std::chrono::duration<double>(kernel_end - kernel_start);


    q.enqueueMigrateMemObjects({rst_mat_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    
    // Wait for all scheduled operations to finish
    q.finish();

    // get the kernel execution time
    kernel_time_in_sec = kernel_time.count();
    std::cout << "kernel_time_in_sec = " << kernel_time_in_sec << std::endl;

    // ------------------------------------------------------------------------------------
    // Step 4: Check Results and Release Allocated Resources
    // ------------------------------------------------------------------------------------
    void* cor_rst_void;
    posix_memalign(&cor_rst_void, 4096, N_NODES*FEATS_OUT * sizeof(TYPE));
    TYPE* cor_rst = (TYPE*)cor_rst_void;

    read_2d_mat("../data/h2_l0.txt", N_NODES, FEATS_OUT, FEATS_OUT*32, cor_rst);
    check_rst(N_NODES*FEATS_OUT, rst_mat, cor_rst);
}

// ------------------------------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------------------------------
std::vector<cl::Device> get_xilinx_devices()
{
    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    cl::Platform platform;
    for (i = 0; i < platforms.size(); i++)
    {
        platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err);
        if (platformName == "Xilinx")
        {
            std::cout << "INFO: Found Xilinx Platform" << std::endl;
            break;
        }
    }
    if (i == platforms.size())
    {
        std::cout << "ERROR: Failed to find Xilinx platform" << std::endl;
        exit(EXIT_FAILURE);
    }

    //Getting ACCELERATOR Devices and selecting 1st such device
    std::vector<cl::Device> devices;
    err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
    return devices;
}

char *read_binary_file(const std::string &xclbin_file_name, unsigned &nb)
{
    if (access(xclbin_file_name.c_str(), R_OK) != 0)
    {
        printf("ERROR: %s xclbin not available please build\n", xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    //Loading XCL Bin into char buffer
    std::cout << "INFO: Loading '" << xclbin_file_name << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char *buf = new char[nb];
    bin_file.read(buf, nb);
    return buf;
}
