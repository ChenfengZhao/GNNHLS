# GNNHLS

## Overview
With the ever-growing popularity of Graph Neural Networks (GNNs), efficient GNN inference is gaining tremendous attention. Field-Programming Gate Arrays (FPGAs) are a promising execution platform due to their fine-grained parallelism, low-power consumption, reconfigurability, and concurrent execution. Even better, High-Level Synthesis (HLS) tools bridge the gap between the non-trivial FPGA development efforts and rapid emergence of new GNN models. To enable investigation into how effectively modern HLS tools can accelerate GNN inference, we present GNNHLS, a benchmark suite containing a [software stack](./benchmarking-gnns) extended from [this work](https://arxiv.org/pdf/2003.00982.pdf) for data generation and baseline deployment (i.e., CPU and GPU platforms) and FPGA implementations of 6 well-tuned [GNN HLS kernels](./gnnhls) (i.e., GCN, GraphSage, GIN, GAT, MoNet, and GatedGCN). We use the datasets in [Open Graph Benchmarks (OGB)](https://ogb.stanford.edu/docs/dataset_overview/).

## Folder Hierarchy
In the top level directory (GNNHLS):

   - LICENSE - license file of this project.
   - benchmarking-gnns - directory of inference software stack based on 
     PyTorch and DGL for GNN beseline deployment and data generation.
   - gnnhls - directory of 6 FPGA implementations.
      - gcn - directory of graph convolutional network (GCN) kernel.
      - graphsage_mean - directory of GraphSage kernel.
      - gin_sum - directory of Graph Isomorphism Network (GIN) kernel.
      - gat - directory of Graph Attention Network (GAT) kernel.
      - monet - directory of Mixture Model Networks (MoNet) kernel.
      - gatedgcn - directory of Gated Graph ConvNet (GatedGCN) kernel.
   - GNNGLSSupplementalMaterial.pdf - detailed description of GNN kernels, 
     experimental methodology, characterization results, and absolute
     experiment results.

The directory for each kernel contains
a header file defines.h and the following sub-directories:
   - config - configuration files for the kernel
   - host - host files for for the kernel
   - kernel - HLS source code files for the kernel
   - sw_emu - software emulation files for the kernel
   - hw - hardware implementation files for the kernel
   - data - data files for the kernel

## Recommended Requirements
### Software Requrements
- OS: Linux Ubuntu >= 16.04
- Software stack dependencies: Pytorch; DGL == 0.4; CUDA == 10.0, OGB==1.1.1 (We provid conda and images to set up environments for CPU and GPU platforms)
- HLS tools: Xilinx Vitis == 2020.2 and its dependencies (e.g. XRT driver)

### HardWare Requirements
- Host: Multi-core X86 CPUs and/or NVIDIA RTX GPU
- FPGA: We perform our HLS designs on Xilinx Alveo U280, but other Xilinx FPGAs supporting Vitis should also be fine

## Installation Guide
In order to use GNNHLS, the HLS tool, the software stack, and their dependencies need to be installed as follows.

### Software Stack
The source code of the software stack is placed under `./benchmarking-gnns/`. Users could either directly use [conda](#conda_install) or use [docker images](#docker_install) to build containers to easily set up its environments for CPU and GPU platforms. 

#### <span id="conda_install">Install via Conda</span>

1. Install Conda (If conda has been installed, skip this step.)

```
# Conda installation

# For Linux
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

# For OSX
curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

chmod +x ~/miniconda.sh    
./miniconda.sh  

source ~/.bashrc          # For Linux
source ~/.bash_profile    # For OSX
```
2. Set up enviroments
```
# Clone the Github repo:
git clone https://github.com/ChenfengZhao/GNNHLS.git
cd ./GNNHLS
```

- For CPU platforms: 
```
# Install python environment
cd ./benchmarking-gnns/dockerfile
./env_cpu.sh

# Activate environment
conda activate benchmark_gnn
```

- For GPU platforms:

DGL requires CUDA 10.0

For Ubuntu 16.04:

```
# Setup CUDA 10.0 on Ubuntu 16.04
sudo apt-get --purge remove "*cublas*" "cuda*"
sudo apt --purge remove "nvidia*"
sudo apt autoremove
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt update
sudo apt install -y cuda-10-0
sudo reboot
cat /usr/local/cuda/version.txt # Check CUDA version is 10.0

# Install python environment
cd ./benchmarking-gnns/dockerfile
./env_gpu.sh

# Activate environment
conda activate benchmark_gnn
```

For Ubuntu 18.04:

```
# Setup CUDA 10.0 on Ubuntu 18.04
sudo apt-get --purge remove "*cublas*" "cuda*"
sudo apt --purge remove "nvidia*"
sudo apt autoremove
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb 
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt update
sudo apt install -y cuda-10-0
sudo reboot
cat /usr/local/cuda/version.txt # Check CUDA version is 10.0

# Install python environment
cd ./benchmarking-gnns/dockerfile
./env_gpu.sh

# Activate environment
conda activate benchmark_gnn
```

#### <span id="docker_install">Install via docker images</span>
If you don't want to set up the environment by yourself, feel free to use the docker image to build a container in which dependencies has already been installed

1. Download the source code
```
# Clone the Github repo:
git clone https://github.com/ChenfengZhao/GNNHLS.git
cd ./GNNHLS
pwd # this is the path to GNNHLS
```

2. Set up enviroments
For CPU platforms:

```
# Download the docker
docker pull chenfengzhao/gnn_bench:bench_gnns_cpu

# Build a container
docker run -i -t --name="bench-gnns-cpu" -v <path to GNNHLS>:/GNNHLS --net="host" chenfengzhao/gnn_bench:bench_gnns_cpu /bin/bash

# Start the container
docker start bench-gnns-cpu -i

# Activate environment
conda activate benchmark_gnn
```

For GPU platforms:

```
# Download the docker
docker pull chenfengzhao/gnn_bench:bench_gnns_gpu

# Build a container
docker run -i -t --name="bench-gnns-gpu" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -v <Path to GNNHLS>:/GNNHLS --net="host" chenfengzhao/gnn_bench:bench_gnns_gpu /bin/bash

# Start the container
docker start bench-gnns-gpu -i

# Activate environment
conda activate benchmark_gnn
```

### HLS tool

- Xilinx Vitis 2020.2 is installed following the [offical document](https://docs.xilinx.com/r/2020.2-English/ug1393-vitis-application-acceleration/Installation).



## Usage Example

Take GCN as an example.

1. Execute the sofware stack for input data generation and baseline deployment

All the execution commands are listed in the following shell scripts in `./benchmarking-gnns`. Although only commands related to GCN is not commented out, users could take a peek at these shell scripts and uncommnet the corresponding commands to execute other GNN models with various OGB datasets.

```
# Enter the software stack path
cd ./GNNHLS/
# Execute GCN on CPU plaforms with ogbg-moltox21
./run_cpu_infer_OGBG.sh 

# Execute GCN on GPU plaforms with ogbg-moltox21
./run_gpu_infer_OGBG.sh

# Execute GCN on CPU plaforms with ogbn-arxiv
./run_cpu_infer_OGBN.sh

# Execute GCN on GPU plaforms with ogbn-arxiv
./run_gpu_infer_OGBN.sh
```

The above scripts contains several options: `DATASET` means the name of datasets. `ONLY_INFER` denotes whether to perform inference alone (comment it to perform both training and inference sequentially). `GPU_CONFIG` represents the index of the GPU.

The results of GCN on ogbg-moltox21 dataset are located at `./out_new/OGBG_graph_classification/ogbg-moltox21/GCN/data/` in which `infer_time.log` records the execution of the inference step. This folder should be copied to HLS kernel path for the FPGA implementation.

2. Execute HLS kernels on FPGA platforms

```
# Use GCN and ogbg-moltox21 as an example.

# copy generated data folder to HLS kernel path (./gnnhls/gcn/)
cd <path to GNNHLS>
cp -r ./benchmarking-gnns/out_new/OGBG_graph_classification/ogbg-moltox21/GCN/data/ ./gnnhls/gcn/

# build and run the GCN HLS kernel for software emulation
cd ./gnnhls/gcn/sw_emu
./build_and_run.sh

# build the bitstream for FPGA implementation
cd ../hw
./build.sh

# execute the bitstream with host binary on FPGA
./app.exe
```

You can modify the following codes for other GNN HLS kernels.

## License
[MIT_license]: https://spdx.org/licenses/MIT.html

The input data set is in the public domain. The source code of this project is released under the [MIT License][MIT_license]


## Citation
If you think GNNHLS is helpful for your research, please cite the following paper:

Chenfeng Zhao, Zehao Dong, Yixin Chen, Xuan Zhang, and Roger D. Chamberlain. 2023. GNNHLS: Evaluating Graph Neural Network Inference via High-Level Synthesis. In Proc. of 41st IEEE International Conference on Computer Design (ICCD), November 6-8, 2023, Washington, DC, USA