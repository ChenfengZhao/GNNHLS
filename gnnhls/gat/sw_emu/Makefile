APP_NAME=gat

HOST_PATH=../host_2kern
HOST_SRCS=$(HOST_PATH)/host.cpp $(HOST_PATH)/util.cpp
HOST_FILE=$(HOST_SRCS) $(HOST_PATH)/util.h ../defines.h ../data/defines_$(APP_NAME).h

KRNL_PATH=../kernel_2kern
# source files of kernel1
KRNL_SRCS_K1=$(KRNL_PATH)/$(APP_NAME)_kern1.cpp
KRNL_FILE_K1=$(KRNL_SRCS_K1) $(KRNL_PATH)/$(APP_NAME).h ../defines.h ../data/defines_$(APP_NAME).h
# source files of kernel2
KRNL_SRCS_K2=$(KRNL_PATH)/$(APP_NAME)_kern2.cpp
KRNL_FILE_K2=$(KRNL_SRCS_K2) $(KRNL_PATH)/$(APP_NAME).h ../defines.h ../data/defines_$(APP_NAME).h

# compile objects for all kernels
OX_OBJS += $(APP_NAME)_kern1.xo
OX_OBJS += $(APP_NAME)_kern2.xo

CFG_PATH=../config
# CFG_FILE=$(CFG_PATH)/u280.cfg
# CFG_FILE=$(CFG_PATH)/u280_plram.cfg

# config file for kernel1
# CFG_FILE_K1=$(CFG_PATH)/1k1_1k2/u280_plram_v2_1cu_kern1.cfg
# CFG_FILE_K1=$(CFG_PATH)/2k1_2k2/u280_plram_v2_2cu_kern1.cfg
# CFG_FILE_K1=$(CFG_PATH)/4k1_4k2/u280_plram_v2_4cu_kern1.cfg
# CFG_FILE_K1=$(CFG_PATH)/4k1_2k2/u280_plram_v2_4cu_kern1.cfg
# CFG_FILE_K1=$(CFG_PATH)/3k1_3k2/u280_plram_v2_3cu_kern1.cfg
# CFG_FILE_K1=$(CFG_PATH)/4k1_4k2/u280_plram_v2_4cu_kern1.cfg

# CFG_FILE_K1=$(CFG_PATH)/1k1_1k2_v4/u280_plram_v4_1cu_kern1.cfg
# CFG_FILE_K1=$(CFG_PATH)/1k1_1k2_v4_large_hbm/u280_plram_v4_1cu_kern1.cfg
CFG_FILE_K1=$(CFG_PATH)/1k1_1k2_v5_large_hbm/u280_plram_v5_1cu_kern1.cfg

# config file for kernel2
# CFG_FILE_K2=$(CFG_PATH)/1k1_1k2/u280_plram_v2_1cu_kern2.cfg
# CFG_FILE_K2=$(CFG_PATH)/2k1_2k2/u280_plram_v2_2cu_kern2.cfg
# CFG_FILE_K2=$(CFG_PATH)/4k1_4k2/u280_plram_v2_4cu_kern2.cfg
# CFG_FILE_K2=$(CFG_PATH)/4k1_2k2/u280_plram_v2_2cu_kern2_4k1_2k2.cfg
# CFG_FILE_K2=$(CFG_PATH)/3k1_3k2/u280_plram_v2_3cu_kern2.cfg
# CFG_FILE_K2=$(CFG_PATH)/4k1_4k2/u280_plram_v2_4cu_kern2.cfg

# CFG_FILE_K2=$(CFG_PATH)/1k1_1k2_v4/u280_plram_v4_1cu_kern2.cfg
# CFG_FILE_K2=$(CFG_PATH)/1k1_1k2_v4_large_hbm/u280_plram_v4_1cu_kern2.cfg
CFG_FILE_K2=$(CFG_PATH)/1k1_1k2_v5_large_hbm/u280_plram_v5_1cu_kern2.cfg

# config file for link
# CFG_FILE_LD=$(CFG_PATH)/1k1_1k2/u280_plram_v2_1k1_1k2_2kern.cfg
# CFG_FILE_LD=$(CFG_PATH)/2k1_2k2/u280_plram_v2_2k1_2k2_2kern.cfg
# CFG_FILE_LD=$(CFG_PATH)/4k1_4k2/u280_plram_v2_4k1_4k2_2kern.cfg
# CFG_FILE_LD=$(CFG_PATH)/4k1_2k2/u280_plram_v2_4k1_2k2_2kern.cfg
# CFG_FILE_LD=$(CFG_PATH)/3k1_3k2/u280_plram_v2_3k1_3k2_2kern.cfg
# CFG_FILE_LD=$(CFG_PATH)/4k1_4k2/u280_plram_v2_4k1_4k2_2kern.cfg

# CFG_FILE_LD=$(CFG_PATH)/1k1_1k2_v4/u280_plram_v4_1k1_1k2_2kern.cfg
# CFG_FILE_LD=$(CFG_PATH)/1k1_1k2_v4_large_hbm/u280_plram_v4_1k1_1k2_2kern.cfg
CFG_FILE_LD=$(CFG_PATH)/1k1_1k2_v5_large_hbm/u280_plram_v5_1k1_1k2_2kern.cfg


TARGET=sw_emu

all: app.exe emconfig.json $(APP_NAME).xclbin

app.exe: $(HOST_FILE)
	g++ -Wall -g -std=c++11 $(HOST_SRCS) -o app.exe \
		-I${XILINX_XRT}/include/ \
		-L${XILINX_XRT}/lib/ -lOpenCL -pthread -lrt -lstdc++

# compile kernel1
$(APP_NAME)_kern1.xo: $(KRNL_FILE_K1)
	v++ -c -t ${TARGET} --config $(CFG_FILE_K1) -k $(APP_NAME)_hls_kern1 -I$(KRNL_PATH) -I../ -I../data $(KRNL_SRCS_K1) -o $(APP_NAME)_kern1.xo 

# compile kernel2
$(APP_NAME)_kern2.xo: $(KRNL_FILE_K2)
	v++ -c -t ${TARGET} --config $(CFG_FILE_K2) -k $(APP_NAME)_hls_kern2 -I$(KRNL_PATH) -I../ -I../data $(KRNL_SRCS_K2) -o $(APP_NAME)_kern2.xo 

$(APP_NAME).xclbin: ./$(OX_OBJS)
	v++ -l -t ${TARGET} --config $(CFG_FILE_LD) ./$(OX_OBJS) -o $(APP_NAME).xclbin

emconfig.json:
	emconfigutil --platform xilinx_u280_xdma_201920_3 --nd 1

clean:
	rm -rf $(APP_NAME)* app.exe *json *csv *log *summary _x xilinx* .run .Xil .ipcache *.jou

# Unless specified, use the current directory name as the v++ build target
TARGET ?= $(notdir $(CURDIR))
