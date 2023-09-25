APP_NAME=monet

HOST_PATH=../host
HOST_SRCS=$(HOST_PATH)/host.cpp $(HOST_PATH)/util.cpp
HOST_FILE=$(HOST_SRCS) $(HOST_PATH)/util.h ../defines.h ../data/defines_$(APP_NAME).h

KRNL_PATH=../kernel
KRNL_SRCS=$(KRNL_PATH)/$(APP_NAME).cpp
KRNL_FILE=$(KRNL_SRCS) $(KRNL_PATH)/$(APP_NAME).h ../defines.h ../data/defines_$(APP_NAME).h

CFG_PATH=../config
# CFG_FILE=$(CFG_PATH)/u280.cfg
# CFG_FILE=$(CFG_PATH)/u280_plram.cfg
# CFG_FILE=$(CFG_PATH)/u280_plram_v2_1cu.cfg
# CFG_FILE=$(CFG_PATH)/u280_plram_v2_2cu.cfg
CFG_FILE=$(CFG_PATH)/u280_plram_v2_2cu_large_hbm.cfg


all: app.exe emconfig.json $(APP_NAME).xclbin

app.exe: $(HOST_FILE)
	g++ -Wall -g -std=c++11 $(HOST_SRCS) -o app.exe \
		-I${XILINX_XRT}/include/ \
		-L${XILINX_XRT}/lib/ -lOpenCL -pthread -lrt -lstdc++
	
$(APP_NAME).xo: $(KRNL_FILE)
	v++ -c -t ${TARGET} --config $(CFG_FILE) -k $(APP_NAME)_hls -I$(KRNL_PATH) -I../ -I../data $(KRNL_SRCS) -o $(APP_NAME).xo 

$(APP_NAME).xclbin: ./$(APP_NAME).xo
	v++ -l -t ${TARGET} --config $(CFG_FILE) ./$(APP_NAME).xo -o $(APP_NAME).xclbin

emconfig.json:
	emconfigutil --platform xilinx_u280_xdma_201920_3 --nd 1

clean:
	rm -rf $(APP_NAME)* app.exe *json *csv *log *summary _x xilinx* .run .Xil .ipcache *.jou

# Unless specified, use the current directory name as the v++ build target
TARGET ?= $(notdir $(CURDIR))
