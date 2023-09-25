# Setup PLRAM 
sdx_memory_subsystem::update_plram_specification [get_bd_cells /memory_subsystem] PLRAM_MEM00 { SIZE 256K AXI_DATA_WIDTH 512 
SLR_ASSIGNMENT SLR0 READ_LATENCY 1 MEMORY_PRIMITIVE URAM} 

sdx_memory_subsystem::update_plram_specification [get_bd_cells /memory_subsystem] PLRAM_MEM01 { SIZE 256K AXI_DATA_WIDTH 512 
SLR_ASSIGNMENT SLR0 READ_LATENCY 1 MEMORY_PRIMITIVE URAM} 

sdx_memory_subsystem::update_plram_specification [get_bd_cells /memory_subsystem] PLRAM_MEM02 { SIZE 256K AXI_DATA_WIDTH 512 
SLR_ASSIGNMENT SLR1 READ_LATENCY 1 MEMORY_PRIMITIVE URAM} 

sdx_memory_subsystem::update_plram_specification [get_bd_cells /memory_subsystem] PLRAM_MEM03 { SIZE 256K AXI_DATA_WIDTH 512 
SLR_ASSIGNMENT SLR1 READ_LATENCY 1 MEMORY_PRIMITIVE URAM} 

sdx_memory_subsystem::update_plram_specification [get_bd_cells /memory_subsystem] PLRAM_MEM04 { SIZE 256K AXI_DATA_WIDTH 512 
SLR_ASSIGNMENT SLR2 READ_LATENCY 1 MEMORY_PRIMITIVE URAM} 

sdx_memory_subsystem::update_plram_specification [get_bd_cells /memory_subsystem] PLRAM_MEM05 { SIZE 256K AXI_DATA_WIDTH 512 
SLR_ASSIGNMENT SLR2 READ_LATENCY 1 MEMORY_PRIMITIVE URAM} 


validate_bd_design -force
save_bd_design