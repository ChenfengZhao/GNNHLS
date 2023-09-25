#!/bin/bash

# Exit when any command fails
set -e

# Make sure everything is up to date
make all 

# Run the application in HW emulation mode
export XCL_EMULATION_MODE=sw_emu
./app.exe 

