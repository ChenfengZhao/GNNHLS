#!/bin/bash

# Exit when any command fails
set -e

# Make sure everything is up to date
make all | tee report_build.log

