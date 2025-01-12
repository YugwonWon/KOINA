#!/bin/bash

# Set the PYTHONPATH to include the src directory
export PYTHONPATH=$(pwd)/src:src/lib

# Run the transcriber script with nohup
nohup python3 src/analyze/transcriber.py > trans.log 2>&1 &
