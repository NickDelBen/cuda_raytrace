#!/bin/bash

export CUDAROOT=/usr/local/cuda-8.0/
export PATH=$CUDAROOT/bin/:$PATH
export LD_LIBRARY_PATH=$CUDAROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDAROOT/lib64:$LD_LIBRARY_PATH
