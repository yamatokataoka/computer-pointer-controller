#!/bin/bash

exec 1>/output/stdout.log 2>/output/stderr.log

DEVICE=$1
OUTPUT=$2
PRECISION=$3

rm -rf benchmark.zip computer-pointer-controller-benchmark
curl -OL https://github.com/yamatokataoka/computer-pointer-controller/archive/benchmark.zip

unzip benchmark.zip

cd computer-pointer-controller-benchmark/src/

mkdir -p ${OUTPUT}

if echo "$DEVICE" | grep -q "FPGA"; then # if device passed in is FPGA, load bitstream to program FPGA
    #Environment variables and compilation for edge compute nodes with FPGAs
    export AOCL_BOARD_PACKAGE_ROOT=/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2

    source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/2020-2_PL2_FP16_MobileNet_Clamp.aocx

    export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
fi

python3 main.py  -b \
                 --device ${DEVICE} \
                 --output_path ${OUTPUT} \
                 --accuracy ${PRECISION}

cd /output

tar zcvf output.tgz *