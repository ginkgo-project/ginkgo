#!/bin/bash

HIPIFY=/opt/rocm/hip/bin/hipify-perl

if [ "$0" != "dev_tools/scripts/cuda2hip.sh" ]; then
    echo "Only allow to run dev_tools/scripts/cuda2hip.sh at ginkgo source folder."
    exit 1
fi

if [ -z "$1" ]; then
    echo "Usage: $0 path/to/cuda/file"
    exit 2
fi

ORIGIN_FILE=$1
echo "CUDA: ${ORIGIN_FILE}"
NEW_FILE=$(echo ${ORIGIN_FILE} | sed -E "s/^cuda/hip/g;s/(cuh|hpp)$/hip\.hpp/g;s/(cpp|cu)$/hip\.cpp/g")
echo "HIP: ${NEW_FILE}"
${HIPIFY} "${ORIGIN_FILE}" > "${NEW_FILE}"

# String replacement
# header file
REG="s/(cuda[a-z\/_]*)(\.hpp|\.cuh)/\1.hip.hpp/g"
# cuda -> hip
REG="${REG};s/cuda/hip/g;s/Cuda/Hip/g;s/CUDA/HIP/g"
# cublas -> hipblas
REG="${REG};s/cublas/hipblas/g;s/Cublas/Hipblas/g;s/CUBLAS/HIPBLAS/g"
# cusparse -> hipsparse
REG="${REG};s/cusparse/hipsparse/g;s/Cusparse/Hipsparse/g;s/CUSPARSE/HIPSPARSE/g"
# culibs -> hiplibs
REG="${REG};s/culibs/hiplibs/g"
# header definition
REG="${REG};s/(CUH_|HPP_)$/HIP_HPP_/g"

sed -i -E "${REG}" "${NEW_FILE}"
