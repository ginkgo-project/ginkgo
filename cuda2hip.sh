#!/bin/bash

HIPIFY=/opt/rocm/hip/bin/hipify-perl
ORIGIN_FILE=$1
echo "cudafile: ${ORIGIN_FILE}"
NEW_FILE=$(echo ${ORIGIN_FILE} | sed -E "s/^cuda/hip/g;s/(cuh|hpp)$/hip\.hpp/g;s/(cpp|cu)$/hip\.cpp/g")
echo "hipfile: ${NEW_FILE}"
${HIPIFY} "${ORIGIN_FILE}" > "${NEW_FILE}"
sed -i -E "s/(cuda[a-z\/_]*)(\.hpp|\.cuh)/\1.hip.hpp/g;s/culibs/hiplibs/g;s/cuda/hip/g;s/Cuda/Hip/g;s/CUDA/HIP/g;s/cublas/hipblas/g;s/Cublas/Hipblas/g;s/CUBLAS/HIPBLAS/g;s/cusparse/hipsparse/g;s/Cusparse/Hipsparse/g;s/CUSPARSE/HIPSPARSE/g;s/(CUH_|HPP_)$/HIP_HPP_/g" "${NEW_FILE}"

