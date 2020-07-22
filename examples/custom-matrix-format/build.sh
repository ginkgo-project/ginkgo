#!/bin/bash

# set up script
if [ $# -ne 1 ]; then
    echo -e "Usage: $0 GINKGO_BUILD_DIRECTORY"
    exit 1
fi
BUILD_DIR=$1
THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

source ${THIS_DIR}/../build-setup.sh

CXX="nvcc"

# figure out correct compiler flags
if ls ${THIS_DIR} | grep -F "libginkgo." >/dev/null; then
    LINK_FLAGS="-lginkgo -lginkgo_omp -lginkgo_cuda -lginkgo_reference -lginkgo_hip -Xlinker -rpath -Xlinker ${THIS_DIR}"
else
    LINK_FLAGS="-lginkgod -lginkgo_ompd -lginkgo_cudad -lginkgo_referenced -lginkgo_hipd -Xlinker -rpath -Xlinker ${THIS_DIR}"
fi


# build
${CXX} -std=c++11 -o ${THIS_DIR}/custom-matrix-format \
       ${THIS_DIR}/custom-matrix-format.cpp ${THIS_DIR}/stencil_kernel.cu \
       -I${THIS_DIR}/../../include -I${BUILD_DIR}/include \
       -L${THIS_DIR} ${LINK_FLAGS}
