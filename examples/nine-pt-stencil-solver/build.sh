#!/bin/bash

# set up script
if [ $# -ne 1 ]; then
    echo -e "Usage: $0 GINKGO_BUILD_DIRECTORY"
    exit 1
fi
BUILD_DIR=$1
THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

source ${THIS_DIR}/../build-setup.sh

# build
${CXX} -std=c++14 -o ${THIS_DIR}/nine-pt-stencil-solver \
       ${THIS_DIR}/nine-pt-stencil-solver.cpp \
       -I${THIS_DIR}/../../include -I${BUILD_DIR}/include \
       -L${THIS_DIR} ${LINK_FLAGS}
