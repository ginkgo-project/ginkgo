#!/bin/bash

# set up script
if [ $# -ne 1 ]; then
    echo -e "Usage: $0 GINKGO_BUILD_DIRECTORY"
    exit 1
fi
BUILD_DIR=$1
THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

# copy libraries
LIBRARY_DIRS="core core/device_hooks reference cpu gpu"
LIBRARY_NAMES="ginkgo ginkgo_reference ginkgo_cpu ginkgo_gpu"
SUFFIXES=".so .dylib .dll d.so d.dylib d.dll"
for prefix in ${LIBRARY_DIRS}; do
    for name in ${LIBRARY_NAMES}; do
        for suffix in ${SUFFIXES}; do
            cp ${BUILD_DIR}/${prefix}/lib${name}${suffix} \
                ${THIS_DIR}/lib${name}${suffix} 2>/dev/null
        done
    done
done

# figure out correct compiler flags
if ls ${THIS_DIR} | grep -F "libginkgo."; then
    LINK_FLAGS="-lginkgo -lginkgo_cpu -lginkgo_gpu -lginkgo_reference"
else
    LINK_FLAGS="-lginkgod -lginkgo_cpud -lginkgo_gpud -lginkgo_referenced"
fi
if [ -z "${CXX}" ]; then
    CXX="c++"
fi

# build
${CXX} -std=c++11 -o ${THIS_DIR}/simple_solver ${THIS_DIR}/simple_solver.cpp \
    -I${THIS_DIR}/../..                                                      \
    -L${THIS_DIR} ${LINK_FLAGS} -Wl,-rpath=${THIS_DIR}
