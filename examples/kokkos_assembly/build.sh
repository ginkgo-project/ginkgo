#!/bin/bash

# set up script
if [ $# -ne 1 ]; then
    echo -e "Usage: $0 GINKGO_BUILD_DIRECTORY"
    exit 1
fi
BUILD_DIR=$1
THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )

cmake ${THIS_DIR} -DGinkgo_ROOT=${BUILD_DIR}
cmake --build .
