#!/bin/bash
# Build the scale-correction-mg-solver example against a Ginkgo build tree.
# Usage: ./build.sh <GINKGO_BUILD_DIR>
# Example: ./build.sh /Users/go/Code/ginkgo/build/develop

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 GINKGO_BUILD_DIRECTORY"
    exit 1
fi

BUILD_DIR="$1"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${THIS_DIR}/../../../build-setup.sh"  # not needed; link manually

SDK="$(xcrun --show-sdk-path 2>/dev/null || echo "")"
ISYSROOT="${SDK:+-isysroot $SDK}"

LIB_DIR="${BUILD_DIR}/lib"
INC_DIR="${THIS_DIR}/../../include"
BUILD_INC="${BUILD_DIR}/include"

: "${CXX:=c++}"

echo "Compiler : ${CXX}"
echo "Build dir: ${BUILD_DIR}"
echo "SDK      : ${SDK}"

${CXX} -std=c++17 -O2 \
    ${ISYSROOT} \
    -I"${INC_DIR}" -I"${BUILD_INC}" \
    "${THIS_DIR}/scale-correction-mg-solver.cpp" \
    -L"${LIB_DIR}" \
    -lginkgo -lginkgo_omp -lginkgo_reference -lginkgo_device \
    -lginkgo_cuda -lginkgo_hip -lginkgo_dpcpp \
    -Wl,-rpath,"${LIB_DIR}" \
    -o "${THIS_DIR}/scale-correction-mg-solver"

echo "Built: ${THIS_DIR}/scale-correction-mg-solver"
echo ""
echo "Run from the ginkgo root with:"
echo "  ${THIS_DIR}/scale-correction-mg-solver omp 1 gko_export"
echo "  ${THIS_DIR}/scale-correction-mg-solver omp 0 gko_export   # no scale correction"
