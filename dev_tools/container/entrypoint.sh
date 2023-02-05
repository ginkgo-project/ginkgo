#!/bin/bash
. /root/spack/share/spack/setup-env.sh
spack env activate ginkgo
# Intel's compiler packages set these variables, reset them
export CC=gcc CXX=g++ CUDACXX=`spack location -i cuda`/bin/nvcc

/bin/bash "$@"
