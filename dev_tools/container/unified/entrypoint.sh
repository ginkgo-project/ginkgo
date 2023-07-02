#!/bin/bash
. /spack/share/spack/setup-env.sh
spack env activate ginkgo
export CC=gcc CXX=g++ CUDACXX=`spack location -i cuda`/bin/nvcc PATH=$PATH:`spack location -i cuda`/bin
exec "$@"
