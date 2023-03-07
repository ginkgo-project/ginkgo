FROM ginkgodev/unified:latest

SHELL ["/root/entrypoint.sh", "-c"]

RUN git clone https://github.com/ginkgo-project/ginkgo && cd ginkgo && mkdir build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_LINK_DEPENDS_NO_SHARED=ON \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CUDA_COMPILER=`spack location -i cuda`/bin/nvcc \
    #-DCMAKE_CXX_FLAGS= \
    #-DCMAKE_C_FLAGS= \
    #-DCMAKE_CUDA_FLAGS= \
    -DGINKGO_HIP_AMDGPU=gfx906 \
    -DCMAKE_CUDA_ARCHITECTURES=OFF \
    -DGINKGO_CUDA_ARCHITECTURES='61;70' \
    -DGINKGO_BUILD_OMP=ON \
    -DGINKGO_BUILD_CUDA=ON \
    -DGINKGO_BUILD_HIP=ON \
    -DGINKGO_BUILD_MPI=ON \
    -DGINKGO_BUILD_TESTS=ON \
    -DGINKGO_BUILD_EXAMPLES=ON \
    -DGINKGO_BUILD_BENCHMARKS=ON \
    -DGINKGO_MIXED_PRECISION=ON \
    -DGINKGO_DEVEL_TOOLS=ON \
    -DGINKGO_WITH_CCACHE=ON \
    -GNinja

RUN cd ginkgo/build && ninja install

WORKDIR /root/ginkgo