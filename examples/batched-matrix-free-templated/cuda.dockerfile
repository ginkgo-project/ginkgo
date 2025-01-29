FROM ginkgohub/nvhpc:233-cuda120-openmpi-gnu12-llvm16

RUN git clone -b 4.5.01 --depth=1 https://github.com/kokkos/kokkos.git /tmp/kokkos
ENV NVCC_WRAPPER_DEFAULT_COMPILER=nvc++
RUN cmake -S /tmp/kokkos -B /tmp/kokkos/build  \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
    -DKokkos_ARCH_AMPERE80=ON \
    -G Ninja
RUN cmake --build /tmp/kokkos/build
RUN cmake --install /tmp/kokkos/build
RUN rm -rf /tmp/kokkos
