FROM rocm/dev-ubuntu-22.04:6.2.4-complete

RUN apt update && apt install -y git cmake ninja-build
RUN git clone -b 4.5.01 --depth=1 https://github.com/kokkos/kokkos.git /tmp/kokkos
ENV ROCM_PATH=/opt/rocm
ENV HIP_PATH=$ROCM_PATH
ENV CMAKE_PREFIX_PATH=$ROCM_PATH
RUN cmake -S /tmp/kokkos -B /tmp/kokkos/build  \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_HIP=ON \
    -DKokkos_ARCH_AMD_GFX908=ON \
    -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ \
    -G Ninja
RUN cmake --build /tmp/kokkos/build
RUN cmake --install /tmp/kokkos/build
RUN rm /tmp/kokkos -rf
