FROM ubuntu:22.04

# install spack prerequisites and basics
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    coreutils \
    curl \
    environment-modules \
    gfortran \
    git \
    gpg \
    gpg-agent \
    lsb-release \
    python3 \
    python3-distutils \
    python3-venv \
    unzip \
    zip \
    vim-nox \
    wget \
    libopencv-dev \
    pkg-config \
    ninja-build \
    # Dependencies for our spack packages
    autoconf automake bzip2 diffutils libtool m4 tar xz-utils perl pigz re2c openssh-client \    
    && rm -rf /var/lib/apt/lists/*

RUN (echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-15 main" > /etc/apt/sources.list.d/llvm15.list) && \
    (curl https://apt.llvm.org/llvm-snapshot.gpg.key > /etc/apt/trusted.gpg.d/apt.llvm.org.asc) && apt-get update && \
    apt-get install -y --no-install-recommends clang-15 clang-tools-15 clang-15-doc clangd-15 clang-tidy-15 clang-format-15 \
    lldb-15 lld-15 libclang-common-15-dev libclang-15-dev libclang1-15 python3-clang-15 libclang-rt-15-dev \
    libc++-15-dev libc++abi-15-dev libomp-15-dev libunwind-15-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && wget https://repo.radeon.com/amdgpu-install/5.4/ubuntu/jammy/amdgpu-install_5.4.50400-1_all.deb \
    && apt-get install -y --no-install-recommends ./amdgpu-install_5.4.50400-1_all.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
    rocm-dev hipblas-dev hipfft-dev hipsparse-dev rocfft-dev rocrand-dev rocsolver-dev rocthrust-dev roctracer-dev \
    && rm ./amdgpu-install_5.4.50400-1_all.deb \
    && rm -rf /var/lib/apt/lists/*

RUN (echo "deb https://apt.repos.intel.com/oneapi all main" > /etc/apt/sources.list.d/oneapi.list) && \
    (curl https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    > /etc/apt/trusted.gpg.d/apt.repos.intel.com.asc) && apt-get update && \
    apt-get install -y \
    intel-oneapi-ccl-devel \
    intel-oneapi-common-licensing \
    intel-oneapi-common-vars \
    intel-oneapi-compiler-dpcpp-cpp \
    intel-oneapi-dev-utilities \
    intel-oneapi-dpcpp-ct \
    intel-oneapi-dpcpp-debugger \
    intel-oneapi-ipp-devel \
    intel-oneapi-libdpstd-devel \
    intel-oneapi-mkl-devel \
    intel-oneapi-tbb-devel \
    intel-oneapi-vtune \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

WORKDIR /root

RUN git clone --depth=1 -c feature.manyFiles=true https://github.com/upsj/spack.git -b gingko_dev_container_patches && \
    source spack/share/spack/setup-env.sh && \
    spack compiler find && \
    spack external find --not-buildable curl autoconf automake \
    bzip2 diffutils libtool m4 \
    pigz re2c source-highlight opencv \
    openssh perl tar xz llvm gettext

ENV PATH="${PATH}:/root/spack/bin"

RUN spack env create ginkgo

COPY spack.yaml spack/var/spack/environments/ginkgo/spack.yaml
COPY packages.yaml spack/etc/spack/packages.yaml

ENV CMAKE_GENERATOR=Ninja

RUN source spack/share/spack/setup-env.sh && \
    spack env activate ginkgo && \
    spack install --use-cache --reuse

COPY entrypoint.sh /root/entrypoint.sh

ENV CMAKE_EXPORT_COMPILE_COMMANDS=ON

# allow mpi to run with root user
ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# fixup broken rocm package
RUN cd /opt/rocm-5.4.0/hiprand/lib/cmake/ && ln -s ../../../lib/cmake/hiprand/hiprand-fortran-config.cmake && \
    cd /opt/rocm-5.4.0/rocrand/lib/cmake/ && ln -s ../../../lib/cmake/rocrand/rocrand-fortran-config.cmake

ENTRYPOINT ["/bin/bash", "-c", "/root/entrypoint.sh"]
