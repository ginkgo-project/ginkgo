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
    lsb-release \
    python3 \
    python3-distutils \
    python3-venv \
    unzip \
    zip \
    vim-nox \
    wget \
    pkg-config \
    ninja-build \
    # for ginkgo
    libopencv-dev \
    # for rocm
    gnupg2 \
    gawk \
    # for spack
    perl \
    && rm -rf /var/lib/apt/lists/*

RUN (echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-15 main" > /etc/apt/sources.list.d/llvm15.list) && \
    (curl https://apt.llvm.org/llvm-snapshot.gpg.key > /etc/apt/trusted.gpg.d/apt.llvm.org.asc) && apt-get update && \
    apt-get install -y libllvm-15-ocaml-dev libllvm15 llvm-15 llvm-15-dev llvm-15-doc llvm-15-examples llvm-15-runtime \
    clang-15 clang-tools-15 clang-15-doc libclang-common-15-dev libclang-15-dev libclang1-15 clang-format-15 python3-clang-15 \
    clangd-15 clang-tidy-15 lldb-15 lld-15 libc++-15-dev libc++abi-15-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    wget https://repo.radeon.com/amdgpu-install/5.5/ubuntu/jammy/amdgpu-install_5.5.50500-1_all.deb && \
    apt-get install -y --no-install-recommends ./amdgpu-install_5.5.50500-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
    rocm-dev hipblas-dev hipfft-dev hipsparse-dev rocfft-dev rocrand-dev rocsolver-dev rocthrust-dev roctracer-dev && \
    rm ./amdgpu-install_5.5.50500-1_all.deb && \
    rm -rf /var/lib/apt/lists/*

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

RUN cd /usr/ && wget -O cmake.tar.gz https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.tar.gz && \
    tar xzf cmake.tar.gz --strip-components 1 && rm cmake.tar.gz

SHELL ["/bin/bash", "-c"]

WORKDIR /
