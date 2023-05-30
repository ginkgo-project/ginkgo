FROM ginkgodev/base:latest

RUN git clone --depth=1 -c feature.manyFiles=true https://github.com/upsj/spack.git -b gingko_dev_container_patches && \
    echo '. /spack/share/spack/setup-env.sh; spack env activate ginkgo; export CC=gcc CXX=g++ CUDACXX=`spack location -i cuda`/bin/nvcc' >> /etc/bash.bashrc

ENV PATH="${PATH}:/spack/bin"
    
RUN spack compiler find && \
    spack external find --not-buildable opencv openssh perl llvm cmake ninja && \
    spack env create ginkgo && spack bootstrap now

COPY spack.yaml /spack/var/spack/environments/ginkgo/spack.yaml
COPY packages.yaml /spack/etc/spack/packages.yaml

ENV CMAKE_GENERATOR=Ninja CMAKE_EXPORT_COMPILE_COMMANDS=ON 
ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN --mount=type=cache,target=/cache \
    . /spack/share/spack/setup-env.sh && \
    if [ -f "/cache/key" ]; then \
        echo Reusing existing buildcache && \
        spack gpg trust /cache/key && \
        spack mirror add buildcache /cache && \
        spack buildcache keys --install --trust; \
    else \
        echo Creating new buildcache && \
        spack gpg create buildcache build@cache && \
        spack gpg export --secret /cache/key; \
    fi && \
    spack env activate ginkgo && \
    spack install --use-cache --reuse && \
    spack buildcache create -a /cache
