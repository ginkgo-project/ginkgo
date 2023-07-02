FROM ginkgodev/base:latest

RUN git clone --depth=1 -c feature.manyFiles=true https://github.com/spack/spack.git /spack
    
RUN /spack/bin/spack compiler find && \
    /spack/bin/spack external find --not-buildable opencv openssh perl llvm cmake ninja && \
    /spack/bin/spack env create ginkgo && \
    /spack/bin/spack bootstrap now && \
    /spack/bin/spack gpg create buildcache build@cache

COPY spack.yaml /spack/var/spack/environments/ginkgo/spack.yaml
COPY packages.yaml /spack/etc/spack/packages.yaml
COPY entrypoint.sh /usr/bin/

ENV CMAKE_GENERATOR=Ninja CMAKE_EXPORT_COMPILE_COMMANDS=ON 
ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN --mount=type=cache,target=/cache \
    . /spack/share/spack/setup-env.sh && \
    if [ -d "/cache/build_cache" ]; then \
        echo Reusing buildcache && \
        spack mirror add buildcache /cache && \
        spack buildcache keys --install --trust; \
    fi && \
    spack env activate ginkgo && \
    spack install --use-cache --reuse && \
    spack buildcache create -a /cache

# remove the generated key to prevent malicious package injection
RUN rm -rf /spack/opt/spack/gpg

ENTRYPOINT ["/usr/bin/entrypoint.sh"]
