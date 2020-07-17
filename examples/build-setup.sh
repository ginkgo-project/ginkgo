#!/bin/bash

# copy libraries
LIBRARY_DIRS="core core/device_hooks reference omp cuda hip"
LIBRARY_NAMES="ginkgo ginkgo_reference ginkgo_omp ginkgo_cuda ginkgo_hip"
SUFFIXES=".so .dylib .dll d.so d.dylib d.dll"
VERSION="1.2.0"
for prefix in ${LIBRARY_DIRS}; do
    for name in ${LIBRARY_NAMES}; do
        for suffix in ${SUFFIXES}; do
            cp ${BUILD_DIR}/${prefix}/lib${name}${suffix}.${VERSION} \
                ${THIS_DIR} 2>/dev/null
            if [ -e "${THIS_DIR}/lib${name}${suffix}.${VERSION}" ]
            then
                ln -s ${THIS_DIR}/lib${name}${suffix}.${VERSION} ${THIS_DIR}/lib${name}${suffix} 2>/dev/null
            fi
        done
    done
done

# figure out correct compiler flags
if ls ${THIS_DIR} | grep -F "libginkgo." >/dev/null; then
    LINK_FLAGS="-lginkgo -lginkgo_omp -lginkgo_cuda -lginkgo_reference -lginkgo_hip -Wl,-rpath,${THIS_DIR}"
else
    LINK_FLAGS="-lginkgod -lginkgo_ompd -lginkgo_cudad -lginkgo_referenced -lginkgo_hipd -Wl,-rpath,${THIS_DIR}"
fi
if [ -z "${CXX}" ]; then
    CXX="c++"
fi
