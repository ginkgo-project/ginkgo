#!/ bin / bash

#set up script
if
    [$ # - ne 1]; then
    echo -e "Usage: $0 GINKGO_BUILD_DIRECTORY"
    exit 1
fi
BUILD_DIR=$1
THIS_DIR=$( cd "$( dirname "${
    BASH_SOURCE[0]}" )" &>/dev/null && pwd )

#copy libraries
LIBRARY_DIRS="core core/device_hooks reference omp cuda"
LIBRARY_NAMES="ginkgo ginkgo_reference ginkgo_omp ginkgo_cuda"
SUFFIXES=".so .dylib .dll d.so d.dylib d.dll"
for prefix in ${LIBRARY_DIRS};
do
    for
        name in ${LIBRARY_NAMES};
do
        for
            suffix in ${SUFFIXES};
do
    cp ${BUILD_DIR} / ${prefix} / lib${name} ${suffix} ${THIS_DIR} /
        lib${name} $
    {
        suffix
    }
2 > / dev /
            null done done done

#figure out correct compiler flags
            if ls ${THIS_DIR} |
    grep - F "libginkgo." > / dev / null;
then LINK_FLAGS =
    "-lginkgo -lginkgo_omp -lginkgo_cuda -lginkgo_reference" else LINK_FLAGS =
        "-lginkgod -lginkgo_ompd -lginkgo_cudad -lginkgo_referenced" fi if
            [-z "${CXX}"];
then CXX = "c++" fi

#build
               ${CXX} -
           std = c++ 11 - o ${THIS_DIR} / nine - pt - stencil -
                 solver ${THIS_DIR} / nine - pt - stencil - solver.cpp -
                 I${THIS_DIR} /../../ include - I${BUILD_DIR} / include -
                 L${THIS_DIR} $
{
    LINK_FLAGS
}
