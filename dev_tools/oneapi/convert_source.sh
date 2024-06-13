#!/bin/bash

# convert_source.sh converts cuda (and c++ code) to dpcpp code with ginkgo design.

# Usage:
# EnvironmentSet ./dev_tools/oneapi/convert_source.sh <the_file_in_cuda>
# <the_file_in_cuda> can be .hpp/.cpp/.cu/.cuh

# the following are parameters set by environment variables
# REQUIRED:
#   CUDA_HEADER_DIR: contains the cuda headers
# OPTIONAL:
#   ROOT_DIR: the ginkgo folder. The default is current path
#   BUILD_DIR: the ginkgo build folder, which needs cmake to generate config.hpp and gtest include. The default is "build"
#              Note. It requires GINKGO_BUILD_TESTS=ON to download gtest but it is not required to compile ginkgo.
#                    If GTEST_HEADER_DIR is available elsewhere GINKGO_BUILD_TESTS is not required.
#                    CMake's step is not required if copying the ginkgo config.hpp from another ginkgo build into "${ROOT_DIR}/include/ginkgo/".
#   ROOT_BUILD_DIR: the complete path for build folder. The default is "${ROOT_DIR}/${BUILD_DIR}"
#   GTEST_HEADER_DIR: the gtest header folder. The default is "${ROOT_BUILD_DIR}/_deps/googletest-src/googletest/include"
#   CLANG_FORMAT: the clang-format exec. The default is "clang-format"
#   VERBOSE: if it is set as 1, script will output the path information
CURRENT_DIR="$( pwd )"
cd "$( dirname "${BASH_SOURCE[0]}" )"
SCRIPT_DIR="$( pwd )"

source "${SCRIPT_DIR}/shared.sh"

ROOT_DIR="${ROOT_DIR:="${CURRENT_DIR}"}"
BUILD_DIR="${BUILD_DIR:="build"}"
ROOT_BUILD_DIR="${ROOT_BUILD_DIR:="${ROOT_DIR}/${BUILD_DIR}"}"
CUDA_HEADER_DIR="${CUDA_HEADER_DIR}"
GTEST_HEADER_DIR="${GTEST_HEADER_DIR:="${ROOT_BUILD_DIR}/_deps/googletest-src/googletest/include"}"
CLANG_FORMAT=${CLANG_FORMAT:="clang-format"}
if [[ "${VERBOSE}" == 1 ]]; then
    echo "#####################"
    echo "# Environment Setting:"
    echo "CURRENT_DIR ${CURRENT_DIR}"
    echo "SCRIPT_DIR ${SCRIPT_DIR}"
    echo "ROOT_DIR ${ROOT_DIR}"
    echo "ROOT_BUILD_DIR ${ROOT_BUILD_DIR}"
    echo "GTEST_HEADER_DIR ${GTEST_HEADER_DIR}"
    echo "CUDA_HEADER_DIR ${CUDA_HEADER_DIR}"
    echo "CLANG_FORMAT ${CLANG_FORMAT}"
    echo "#####################"
fi
if [[ "${CUDA_HEADER_DIR}" == "" ]]; then
    echo "Please set the environment variable CUDA_HEADER_DIR"
    exit 1
fi
# move to working_directory
cd working_directory


KERNEL_SYNTAX_START="<<<"
KERNEL_SYNTAX_END=">>>"
DEVICE_CODE_SYNTAX="#include \"(common.*)\""
FUNCTION_END=");"
EXTRACT_KERNEL="false"
GLOBAL_FILE="global_kernel"

# Converts a CUDA kernel call to the DPC++ equivalent. Also takes care of DPC++ queue manipulation if needed.
convert_cuda_kernel_call() {
    # the syntax_regex turn the string into the following
    # [1]: kernel name
    # [2]: template parameters if it exists. if template contain template struct, it will fail.
    # [3]: CUDA parameters
    # [4]: the rest of the function call (parameters, ...)
    local syntax_regex="([^<>]*)(<[^<>]*>)?<<<(.*)>>>(.*)"
    local str="$1"
    str=$(echo "${str}" | sed -E 's/ +/ /g')
    local temp=""
    local num=0
    local var=""
    if [[ "${str}" =~ ${syntax_regex} ]]; then
        local kernel_name="${BASH_REMATCH[1]}"
        local template_param="${BASH_REMATCH[2]}"
        local cuda_arg="${BASH_REMATCH[3]}"
        local arg="${BASH_REMATCH[4]}"
        temp=${cuda_arg//\[*\]/}
        temp=${temp//\(*\)/}
        temp=${temp//<*>/}
        temp=${temp//[^,]}
        num=$((${#temp} + 1))
        if [[ ${num} -eq 4 ]]; then
            local split_regex="(.*),([^,]*)$"
            # avoid overwriting the BASH_REMATCH
            if [[ "${cuda_arg}" =~ ${split_regex} ]]; then
                var="${BASH_REMATCH[1]}"
                if [[ "${BASH_REMATCH[2]}" -eq 0 ]]; then
                    var="${var}, GET_QUEUE"
                else
                    var="${var}, Error"
                fi
            else
                var="Error"
            fi
        elif [[ "${num}" -lt 2 ]]; then
            var="Error"
        elif [[ ${num} -eq 2 ]]; then
            var="${cuda_arg}, 0, GET_QUEUE"
        elif [[ ${num} -eq 3 ]]; then
            var="${cuda_arg}, GET_QUEUE"
        fi
        local suffix=""
        local function_name=$(echo "${kernel_name}" | sed -E 's/(.*::)| //g')
        suffix=$(cat "${MAP_FILE}" | sed -nE "s/${function_name} -> ${function_name}(.*)/\1/p")
        local suffix_matches=$(echo "${suffix}" | wc -l)
        if [[ "${suffix_matches}" -gt 1 ]]; then
            echo "static_assert(false, \"Has ${suffix_matches} matches for ${function_name}\");"
            suffix=$(echo "${suffix}" | head -1)
        fi

        local result="${kernel_name}${suffix}${template_param}(${var})${arg}"
        result=$(echo "${result}" | sed -E 's/\) *\(+/,/g;s/, *\)/\)/g')
        if [[ "${suffix}" = "" ]]; then
            # if the function does not exist in the file, comment it.
            echo "//remove//${result}"
        else
            echo "${result}"
        fi
    fi
}


convert_regex_allowed() {
    local str="$1"
    str=$(echo "${str}" | sed -E 's~([/().])~\\\1~g')
    echo "$str"
}

# Transfer header file to the correct location
# cuda -> dpcpp
# add sycl header

input="$1"
filename="${ROOT_DIR}/$input"
if [[ "${VERBOSE}" == 1 ]]; then
    echo "Porting file ${filename}"
fi

# check file exists
if [ ! -f "${filename}" ]; then
    echo "${filename} does not exist"
    exit 1
fi


temp=""
IN_SYNTAX="false"
DEVICE_FILE=""
UNFORMAT_FILE="unformat.cpp"
FORMAT_FILE="format.cpp"
cp "${filename}" "${UNFORMAT_FILE}"
OUTPUT_FILE="source.cpp"
EMBED_FILE="embed.cu"
EMBED_HOST_FILE="embed_host.cu"
if [[ "${VERBOSE}" == 1 ]]; then
    echo "###################################"
    echo "# Some generated file for debugging"
    echo "the original file ${UNFORMAT_FILE}"
    echo "the formatted file ${FORMAT_FILE}"
    echo "collect common/*.inc in file ${EMBED_FILE}"
    echo "add autohost func in file ${EMBED_HOST_FILE}"
    echo "convert original CUDA call in file ${OUTPUT_FILE}, which is the file for dpct"
    echo "###################################"
fi
rm "${OUTPUT_FILE}"
echo "#define GET_QUEUE 0" >> "${OUTPUT_FILE}"
# add empty ginkgo license such that format_header recognize some header before header def macro
CURRENT_YEAR=$(date +%Y)
echo "${GINKGO_LICENSE_BEGIN} 2017-${CURRENT_YEAR} The Ginkgo authors" >> "${OUTPUT_FILE}"
echo "//" >> "${OUTPUT_FILE}"
echo "${GINKGO_LICENSE_END} BSD-3-Clause" >> "${OUTPUT_FILE}"
rm "${GLOBAL_FILE}"
rm "${EMBED_FILE}"

# Embed shared kernels into ${EMBED_FILE}
while IFS='' read -r line; do
    if [[ "${line}" =~ $DEVICE_CODE_SYNTAX ]]; then
        # hold the command to easy replace
        device_file="${BASH_REMATCH[1]}"
        [ "${EXTRACT_KERNEL}" == "true" ] && echo "/**** ${device_file} - start ****/" >> "${EMBED_FILE}"
        cat "${ROOT_DIR}/${device_file}" >> "${EMBED_FILE}"
        [ "${EXTRACT_KERNEL}" == "true" ] && echo "/**** ${device_file} - end ****/" >> "${EMBED_FILE}"
        if [ -n "${DEVICE_FILE}" ]; then
            DEVICE_FILE="${DEVICE_FILE};"
        fi
        DEVICE_FILE="${DEVICE_FILE}${device_file}"
    else
        echo "${line}" >> "${EMBED_FILE}"
    fi
done < "${UNFORMAT_FILE}"

# Call clang-format for better formatting.
${CLANG_FORMAT} -style=file "${EMBED_FILE}" > "${FORMAT_FILE}"

# Add an extra host function so that the converted DPC++ code will look like CUDA.
"${SCRIPT_DIR}/add_host_function.sh" "${FORMAT_FILE}" > "${EMBED_HOST_FILE}"

while IFS='' read -r line; do
    if [[ "${line}" =~ ${KERNEL_SYNTAX_START} ]] || [[ "${IN_SYNTAX}" = "true" ]]; then
        temp="${temp} ${line}"
        IN_SYNTAX="true"
        if [[ "${line}" =~ ${FUNCTION_END} ]]; then
            IN_SYNTAX="false"
            modified=""
            if [[ "${temp}" = *"/*KEEP*/"* ]]; then
                modified="${temp/\/\*KEEP\*\//}"
            else
                # change <<<>>> to (grid, block, dynamic, queue)
                modified=$(convert_cuda_kernel_call "$temp")
            fi
            echo "${modified}" >> "${OUTPUT_FILE}"
            temp=""
        fi
    else
        echo "${line}" >> "${OUTPUT_FILE}"
    fi
done < "${EMBED_HOST_FILE}"

# Other fix on OUTPUT_FILE
# dim3 -> dim3_t (for easy replace)
# this_thread_block -> this_thread_block_t
# tiled_partition -> tiled_partition_t
# thread_id.cuh -> use local
# cooperative_group.cuh -> use local
replace_regex="s/dim3/dim3_t/g"
replace_regex="${replace_regex};s/this_thread_block/this_thread_block_t/g"
replace_regex="${replace_regex};s/this_grid/this_grid_t/g"
replace_regex="${replace_regex};s/tiled_partition/tiled_partition_t/g"
replace_regex="${replace_regex};s/thread::/thread_t::/g"
replace_regex="${replace_regex};s/bitonic_sort/bitonic_sort_t/g"
replace_regex="${replace_regex};s/reduction_array/reduction_array_t/g"
replace_regex="${replace_regex};s|cuda/components/thread_ids\.cuh|trick/thread_ids.hpp|g"
replace_regex="${replace_regex};s|cuda/components/cooperative_groups\.cuh|trick/cooperative_groups.hpp|g"
replace_regex="${replace_regex};s|cuda/components/sorting\.cuh|trick/sorting.hpp|g"
replace_regex="${replace_regex};s|cuda/components/reduction\.cuh|trick/reduction.hpp|g"
replace_regex="${replace_regex};s|CUH_|DP_HPP_|g"
# keep using original xxx.sync(); ->xxx;//.sync();
replace_regex="${replace_regex};s|(\.sync\(\);)|;//.sync()|g"
# template macro(); lead std::length_error
replace_regex="${replace_regex};s|(template GKO.*;)|// \1|g"
sed -i -E "${replace_regex}" "${OUTPUT_FILE}"

# Add the trick dim3_t header if dim3 is used.
if grep -Eq "dim3" ${OUTPUT_FILE}; then
    # Found
    sed -i '1 i#include "trick/dim3_t.hpp"' ${OUTPUT_FILE}
fi
# add the cooperative group header according to group:: because some sources forget to add it
if grep -Eq "group::" ${OUTPUT_FILE}; then
    # Found
    sed -i '1 i#include "trick/cooperative_groups.hpp"' ${OUTPUT_FILE}
fi

OUTPUT_FOLDER="output"
if [[ "${VERBOSE}" == 1 ]]; then
    echo "The dpct calling:"
    echo "dpct --extra-arg=\"-std=c++14\" --extra-arg=\"-I ${ROOT_DIR}\" --extra-arg=\"-I ${ROOT_DIR}/include\" --extra-arg=\"-I ${ROOT_BUILD_DIR}/include\" --extra-arg=\"-I ${ROOT_DIR}/dev_tools/oneapi\" --extra-arg=\"-I ${GTEST_HEADER_DIR}\" --cuda-include-path=\"${CUDA_HEADER_DIR}\" --format-range=none ${OUTPUT_FILE} --suppress-warnings=1049 --out-root=${OUTPUT_FOLDER}"
fi

# Delete output/source.cpp
rm "${OUTPUT_FOLDER}/${OUTPUT_FILE}"
rm "${OUTPUT_FOLDER}/${OUTPUT_FILE}.dp.cpp"

# Call DPCT
echo "# Call DPCT on the previously generated file."
echo "############################################"
dpct --extra-arg="-std=c++14" --extra-arg="-I ${ROOT_DIR}" --extra-arg="-I ${ROOT_DIR}/include" --extra-arg="-I ${ROOT_BUILD_DIR}/include" --extra-arg="-I ${ROOT_DIR}/dev_tools/oneapi" --extra-arg="-I ${GTEST_HEADER_DIR}" --cuda-include-path="${CUDA_HEADER_DIR}" --format-range=none ${OUTPUT_FILE} --suppress-warnings=1049 --out-root=${OUTPUT_FOLDER}
echo "############################################"
echo "# The DPCT process is end"

dpct_file=""
if [ -f "${OUTPUT_FOLDER}/${OUTPUT_FILE}.dp.cpp" ]; then
    dpct_file="${OUTPUT_FOLDER}/${OUTPUT_FILE}.dp.cpp"
elif [ -f "${OUTPUT_FOLDER}/${OUTPUT_FILE}" ]; then
    dpct_file="${OUTPUT_FOLDER}/${OUTPUT_FILE}"
else
    echo "No file"
    exit 1
fi

cp "${dpct_file}" "${dpct_file}_bkp"
if [[ "${VERBOSE}" == 1 ]]; then
    echo "the dpct result ${dpct_file}_bkp"
    echo "recover the temporary change in the file ${dpct_file}"
fi

# global reverse fix
replace_regex="s/dim3_t/dim3/g"
replace_regex="${replace_regex};s|trick/dim3\.hpp|dpcpp/base/dim3.dp.hpp|g"
replace_regex="${replace_regex};s/this_thread_block_t/this_thread_block/g"
replace_regex="${replace_regex};s/thread_t::/thread::/g"
replace_regex="${replace_regex};s/this_grid_t/this_grid/g"
replace_regex="${replace_regex};s/bitonic_sort_t/bitonic_sort/g"
replace_regex="${replace_regex};s/reduce_array_t/reduce_array/g"
replace_regex="${replace_regex};s/auto dpct_local_range = block;//g"
replace_regex="${replace_regex};s/sycl::nd_range<3>.*, *$/sycl_nd_range(grid, block), /g"
# do not use c-style casting. use .get() to get the pointer.
replace_regex="${replace_regex};s/\(uninitialized_array<.*> \*\)(.*\.get_pointer\(\))/\1.get()/g"
replace_regex="${replace_regex};s/tiled_partition_t/tiled_partition/g"
replace_regex="${replace_regex};s|trick/thread_ids.hpp|dpcpp/components/thread_ids.dp.hpp|g"
replace_regex="${replace_regex};s|trick/cooperative_groups\.hpp|dpcpp/components/cooperative_groups.dp.hpp|g"
replace_regex="${replace_regex};s|trick/sorting\.hpp|dpcpp/components/sorting.dp.hpp|g"
replace_regex="${replace_regex};s|trick/reduction\.hpp|dpcpp/components/reduction.dp.hpp|g"
replace_regex="${replace_regex};s|cuda/base/math\.hpp|limits|g"
replace_regex="${replace_regex};s|device_numeric_limits(<.*>)::inf(,|;)|std::numeric_limits\1::infinity()\3|g"
replace_regex="${replace_regex};s|device_numeric_limits(<.*>)::(.*)(,|;)|std::numeric_limits\1::\2()\3|g"
replace_regex="${replace_regex};s/#define GET_QUEUE 0//g"
replace_regex="${replace_regex};s/GET_QUEUE/exec->get_queue()/g"
replace_regex="${replace_regex};s/cuda/dpcpp/g"
replace_regex="${replace_regex};s/Cuda/Dpcpp/g"
replace_regex="${replace_regex};s/CUDA/DPCPP/g"
replace_regex="${replace_regex};s/(cusparse|cublas)_bindings/onemkl_bindings/g"
replace_regex="${replace_regex};s/\.cuh/.dp.hpp/g"
replace_regex="${replace_regex};s/\.cu/.dp.cpp/g"
replace_regex="${replace_regex};s/${HOST_SUFFIX}//g"
replace_regex="${replace_regex};s|#include <dpct/dpct\.hpp>||g"
replace_regex="${replace_regex};s|#include \"dpcpp/base/types\.hpp\"||g"
replace_regex="${replace_regex};s|#include \"dpcpp/test/utils\.hpp\"|#include \"core/test/utils.hpp\"|g"
# remove as_dpcpp_type\(content\) -> content
# If the content use some brackets, only allowed one nested bracket now.
replace_regex="${replace_regex};s/as_dpcpp_type\((([^()]*(\([^()]*\))[^()]*)*)\)/\1/g"
replace_regex="${replace_regex};s/as_dpcpp_type\(([^()]*)\)/\1/g"
# dcpt can not convert idx in static_cast of nested template
# It should be fixed already, but let's keep it
replace_regex="${replace_regex};s/threadIdx\.x/item_ct1.get_local_id(2)/g"
replace_regex="${replace_regex};s/threadIdx\.y/item_ct1.get_local_id(1)/g"
replace_regex="${replace_regex};s/threadIdx\.z/item_ct1.get_local_id(0)/g"

replace_regex="${replace_regex};s/blockIdx\.x/item_ct1.get_group(2)/g"
replace_regex="${replace_regex};s/blockIdx\.y/item_ct1.get_group(1)/g"
replace_regex="${replace_regex};s/blockIdx\.z/item_ct1.get_group(0)/g"

replace_regex="${replace_regex};s/blockDim\.x/item_ct1.get_local_range().get(2)/g"
replace_regex="${replace_regex};s/blockDim\.y/item_ct1.get_local_range().get(1)/g"
replace_regex="${replace_regex};s/blockDim\.z/item_ct1.get_local_range().get(0)/g"

replace_regex="${replace_regex};s/gridDim\.x/item_ct1.get_group_range(2)/g"
replace_regex="${replace_regex};s/gridDim\.y/item_ct1.get_group_range(1)/g"
replace_regex="${replace_regex};s/gridDim\.z/item_ct1.get_group_range(0)/g"
# Workaround for abs
replace_regex="${replace_regex};s/sycl::fabs/std::abs/g"
# Remove unneed warning DPCT1049 - check the block size
# It should be properly handled by dpct, but let's keep it without hurt
replace_regex="${replace_regex};/\/\*$/{N;N;/ *\/\*\n *DPCT1049.*\n *\*\//d}"
# Recover // template GKO_...
replace_regex="${replace_regex};s|// (template GKO.*;)|\1|g"
# Recover // xxx;//.sync();->xxx.sync();
replace_regex="${replace_regex};s|;//\.sync\(\)|.sync();|g"
replace_regex="${replace_regex};s|//remove//||g"

sed -i -E "${replace_regex}" "${dpct_file}"

# Include dpct.hpp if __dpct_inline__ or __dpct_align__ are found
need_dpct=$(grep -Eq "__(dpct_align|dpct_inline)__" ${dpct_file})
if grep -Eq "__(dpct_align|dpct_inline)__" ${dpct_file}; then
    # Found
    sed -i '1 i#include "dpcpp/base/dpct.hpp"' ${dpct_file}
fi

# extract device_code
if [ "${EXTRACT_KERNEL}" = "true" ]; then
    IFS=';' read -ra individual_device <<< "${DEVICE_FILE}"
    for variable in "${individual_device[@]}"; do
        device_regex=$(convert_regex_allowed "${variable}")
        dpct_device_path=$(echo "${variable}" | sed 's/common/dpcpp_code/g')
        dpct_device_file=$(echo "${dpct_device_path}" | sed 's|/|@|g')
        dpct_device_file="output/${dpct_device_file}"
        cat "${dpct_file}" | sed -n "/${device_regex} - start/,/${device_regex} - end/p" | sed "1d;\$d" > "${dpct_device_file}"
        sed -i "/${device_regex} - start/,/${device_regex} - end/d;s~// *#include \"${device_regex}\"~#include \"${dpct_device_path}\"~g" ${dpct_file}
        dpct_dir=$(dirname "${dpct_device_path}")
        mkdir -p "${ROOT_DIR}/${dpct_dir}"
        cp "${dpct_device_file}" "${ROOT_DIR}/${dpct_device_path}"
    done
fi

# Integrate the new DPC++ file into the correct place
target_file=$(echo "${input}" | sed 's|cuda/|dpcpp/|g;s/\.cuh/\.dp\.hpp/g;s/\.cu/.dp.cpp/g')
target_dir=$(dirname "${target_file}")
mkdir -p "${ROOT_DIR}/${target_dir}"
echo "cp ${dpct_file} ${ROOT_DIR}/${target_file}"
cp "${dpct_file}" "${ROOT_DIR}/${target_file}"
