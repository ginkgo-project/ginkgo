#!/bin/bash

GINKGO_LICENSE_BEACON="******************************<GINKGO LICENSE>******************************"

HEADER="header"
CONTENT="content"
BEFORE="before"
HAS_HIP_RUNTIME="false"
DURING_LICENSE="false"
DURING_CONTENT="false"
MAIN_INCLUDE=""
MAIN_PART_MATCH=$(echo $1 | sed -E "s/^(core|cuda|hip|reference)\/([A-Za-z\/_]*).*/\2/g;s/\/test\//\//g")
echo ${MAIN_PART_MATCH}

Style="\"{"
Style="${Style} Language: Cpp,"
Style="${Style} SortIncludes:    true,"
Style="${Style} IncludeBlocks: Regroup,"
Style="${Style} IncludeCategories: ["
Style="${Style} {Regex:           '^(<)(omp|cu|hip|rapidjson|gflags)\/.*',"
Style="${Style} Priority:        2},"
Style="${Style} {Regex:           '^<ginkgo.*', "
Style="${Style} Priority:        4},"
Style="${Style} {Regex:           '^\\\".*',"
Style="${Style} Priority:        5},"
Style="${Style} {Regex:           '.*',"
Style="${Style} Priority:        1}"
Style="${Style} ],"
Style="${Style} MaxEmptyLinesToKeep: 2,"
Style="${Style} SpacesBeforeTrailingComments: 2"
Style="${Style} }\""
# echo "${Style}"
FORMAT_COMMAND="clang-format -i -style=${Style}"
# igonore /test/
# in core without _kernel/_utils -> #include <ginkgo/path/to/file>
# in core with _kernels/_utils > no
# others try to find core/path/the first * with _kernels.hpp
if [[ "$1" =~ ^core ]]; then
    if [[ "${MAIN_PART_MATCH}" =~ _(kernels|utils)$ ]]; then
        MAIN_PART_MATCH="$"
    else
        MAIN_PART_MATCH="<ginkgo/core/${MAIN_PART_MATCH}.hpp>"
    fi
else
    MAIN_PART_MATCH=$(echo ${MAIN_PART_MATCH} | sed -E "s/^([A-Za-z\/]*).*/\1/g")
    MAIN_PART_MATCH="\"core/${MAIN_PART_MATCH}.*_kernels\.hpp\"$"
fi
INCLUDE_REGEX="^#include.*"
MAIN_PART_MATCH="^#include ${MAIN_PART_MATCH}"
echo "+++ ${MAIN_PART_MATCH}"
SHAPE="^#"
RECORD_HEADER=0
IFNDEF="^#ifndef"
DEFINE="^#define"
while IFS='' read -r line; do
    if [ "${DURING_CONTENT}" = "true" ]; then
        echo "${line}" >> "${CONTENT}"
    elif [ "${line}" = '#include "hip/hip_runtime.h"' ]; then
        HAS_HIP_RUNTIME="true"
    elif [ "${line}" = "/*${GINKGO_LICENSE_BEACON}" ] || [ "${DURING_LICENSE}" = "true" ]; then
        DURING_LICENSE="true"
        if [ "${line}" = "${GINKGO_LICENSE_BEACON}*/" ]; then
            DURING_LICENSE="false"
        fi
    elif [ -z "${line}" ] && [[ ${RECORD_HEADER} < 3 ]]; then
        :
    elif [[ "${line}" =~ ${INCLUDE_REGEX} ]] || [ -z "${line}" ] || [[ "${line}" =~ ${SHAPE} ]]; then
        if [[ "${line}" =~ ${IFNDEF} ]] && [[ ${RECORD_HEADER} == 0 ]]; then
            echo "${line}" >> ${BEFORE}
            RECORD_HEADER=$((RECORD_HEADER+1))
        elif [[ "${line}" =~ ${DEFINE} ]] && [[ ${RECORD_HEADER} == 1 ]]; then
            echo "${line}" >> ${BEFORE}
            RECORD_HEADER=$((RECORD_HEADER+1))
        elif [[ "${line}" =~ ${MAIN_PART_MATCH} ]]; then
            echo "${line}"
            if [ -f ${BEFORE} ]; then
                echo "" >> ${BEFORE}
                echo "" >> ${BEFORE}
            fi
            MAIN_INCLUDE="${line}"
            echo "${line}" >> ${BEFORE}
        else
            echo "${line}" >> "${HEADER}"
            RECORD_HEADER=3
        fi
    else
        DURING_CONTENT="true"
        if [ "HAS_HIP_RUNTIME" = "true" ]; then
            echo '#include <hip/hip_runtime.h>' >> "${HEADER}"
        fi
        echo "${line}" >> "${CONTENT}"
    fi
done < $1
# echo "${MAIN_INCLUDE}"
cp ${HEADER} header2
COMMAND="${FORMAT_COMMAND} ${HEADER}"
eval "${COMMAND}"
echo "/*${GINKGO_LICENSE_BEACON}" > $1
cat LICENSE >> $1
echo "${GINKGO_LICENSE_BEACON}*/" >> $1
echo "" >> $1
if [ -f "${BEFORE}" ]; then
    cat ${BEFORE} >> $1
    echo "" >> ${BEFORE}
    echo "" >> ${BEFORE}
fi
PREV_INC=0
if [ -f "${HEADER}" ]; then
    while IFS='' read -r line; do
        if [[ ${line} =~ ${INCLUDE_REGEX} ]]; then
            if [[ ${PREV_INC} == 1 ]]; then
                echo "" >> $1
            fi
            PREV_INC=0
        else
            if [ -z "${line}" ]; then
                PREV_INC=$((PREV_INC+1))
            fi
        fi
        echo "${line}" >> $1
    done < ${HEADER}
    echo "" >> $1
    echo "" >> $1
    # rm "${HEADER}"
fi
cat ${CONTENT} >> $1

rm "${CONTENT}"