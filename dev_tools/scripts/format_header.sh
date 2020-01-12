#!/bin/bash

convert_header () {
    REGEX="^(#include )(<|\")(.*)(\"|>)$"
    if [[ $@ =~ ${REGEX} ]]; then
        header_file="${BASH_REMATCH[3]}"
        if [ -f "${header_file}" ]; then
            if [[ "${header_file}" =~ ^ginkgo ]]; then
                echo "#include <${header_file}>"
            else
                echo "#include \"${header_file}\""
            fi
        elif [ "${header_file}" = "matrices/config.hpp" ]; then 
            echo "#include \"${header_file}\""
        else
            echo "#include <${header_file}>"
        fi
    else
        echo "$@"
    fi
}

get_header_def () {
    local regex="\.(hpp|cuh)"
    if [[ $@ =~ $regex ]]; then
        local def=$(echo "$@" | sed -E "s~include/ginkgo/~~g;s~/|\.~_~g")
        def=$(echo GKO_${def^^}_)
        echo $def
    else
        echo ""
    fi
}



GINKGO_LICENSE_BEACON="******************************<GINKGO LICENSE>******************************"

HEADER="header" # Store the included header except main header
CONTENT="content" # Store the residual part (start from namespace)
BEFORE="before" # Store the main header and the #ifdef/#define of header file
HAS_HIP_RUNTIME="false"
DURING_LICENSE="false"
DURING_CONTENT="false"
MAIN_INCLUDE=""

Style="\"{"
Style="${Style} Language: Cpp,"
Style="${Style} SortIncludes: true,"
Style="${Style} IncludeBlocks: Regroup,"
Style="${Style} IncludeCategories: ["
Style="${Style} {Regex: '^(<)(omp|cu|hip|rapidjson|gflags|gtest|thrust|papi).*',"
Style="${Style}  Priority: 2},"
Style="${Style} {Regex: '^<ginkgo.*', "
Style="${Style}  Priority: 4},"
Style="${Style} {Regex: '^\\\".*',"
Style="${Style}  Priority: 5},"
Style="${Style} {Regex: '.*',"
Style="${Style}  Priority: 1}"
Style="${Style} ],"
Style="${Style} MaxEmptyLinesToKeep: 2,"
Style="${Style} SpacesBeforeTrailingComments: 2,"
Style="${Style} IndentWidth: 4,"
Style="${Style} AlignEscapedNewlines: Left"
Style="${Style} }\""
FORMAT_COMMAND="clang-format -i -style=${Style}"

INCLUDE_REGEX="^#include.*"
RECORD_HEADER=0
NAMESPACE="^namespace"
MAIN_PART_MATCH=$(dev_tools/scripts/temp.sh $1)
HEADER_DEF=$(get_header_def $1)
IFNDEF=""
DEFINE=""
IFNDEF_REGEX="^#ifndef GKO_"
DEFINE_REGEX="^#define GKO_"
HEADER_REGEX="\.(hpp|cuh)"
SKIP="true"
IF_REX="^#if"
ENDIF_REX="^#endif"
IN_IF="false"
KEEP_LINES=0
while IFS='' read -r line || [ -n "$line" ]; do
    if [ "${DURING_CONTENT}" = "true" ]; then
        echo "${line}" >> "${CONTENT}"
    elif [ "${line}" = '#include "hip/hip_runtime.h"' ]; then
        HAS_HIP_RUNTIME="true"
    elif [ "${line}" = "/*${GINKGO_LICENSE_BEACON}" ] || [ "${DURING_LICENSE}" = "true" ]; then
        DURING_LICENSE="true"
        if [ "${line}" = "${GINKGO_LICENSE_BEACON}*/" ]; then
            DURING_LICENSE="false"
        fi
    elif [ -z "${line}" ] && [ "${SKIP}" = "true" ]; then
    # Ignore all empty lines beteen LICENSE and Header
        :
    elif [[ ! "${line}" =~ ${NAMESPACE} ]]; then
        if [ -z "$line" ]; then
            KEEP_LINES=$((KEEP_LINES+1))
        else
            KEEP_LINES=0
        fi
        if [[ $1 =~ ${HEADER_REGEX} ]] && [[ "${line}" =~ ${IFNDEF_REGEX} ]] && [ "${SKIP}" = "true" ] && [ -z "${DEFINE}" ]; then
            IFNDEF="${line}"
        elif [[ $1 =~ ${HEADER_REGEX} ]] && [[ "${line}" =~ ${DEFINE_REGEX} ]] && [ "${SKIP}" = "true" ]; then
            DEFINE="${line}"
        elif [[ "${line}" =~ $IF_REX ]] || [ "$IN_IF" = "true" ]; then
            # make sure that the header in #if is not extracted
            echo "${line}" >> "${HEADER}"
            IN_IF="true"
            if [[ "${line}" =~ $ENDIF_REX ]]; then
                IN_IF="false"
            fi
            SKIP="false"
        elif [[ "${line}" =~ ${MAIN_PART_MATCH} ]]; then
            line="$(convert_header ${line})"
            if [ ! "${line}" = "${MAIN_INCLUDE}" ]; then
                if [ ! -z "${MAIN_INCLUDE}" ]; then
                    echo "Warning there are different main headers matches: ${MAIN_INCLUDE}, ${line}"
                fi
                echo "${line}" >> ${BEFORE}
                MAIN_INCLUDE="${line}"
            fi
        else 
            if [[ "${line}" =~ $INCLUDE_REGEX ]]; then
                line="$(convert_header ${line})"
            fi
            echo "${line}" >> "${HEADER}"
            SKIP="false"
            # echo "${line}"
        fi
    else
        DURING_CONTENT="true"
        if [ "HAS_HIP_RUNTIME" = "true" ]; then
            echo '#include <hip/hip_runtime.h>' >> "${HEADER}"
        fi
        echo "${line}" >> "${CONTENT}"
    fi
done < $1
# echo "final ${MAIN_INCLUDE}"
# if [ ! "${MAIN_INCLUDE}" = "${FINAL_CONFIG}" ]; then
#     echo "$1 config_regex ${config_regex}"
#     echo "${MAIN_INCLUDE} config ${FINAL_CONFIG}"
# fi
# cp ${HEADER} header2
echo "/*${GINKGO_LICENSE_BEACON}" > $1
cat LICENSE >> $1
echo "${GINKGO_LICENSE_BEACON}*/" >> $1
echo "" >> $1
if [ ! -z "${IFNDEF}" ] && [ ! -z "${DEFINE}" ]; then
    IFNDEF="#ifndef ${HEADER_DEF}"
    DEFINE="#define ${HEADER_DEF}"
elif [ -z "${IFNDEF}" ] && [ -z "${DEFINE}" ]; then
    :
else
    echo "Warning: only #ifndef GKO_ or #define GKO_ is in the header"
fi
if [ ! -z "${IFNDEF}" ]; then
    echo "${IFNDEF}" >> $1
fi
if [ ! -z "${DEFINE}" ]; then
    echo "${DEFINE}" >> $1
    echo "" >> $1
    echo "" >> $1
fi
if [ -f "${BEFORE}" ]; then
    cat ${BEFORE} >> $1
    if [ -f "${HEADER}" ]; then
        echo "" >> $1
        echo "" >> $1
    elif [ -f "${CONTENT}" ]; then
        echo "" >> $1
        echo "" >> $1
    fi
    rm ${BEFORE}
fi
PREV_INC=0
IN_IF="false"
if [ -f "${HEADER}" ]; then
    COMMAND="${FORMAT_COMMAND} ${HEADER}"
    eval "${COMMAND}"
    while IFS='' read -r line; do
        if [[ "${line}" =~ $IF_REX ]] || [ "$IN_IF" = "true" ]; then
            # make sure that the header in #if is not extracted
            echo "${line}" >> $1
            IN_IF="true"
            if [[ "${line}" =~ $ENDIF_REX ]]; then
                IN_IF="false"
                # To keep the original lines
                PREV_INC=-3
            fi
        else
            if [[ ${line} =~ ${INCLUDE_REGEX} ]]; then
                if [[ ${PREV_INC} == 1 ]]; then
                    echo "" >> $1
                fi
                PREV_INC=0
            else
                if [ -z "${line}" ]; then
                    PREV_INC=$((PREV_INC+1))
                else
                    # To keep the original lines
                    PREV_INC=-3
                fi
            fi
            echo "${line}" >> $1
        fi
        
    done < ${HEADER}
    if [ -f "${CONTENT}" ]; then
        for (( i=0; i < ${KEEP_LINES}; i++ ))
        do
            echo "" >> $1
        done
    fi
    rm "${HEADER}"
fi
if [ -f "${CONTENT}" ]; then
    if [ ! -z "${IFNDEF}" ] && [ ! -z "${DEFINE}" ]; then
        head -n -1 ${CONTENT} >> $1
        echo "#endif  // $HEADER_DEF" >> $1
    else
        cat ${CONTENT} >> $1
    fi
    rm "${CONTENT}"
fi
