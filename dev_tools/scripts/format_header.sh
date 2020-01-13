#!/bin/bash

convert_header () {
    local regex="^(#include )(<|\")(.*)(\"|>)$"
    if [[ $@ =~ ${regex} ]]; then
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

add_regroup () {
    cp .clang-format .clang-format.temp
    sed -i "s~\.\.\.~~g" .clang-format
    cat dev_tools/scripts/regroup >> .clang-format
    echo "..." >> .clang-format
}

remove_regroup () {
    mv .clang-format.temp .clang-format
}

get_include_regex () {
    local file="$1"
    local core_suffix=""
    local path_prefix=""
    local path_ignore="0"
    local fix_include=""
    local remove_test="false"
    local item_regex="^-\ +\"(.*)\""
    local path_prefix_regex="PathPrefix:\ +\"(.*)\""
    local core_suffix_regex="CoreSuffix:\ +\"(.*)\""
    local path_ignore_regex="PathIgnore:\ +\"(.*)\""
    local fix_include_regex="FixInclude:\ +\"(.*)\""
    local remove_test_regex="RemoveTest:\ +\"(.*)\""
    local match="false"
    while IFS='' read -r line; do
        if [[ "$line" =~ $item_regex ]]; then
            file_regex="${BASH_REMATCH[1]}"
            if [[ "$match" = "true" ]]; then
                break
            elif [[ $file =~ $file_regex ]]; then
                match="true"
            fi
        elif [ "$match" = "true" ]; then
            if [[ "$line" =~ $path_prefix_regex ]]; then
                path_prefix="${BASH_REMATCH[1]}"
            elif [[ "$line" =~ $core_suffix_regex ]]; then
                core_suffix="${BASH_REMATCH[1]}"
            elif [[ "$line" =~ $path_ignore_regex ]]; then
                path_ignore="${BASH_REMATCH[1]}"
            elif [[ "$line" =~ $fix_include_regex ]]; then
                fix_include="${BASH_REMATCH[1]}"
            elif [[ "$line" =~ $remove_test_regex ]]; then
                remove_test="${BASH_REMATCH[1]}"
            else
                echo "wrong: ${line}"
            fi
        fi
    done < "dev_tools/scripts/config"
    output=""
    if [ -z "${fix_include}" ]; then
        local path_regex="([a-zA-Z_]*\/){${path_ignore}}(.*)\.(cpp|hpp|cu|cuh)"
        if [ ! -z "${path_prefix}" ]; then
            path_prefix="${path_prefix}/"
        fi
        output=$(echo "${file}" | sed -E "s~\.hip~~g;s~$path_regex~$path_prefix\2~g")
        output=$(echo "${output}" | sed -E "s~$core_suffix$~~g")
        output="#include (<|\")$output\.(hpp|hip\.hpp|cuh)(\"|>)"
        if [ "${remove_test}" = "true" ]; then
            output=$(echo "${output}" | sed -E "s~test/~~g")
        fi
    else
        output="#include (<|\")$fix_include(\"|>)"
    fi
    echo "$output"
}

GINKGO_LICENSE_BEACON="******************************<GINKGO LICENSE>******************************"

CONTENT="content" # Store the residual part (start from namespace)
BEFORE="before" # Store the main header and the #ifdef/#define of header file
HAS_HIP_RUNTIME="false"
DURING_LICENSE="false"
INCLUDE_REGEX="^#include.*"
INCLUDE_INC="\.inc"
MAIN_PART_MATCH="$(get_include_regex $1)"
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
LAST_NONEMPTY=""
ALARM=""
COMMENT_REGEX="^ *(\/\/|\/\*)"

while IFS='' read -r line || [ -n "$line" ]; do
    if [ "${line}" = '#include "hip/hip_runtime.h"' ] && [ "${SKIP}" = "true" ]; then
        HAS_HIP_RUNTIME="true"
    elif [ "${line}" = "/*${GINKGO_LICENSE_BEACON}" ] || [ "${DURING_LICENSE}" = "true" ]; then
        DURING_LICENSE="true"
        if [ "${line}" = "${GINKGO_LICENSE_BEACON}*/" ]; then
            DURING_LICENSE="false"
        fi
    elif [ -z "${line}" ] && [ "${SKIP}" = "true" ]; then
    # Ignore all empty lines beteen LICENSE and Header
        :
    else
        if [ -z "${line}" ]; then
            KEEP_LINES=$((KEEP_LINES+1))
        else
            LAST_NONEMPTY="${line}"
            KEEP_LINES=0
        fi
        if [[ $1 =~ ${HEADER_REGEX} ]] && [[ "${line}" =~ ${IFNDEF_REGEX} ]] && [ "${SKIP}" = "true" ] && [ -z "${DEFINE}" ]; then
            IFNDEF="${line}"
        elif [[ $1 =~ ${HEADER_REGEX} ]] && [[ "${line}" =~ ${DEFINE_REGEX} ]] && [ "${SKIP}" = "true" ] && [ ! -z "${IFNDEF}" ]; then
            DEFINE="${line}"
        elif [[ "${line}" =~ $IF_REX ]] || [ "$IN_IF" = "true" ]; then
            # make sure that the header in #if is not extracted
            echo "${line}" >> "${CONTENT}"
            IN_IF="true"
            if [[ "${line}" =~ $ENDIF_REX ]]; then
                IN_IF="false"
            fi
            SKIP="false"
            if [ -z "${ALARM}" ]; then
                ALARM="set"
            fi
        elif [ ! -z "${MAIN_PART_MATCH}" ] && [[ "${line}" =~ ${MAIN_PART_MATCH} ]]; then
            line="$(convert_header ${line})"
            echo "${line}" >> ${BEFORE}
        else 
            if [ -z "${ALARM}" ] && [[ "${line}" =~ $COMMENT_REGEX ]]; then
                ALARM="set"
            elif [[ "${line}" =~ $INCLUDE_REGEX ]]; then
                line="$(convert_header ${line})"
                if [[ ! "${line}" =~ $INCLUDE_INC ]] && [ "${ALARM}" = "set" ]; then
                    ALARM="true"
                fi
            fi
            echo "${line}" >> "${CONTENT}"
            SKIP="false"
        fi
    fi
done < $1
if [ "${ALARM}" = "true" ]; then
    echo "ALARM $1 may not be sorted correctly"
fi
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
    # sort or remove the duplication
    clang-format -i ${BEFORE}
    if [ $(wc -l < ${BEFORE}) -gt "1" ]; then
        echo "Warning there are multiple main header matched"
    fi
    cat ${BEFORE} >> $1
    if [ -f "${CONTENT}" ]; then
        echo "" >> $1
        echo "" >> $1
    fi
    rm ${BEFORE}
fi

if [ -f "${CONTENT}" ]; then
    add_regroup
    if [ "${HAS_HIP_RUNTIME}" = "true" ]; then
        echo "#include <hip/hip_runtime.h>" > temp
    fi
    head -n -${KEEP_LINES} ${CONTENT} >> temp
    if [ ! -z "${IFNDEF}" ] && [ ! -z "${DEFINE}" ]; then
        # Ignore the last line #endif
        if [[ "${LAST_NONEMPTY}" =~ $ENDIF_REX ]]; then
            head -n -1 temp > ${CONTENT}
            echo "#endif  // $HEADER_DEF" >> ${CONTENT}
        else 
            echo "Warning - Found the begin header_def but do not find the end of header_def"
            cat temp > ${CONTENT}
        fi
    else
        cat temp > ${CONTENT}
    fi
    clang-format -i ${CONTENT}
    rm temp
    remove_regroup
    PREV_INC=0
    IN_IF="false"
    while IFS='' read -r line; do
        if [[ ${line} =~ ${INCLUDE_REGEX} ]] && [[ ! ${line} =~ ${INCLUDE_INC} ]]; then
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
    done < ${CONTENT}
    rm ${CONTENT}
fi