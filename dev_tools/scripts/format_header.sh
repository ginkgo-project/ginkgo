#!/usr/bin/env bash

CLANG_FORMAT=${CLANG_FORMAT:="clang-format"}

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
        local def=$(echo "$@" | sed -E "s~include/ginkgo/~PUBLIC_~g;s~/|\.~_~g")
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

# It reads "dev_tools/scripts/config" to generate the corresponding main header
# The setting setting:
# - "file_regex"
#   - CoreSuffix: "core_suffix_regex"           (default "")
#   - PathPrefix: "path_prefix_regex"           (default "")
#   - PathIgnore: "path_ignore_number"          (default "0")
#   - RemoveTest: "false/true"                  (default "test")
#   - FixInclude: "the specific main header"    (default "")
# Only "file_regex" without any setting is fine, and it means find the same name with header suffix
# For example, /path/to/file.cpp will change to /path/to/file.hpp
# file_regex : selecting which file apply this rule
# CoreSuffix : remove the pattern which passes the "core_suffix_regex" of file
# PathPrefix : adds "path_prefix_regex" before path, and the position depends on PathIgnore
# PathIgnore : ignore the number "path_ignore_number" folder from top level, and then add "path_prefix_regex" into path
# RemoveTest : Decide whether ignore /test/ in the path
# FixInclude : Specify the main header. If it is set, ignore others setting
# Note: This script picks the first fitting "file_regex" rules according the ordering in config
get_include_regex () {
    local file="$1"
    declare -n local_output=$2
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
                echo "Ignore unknow setting: \"${file_regex}\" - ${line}"
            fi
        fi
    done < "dev_tools/scripts/config"
    local_output=""
    if [ -z "${fix_include}" ]; then
        local path_regex="([a-zA-Z_]*\/){${path_ignore}}(.*)\.(cpp|hpp|cu|cuh)"
        if [ ! -z "${path_prefix}" ]; then
            path_prefix="${path_prefix}/"
        fi
        local_output=$(echo "${file}" | sed -E "s~\.(hip|dp)~~g;s~$path_regex~$path_prefix\2~g")
        local_output=$(echo "${local_output}" | sed -E "s~$core_suffix$~~g")
        local_output="#include (<|\")$local_output\.(hpp|hip\.hpp|dp\.hpp|cuh)(\"|>)"
        if [ "${remove_test}" = "true" ]; then
            local_output=$(echo "${local_output}" | sed -E "s~test/~~g")
        fi
    else
        local_output="#include (<|\")$fix_include(\"|>)"
    fi
}

GINKGO_LICENSE_BEACON="******************************<GINKGO LICENSE>******************************"

CONTENT="content.cpp" # Store the residual part (start from namespace)
BEFORE="before.cpp" # Store the main header and the #ifdef/#define of header file
HAS_HIP_RUNTIME="false"
DURING_LICENSE="false"
INCLUDE_REGEX="^#include.*"
INCLUDE_INC="\.inc"
MAIN_PART_MATCH=""

# FORCE_TOP_ON/OFF is only valid before other #include
FORCE_TOP_ON="// force-top: on"
FORCE_TOP_OFF="// force-top: off"
FORCE_TOP="force_top"
DURING_FORCE_TOP="false"

get_include_regex $1 MAIN_PART_MATCH
HEADER_DEF=$(get_header_def $1)

IFNDEF=""
DEFINE=""
IFNDEF_REGEX="^#ifndef GKO_"
DEFINE_REGEX="^#define GKO_"
HEADER_REGEX="\.(hpp|cuh)"
SKIP="true"
START_BLOCK_REX="^(#if| *\/\*)"
END_BLOCK_REX="^#endif|\*\/$"
ENDIF_REX="^#endif"
IN_BLOCK=0
KEEP_LINES=0
LAST_NONEMPTY=""
ALARM=""
COMMENT_REGEX="^ *\/\/"
CONSIDER_REGEX="${START_BLOCK_REX}|${END_BLOCK_REX}|${COMMENT_REGEX}|${INCLUDE_REGEX}"

# This part capture the main header and give the possible fail arrangement information
while IFS='' read -r line || [ -n "$line" ]; do
    if [ "${line}" = '#include "hip/hip_runtime.h"' ] && [ "${SKIP}" = "true" ]; then
        HAS_HIP_RUNTIME="true"
    elif [ "${line}" = "/*${GINKGO_LICENSE_BEACON}" ] || [ "${DURING_LICENSE}" = "true" ]; then
        DURING_LICENSE="true"
        if [ "${line}" = "${GINKGO_LICENSE_BEACON}*/" ]; then
            DURING_LICENSE="false"
        fi
    elif [ "${SKIP}" = "true" ] && ([ "$line" = "${FORCE_TOP_ON}" ] || [ "${DURING_FORCE_TOP}" = "true" ]); then
        DURING_FORCE_TOP="true"
        if [ "$line" = "${FORCE_TOP_OFF}" ]; then
            DURING_FORCE_TOP="false"
        fi
        if [[ "${line}" =~ $INCLUDE_REGEX ]]; then
            line="$(convert_header ${line})"
        fi
        echo "$line" >> "${FORCE_TOP}"
    elif [ -z "${line}" ] && [ "${SKIP}" = "true" ]; then
    # Ignore all empty lines between LICENSE and Header
        :
    else
        if [[ "${line}" =~ $INCLUDE_REGEX ]]; then
            line="$(convert_header ${line})"
        fi
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
        elif [ -z "${MAIN_PART_MATCH}" ] || [[ ! "${line}" =~ ${MAIN_PART_MATCH} ]] || [[ "${IN_BLOCK}" -gt 0 ]]; then
            echo "${line}" >> "${CONTENT}"
            SKIP="false"
            if [[ "${line}" =~ $START_BLOCK_REX ]]; then
                # keep everythin in #if block and /* block
                IN_BLOCK=$((IN_BLOCK+1))
                if [ -z "${ALARM}" ]; then
                    ALARM="set"
                fi
            fi
            if [[ "${IN_BLOCK}" = "0" ]] && [ ! -z "${line}" ] && [[ ! "${line}" =~ ${CONSIDER_REGEX} ]]; then
                if [ "${ALARM}" = "set" ]; then
                    ALARM="true"
                elif [ -z "${ALARM}" ]; then
                    ALARM="false"
                fi
            fi
            if [[ "${line}" =~ $END_BLOCK_REX ]]; then
                IN_BLOCK=$((IN_BLOCK-1))
            fi
        else
            echo "${line}" >> ${BEFORE}
        fi
    fi
done < $1
if [ "${ALARM}" = "true" ]; then
    echo "Warning $1: sorting is probably incorrect"
fi

# Wrtie license
echo "/*${GINKGO_LICENSE_BEACON}" > $1
cat LICENSE >> $1
echo "${GINKGO_LICENSE_BEACON}*/" >> $1
echo "" >> $1

# Wrtie the definition of header according to path
if [ ! -z "${IFNDEF}" ] && [ ! -z "${DEFINE}" ]; then
    IFNDEF="#ifndef ${HEADER_DEF}"
    DEFINE="#define ${HEADER_DEF}"
elif [ -z "${IFNDEF}" ] && [ -z "${DEFINE}" ]; then
    :
else
    echo "Warning $1: only #ifndef GKO_ or #define GKO_ is in the header"
fi
if [ ! -z "${IFNDEF}" ]; then
    echo "${IFNDEF}" >> $1
fi
if [ ! -z "${DEFINE}" ]; then
    echo "${DEFINE}" >> $1
    echo "" >> $1
    echo "" >> $1
fi

# Write the force-top header
if [ -f "${FORCE_TOP}" ]; then
    cat "${FORCE_TOP}" >> $1
    echo "" >> $1
    echo "" >> $1
    rm "${FORCE_TOP}"
fi

# Write the main header and give warnning if there are multiple matches
if [ -f "${BEFORE}" ]; then
    # sort or remove the duplication
    "${CLANG_FORMAT}" -i -style=file ${BEFORE}
    if [ $(wc -l < ${BEFORE}) -gt "1" ]; then
        echo "Warning $1: there are multiple main header matchings"
    fi
    cat ${BEFORE} >> $1
    if [ -f "${CONTENT}" ]; then
        echo "" >> $1
        echo "" >> $1
    fi
    rm "${BEFORE}"
fi

# Arrange the remain files and give
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
            echo "Warning $1: Found the begin header_def but did not find the end of header_def"
            cat temp > ${CONTENT}
        fi
    else
        cat temp > "${CONTENT}"
    fi
    "${CLANG_FORMAT}" -i -style=file "${CONTENT}"
    rm temp
    remove_regroup
    PREV_INC=0
    IN_IF="false"
    SKIP="true"
    while IFS='' read -r line; do
        # Skip the empty line in the beginning
        if [ "${SKIP}" = "true" ] && [[ -z "${line}" ]]; then
            continue
        else
            SKIP="false"
        fi
        # Insert content with correct number empty lines
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
    done < "${CONTENT}"
    rm "${CONTENT}"
fi
