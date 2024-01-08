#!/usr/bin/env bash

CLANG_FORMAT=${CLANG_FORMAT:="clang-format"}

convert_header () {
    local regex="^(#include )(<|\")(.*)(\"|>)$"
    local jacobi_regex="^(cuda|hip|dpcpp)\/preconditioner\/jacobi_common(\.hip)?\.hpp"
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
	    elif [[ "${header_file}" =~ ${jacobi_regex} ]]; then
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
	# Used to get rid of \r in Windows
        def=$(echo "GKO_${def^^}_")
        echo "$def"
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
#   - RemoveTest: "false/true"                  (default "false")
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
            if [ "$FORMAT_HEADER_DEBUG" = "1" ]; then
                echo "DEBUG: Checking pattern $line"
            fi
            if [[ "$match" = "true" ]]; then
                break
            elif [[ $file =~ $file_regex ]]; then
                if [ "$FORMAT_HEADER_DEBUG" = "1" ]; then
                    echo "DEBUG: Matching pattern $line for $file"
                fi
                match="true"
            fi
        elif [ "$match" = "true" ]; then
            if [[ "$line" =~ $path_prefix_regex ]]; then
                path_prefix="${BASH_REMATCH[1]}"
                if [ "$FORMAT_HEADER_DEBUG" = "1" ]; then
                    echo "DEBUG: Path prefix set to $path_prefix"
                fi
            elif [[ "$line" =~ $core_suffix_regex ]]; then
                core_suffix="${BASH_REMATCH[1]}"
                if [ "$FORMAT_HEADER_DEBUG" = "1" ]; then
                    echo "DEBUG: Core suffix set to $core_suffix"
                fi
            elif [[ "$line" =~ $path_ignore_regex ]]; then
                path_ignore="${BASH_REMATCH[1]}"
                if [ "$FORMAT_HEADER_DEBUG" = "1" ]; then
                    echo "DEBUG: Ignoring $path_ignore top-level dirs"
                fi
            elif [[ "$line" =~ $fix_include_regex ]]; then
                fix_include="${BASH_REMATCH[1]}"
                if [ "$FORMAT_HEADER_DEBUG" = "1" ]; then
                    echo "DEBUG: Fixed include $fix_include"
                fi
            elif [[ "$line" =~ $remove_test_regex ]]; then
                remove_test="${BASH_REMATCH[1]}"
                if [ "$FORMAT_HEADER_DEBUG" = "1" ]; then
                    echo "DEBUG: Remove test $remove_test"
                fi
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
        if [ "$FORMAT_HEADER_DEBUG" = "1" ]; then
            echo "DEBUG: Handling $file"
        fi
        local_output=$(echo "${file}" | sed -E "s~\.(hip|dp)~~g;s~$path_regex~$path_prefix\2~g")
        if [ "$FORMAT_HEADER_DEBUG" = "1" ]; then
            echo "DEBUG: After removing path_ignore and path_prefix: $local_output"
        fi
        local_output=$(echo "${local_output}" | sed -E "s~$core_suffix$~~g")
        if [ "$FORMAT_HEADER_DEBUG" = "1" ]; then
            echo "DEBUG: After removing core_suffix: $local_output"
        fi
        local_output="#include (<|\")$local_output\.(hpp|hip\.hpp|dp\.hpp|cuh)(\"|>)"
        if [ "${remove_test}" = "true" ]; then
            local_output=$(echo "${local_output}" | sed -E "s~test/~~g")
            if [ "$FORMAT_HEADER_DEBUG" = "1" ]; then
                echo "DEBUG: After removing test: ${local_output}"
            fi
        fi
    else
        if [ "$FORMAT_HEADER_DEBUG" = "1" ]; then
            echo "DEBUG: Fixing include $fix_include to the top"
        fi
        local_output="#include (<|\")$fix_include(\"|>)"
    fi
}

# Test if required commands are present on the system:
if ! command -v "$CLANG_FORMAT" &> /dev/null; then
    echo "The command 'clang-format' is required for this script to work, but not supported by your system. It can be set via environment parameter CLANG_FORMAT=<clang-format path>" 1>&2
    exit 1
fi

# Test the command on MacOS
if ! declare -n &> /dev/null; then
    echo "The command 'declare' needs to support the '-n' option. Please update bash or use 'brew install bash' if on MacOS" 1>&2
    exit 1
fi

touch .dummy_file
if ! sed -i 's///g' .dummy_file &> /dev/null; then
    echo "The command 'sed' needs to support the '-i' option without suffix. Please use gnu sed or use 'brew install gnu-sed' if on MacOS" 1>&2
    rm .dummy_file
    exit 1
fi

if ! head -n -1 .dummy_file &> /dev/null; then
    echo "The command 'head' needs to support '-NUM' option, Please use gnu head or use 'brew install coreutils' if on MacOS" 1>&2
    rm .dummy_file
    exit 1
fi
rm .dummy_file

for current_file in $@; do
    if [ -z "${current_file}" ]; then
        echo "Usage: $0 path/to/fileA path/to/fileB ..."
        exit 1
    fi

    if [ ! -f "${current_file}" ]; then
        echo "${current_file} does not exist or it is not a file."
        exit 1
    fi

    GINKGO_LICENSE_BEGIN="// SPDX-FileCopyrightText:"
    GINKGO_LICENSE_END="// SPDX-License-Identifier:"

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

    get_include_regex "${current_file}" MAIN_PART_MATCH
    HEADER_DEF=$(get_header_def "${current_file}")

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
        if [[ "${line}" =~ ${GINKGO_LICENSE_BEGIN}  ]] || [ "${DURING_LICENSE}" = "true" ]; then
            DURING_LICENSE="true"
            if [[ "${line}" =~ ${GINKGO_LICENSE_END} ]]; then
                DURING_LICENSE="false"
                SKIP="true"
            fi
        elif [ "${SKIP}" = "true" ] && ([ "$line" = "${FORCE_TOP_ON}" ] || [ "${DURING_FORCE_TOP}" = "true" ]); then
            DURING_FORCE_TOP="true"
            if [ "$line" = "${FORCE_TOP_OFF}" ]; then
                DURING_FORCE_TOP="false"
            fi
            if [[ "${line}" =~ $INCLUDE_REGEX ]]; then
                line="$(convert_header "${line}")"
            fi
            echo "$line" >> "${FORCE_TOP}"
        elif [ -z "${line}" ] && [ "${SKIP}" = "true" ]; then
        # Ignore all empty lines between LICENSE and Header
            :
        else
            if [[ "${line}" =~ $INCLUDE_REGEX ]]; then
                line="$(convert_header "${line}")"
            fi
            if [ -z "${line}" ]; then
                KEEP_LINES=$((KEEP_LINES+1))
            else
                LAST_NONEMPTY="${line}"
                KEEP_LINES=0
            fi
            if [[ "${current_file}" =~ ${HEADER_REGEX} ]] && [[ "${line}" =~ ${IFNDEF_REGEX} ]] && [ "${SKIP}" = "true" ] && [ -z "${DEFINE}" ]; then
                IFNDEF="${line}"
            elif [[ "${current_file}" =~ ${HEADER_REGEX} ]] && [[ "${line}" =~ ${DEFINE_REGEX} ]] && [ "${SKIP}" = "true" ] && [ -n "${IFNDEF}" ]; then
                DEFINE="${line}"
            elif [ -z "${MAIN_PART_MATCH}" ] || [[ ! "${line}" =~ ${MAIN_PART_MATCH} ]] || [[ "${IN_BLOCK}" -gt 0 ]]; then
                echo "${line}" >> "${CONTENT}"
                SKIP="false"
                if [[ "${line}" =~ $START_BLOCK_REX ]]; then
                    # keep everything in #if block and /* block
                    IN_BLOCK=$((IN_BLOCK+1))
                    if [ -z "${ALARM}" ]; then
                        ALARM="set"
                    fi
                fi
                if [[ "${IN_BLOCK}" = "0" ]] && [ -n "${line}" ] && [[ ! "${line}" =~ ${CONSIDER_REGEX} ]]; then
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
    done < "${current_file}"
    if [ "${ALARM}" = "true" ]; then
        echo "Warning ${current_file}: sorting is probably incorrect"
    fi

    # Write license
    CURRENT_YEAR=$(date +%Y)
    echo "${GINKGO_LICENSE_BEGIN} 2017 - ${CURRENT_YEAR} The Ginkgo authors" > "${current_file}"
    echo "//" >> "${current_file}"
    echo "${GINKGO_LICENSE_END} BSD-3-Clause" >> "${current_file}"
    echo "" >> "${current_file}"

    # Write the definition of header according to path
    if [ -n "${IFNDEF}" ] && [ -n "${DEFINE}" ]; then
        IFNDEF="#ifndef ${HEADER_DEF}"
        DEFINE="#define ${HEADER_DEF}"
    elif [ -z "${IFNDEF}" ] && [ -z "${DEFINE}" ]; then
        :
    else
        echo "Warning ${current_file}: only #ifndef GKO_ or #define GKO_ is in the header"
    fi
    if [ -n "${IFNDEF}" ]; then
        echo "${IFNDEF}" >> "${current_file}"
    fi
    if [ -n "${DEFINE}" ]; then
        echo "${DEFINE}" >> "${current_file}"
        echo "" >> "${current_file}"
        echo "" >> "${current_file}"
    fi

    # Write the force-top header
    if [ -f "${FORCE_TOP}" ]; then
        cat "${FORCE_TOP}" >> "${current_file}"
        echo "" >> "${current_file}"
        echo "" >> "${current_file}"
        rm "${FORCE_TOP}"
    fi

    # Write the main header and give warnning if there are multiple matches
    if [ -f "${BEFORE}" ]; then
        # sort or remove the duplication
        "${CLANG_FORMAT}" -i -style=file ${BEFORE}
        if [ "$(wc -l < ${BEFORE})" -gt "1" ]; then
            echo "Warning ${current_file}: there are multiple main header matchings"
        fi
        cat ${BEFORE} >> "${current_file}"
        if [ -f "${CONTENT}" ]; then
            echo "" >> "${current_file}"
            echo "" >> "${current_file}"
        fi
        rm "${BEFORE}"
    fi

    # Arrange the remain files and give
    if [ -f "${CONTENT}" ]; then
        add_regroup
        head -n -${KEEP_LINES} ${CONTENT} >> temp
        if [ -n "${IFNDEF}" ] && [ -n "${DEFINE}" ]; then
            # Ignore the last line #endif
            if [[ "${LAST_NONEMPTY}" =~ $ENDIF_REX ]]; then
                head -n -1 temp > ${CONTENT}
                echo "#endif  // $HEADER_DEF" >> ${CONTENT}
            else
                echo "Warning ${current_file}: Found the begin header_def but did not find the end of header_def"
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
                    echo "" >> "${current_file}"
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
            echo "${line}" >> "${current_file}"
        done < "${CONTENT}"
        rm "${CONTENT}"
    fi
done
