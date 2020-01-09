#!/bin/bash

get_config() {
# declare -a file_regex
# declare -a path_prefix
# declare -a core_suffix
local file="$1"
declare -n core_suffix=$2
declare -n path_prefix=$3
declare -n path_ignore=$4
declare -n remove_test=$5
declare -n fix_include=$6
core_suffix=""
path_prefix=""
path_ignore="0"
fix_include=""
remove_test="false"
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
            return
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

}

get_include_regex() {
    local file=$1
    local core_suffix="$2"
    local path_prefix="$3"
    local path_ignore="$4"
    local remove_test="$5"
    local fix_include="$6"
    declare -n output="$7"
    # echo "${fix_include}"
    if [ -z "${fix_include}" ]; then
        local path_regex="([a-zA-Z_]*\/){${path_ignore}}(.*)\.(cpp|hpp|cu|cuh)"
        if [ ! -z "${path_prefix}" ]; then
            path_prefix="${path_prefix}/"
        fi
        # echo "$output"
        # echo "s~$path_regex~$path_prefix\2~g"
        output=$(echo "${file}" | sed -E "s~\.hip~~g;s~$path_regex~$path_prefix\2~g")
        # echo "$output"
        output=$(echo "${output}" | sed -E "s~$core_suffix$~~g")
        # echo "$output"
        output="#include (<|\")$output\.(hpp|hip\.hpp|cuh)(\"|>)"
        if [ "${remove_test}" = "true" ]; then
            output=$(echo "${output}" | sed -E "s~test/~~g")
        fi
    else
        output="#include (<|\")$fix_include(\"|>)"
    fi
}


# declare -a file_regex2
# declare -a path_prefix2
# declare -a core_suffix2
# read_config file_regex2 path_prefix2 core_suffix2
# declare -p file_regex2
# declare -p path_prefix2
# declare -p core_suffix2

g_core_suffix=""
g_path_prefix=""
g_path_ignore="0"
g_remove_test="false"
g_fix_include=""
get_config $1 g_core_suffix g_path_prefix g_path_ignore g_remove_test g_fix_include
# echo "$1 $g_core_suffix $g_path_prefix $g_path_ignore"
g_output=""
get_include_regex $1 "$g_core_suffix" "$g_path_prefix" "$g_path_ignore" "$g_remove_test" "$g_fix_include" g_output
echo "$g_output"
