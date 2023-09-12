#!/bin/bash

# add_host_function adds a host function to wrap the cuda kernel call with template and parameter configuration.
#
# For example
# ````
# template <int config = 1, int info, typename ValueType>
# __global__ kernel(ValueType a) {...}
# ```
# add_host_function will add another host function with the same template and calling the cuda call
# ```
# template <int config = 1, int info, typename ValueType>
# void kernel_AUTOHOSTFUNC(dim3 grid, dim3 block, size_type dynamic_shared_memory, cudaStream_t stream, ValueType a) {
#     /*KEEP*/kernel<config, info><<<grid, block, dynamic_shared_memory, stream>>>(a);
# }
# ```
# _AUTOHOSTFUNC and /*KEEP*/ is internal step and they are removed in the end.
# It will use the same template as original cuda call and pust the kernel args into input args.
# Note. This script does not translate original cuda kernel call to corresponding call.
#       convert_source.sh will handle it later.


SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
source "${SCRIPT_DIR}/shared.sh"

extract_varname() {
    local str="$1"
    # local GET_PARAM=" *([^ ]*) *$"
    # Need to remove the  = ....
    # note. it only remove the simple one
    local GET_PARAM=" *([^ =\*]*) *(= *.*)* *$"
    local parameter=""
    local temp=""
    IFS=',' read -ra par <<< "$str"
    for var in "${par[@]}"; do
        if [ -n "${temp}" ]; then
            temp="${temp},"
        fi
        temp="${temp}${var}"
        # only handle one pair <> currently
        if [[ "${temp}" =~ "<" ]] && [[ ! "${temp}" =~ ">" ]]; then
            continue
        fi
        # If the string contains typename, do not extract it.
        # It should automatically be decided from argument
        # Also need to ignore = ...
        if [[ "${temp}" =~ "typename" ]]; then
            :
        elif [[  "${temp}" =~ $GET_PARAM ]]; then
            if [ -n "${parameter}" ]; then
                parameter="${parameter}, "
            fi
            parameter="${parameter}${BASH_REMATCH[1]}"
        fi
        temp=""
    done
    echo "$parameter"
}


GLOBAL_KEYWORD="__global__"
TEMPLATE_REGEX="^ *template <*"
FUNCTION_START="^ *(template *<|${GLOBAL_KEYWORD}|void)"
FUNCTION_NAME_END=".*\{.*"
SCOPE_START="${FUNCTION_NAME_END}"
SCOPE_END=".*\}.*"
CHECK_GLOBAL_KEYWORD=".*${GLOBAL_KEYWORD}.*"
FUNCTION_HANDLE=""
DURING_FUNCNAME="false"
ANAYSIS_FUNC=" *(template *<(.*)>)?.* (.*)\((.*)\)"
START_BLOCK_REX="^( *\/\*| *\/\/)"
END_BLOCK_REX="\*\/$| *\/\/"
IN_BLOCK=0
IN_FUNC=0
STORE_LINE=""
STORE_REGEX="__ *$"
EXTRACT_KERNEL="false"
DURING_LICENSE="false"
SKIP="false"

rm "${MAP_FILE}"
while IFS='' read -r line || [ -n "$line" ]; do
    if [ "${EXTRACT_KERNEL}" = "false" ] && ([[ "${line}" =~ {GINKGO_LICENSE_BEGIN} ]] ||  [ "${DURING_LICENSE}" = "true" ]); then
        DURING_LICENSE="true"
        if [[ "${line}" =~ ${GINKGO_LICENSE_END} ]]; then
            DURING_LICENSE="false"
            SKIP="true"
        fi
        continue
    fi
    # When do not need the license, do not need the space between license and other codes, neither.
    if [ ${SKIP} = "true" ] && [ -z "${line}" ]; then
        continue
    fi
    SKIP="false"
    # It prints the original text into new file.
    if [[ "$line" =~ ${STORE_REGEX} ]]; then
        STORE_LINE="${STORE_LINE} ${line}"
    elif [[ -n "${STORE_LINE}" ]]; then
        echo "${STORE_LINE} ${line}"
        STORE_LINE=""
    else
        echo "${line}"
    fi

    # handle comments
    if [[ "$line" =~ ${START_BLOCK_REX} ]] || [[ "${IN_BLOCK}" -gt 0 ]]; then
        if [[ "$line" =~ ${START_BLOCK_REX} ]]; then
            IN_BLOCK=$((IN_BLOCK+1))
        fi
        if [[ "$line" =~ ${END_BLOCK_REX} ]]; then
            IN_BLOCK=$((IN_BLOCK-1))
        fi
        # output to new file
        continue
    fi
    # handle functions
    if [[ "${line}" =~ $FUNCTION_START ]] || [[ $DURING_FUNCNAME = "true" ]]; then
        DURING_FUNCNAME="true"
        FUNCTION_HANDLE="${FUNCTION_HANDLE} $line"
        if [[ "${line}" =~ ${FUNCTION_NAME_END} ]]; then
            DURING_FUNCNAME="false"
        fi
        if [[ "${line}" =~ ${SCOPE_START} ]]; then
            IN_FUNC=$((IN_FUNC+1))
        fi
        if [[ "${line}" =~ ${SCOPE_END} ]]; then
            IN_FUNC=$((IN_FUNC-1))
        fi
        # output to new file
        continue
    fi

    if [ -n "${FUNCTION_HANDLE}" ] && [[ ${DURING_FUNCNAME} = "false" ]]; then
        if [[ "${line}" =~ ${SCOPE_START} ]]; then
            IN_FUNC=$((IN_FUNC+1))
        fi
        if [[ "${line}" =~ ${SCOPE_END} ]]; then
            IN_FUNC=$((IN_FUNC-1))
        fi

        # make sure the function is end
        if [[ "${IN_FUNC}" -eq 0 ]]; then

            if [[ "${FUNCTION_HANDLE}" =~ $CHECK_GLOBAL_KEYWORD ]]; then
                echo ""
                # remove additional space
                FUNCTION_HANDLE=$(echo "${FUNCTION_HANDLE}" | sed -E 's/ +/ /g;')

                if [[ "${FUNCTION_HANDLE}" =~ $ANAYSIS_FUNC ]]; then
                    TEMPLATE="${BASH_REMATCH[1]}"
                    TEMPLATE_CONTENT="${BASH_REMATCH[2]}"
                    NAME="${BASH_REMATCH[3]}"
                    VARIABLE="${BASH_REMATCH[4]}"
                    VARIABLE=$(echo "${VARIABLE}" | sed 's/__restrict__ //g')
                    VAR_INPUT=$(extract_varname "${VARIABLE}")
                    TEMPLATE_INPUT=$(extract_varname "${TEMPLATE_CONTENT}")
                    if [ -n "${TEMPLATE_INPUT}" ]; then
                        TEMPLATE_INPUT="<${TEMPLATE_INPUT}>"
                    fi
                    echo "${TEMPLATE} void ${NAME}${HOST_SUFFIX} (dim3 grid, dim3 block, size_type dynamic_shared_memory, cudaStream_t queue, ${VARIABLE}) {
                        /*KEEP*/${NAME}${TEMPLATE_INPUT}<<<grid, block, dynamic_shared_memory, queue>>>(${VAR_INPUT});
                        }"
                    echo "${NAME} -> ${NAME}${HOST_SUFFIX}" >> ${MAP_FILE}
                fi
            fi
            FUNCTION_HANDLE=""
        fi
    fi


done < "$1"

# Maybe it only works in Linux
sort "${MAP_FILE}" | uniq > "${MAP_FILE}_temp"
mv "${MAP_FILE}_temp"  "${MAP_FILE}"
