#!/bin/bash
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
    if [ "${EXTRACT_KERNEL}" = "false" ] && ([ "${line}" = "/*${GINKGO_LICENSE_BEACON}" ] ||  [ "${DURING_LICENSE}" = "true" ]); then
        DURING_LICENSE="true"
        if [ "${line}" = "${GINKGO_LICENSE_BEACON}*/" ]; then
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
    # echo "Handle___ ${line}"
    if [[ "$line" =~ ${START_BLOCK_REX} ]] || [[ "${IN_BLOCK}" -gt 0 ]]; then
        if [[ "$line" =~ ${START_BLOCK_REX} ]]; then
            IN_BLOCK=$((IN_BLOCK+1))
        fi
        if [[ "$line" =~ ${END_BLOCK_REX} ]]; then
            IN_BLOCK=$((IN_BLOCK-1))
        fi
        # echo ""
        # echo "IN BLOCK ${IN_BLOCK}"
        # output to new file
        continue
    fi
    # echo "Handle ${line}"
    # handle comments
    if [[ "${line}" =~ $FUNCTION_START ]] || [[ $DURING_FUNCNAME = "true" ]]; then
        # echo "line ${line}"
        # echo "${FUNCTION_NAME_END}"
        DURING_FUNCNAME="true"
        FUNCTION_HANDLE="${FUNCTION_HANDLE} $line"
        if [[ "${line}" =~ ${FUNCTION_NAME_END} ]]; then
            # echo "end"
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
    # echo "Handle ${line}"
    if [ -n "${FUNCTION_HANDLE}" ] && [[ ${DURING_FUNCNAME} = "false" ]]; then
        if [[ "${line}" =~ ${SCOPE_START} ]]; then
            IN_FUNC=$((IN_FUNC+1))
        fi
        if [[ "${line}" =~ ${SCOPE_END} ]]; then
            IN_FUNC=$((IN_FUNC-1))
        fi
        # echo "IN FUNC ${IN_FUNC}"

        # make sure the function is end
        if [[ "${IN_FUNC}" -eq 0 ]]; then
            # echo "check ${FUNCTION_HANDLE}"

            if [[ "${FUNCTION_HANDLE}" =~ $CHECK_GLOBAL_KEYWORD ]]; then
                echo ""
                # echo "${FUNCTION_HANDLE}"
                # remove additional space
                FUNCTION_HANDLE=$(echo "${FUNCTION_HANDLE}" | sed -E 's/ +/ /g;')
                # echo "->"
                # echo "${FUNCTION_HANDLE}"
                # echo "->"

                if [[ "${FUNCTION_HANDLE}" =~ $ANAYSIS_FUNC ]]; then
                    TEMPLATE="${BASH_REMATCH[1]}"
                    TEMPLATE_CONTENT="${BASH_REMATCH[2]}"
                    NAME="${BASH_REMATCH[3]}"
                    VARIABLE="${BASH_REMATCH[4]}"
                    VARIABLE=$(echo ${VARIABLE} | sed 's/__restrict__ //g')
                    VAR_INPUT=$(extract_varname "${VARIABLE}")
                    TEMPLATE_INPUT=$(extract_varname "${TEMPLATE_CONTENT}")
                    if [ -n "${TEMPLATE_INPUT}" ]; then
                        TEMPLATE_INPUT="<${TEMPLATE_INPUT}>"
                    fi
                    echo "${TEMPLATE} void ${NAME}${HOST_SUFFIX} (dim3 grid, dim3 block, size_t dynamic_shared_memory, cudaStream_t stream, ${VARIABLE}) {
                        /*KEEP*/${NAME}${TEMPLATE_INPUT}<<<grid, block, dynamic_shared_memory, stream>>>(${VAR_INPUT});
                        }"
                    echo "${NAME} -> ${NAME}${HOST_SUFFIX}" >> ${MAP_FILE}
                fi
                # echo ""
                # check the property
                # extract template
                # maybe remove any [[ ]]
                # extract function name
                # extract function variables

                # check Config
                # extract bool, int, size_type, typename before Config
                # add one function like original one
                # add selection_config macro
                # need to keep the map
            fi
            FUNCTION_HANDLE=""
        fi
    fi


done < "$1"

# Maybe it only works in Linux
sort "${MAP_FILE}" | uniq > "${MAP_FILE}_temp"
mv "${MAP_FILE}_temp"  "${MAP_FILE}"
